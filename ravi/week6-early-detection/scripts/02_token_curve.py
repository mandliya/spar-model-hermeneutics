#!/usr/bin/env python3
"""Token lead time before the bad reply token. See AGENTS.md.

    python -u scripts/02_token_curve.py --mark-only
    python -u scripts/02_token_curve.py --extract-only --smoke 8
    python -u scripts/02_token_curve.py --extract-only
    python -u scripts/02_token_curve.py --fit-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Set before huggingface_hub is imported. The container disk is ~16 GB.
os.environ["HF_HOME"] = "/dev/shm/hf_home"
os.environ["HF_HUB_CACHE"] = "/dev/shm/hf_home/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/dev/shm/hf_home/hub"

import numpy as np

WEEK6 = Path(__file__).resolve().parents[1]
WEEK3 = WEEK6.parent / "week3-noninvasive-probe"
WEEK1 = WEEK6.parent / "week1-dyl-vs-oa"
sys.path.insert(0, str(WEEK6 / "src"))
sys.path.insert(0, str(WEEK3 / "src"))
sys.path.insert(0, str(WEEK1 / "src"))

from early_detection.token_curve import flavor_tables, render_curve
from early_detection.token_mark import EXPECTED_COUNTS, OFFSETS, reply_index_at_offset, t_star_reply_index
from noninvasive_probe.categories import FLAVOR_ORDER, load_category
from noninvasive_probe.tianq_read import LAYER, reply_token_ids

RESULTS = WEEK6 / "results"
FULL_CACHE = RESULTS / "token_cache"
SMOKE_CACHE = RESULTS / "token_cache_smoke"
CURVE_JSON = RESULTS / "token_curve.json"
CURVE_MD = WEEK6 / "CURVE.md"
LOG = WEEK6 / "LOG.md"
MODEL_NAME = "google/gemma-3-27b-it"
MAX_LENGTH = 8192


def _load_env() -> None:
    for path in (
        WEEK6 / ".env",
        WEEK1 / "dyl-probe-repro" / ".env",
        WEEK3 / ".env",
        WEEK1 / ".env",
        Path("/workspace/.env"),
    ):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    # The container overlay is ~16 GB. Gemma 27B does not fit there.
    os.environ["HF_HOME"] = "/dev/shm/hf_home"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/dev/shm/hf_home/hub"


def _log(title: str, body: str = "") -> None:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("# Week 6 run log\n\n")
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {stamp} — {title}\n\n{body.rstrip()}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _rows(flavors: list[str], max_n: int | None) -> dict[str, list]:
    loaded = {}
    for flavor in flavors:
        loaded[flavor] = load_category(flavor, max_n)
    return loaded


def _check_counts(loaded: dict[str, list]) -> str:
    lines = []
    bad = []
    for flavor, rows in loaded.items():
        n = len(rows)
        n_lie = sum(label for _, label, _ in rows)
        n_honest = n - n_lie
        expected = EXPECTED_COUNTS[flavor]
        ok = (n, n_lie, n_honest) == expected
        lines.append(f"{flavor}: {n} rows, {n_lie} lies, {n_honest} truthful" + ("" if ok else f" EXPECTED {expected}"))
        if not ok:
            bad.append(flavor)
    if bad:
        raise SystemExit("label counts disagree with week 3 for: " + ", ".join(bad))
    return "\n".join(lines)


def _length_report(tokenizer, loaded: dict[str, list]) -> str:
    lines = [
        "t* is reply token 0 for every flavor. The committed act is the opening of the reply.",
        "Offsets 1 and above are empty unless a reply somehow started after token 0, which this rule does not produce.",
    ]
    for flavor, rows in loaded.items():
        lengths = []
        long = 0
        for messages, _label, _uid in rows:
            _ids, _positions, span = reply_token_ids(tokenizer, messages)
            n_reply = span[1] - span[0]
            lengths.append(n_reply)
            if len(_ids) > MAX_LENGTH:
                long += 1
        lengths.sort()
        def pct(q: float) -> int:
            return lengths[min(len(lengths) - 1, int(q * (len(lengths) - 1)))]

        one = sum(n == 1 for n in lengths)
        lines.append(
            f"{flavor}: reply tokens min={lengths[0]} p50={pct(0.5)} p90={pct(0.9)} max={lengths[-1]}; "
            f"one-token={one}; sequences longer than {MAX_LENGTH}={long}; "
            f"in-reply offsets 1–32 before t*: 0 (t* is token 0)"
        )
    return "\n".join(lines)


def mark(flavors: list[str]) -> None:
    _load_env()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is missing. Stage 1 counts can run without it; tokenizer lengths cannot.")
    loaded = _rows(flavors, None)
    counts = _check_counts(loaded)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    lengths = _length_report(tokenizer, loaded)
    body = counts + "\n\n" + lengths
    print(body, flush=True)
    _log("mark tokens", body)


def _cache_path(cache_dir: Path, flavor: str) -> Path:
    return cache_dir / f"{flavor}.npz"


def _read_flavor(path: Path) -> dict | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        acts = {key[4:]: np.array(data[key]) for key in data.files if key.startswith("act_")}
        return {
            "uids": [str(uid) for uid in data["uids"]],
            "y": np.array(data["y"]),
            "reply_len": np.array(data["reply_len"]),
            "n_lost": int(data["n_lost"]),
            "acts": acts,
        }


def _write_flavor(path: Path, uids: list[str], y: list[int], reply_len: list[int], acts: dict[str, list], n_lost: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "uids": np.array(uids),
        "y": np.array(y, dtype=np.int64),
        "reply_len": np.array(reply_len, dtype=np.int64),
        "n_lost": np.array(n_lost),
    }
    for name, rows in acts.items():
        payload["act_" + name] = np.stack(rows).astype(np.float32)
    np.savez_compressed(path, **payload)


def _layer_module(model, layer: int):
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if layers is not None and len(layers) > layer:
            return layers[layer]
    raise RuntimeError(f"could not find decoder layer {layer}")


def extract(flavors: list[str], cache_dir: Path, max_n: int | None, batch_size: int, max_length: int) -> None:
    import torch
    from dyl_vs_oa.model import load_subject

    _load_env()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is missing. Not loading Gemma.")
    loaded = _rows(flavors, max_n)
    if max_n is None:
        _check_counts(loaded)
    _log("extract tokens start", f"cache={cache_dir} flavors={', '.join(flavors)} max_n={max_n} batch={batch_size} max_length={max_length}")
    print("loading " + MODEL_NAME, flush=True)
    subject = load_subject(MODEL_NAME)
    tok = subject.tokenizer
    positions = ["prompt_last"] + [str(offset) for offset in OFFSETS]
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden
        return output

    handle = _layer_module(subject.model, LAYER).register_forward_hook(_hook)
    try:
        for flavor, rows in loaded.items():
            path = _cache_path(cache_dir, flavor)
            have = _read_flavor(path)
            done = set(have["uids"]) if have else set()
            pending = [(messages, label, uid) for messages, label, uid in rows if uid not in done]
            print(f"{flavor}: cached {len(done)}; extracting {len(pending)}", flush=True)
            uids = list(have["uids"]) if have else []
            y = list(map(int, have["y"])) if have else []
            reply_len = list(map(int, have["reply_len"])) if have else []
            acts = {name: [row for row in have["acts"][name]] for name in positions} if have else {name: [] for name in positions}
            n_lost = have["n_lost"] if have else 0
            hidden_size = None
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                meta = []
                kept_ids = []
                for messages, label, uid in chunk:
                    ids, tianq_pos, span = reply_token_ids(tok, messages)
                    t_star = span[0] + t_star_reply_index(flavor, messages[-1]["content"])
                    # Tokens after t* do not change the residual at t* under causal attention.
                    ids = ids[: t_star + 1]
                    wanted = {"prompt_last": tianq_pos["prompt_last"]}
                    for offset in OFFSETS:
                        reply_at = reply_index_at_offset(t_star_reply_index(flavor, messages[-1]["content"]), offset)
                        wanted[str(offset)] = None if reply_at is None else span[0] + reply_at
                    drop = max(0, len(ids) - max_length)
                    meta.append((uid, label, span[1] - span[0], wanted, drop, t_star))
                    kept_ids.append(ids[drop:])
                width = max(len(row) for row in kept_ids)
                pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
                input_ids = torch.full((len(kept_ids), width), pad_id, dtype=torch.long)
                attention_mask = torch.zeros_like(input_ids)
                for index, row in enumerate(kept_ids):
                    input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
                    attention_mask[index, : len(row)] = 1
                enc = {"input_ids": input_ids.to(subject.device), "attention_mask": attention_mask.to(subject.device)}
                with torch.inference_mode():
                    subject.model(**enc, use_cache=False)
                hidden = captured["hidden"]
                hidden_size = int(hidden.shape[-1])
                for index, (uid, label, n_reply, wanted, drop, _t_star) in enumerate(meta):
                    uids.append(uid)
                    y.append(int(label))
                    reply_len.append(n_reply)
                    for name in positions:
                        absolute = wanted[name]
                        if absolute is None:
                            acts[name].append(np.full(hidden_size, np.nan, dtype=np.float32))
                            continue
                        local = absolute - drop
                        if local < 0 or local >= int(attention_mask[index].sum()):
                            n_lost += 1
                            acts[name].append(np.full(hidden_size, np.nan, dtype=np.float32))
                            continue
                        vector = hidden[index, local].float().cpu().numpy()
                        if not np.isfinite(vector).all():
                            raise RuntimeError(f"non-finite activation for {flavor} {uid} {name}")
                        acts[name].append(vector)
                _write_flavor(path, uids, y, reply_len, acts, n_lost)
                print(f"  {flavor} {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)
            if n_lost:
                raise SystemExit(
                    f"{flavor}: lost {n_lost} target tokens to the {max_length} limit. "
                    "Raise --max-length, delete this cache, and extract again. Do not fit it."
                )
        _log("extract tokens", f"wrote {cache_dir} for {', '.join(flavors)}")
    finally:
        handle.remove()


def fit(cache_dir: Path) -> None:
    if cache_dir == SMOKE_CACHE:
        raise SystemExit("Refusing to fit the smoke cache.")
    cache = {}
    for flavor in FLAVOR_ORDER:
        saved = _read_flavor(_cache_path(cache_dir, flavor))
        if saved is None:
            raise SystemExit(f"missing cache for {flavor}. Run --extract-only first.")
        if saved["n_lost"]:
            raise SystemExit(f"{flavor} cache lost {saved['n_lost']} tokens. Delete it and re-extract with a higher --max-length.")
        cache[flavor] = saved
    tables = flavor_tables(cache)
    text = render_curve(tables, monitor_measured=False)
    payload = {"flavors": tables, "monitor_measured": False}
    CURVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CURVE_JSON.write_text(json.dumps(payload, indent=2))
    CURVE_MD.write_text(text)
    print(text, flush=True)
    _log("fit tokens", text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mark-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--smoke", type=int, default=None)
    parser.add_argument("--flavors", default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    modes = (args.mark_only, args.extract_only, args.fit_only)
    if sum(bool(mode) for mode in modes) != 1:
        raise SystemExit("pass exactly one of --mark-only, --extract-only, --fit-only")
    from noninvasive_probe.categories import resolve_flavors

    flavors = resolve_flavors(args.flavors)
    try:
        if args.mark_only:
            mark(flavors)
            return
        if args.extract_only:
            if args.smoke:
                cache_dir = SMOKE_CACHE
                flavors = resolve_flavors("alibi,mask_known_facts")
                extract(flavors, cache_dir, args.smoke, args.batch_size, args.max_length)
            else:
                extract(flavors, FULL_CACHE, None, args.batch_size, args.max_length)
            return
        fit(FULL_CACHE)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            _log("STOPPED", str(exc))
        raise


if __name__ == "__main__":
    main()
