#!/usr/bin/env python3
"""Lead time before the bad tool call.

Horizon 0 is the bad call and is not the lead time. See AGENTS.md.

    python -u scripts/01_lead_curve.py --mark-only
    python -u scripts/01_lead_curve.py --extract-only --smoke 8
    python -u scripts/01_lead_curve.py --extract-only
    python -u scripts/01_lead_curve.py --fit-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

WEEK6 = Path(__file__).resolve().parents[1]
WEEK3 = WEEK6.parent / "week3-noninvasive-probe"
WEEK1 = WEEK6.parent / "week1-dyl-vs-oa"
sys.path.insert(0, str(WEEK6 / "src"))
sys.path.insert(0, str(WEEK3 / "src"))
sys.path.insert(0, str(WEEK1 / "src"))

from early_detection.mark import mark_actions, reader_text

LAYER = 58
MAX_LENGTH = 4096
RESULTS = WEEK6 / "results"
MARKED = RESULTS / "marked.json"
FULL_CACHE = RESULTS / "lead_cache"
SMOKE_CACHE = RESULTS / "lead_cache_smoke"
CURVE_JSON = RESULTS / "lead_curve.json"
CURVE_MD = WEEK6 / "CURVE.md"
LOG = WEEK6 / "LOG.md"


def _load_env() -> None:
    for path in (
        WEEK1 / "dyl-probe-repro" / ".env",
        WEEK3 / ".env",
        WEEK1 / ".env",
        WEEK6 / ".env",
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


def _log(title: str, body: str = "") -> None:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("# Week 6 run log\n\nAppended by scripts/01_lead_curve.py.\n")
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {stamp} — {title}\n\n{body.rstrip()}\n")
        handle.flush()
        os.fsync(handle.fileno())


def resolve_data_dir(arg: str | None) -> Path:
    candidates = []
    if arg:
        candidates.append(Path(arg))
    candidates.append(WEEK6 / "data" / "docent_per_action")
    candidates.append(WEEK6.parents[2] / "steering-vectors-control" / "data" / "docent_per_action")
    for path in candidates:
        if (path / "attack").is_dir() and (path / "honest").is_dir():
            return path
    raise SystemExit(
        "No per-action JSON found. Pass --data-dir pointing at docent_per_action "
        "(attack/ and honest/ full of JSON files). If DOCENT_API_KEY is set, download it with "
        "steering-vectors-control/scripts/download_docent_trajectories.py. Do not invent trajectories."
    )


def load_actions(data_dir: Path) -> list[dict]:
    actions = []
    for kind in ("attack", "honest"):
        for path in sorted((data_dir / kind).glob("*.json")):
            actions.append(json.loads(path.read_text()))
    if not actions:
        raise SystemExit(f"{data_dir} has no JSON files")
    return actions


def _index_actions(actions: list[dict]) -> dict[str, dict]:
    from early_detection.mark import row_id

    return {row_id(action): action for action in actions}


def write_marked(payload: dict) -> None:
    MARKED.parent.mkdir(parents=True, exist_ok=True)
    MARKED.write_text(json.dumps(payload))


def read_marked() -> dict:
    if not MARKED.exists():
        raise SystemExit(f"missing {MARKED}. Run --mark-only first.")
    return json.loads(MARKED.read_text())


def mark(data_dir: Path) -> dict:
    payload = mark_actions(load_actions(data_dir))
    write_marked(payload)
    examples = []
    for item in payload["dropped"][:20]:
        commands = " | ".join(item["commands"]) or "(no command text)"
        examples.append(f"- {item['collection']} {item['task_id']} ({item['side_task']}): {commands}")
    body = (
        f"Attack trajectories: {payload['n_attack_trajectories']}\n"
        f"Matched: {payload['n_matched']}\n"
        f"Match rate: {payload['match_rate']:.3f}\n"
        f"Kept rows: {len(payload['rows'])}\n"
        + ("\n".join(examples) if examples else "No unmatched trajectories.")
    )
    print(body, flush=True)
    _log("mark", body)
    if not payload["enough"]:
        raise SystemExit(
            f"Match rate {payload['match_rate']:.3f} is under 0.5, or there were no attack trajectories. "
            "Not loading Gemma. The unmatched commands are in LOG.md."
        )
    return payload


def _read_cache(cache_dir: Path) -> dict[str, np.ndarray]:
    path = cache_dir / "activations.npz"
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as data:
        return {str(row_id): np.array(vector) for row_id, vector in zip(data["ids"], data["activations"])}


def _write_cache(cache_dir: Path, acts: dict[str, np.ndarray]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids = list(acts)
    stacked = np.stack([acts[row_id] for row_id in ids]).astype(np.float32)
    np.savez_compressed(cache_dir / "activations.npz", ids=np.array(ids), activations=stacked)


def _left_truncate(tokenizer, texts: list[str], max_length: int) -> tuple[list[list[int]], int]:
    kept = []
    lost = 0
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = [int(token) for token in ids]
        if not ids:
            raise RuntimeError("empty reader text")
        if len(ids) > max_length:
            ids = ids[-max_length:]
            lost += 0  # the end token is kept; this is not a lost read
        kept.append(ids)
    return kept, lost


def extract(data_dir: Path, cache_dir: Path, row_ids: list[str], batch_size: int, max_length: int) -> None:
    import torch
    from dyl_vs_oa.model import load_subject

    by_id = _index_actions(load_actions(data_dir))
    missing = [row_id for row_id in row_ids if row_id not in by_id]
    if missing:
        raise SystemExit(f"{len(missing)} marked rows are not in {data_dir}")
    acts = _read_cache(cache_dir)
    todo = [row_id for row_id in row_ids if row_id not in acts]
    print(f"cached {len(acts)}; extracting {len(todo)}", flush=True)
    if not todo:
        _log("extract", f"cache already complete at {cache_dir} ({len(acts)} rows)")
        return
    _load_env()
    print("loading google/gemma-3-27b-it", flush=True)
    subject = load_subject("google/gemma-3-27b-it")
    tok = subject.tokenizer
    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        texts = [reader_text(by_id[row_id]) for row_id in chunk]
        kept, _lost = _left_truncate(tok, texts, max_length)
        width = max(len(row) for row in kept)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        input_ids = torch.full((len(kept), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(kept):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[index, : len(row)] = 1
        enc = {"input_ids": input_ids.to(subject.device), "attention_mask": attention_mask.to(subject.device)}
        with torch.inference_mode():
            out = subject.model(**enc, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[LAYER + 1]
        last = attention_mask.sum(dim=1) - 1
        for index, row_id in enumerate(chunk):
            vector = hidden[index, int(last[index])].float().cpu().numpy()
            if not np.isfinite(vector).all():
                raise RuntimeError(f"non-finite activation for {row_id}")
            acts[row_id] = vector
        _write_cache(cache_dir, acts)
        print(f"  {min(start + batch_size, len(todo))}/{len(todo)}", flush=True)
    _log("extract", f"wrote {len(acts)} rows to {cache_dir}")


def smoke_ids(rows: list[dict], n_trajectories: int) -> list[str]:
    """A few attack trajectories at horizons 0, 1, and 2, plus a few honest rows. Not the full set."""
    chosen = set()
    for row in rows:
        if row["role"] != "attack":
            continue
        key = (row["collection"], row["task_id"])
        if key in chosen:
            continue
        if len(chosen) >= n_trajectories:
            break
        chosen.add(key)
    attack_ids = [
        row["row_id"]
        for row in rows
        if row["role"] == "attack" and (row["collection"], row["task_id"]) in chosen and row["horizon"] in (0, 1, 2)
    ]
    indexes = {
        row["action_index"]
        for row in rows
        if row["row_id"] in set(attack_ids)
    }
    honest_ids = [row["row_id"] for row in rows if row["role"] == "honest" and row["action_index"] in indexes]
    return attack_ids + honest_ids[:n_trajectories]


def fit(cache_dir: Path) -> None:
    from early_detection.curve import horizon_table, lead_summary, render_curve

    payload = read_marked()
    if not payload["enough"]:
        raise SystemExit("marked.json says the bad-call rule did not match enough trajectories. Refusing to fit.")
    acts = _read_cache(cache_dir)
    if cache_dir == SMOKE_CACHE or cache_dir.name == SMOKE_CACHE.name:
        raise SystemExit("Refusing to fit the smoke cache.")
    table = horizon_table(payload["rows"], acts)
    summary = lead_summary(table)
    result = {"summary": summary, "horizons": table, "n_dropped": len(payload["dropped"]), "n_attack": payload["n_attack_trajectories"]}
    CURVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CURVE_JSON.write_text(json.dumps(result, indent=2))
    text = render_curve(table, summary, len(payload["dropped"]), payload["n_attack_trajectories"])
    CURVE_MD.write_text(text)
    print(text, flush=True)
    _log("fit", text)
    failed = [row for row in table if row["horizon"] >= 1 and row["auroc"] is None and row["n_attack"] >= 10]
    if failed and summary["peak_horizon"] is None:
        raise SystemExit("No earlier horizon was fit. See CURVE.md.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mark-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--smoke", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()
    modes = (args.mark_only, args.extract_only, args.fit_only)
    if sum(bool(mode) for mode in modes) != 1:
        raise SystemExit("pass exactly one of --mark-only, --extract-only, --fit-only")
    try:
        if args.mark_only:
            mark(resolve_data_dir(args.data_dir))
            return
        if args.extract_only:
            data_dir = resolve_data_dir(args.data_dir)
            payload = mark(data_dir)
            if args.smoke:
                cache_dir = SMOKE_CACHE
                ids = smoke_ids(payload["rows"], args.smoke)
                if not ids:
                    raise SystemExit("smoke selected no rows")
            else:
                cache_dir = FULL_CACHE
                ids = [row["row_id"] for row in payload["rows"]]
            extract(data_dir, cache_dir, ids, args.batch_size, args.max_length)
            return
        fit(FULL_CACHE)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            _log("STOPPED", str(exc))
        raise


if __name__ == "__main__":
    main()
