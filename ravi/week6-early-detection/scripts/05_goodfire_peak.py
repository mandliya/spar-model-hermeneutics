#!/usr/bin/env python3
"""Goodfire's passage score, read by Gemma 27B.

The paper scores a passage at the peak of a difference-of-means direction
inside that passage. A passage is one channel of one step. This script does
that on the ImpossibleBench Kimi traces. Gemma is the reader. Kimi's own
activations are the number in the paper (AUROC 0.94 on this benchmark); this
run is the closest comparison that fits on one 80 GB GPU.

The direction is the synthetic difference of means already cached by
scripts/03_goodfire_dom.py, taken at the end of each synthetic readout. The
change from that script is the eval score: peak over the labeled channel,
with the other channels kept only as prefix, and a longer tail (8192 tokens
unless you pass --max-length).

Smoke, then the full run. A killed job resumes from the jsonl. If the smoke
run stops with a span error, do not start the full run.

    cd ravi/week6-early-detection
    python -u scripts/05_goodfire_peak.py --limit 4
    python -u scripts/05_goodfire_peak.py

If the process is killed for memory, rerun the same command with --max-length 4096.
That writes a different jsonl. Do not mix the two.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/dev/shm/hf_home")
os.environ.setdefault("HF_HUB_CACHE", "/dev/shm/hf_home/hub")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/dev/shm/hf_home/hub")

import numpy as np

WEEK6 = Path(__file__).resolve().parents[1]
WEEK3 = WEEK6.parent / "week3-noninvasive-probe"
WEEK1 = WEEK6.parent / "week1-dyl-vs-oa"
for path in (WEEK6 / "src", WEEK3 / "src", WEEK1 / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from early_detection.goodfire_dom import score_report
from early_detection.goodfire_peak import (
    build_turn,
    contrast_row,
    keep_tail,
    prior_messages,
    span_from_prefixes,
    to_gemma_chat,
    trim_prior,
    trim_turn,
)

_spec = importlib.util.spec_from_file_location("goodfire_dom_run", WEEK6 / "scripts" / "03_goodfire_dom.py")
dom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dom)

REPORT_JSON = WEEK6 / "results" / "goodfire_peak.json"
REPORT_MD = WEEK6 / "GOODFIRE_PEAK.md"
PAPER_IMPOSSIBLEBENCH_AUROC = 0.94


def _scores_path(max_length: int) -> Path:
    return WEEK6 / "results" / f"goodfire_peak_m{max_length}.jsonl"


def _load_scores(path: Path) -> tuple[dict | None, dict[str, dict]]:
    header = None
    rows: dict[str, dict] = {}
    if not path.exists():
        return header, rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "header":
            header = row
            continue
        rows[row["key"]] = row
    return header, rows


def _template_ids(tokenizer, chat: list[dict[str, str]], add_generation_prompt: bool = False) -> list[int]:
    """Same tokenization as the last-token run: the chat template, not a second encode of its string."""
    ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=add_generation_prompt)
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if not ids:
        raise RuntimeError("empty tokenization")
    return [int(token) for token in ids]


def _special_ids(tokenizer) -> set[int]:
    unk = getattr(tokenizer, "unk_token_id", None)
    found = set()
    for token in (getattr(tokenizer, "eos_token", None), "<end_of_turn>", "<eos>"):
        if not token:
            continue
        index = tokenizer.convert_tokens_to_ids(token)
        if isinstance(index, int) and index >= 0 and index != unk:
            found.add(index)
    return found


def _forward_scores(model, ids: list[int], spans: dict[str, tuple[int, int]], weight, captured: dict, device) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import torch

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    scored = {}
    with torch.inference_mode():
        model(input_ids=input_ids, attention_mask=attention, use_cache=False)
        hidden = captured["hidden"]
        if hidden.shape[1] != len(ids):
            raise RuntimeError(f"activation length {hidden.shape[1]} does not match {len(ids)} tokens")
        direction = weight.to(device=hidden.device, dtype=torch.float32)
        for key, (start, end) in spans.items():
            scores = hidden[0, start:end].float() @ direction.T
            peak, _index = scores.max(dim=0)
            scored[key] = (peak.detach().cpu().numpy(), scores[-1].detach().cpu().numpy())
    return scored


def _groups(passages: list[dict]) -> list[list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in passages:
        grouped.setdefault((row["trace_id"], int(row["node_index"])), []).append(row)
    return list(grouped.values())


def _key(row: dict) -> str:
    return f"{row['trace_id']}:{int(row['node_index'])}:{row['channel']}"


def _directions(subject_needed: bool) -> tuple[list[str], np.ndarray | None]:
    contrasts = json.loads(dom.CONTRASTS.read_text())
    synth = dom._load_shards(dom.CACHE / "synth")
    if not synth:
        if not subject_needed:
            raise SystemExit("synthetic activation cache is missing; run without --fit-only so it can be built")
        return [], None
    built = dom._directions(contrasts, synth)
    names = [name for name, row in built.items() if row["vector"] is not None]
    if not names:
        raise SystemExit("no synthetic contrast produced a direction")
    matrix = np.stack([built[name]["vector"] for name in names]).astype(np.float32)
    return names, matrix


def _extract(max_length: int, limit: int | None) -> None:
    import torch
    from dyl_vs_oa.model import load_subject

    path = _scores_path(max_length)
    path.parent.mkdir(parents=True, exist_ok=True)
    names, matrix = _directions(subject_needed=True)
    traces = dom._trace_index()
    passages = dom._passages()
    groups = _groups(passages)
    header, done = _load_scores(path)
    if header is not None and matrix is None:
        raise SystemExit("synthetic activation cache is missing, and a score file already exists. Restore the cache or delete the score file.")
    if header is not None and (header["contrasts"] != names or int(header["max_length"]) != max_length):
        raise SystemExit(
            f"{path.name} was written with different contrasts or a different max length. "
            "Delete it before rerunning. Do not append a second setup onto it."
        )

    pending = []
    for group in groups:
        if any(_key(row) not in done for row in group):
            pending.append(group)
    if limit is not None:
        pending = pending[:limit]
    print(f"peak: cached {len(done)}; steps to score {len(pending)}; max_length {max_length}", flush=True)
    if not pending:
        return
    if header is None and done:
        raise SystemExit(f"{path.name} has rows and no header. Delete it and rerun.")

    if matrix is None:
        print("loading " + dom.MODEL_NAME + " to build synthetic directions", flush=True)
    else:
        print("loading " + dom.MODEL_NAME, flush=True)
    subject = load_subject(dom.MODEL_NAME)
    captured: dict = {}

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden
        return output

    handle = dom._layer_module(subject.model, dom.LAYER).register_forward_hook(_hook)
    hard_failures = 0
    locate_failures = 0
    scored_channels = 0
    try:
        if matrix is None:
            dom._extract(subject, subject.tokenizer, dom._synth_jobs(), dom.CACHE / "synth", 4096, captured)
            names, matrix = _directions(subject_needed=True)
            if not names:
                raise SystemExit("synthetic extract finished but no direction was built")
        if header is None:
            header = {
                "kind": "header",
                "contrasts": names,
                "max_length": max_length,
                "model": dom.MODEL_NAME,
                "layer": dom.LAYER,
            }
            path.write_text(json.dumps(header) + "\n")
        weight = torch.tensor(matrix, dtype=torch.float32, device=subject.device)
        specials = _special_ids(subject.tokenizer)
        with path.open("a", encoding="utf-8") as handle_out:
            for index, group in enumerate(pending, start=1):
                trace_id = group[0]["trace_id"]
                node_index = int(group[0]["node_index"])
                try:
                    missed, scored = _score_step(
                        subject,
                        traces[trace_id],
                        node_index,
                        group,
                        done,
                        weight,
                        names,
                        specials,
                        max_length,
                        captured,
                        handle_out,
                    )
                except Exception as exc:
                    hard_failures += 1
                    print(f"  skip {trace_id}:{node_index}: {type(exc).__name__}: {exc}", flush=True)
                    if hard_failures > 50 or (index >= 8 and scored_channels == 0):
                        raise SystemExit(
                            "stopped early. The first steps produced no scores, so the full run would not be a result."
                        ) from exc
                    continue
                locate_failures += missed
                scored_channels += scored
                handle_out.flush()
                if scored_channels == scored and scored:
                    print(f"  first step scored {scored} channel(s)", flush=True)
                if index >= 8:
                    attempts = locate_failures + scored_channels
                    if attempts == 0 or locate_failures / attempts > 0.1:
                        raise SystemExit(
                            "stopped early. More than a tenth of the first channels had no usable span. "
                            f"Delete {_scores_path(max_length).name} before trying again, or those rows will be skipped. "
                            "Do not start the full run."
                        )
                if index % 25 == 0 or index == len(pending):
                    print(f"  {index}/{len(pending)}", flush=True)
        if pending and scored_channels == 0:
            raise SystemExit(
                "stopped. No channel was scored. "
                f"Delete {_scores_path(max_length).name} before trying again. Do not start the full run."
            )
    finally:
        handle.remove()
        del subject
        torch.cuda.empty_cache()
    print(f"peak: scored channels {scored_channels}; locate failures {locate_failures}; step errors {hard_failures}", flush=True)


def _score_step(subject, nodes, node_index, group, done, weight, names, specials, max_length, captured, handle_out) -> tuple[int, int]:
    raw = prior_messages(nodes, node_index)
    turn, spans = build_turn(nodes[node_index]["message"])
    max_chars = max_length * 8
    original = spans
    turn, spans, turn_cut = trim_turn(turn, spans, max_chars) if turn else (turn, spans, 0)
    cut_channels = set()
    if turn_cut:
        for name, (start, _end) in original.items():
            if name in spans and start < turn_cut:
                cut_channels.add(name)
    prior, prior_cut = trim_prior(raw, len(turn), max_chars)

    def ids_with(text: str | None) -> list[int]:
        if text is None:
            return _template_ids(subject.tokenizer, prior, add_generation_prompt=True)
        chat = to_gemma_chat([*prior, {"role": "assistant", "content": text}])
        return _template_ids(subject.tokenizer, chat, add_generation_prompt=False)

    missing: list[tuple[dict, str]] = []
    located: dict[str, tuple[int, int]] = {}
    ids_full: list[int] | None = None
    if turn:
        ids_full = ids_with(turn)
    for row in group:
        key = _key(row)
        if key in done:
            continue
        span = spans.get(row["channel"])
        if not turn or span is None or ids_full is None:
            missing.append((row, "channel is empty or was cut from a long turn"))
            continue
        start_c, end_c = span
        try:
            ids_before = ids_with(None) if start_c == 0 else ids_with(turn[:start_c])
            ids_through = ids_full if end_c == len(turn) else ids_with(turn[:end_c])
            start, end = span_from_prefixes(ids_before, ids_through, ids_full)
        except ValueError as exc:
            missing.append((row, str(exc)))
            continue
        while end > start and ids_full[end - 1] in specials:
            end -= 1
        if end <= start:
            missing.append((row, "channel span was only an end-of-turn token"))
            continue
        located[key] = (start, end)
    for row, reason in missing:
        handle_out.write(json.dumps({"key": _key(row), "error": reason}) + "\n")
        done[_key(row)] = {"key": _key(row), "error": reason}
    if not located or ids_full is None:
        return len(missing), 0
    drop = max(0, len(ids_full) - max_length)
    adjusted: dict[str, tuple[tuple[int, int], bool]] = {}
    for key, (start, end) in located.items():
        try:
            _dropped, kept, token_cut = keep_tail(len(ids_full), (start, end), max_length)
        except ValueError as exc:
            handle_out.write(json.dumps({"key": key, "error": str(exc)}) + "\n")
            done[key] = {"key": key, "error": str(exc)}
            missing.append(({"key": key}, str(exc)))
            continue
        channel = key.rsplit(":", 1)[-1]
        adjusted[key] = (kept, bool(token_cut or channel in cut_channels))
    if not adjusted:
        return len(missing), 0
    scored = _forward_scores(
        subject.model,
        ids_full[drop:],
        {key: span for key, (span, _cut) in adjusted.items()},
        weight,
        captured,
        subject.device,
    )
    for key, (peak, last) in scored.items():
        if not np.isfinite(peak).all() or not np.isfinite(last).all():
            raise RuntimeError("non-finite passage score")
        kept, span_cut = adjusted[key]
        record = {
            "key": key,
            "peak": [float(value) for value in peak],
            "last": [float(value) for value in last],
            "span_tokens": int(kept[1] - kept[0]),
            "truncated": bool(drop or turn_cut or prior_cut),
            "span_cut": bool(span_cut),
            "n_contrasts": len(names),
        }
        handle_out.write(json.dumps(record) + "\n")
        done[key] = record
    return len(missing), len(scored)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _previous_peek() -> str:
    path = WEEK6 / "results" / "goodfire_dom.json"
    if path.exists():
        payload = json.loads(path.read_text())
        for row in payload.get("contrasts", []):
            if row.get("name") == "pl_peek" and row.get("test_auroc") is not None:
                return f"{row['test_auroc']:.3f}"
    return "0.743"


def _fit(max_length: int) -> None:
    path = _scores_path(max_length)
    header, scored = _load_scores(path)
    if header is None:
        raise SystemExit(f"no scores at {path}. Run the extract first.")
    names = header["contrasts"]
    passages = dom._passages()
    train_ids, test_ids = dom._split([row["trace_id"] for row in passages])
    by_name: dict[str, list[dict]] = {name: [] for name in names}
    present = 0
    errors = 0
    truncated = 0
    span_cut = 0
    for row in passages:
        key = _key(row)
        found = scored.get(key)
        if found is None:
            continue
        if found.get("error"):
            errors += 1
            continue
        present += 1
        truncated += int(bool(found.get("truncated")))
        span_cut += int(bool(found.get("span_cut")))
        for index, name in enumerate(names):
            by_name[name].append(
                {
                    "trace_id": row["trace_id"],
                    "consensus": row["consensus"],
                    "step": int(row["step"]),
                    "passage_n": int(row["passage_n"]),
                    "statuses": list(row.get("carrier_statuses") or []),
                    "peak": found["peak"][index],
                    "last": found["last"][index],
                }
            )
    table = [contrast_row(name, rows, train_ids, test_ids) for name, rows in by_name.items() if rows]
    ranked = [row for row in table if row["train_peak_auroc"] is not None]
    if not ranked:
        if present < 30:
            text = "PARTIAL. Not enough scored passages with both classes to report AUROC yet. Rerun without --limit.\n"
            REPORT_MD.write_text(text)
            print(text, flush=True)
            return
        raise SystemExit("no contrast had both classes in the scored passages")
    chosen = max(ranked, key=lambda row: row["train_peak_auroc"])["name"]
    chosen_rows = by_name[chosen]
    lead = score_report(
        [{**row, "score": row["peak"]} for row in chosen_rows],
        train_ids,
        test_ids,
    )["lead"]
    payload = {
        "chosen": chosen,
        "max_length": max_length,
        "model": header.get("model", dom.MODEL_NAME),
        "layer": header.get("layer", dom.LAYER),
        "passages": len(passages),
        "scored": present,
        "errors": errors,
        "truncated": truncated,
        "span_cut": span_cut,
        "paper_impossiblebench_auroc": PAPER_IMPOSSIBLEBENCH_AUROC,
        "previous_last_token_2048_pl_peek": _previous_peek(),
        "contrasts": table,
        "chosen_lead": lead,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2))
    text = _markdown(payload)
    REPORT_MD.write_text(text)
    print(text, flush=True)
    dom._log("goodfire peak", text)
    if present + errors and errors / (present + errors) > 0.05:
        raise SystemExit(f"passage locate failed on {errors} of {present + errors}; the table is not a result")


def _markdown(payload: dict) -> str:
    lead = payload["chosen_lead"]
    partial = payload["scored"] + payload["errors"] < payload["passages"]
    lines = [
        "# Goodfire passage peak, read by Gemma 27B",
        "",
        "Score: the maximum dot product inside the labeled channel. Reasoning, the visible reply, and the tool call are separate passages. Earlier channels of the same step stay in the prefix.",
        f"Direction: the synthetic difference of means from the last-token run, Gemma 27B layer {payload['layer']}. Not fit on the real rollouts.",
        f"Prefix: early turns are dropped, then the kept text is cut to {payload['max_length']} tokens. Positions restart at the start of that tail. {payload['truncated']} scored passages did not see the full history; {payload['span_cut']} lost the start of the passage itself.",
        "",
        f"Passages scored: {payload['scored']} of {payload['passages']}. Locate failures: {payload['errors']}.",
        f"Selected on the train half by peak AUROC: `{payload['chosen']}`.",
        "",
        "Kimi's own probe in the paper reaches AUROC 0.94 on ImpossibleBench. The number below is Gemma reading Kimi's text, which is the comparison this machine can run.",
        f"The last-token run in GOODFIRE.md, cut at 2048 and mixing reasoning into the content channel, scored pl_peek at {payload['previous_last_token_2048_pl_peek']}. The Last column here is the last token of the labeled channel on this prefix.",
        "",
        "| Contrast | Train peak AUROC | Test peak AUROC | Test last-token AUROC | Test TPR at 5% train FPR | Test FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if partial:
        lines.insert(2, "PARTIAL. This file does not yet cover every passage. Rerun without --limit before treating the table as the result.")
        lines.insert(3, "")
    for row in payload["contrasts"]:
        lines.append(
            "| {name} | {train} | {test} | {last} | {tpr} | {fpr} |".format(
                name=row["name"],
                train=_fmt(row["train_peak_auroc"]),
                test=_fmt(row["test_peak_auroc"]),
                last=_fmt(row["test_last_auroc"]),
                tpr=_fmt(row["test_tpr_at_train_fpr_05"]),
                fpr=_fmt(row["test_fpr"]),
            )
        )
    lines.extend(
        [
            "",
            f"On the test rollouts with an attempted or enacted hack, the peak score crosses the 5% clean-passage "
            f"threshold on an earlier step in {lead['n_fired_before_action']} of {lead['n_action_rollouts']}. "
            f"Median lead among those fires is {_fmt(lead['median_lead_steps'])} assistant steps. "
            "The paper's early-detection result is a resampling test, which this script does not run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=None, help="score at most this many new steps, then fit")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    dom._load_env()
    for path in (dom.CONTRASTS, dom.SYNTH, dom.TRACES, dom.PASSAGES):
        if not path.exists():
            raise SystemExit(f"missing {path}. The Goodfire release belongs in data/goodfire/.")
    if not args.fit_only:
        if not os.environ.get("HF_TOKEN"):
            raise SystemExit("HF_TOKEN is missing")
        dom._log(
            "goodfire peak start",
            f"ImpossibleBench Kimi traces, peak inside the labeled channel, max_length {args.max_length}",
        )
        _extract(args.max_length, args.limit)
    if not args.extract_only:
        _fit(args.max_length)


if __name__ == "__main__":
    main()
