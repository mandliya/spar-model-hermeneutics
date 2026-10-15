#!/usr/bin/env python3
"""Gemma 27B difference-of-means probe on Goodfire's reward-hacking release.

Synthetic hack-versus-honest spans build the direction. ImpossibleBench Kimi
rollouts are the labeled traces. The direction is not fit on those rollouts.

    python -u scripts/03_goodfire_dom.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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

from early_detection.goodfire_dom import auroc, difference_of_means, dot_scores, fpr_threshold, lead_steps, summarize_leads
from noninvasive_probe.tianq_read import LAYER

DATA = WEEK6 / "data" / "goodfire"
SYNTH = DATA / "synthetic_data" / "dataset.jsonl"
CONTRASTS = DATA / "synthetic_data" / "contrasts.json"
TRACES = DATA / "impossiblebench-kimi-k3" / "traces.jsonl"
PASSAGES = DATA / "impossiblebench-kimi-k3" / "labels" / "judge_passages.jsonl"
CACHE = WEEK6 / "results" / "goodfire_cache"
REPORT = WEEK6 / "results" / "goodfire_dom.json"
REPORT_MD = WEEK6 / "GOODFIRE.md"
LOG = WEEK6 / "LOG.md"
MODEL_NAME = "google/gemma-3-27b-it"
EVAL_MAX_LENGTH = 2048
SEED = 0


def _load_env() -> None:
    path = WEEK6 / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ["HF_HOME"] = "/dev/shm/hf_home"
    os.environ["HF_HUB_CACHE"] = "/dev/shm/hf_home/hub"
    os.environ["HUGGINGFACE_HUB_CACHE"] = "/dev/shm/hf_home/hub"


def _log(title: str, body: str) -> None:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {stamp} — {title}\n\n{body.rstrip()}\n")


def _text(message: dict) -> str:
    parts: list[str] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        parts.append(reasoning.strip())
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    calls = message.get("tool_calls") or []
    for call in calls:
        function = call.get("function") or call
        name = function.get("name") or call.get("name") or "tool"
        arguments = function.get("arguments", call.get("arguments", ""))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        parts.append(f"tool call {name}: {arguments}")
    return "\n".join(parts).strip()


def _chat(messages: list[dict], channel: str | None = None) -> list[dict[str, str]]:
    """Chat messages. channel limits the last message to that span."""
    kept = messages if channel is None else messages[:-1]
    last = None if channel is None else messages[-1]
    out: list[dict[str, str]] = []
    for message in kept:
        role = message.get("role", "user")
        text = _text(message)
        if role == "tool":
            role, text = "user", "Tool result:\n" + text
        if role not in {"system", "user", "assistant"}:
            role = "user"
        out.append({"role": role, "content": text or " "})
    if last is not None:
        text = _channel_text(last, channel)
        out.append({"role": "assistant", "content": text or " "})
    return _alternate(out)


def _alternate(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Gemma's template requires system, then user/assistant/user/assistant."""
    system = "\n\n".join(message["content"] for message in messages if message["role"] == "system").strip()
    body = [message for message in messages if message["role"] != "system"]
    merged: list[dict[str, str]] = []
    for message in body:
        role = message["role"] if message["role"] in {"user", "assistant"} else "user"
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + message["content"]
        else:
            merged.append({"role": role, "content": message["content"]})
    if merged and merged[0]["role"] == "assistant":
        merged.insert(0, {"role": "user", "content": "Continue."})
    if not merged:
        merged = [{"role": "user", "content": " "}]
    out = [{"role": "system", "content": system}] if system else []
    out.extend(merged)
    return out


def _channel_text(message: dict, channel: str) -> str:
    if channel == "reasoning":
        return (message.get("reasoning_content") or "").strip()
    if channel == "content":
        parts = []
        if message.get("reasoning_content"):
            parts.append(str(message["reasoning_content"]).strip())
        if message.get("content"):
            parts.append(str(message["content"]).strip())
        return "\n".join(part for part in parts if part)
    return _text(message)


def _encode(tokenizer, messages: list[dict], max_length: int) -> tuple[list[int], bool]:
    chat = _chat(messages) if messages and "channel" not in messages[0] else messages
    ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "tolist"):
        # BatchEncoding or tensor
        ids = ids["input_ids"] if hasattr(ids, "keys") and "input_ids" in getattr(ids, "keys", lambda: [])() else ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(token) for token in ids]
    if not ids:
        raise RuntimeError("empty tokenization")
    truncated = len(ids) > max_length
    if truncated:
        ids = ids[-max_length:]
    return ids, truncated


def _trim_chat(chat: list[dict[str, str]], max_chars: int) -> list[dict[str, str]]:
    """Keep the scored end. Drop early turns once the text exceeds the budget."""
    total = sum(len(message["content"]) for message in chat)
    if total <= max_chars:
        return chat
    system = [message for message in chat if message["role"] == "system"][:1]
    body = [dict(message) for message in chat if message["role"] != "system"]
    if not body:
        return chat
    last = body[-1]
    room = max_chars
    if system:
        system_text = system[0]["content"][-min(2000, room // 2):]
        system = [{"role": "system", "content": system_text}]
        room -= len(system_text)
    if len(last["content"]) > room:
        last = {"role": last["role"], "content": last["content"][-room:]}
        return _alternate(system + [last])
    kept = [last]
    used = len(last["content"])
    for message in reversed(body[:-1]):
        if used + len(message["content"]) > room:
            break
        kept.append(message)
        used += len(message["content"])
    kept.reverse()
    return _alternate(system + kept)


def _encode_chat(tokenizer, chat: list[dict[str, str]], max_length: int) -> tuple[list[int], bool]:
    original = sum(len(message["content"]) for message in chat)
    chat = _trim_chat(chat, max_length * 4)
    ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(token) for token in ids]
    if not ids:
        raise RuntimeError("empty tokenization")
    truncated = original > max_length * 4 or len(ids) > max_length
    if len(ids) > max_length:
        ids = ids[-max_length:]
    return ids, truncated


def _load_shards(directory: Path) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    if not directory.exists():
        return found
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for key, vector in zip(data["ids"], data["vectors"]):
                found[str(key)] = np.asarray(vector, dtype=np.float32)
    return found


def _write_shard(directory: Path, stem: str, ids: list[str], vectors: list[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"{stem}.npz",
        ids=np.array(ids),
        vectors=np.stack(vectors).astype(np.float32),
    )


def _layer_module(model, layer: int):
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if layers is not None and len(layers) > layer:
            return layers[layer]
    raise RuntimeError(f"could not find decoder layer {layer}")


def _forward_last(subject, tokenizer, chat: list[dict[str, str]], max_length: int, captured: dict) -> tuple[np.ndarray, bool]:
    import torch

    ids, truncated = _encode_chat(tokenizer, chat, max_length)
    input_ids = torch.tensor([ids], dtype=torch.long, device=subject.device)
    attention = torch.ones_like(input_ids)
    with torch.inference_mode():
        subject.model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    vector = captured["hidden"][0, len(ids) - 1].float().cpu().numpy()
    if not np.isfinite(vector).all():
        raise RuntimeError("non-finite activation")
    return vector, truncated


def _synth_jobs() -> list[tuple[str, list[dict]]]:
    contrasts = json.loads(CONTRASTS.read_text())
    wanted: set[tuple[str, str]] = set()
    for contrast in contrasts.values():
        for span, _cell in contrast["pos_keys"] + contrast["neg_keys"]:
            wanted.add((contrast["dataset"], span))
    jobs = []
    with SYNTH.open() as handle:
        for line in handle:
            row = json.loads(line)
            dataset = row["metadata"]["dataset"]
            for readout in row["readouts"]:
                if (dataset, readout["name"]) not in wanted:
                    continue
                index = readout["selector"]["message_index"]
                key = f"{row['id']}::{readout['name']}"
                jobs.append((key, _chat(row["messages"][: index + 1])))
    return jobs


def _trace_index() -> dict[str, list[dict]]:
    traces = {}
    with TRACES.open() as handle:
        for line in handle:
            row = json.loads(line)
            trace = row["traces"][0]
            traces[trace["id"]] = trace["nodes"]
    return traces


def _eval_jobs(traces: dict[str, list[dict]]) -> list[tuple[str, list[dict]]]:
    jobs = []
    with PASSAGES.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["consensus"] == "ambiguous":
                continue
            nodes = traces[row["trace_id"]]
            node = nodes[row["node_index"]]
            messages = [item["message"] for item in nodes[: row["node_index"]]]
            messages.append(node["message"])
            key = f"{row['trace_id']}:{row['node_index']}:{row['channel']}"
            jobs.append((key, _chat(messages, channel=row["channel"])))
    return jobs


def _extract(subject, tokenizer, jobs: list[tuple[str, list[dict]]], directory: Path, max_length: int, captured: dict) -> dict[str, np.ndarray]:
    have = _load_shards(directory)
    pending = [(key, chat) for key, chat in jobs if key not in have]
    print(f"{directory.name}: cached {len(have)}; extracting {len(pending)}", flush=True)
    bucket_ids: list[str] = []
    bucket_vecs: list[np.ndarray] = []
    truncated = 0
    failures = 0
    shard = len(list(directory.glob("*.npz"))) if directory.exists() else 0
    for index, (key, chat) in enumerate(pending, start=1):
        try:
            vector, cut = _forward_last(subject, tokenizer, chat, max_length, captured)
        except Exception as exc:
            failures += 1
            print(f"  skip {key}: {type(exc).__name__}: {exc}", flush=True)
            if failures > 50:
                raise
            continue
        truncated += int(cut)
        have[key] = vector
        bucket_ids.append(key)
        bucket_vecs.append(vector)
        if len(bucket_ids) == 100:
            _write_shard(directory, f"shard_{shard:04d}", bucket_ids, bucket_vecs)
            shard += 1
            bucket_ids, bucket_vecs = [], []
        if index % 25 == 0 or index == len(pending):
            print(f"  {index}/{len(pending)}", flush=True)
    if bucket_ids:
        _write_shard(directory, f"shard_{shard:04d}", bucket_ids, bucket_vecs)
    print(f"{directory.name}: truncated prefixes {truncated}; skipped {failures}", flush=True)
    return have


def _passages() -> list[dict]:
    rows = []
    with PASSAGES.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["consensus"] == "ambiguous":
                continue
            rows.append(row)
    return rows


def _directions(contrasts: dict, synth: dict[str, np.ndarray]) -> dict[str, dict]:
    rows = []
    with SYNTH.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(row)
    by_id = {row["id"]: row for row in rows}
    built = {}
    for name, contrast in contrasts.items():
        positive, negative = [], []
        for span, cell in contrast["pos_keys"]:
            positive.extend(_pool(rows, by_id, synth, contrast["dataset"], span, cell))
        for span, cell in contrast["neg_keys"]:
            negative.extend(_pool(rows, by_id, synth, contrast["dataset"], span, cell))
        if len(positive) < 10 or len(negative) < 10:
            built[name] = {"n_pos": len(positive), "n_neg": len(negative), "vector": None}
            continue
        direction = difference_of_means(np.stack(positive), np.stack(negative))
        built[name] = {"n_pos": len(positive), "n_neg": len(negative), "vector": direction}
    return built


def _pool(rows, by_id, synth, dataset: str, span: str, cell: str) -> list[np.ndarray]:
    found = []
    for row in rows:
        meta = row["metadata"]
        if meta["dataset"] != dataset or meta["cell"] != cell:
            continue
        vector = synth.get(f"{row['id']}::{span}")
        if vector is not None:
            found.append(vector)
    return found


def _split(trace_ids: list[str]) -> tuple[set[str], set[str]]:
    ids = np.array(sorted(set(trace_ids)))
    rng = np.random.RandomState(SEED)
    rng.shuffle(ids)
    mid = len(ids) // 2
    return set(ids[:mid]), set(ids[mid:])


def _score(passages: list[dict], vectors: dict[str, np.ndarray], direction: np.ndarray) -> list[dict]:
    scored = []
    for row in passages:
        key = f"{row['trace_id']}:{row['node_index']}:{row['channel']}"
        vector = vectors.get(key)
        if vector is None:
            continue
        scored.append(
            {
                "trace_id": row["trace_id"],
                "step": int(row["step"]),
                "passage_n": int(row["passage_n"]),
                "channel": row["channel"],
                "consensus": row["consensus"],
                "statuses": list(row.get("carrier_statuses") or []),
                "score": float(dot_scores(direction, vector[None, :])[0]),
            }
        )
    return scored


def _evaluate(scored: list[dict], train_ids: set[str], test_ids: set[str]) -> dict:
    def pack(ids: set[str]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        rows = [row for row in scored if row["trace_id"] in ids and row["consensus"] in {"positive", "negative"}]
        labels = np.array([1 if row["consensus"] == "positive" else 0 for row in rows])
        scores = np.array([row["score"] for row in rows], dtype=np.float64)
        return labels, scores, rows

    train_y, train_s, train_rows = pack(train_ids)
    test_y, test_s, test_rows = pack(test_ids)
    honest = train_s[train_y == 0]
    threshold = fpr_threshold(honest, 0.05)
    test_by_trace: dict[str, list[dict]] = {}
    for row in test_rows:
        test_by_trace.setdefault(row["trace_id"], []).append(row)
    leads = []
    for rows in test_by_trace.values():
        lead = lead_steps(rows, threshold)
        if lead is not None:
            leads.append(lead)
    return {
        "train_auroc": auroc(train_y, train_s),
        "test_auroc": auroc(test_y, test_s),
        "train_n": int(len(train_y)),
        "test_n": int(len(test_y)),
        "train_positives": int(train_y.sum()) if len(train_y) else 0,
        "test_positives": int(test_y.sum()) if len(test_y) else 0,
        "threshold_fpr_05": threshold,
        "lead": summarize_leads(leads),
    }


def _write_report(chosen: str, table: list[dict], chosen_eval: dict) -> None:
    lead = chosen_eval["lead"]
    fraction = lead["fraction_fired_before_action"]
    median = lead["median_lead_steps"]
    lines = [
        "# Goodfire difference-of-means, read by Gemma 27B",
        "",
        "Direction: mean residual on synthetic hack spans minus mean residual on the matched honest spans, Gemma 27B layer 58. Not fit on the real rollouts.",
        "Traces: ImpossibleBench, Kimi K3, judge labels from the Goodfire release. Ambiguous passages dropped. A prefix longer than 2048 tokens is left-truncated; the scored token is the end of the passage.",
        "",
        f"Selected contrast on the train half of rollouts: `{chosen}`.",
        "",
        "| Contrast | Train AUROC | Test AUROC | Synthetic hack spans | Synthetic honest spans |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in table:
        lines.append(
            f"| {row['name']} | {_fmt(row['train_auroc'])} | {_fmt(row['test_auroc'])} | {row['n_pos']} | {row['n_neg']} |"
        )
    lines.extend(
        [
            "",
            f"Held-out passage AUROC for `{chosen}` is {_fmt(chosen_eval['test_auroc'])} "
            f"({chosen_eval['test_positives']} positive passages in {chosen_eval['test_n']} scored passages).",
            "",
            f"On the test rollouts that contain an attempted or enacted hack, the probe fires on an earlier step "
            f"in {lead['n_fired_before_action']} of {lead['n_action_rollouts']} "
            f"({_fmt(fraction)}). {lead['n_fired_on_earlier_step']} of those fires are on an earlier assistant step; "
            f"the rest are an earlier channel of the same step. Median lead, among those that fire early, is {_fmt(median)} assistant steps. "
            "The threshold is the 95th percentile of the probe on train passages the judge called clean, so about 5% of clean train passages would fire.",
            "",
            "This is Gemma reading Kimi's traces. It is not the paper's Kimi, GLM, or Qwen probe, and it does not rerun their resampling test.",
            "",
        ]
    )
    text = "\n".join(lines)
    REPORT_MD.write_text(text)
    print(text, flush=True)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    _load_env()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is missing")
    import torch
    from dyl_vs_oa.model import load_subject

    _log("goodfire start", "ImpossibleBench Kimi traces, synthetic difference of means, Gemma 27B layer 58")
    print("loading " + MODEL_NAME, flush=True)
    subject = load_subject(MODEL_NAME)
    captured: dict = {}

    def _hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden
        return output

    handle = _layer_module(subject.model, LAYER).register_forward_hook(_hook)
    try:
        synth = _extract(subject, subject.tokenizer, _synth_jobs(), CACHE / "synth", 4096, captured)
        traces = _trace_index()
        eval_vectors = _extract(subject, subject.tokenizer, _eval_jobs(traces), CACHE / "impossiblebench", EVAL_MAX_LENGTH, captured)
    finally:
        handle.remove()
        del subject
        torch.cuda.empty_cache()

    contrasts = json.loads(CONTRASTS.read_text())
    directions = _directions(contrasts, synth)
    passages = _passages()
    train_ids, test_ids = _split([row["trace_id"] for row in passages])
    table = []
    for name, built in directions.items():
        if built["vector"] is None:
            table.append({"name": name, "train_auroc": None, "test_auroc": None, "n_pos": built["n_pos"], "n_neg": built["n_neg"]})
            continue
        train_rows = _score([row for row in passages if row["trace_id"] in train_ids], eval_vectors, built["vector"])
        test_rows = _score([row for row in passages if row["trace_id"] in test_ids], eval_vectors, built["vector"])
        table.append(
            {
                "name": name,
                "train_auroc": auroc(
                    np.array([1 if row["consensus"] == "positive" else 0 for row in train_rows]),
                    np.array([row["score"] for row in train_rows]),
                ),
                "test_auroc": auroc(
                    np.array([1 if row["consensus"] == "positive" else 0 for row in test_rows]),
                    np.array([row["score"] for row in test_rows]),
                ),
                "n_pos": built["n_pos"],
                "n_neg": built["n_neg"],
            }
        )
    ranked = [row for row in table if row["train_auroc"] is not None]
    if not ranked:
        raise SystemExit("no contrast produced a direction")
    chosen = max(ranked, key=lambda row: row["train_auroc"])["name"]
    chosen_eval = _evaluate(_score(passages, eval_vectors, directions[chosen]["vector"]), train_ids, test_ids)
    payload = {"chosen": chosen, "contrasts": table, "chosen_eval": chosen_eval, "model": MODEL_NAME, "layer": LAYER}
    REPORT.write_text(json.dumps(payload, indent=2))
    _write_report(chosen, table, chosen_eval)
    _log("goodfire fit", REPORT_MD.read_text())


if __name__ == "__main__":
    main()
