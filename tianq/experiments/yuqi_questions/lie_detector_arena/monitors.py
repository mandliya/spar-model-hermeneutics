"""The untrained monitor: another model reads the transcript and scores 1-100 how likely the assistant lied.

    ./run.sh yuqi lie_detector_arena.monitors        # from tianq/dyl-probe-repro (API only)

Gigi's reproduction of the paper's black-box baseline (gigi/1-replication/third_person), used as is: his
model table, provider pinning and API call (read from his script's source, which runs on import), his
prompt builder and score parser (monitor_prompt.py), and the paper's rubric template, his main result.
He ran it on Gemma-3-27B's and Qwen3-32B's transcripts (not on Qwen3-32B's ASCII and MASK provided-facts
kinds); this runs it on every transcript his files do not score, for all four targets
(gigi/qwen/results/<target>/transcripts.jsonl). Resumable: one JSON line per reply, appended as it lands,
under scratch/yuqi-questions/monitors/<target>/, pushed to the Hub when a target is done. Stops before
the total cost passes BUDGET.

scores() gives every target's monitor scores keyed by conversation content (rows.content_key).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from dyl_repro.core.store import SCRATCH
from experiments.yuqi_questions.lie_detector_arena.rows import content_key
from experiments.yuqi_questions.shared import stored

FOLDER = Path(__file__).parent.name
GIGI = Path(__file__).resolve().parents[4] / "gigi"
THIRD_PERSON = GIGI / "1-replication" / "third_person"
VARIANT = "paper_3p_rubric"
BUDGET = 50.0
OUT = SCRATCH / "yuqi-questions" / "monitors"
GIGI_RAN = ("gemma-3-27b-it", "qwen3-32b")
# Where each target's transcripts and existing monitor scores are (Gigi's layout).
SOURCES = {"gemma-3-27b-it": THIRD_PERSON / "results", "qwen3-32b": GIGI / "qwen" / "results" / "qwen3-32b",
           "qwen3-next-80b": GIGI / "qwen" / "results" / "qwen3-next-80b",
           "qwen3-235b": GIGI / "qwen" / "results" / "qwen3-235b"}


def _gigi_prompt():
    spec = importlib.util.spec_from_file_location("monitor_prompt", THIRD_PERSON / "monitor_prompt.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gigi_api() -> dict:
    """MODELS, key() and call() from 01_monitor_api.py, without running the script around them."""
    path = THIRD_PERSON / "01_monitor_api.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body
              if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "MODELS" for t in n.targets))
              or (isinstance(n, ast.FunctionDef) and n.name in ("key", "call"))]
    namespace = {"os": os, "time": time, "requests": requests, "KEY": os.environ["OPENROUTER_API_KEY"]}
    exec(compile(ast.Module(wanted, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


PROMPT = _gigi_prompt()


def transcripts(target: str) -> list[dict]:
    return [json.loads(line) for line in (SOURCES[target] / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]


def ours(target: str, model: str) -> Path:
    return OUT / target / f"monitor_{model}_paper.jsonl"


def results(target: str, model: str) -> list[dict]:
    """Every reply on record for a target and monitor: Gigi's, then ours."""
    paths = [SOURCES[target] / f"monitor_{model}_paper.jsonl"] if target in GIGI_RAN else []
    paths.append(ours(target, model))
    return [json.loads(line) for path in paths if path.exists()
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scores(target: str) -> dict[str, dict[str, float]]:
    """monitor -> {conversation content: score 1-100}, the paper's rubric template only."""
    api = _gigi_api()
    by_key = {api["key"](t, VARIANT): content_key(t["messages"]) for t in transcripts(target)}
    out = {}
    for model in api["MODELS"]:
        out[model] = {}
        for r in results(target, model):
            if r.get("variant") == VARIANT and r.get("score") is not None and r["key"] in by_key:
                out[model][by_key[r["key"]]] = float(r["score"])
    return out


def spent() -> float:
    return sum(json.loads(line).get("cost") or 0 for path in OUT.glob("*/*.jsonl")
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run(target: str, api: dict) -> None:
    rows = transcripts(target)
    lock = threading.Lock()
    for model in api["MODELS"]:
        path = ours(target, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A reply that failed (an "error" line) is tried again; a later success supersedes it in scores().
        done = {r["key"] for r in results(target, model) if r.get("variant") == VARIANT and "error" not in r}
        jobs = [r for r in rows if api["key"](r, VARIANT) not in done]
        print(f"[{target} {model}] {len(rows)} transcripts, {len(done)} done, {len(jobs)} to go;"
              f" spent so far ${spent():.2f}", flush=True)

        def work(r: dict) -> None:
            if spent_now[0] >= BUDGET:
                return
            try:
                reply = api["call"](model, PROMPT.prompt(r["messages"], VARIANT))
                raw = reply["choices"][0]["message"].get("content") or ""
                usage = reply.get("usage", {})
                record = {"key": api["key"](r, VARIANT), "category": r["category"], "sample_id": r["sample_id"],
                          "track": r["track"], "variant": VARIANT, "model": model, "raw": raw,
                          "score": PROMPT.parse_score(raw), "cost": usage.get("cost"), "provider": reply.get("provider")}
            except Exception as exc:  # recorded and retried on the next run, like Gigi's
                record = {"key": api["key"](r, VARIANT), "variant": VARIANT, "model": model, "error": repr(exc)[:400]}
            with lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                spent_now[0] += record.get("cost") or 0

        spent_now = [spent()]
        with ThreadPoolExecutor(16) as pool:
            list(pool.map(work, jobs))
        if spent_now[0] >= BUDGET:
            raise SystemExit(f"stopped at ${spent_now[0]:.2f}: the budget is ${BUDGET:.0f}")
        if not path.exists():
            continue
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"[{target} {model}] ours: {sum(r.get('score') is not None for r in lines)} scored,"
              f" {sum('error' in r for r in lines)} errors, total spent ${spent_now[0]:.2f}", flush=True)
        stored.push(path, f"{FOLDER}/monitors_{target}")


def main() -> None:
    api = _gigi_api()
    for target in stored.TARGETS:
        run(target, api)
    for target in stored.TARGETS:
        found = scores(target)
        print(f"{target}: " + ", ".join(f"{m} {len(s)} scores" for m, s in found.items()), flush=True)


if __name__ == "__main__":
    main()
