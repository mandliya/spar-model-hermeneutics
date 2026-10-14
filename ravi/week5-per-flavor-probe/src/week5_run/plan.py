"""Week 5 GPU sequence. Counts, smoke, extract, fit, then the report. In that order."""

from __future__ import annotations

from pathlib import Path

# Locked row counts. The run stops if a load disagrees.
EXPECTED_COUNTS = {
    "ascii_sandbagging": (310, 191, 119),
    "sycophancy_answer": (158, 38, 120),
    "sycophancy_feedback": (201, 116, 85),
    "mask_factual": (86, 55, 31),
    "mask_known_facts": (71, 43, 28),
    "mask_statistics": (47, 22, 25),
    "mask_continuations": (87, 73, 14),
    "mask_disinformation": (69, 58, 11),
    "capture_the_secret": (378, 73, 305),
    "alibi": (1276, 784, 492),
    "unanswerable": (93, 86, 7),
}

STAGE_ORDER = ("counts", "smoke", "extract", "fit", "report")
SMOKE_FLAVORS = ("alibi", "mask_known_facts")
SMOKE_POSITIONS = ("mean_answer", "reply_mean")
SMOKE_ROWS = 8

FULL_CACHE = "flavor_cache"
SMOKE_CACHE = "flavor_cache_smoke"


def cache_dirs(results: Path) -> tuple[Path, Path]:
    """Full extract and the smoke extract do not share a directory."""
    return results / FULL_CACHE, results / SMOKE_CACHE


def stages_to_run(state: dict, *, start: str | None, only: str | None) -> list[str]:
    if only is not None:
        if only not in STAGE_ORDER:
            raise SystemExit(f"unknown experiment {only!r}; known: {', '.join(STAGE_ORDER)}")
        return [only]
    if start is not None and start not in STAGE_ORDER:
        raise SystemExit(f"unknown experiment {start!r}; known: {', '.join(STAGE_ORDER)}")
    begin = 0 if start is None else STAGE_ORDER.index(start)
    done = state.get("stages", {})
    return [stage for stage in STAGE_ORDER[begin:] if done.get(stage, {}).get("status") != "ok"]


def run_stages(handlers: dict, state: dict, logbook, stages: list[str]) -> bool:
    """Run each experiment to completion. A failure is logged and later experiments do not start."""
    state.setdefault("stages", {})
    for stage in stages:
        logbook.append(f"experiment {stage} — started")
        try:
            body = handlers[stage]()
        except Exception as exc:
            logbook.append(f"experiment {stage} — STOPPED", f"{type(exc).__name__}: {exc}")
            state["stages"][stage] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            logbook.append("run stopped", f"Failed during {stage}. Later experiments were not started.")
            return False
        state["stages"][stage] = {"status": "ok"}
        logbook.append(f"experiment {stage} — ok", body or "")
    logbook.append("run finished", "All requested experiments finished.")
    return True


def counts_table(observed: dict[str, tuple[int, int, int]]) -> tuple[str, list[str]]:
    """Markdown table and the flavor names whose (n, lies, truthful) differ from EXPECTED_COUNTS."""
    header = "| flavor | n | lies | truthful | expected | |"
    rule = "| --- | ---: | ---: | ---: | --- | --- |"
    lines = [header, rule]
    mismatched = []
    for name, expected in EXPECTED_COUNTS.items():
        got = observed.get(name)
        if got is None:
            mismatched.append(name)
            lines.append(f"| {name} |  |  |  | {expected[0]} ({expected[1]}/{expected[2]}) | missing |")
            continue
        mark = "match" if got == expected else "DIFFERS"
        if got != expected:
            mismatched.append(name)
        lines.append(
            f"| {name} | {got[0]} | {got[1]} | {got[2]} | {expected[0]} ({expected[1]}/{expected[2]}) | {mark} |"
        )
    extra = [name for name in observed if name not in EXPECTED_COUNTS]
    for name in extra:
        mismatched.append(name)
        got = observed[name]
        lines.append(f"| {name} | {got[0]} | {got[1]} | {got[2]} |  | not in the locked table |")
    return "\n".join(lines), mismatched


def format_extract(detail: dict) -> str:
    lines = [
        f"- flavor: {detail['flavor']}",
        f"- rows: {detail['n']} ({detail['n_lie']} lies / {detail['n_honest']} truthful)",
        f"- seconds: {detail['seconds']:.1f}",
        f"- forward pass: {'reused cache' if detail['reused'] else 'ran'}",
    ]
    for family in detail["families"]:
        lines.append(f"- {family['family']} cache: `{family['path']}`")
        lines.append(f"  - prompt tokens lost: {family['n_lost_prompt']}")
        for position, info in family["positions"].items():
            shape = "×".join(str(part) for part in info["shape"])
            lines.append(f"  - {position}: {shape}, finite rows {info['finite']}/{info['rows']}")
    return "\n".join(lines)
