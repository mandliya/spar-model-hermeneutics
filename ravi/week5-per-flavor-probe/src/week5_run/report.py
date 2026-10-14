"""Turn a finished flavor-matrix JSON into the week 5 report, including the curve reading."""

from __future__ import annotations

POSITION_ORDER = (
    "mean_answer",
    "reply_mean",
    "prompt_last",
    "reply_last",
    "last_content",
    "end_of_turn",
    "last_newline",
    "mean_excl_last_5",
    "mean_first_5",
)

TIANQ_REFERENCE = {
    "prompt_last": (0.71, 0.50),
    "reply_last": (0.73, 0.50),
    "reply_mean": (0.80, 0.50),
}


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _diag(matrix: dict, flavor: str) -> dict | None:
    for row in matrix.get("rows", []):
        if row["train"] != flavor:
            continue
        for cell in row["cells"]:
            if cell["test"] == flavor:
                return cell
    return None


def _transfer(matrix: dict, train: str, test: str) -> dict | None:
    for row in matrix.get("rows", []):
        if row["train"] != train:
            continue
        for cell in row["cells"]:
            if cell["test"] == test and cell["kind"] == "transfer":
                return cell
    return None


def curve_table(payload: dict) -> str:
    lines = [
        "| Position | In-category | Transfer | Cells ≥ 0.7 |",
        "| --- | ---: | ---: | ---: |",
    ]
    positions = payload.get("positions", {})
    for position in POSITION_ORDER:
        matrix = positions.get(position)
        if matrix is None:
            lines.append(f"| {position} |  |  |  |")
            continue
        lines.append(
            f"| {position} | {_fmt(matrix.get('mean_in_category'))} | "
            f"{_fmt(matrix.get('mean_transfer'))} | "
            f"{matrix.get('transfer_cells_at_least_0_7', 0)} of {matrix.get('transfer_cells', 0)} |"
        )
    return "\n".join(lines)


def _transfer_sentence(value) -> str:
    if value is None:
        return "`reply_mean` transfer was not fit."
    number = float(value)
    if abs(number - 0.50) <= 0.05:
        verdict = "stays near 0.50, so mean-pooling does not fix flavor-specificity"
    elif number >= 0.60:
        verdict = "clears 0.50, so the gain is the pool rather than training on the category"
    else:
        verdict = "sits between 0.50 and 0.60"
    return f"`reply_mean` transfer is {_fmt(number)}. Tianq's transfer at this position is 0.50. This {verdict}."


def _peaks_sentence(payload: dict) -> str:
    best: dict[str, tuple[float, str]] = {}
    for position, matrix in payload.get("positions", {}).items():
        for row in matrix.get("rows", []):
            cell = _diag(matrix, row["train"])
            if cell is None or cell.get("few") or cell.get("auroc") is None or cell.get("kind") != "in_category":
                continue
            previous = best.get(row["train"])
            if previous is None or cell["auroc"] > previous[0]:
                best[row["train"]] = (float(cell["auroc"]), position)
    if not best:
        return "No in-category diagonal was fit, so there is no per-flavor peak."
    lines = [f"- {name}: {_fmt(score)} at `{position}`" for name, (score, position) in best.items()]
    peaks = {position for _score, position in best.values()}
    if len(peaks) == 1:
        lines.append(f"Every trained flavor peaks at `{peaks.pop()}`.")
    else:
        lines.append("Flavors do not share one peak position. That is the token-position question, answered from the diagonals.")
    return "\n".join(lines)


def _ascii_sentence(payload: dict) -> str:
    positions = payload.get("positions", {})
    wanted = ("last_content", "reply_last", "reply_mean", "mean_excl_last_5")
    found = []
    for position in wanted:
        matrix = positions.get(position)
        if matrix is None:
            continue
        cell = _diag(matrix, "ascii_sandbagging")
        if cell is None or cell.get("auroc") is None:
            continue
        found.append(f"`{position}` {_fmt(cell['auroc'])}")
    if not found:
        return "ASCII diagonals were not in this run."
    last = positions.get("last_content")
    dropped = positions.get("mean_excl_last_5")
    last_cell = _diag(last, "ascii_sandbagging") if last else None
    drop_cell = _diag(dropped, "ascii_sandbagging") if dropped else None
    numbers = ", ".join(found)
    if (
        last_cell
        and drop_cell
        and last_cell.get("auroc") is not None
        and drop_cell.get("auroc") is not None
        and last_cell["auroc"] >= 0.90
        and drop_cell["auroc"] <= 0.60
    ):
        return f"ASCII diagonals: {numbers}. The last-token read is near 1 and `mean_excl_last_5` drops, which is the letter confound."
    return f"ASCII diagonals: {numbers}. This does not match the letter-confound pattern (last token near 1, `mean_excl_last_5` down at 0.60 or below)."


def _pair_sentence(payload: dict) -> str:
    matrix = payload.get("positions", {}).get("reply_mean")
    if matrix is None:
        return "The alibi / MASK-known-facts pair was not scored at `reply_mean`."
    one = _transfer(matrix, "alibi", "mask_known_facts")
    two = _transfer(matrix, "mask_known_facts", "alibi")
    if one is None or two is None or one.get("auroc") is None or two.get("auroc") is None:
        return "One direction of alibi ↔ MASK known facts is missing at `reply_mean`."
    others = [
        cell["auroc"]
        for row in matrix["rows"]
        for cell in row["cells"]
        if cell["kind"] == "transfer" and not cell.get("few") and cell.get("auroc") is not None
        and not (
            (row["train"] == "alibi" and cell["test"] == "mask_known_facts")
            or (row["train"] == "mask_known_facts" and cell["test"] == "alibi")
        )
    ]
    pair = f"alibi → MASK known facts {_fmt(one['auroc'])}, MASK known facts → alibi {_fmt(two['auroc'])}"
    pair_high = one["auroc"] >= 0.60 and two["auroc"] >= 0.60
    others_low = all(score < 0.55 for score in others) if others else False
    if pair_high and others_low:
        return f"At `reply_mean`, {pair}. Those are the only transfer cells clearly above 0.5."
    return f"At `reply_mean`, {pair}. Other transfer cells are not all near 0.5, so this pair is not the only exception."


def render_report(payload: dict) -> str:
    signature = payload.get("signature", {})
    markdowns = payload.get("markdown", {})
    sections = []
    for position in POSITION_ORDER:
        body = markdowns.get(position)
        if body is None:
            continue
        reference = TIANQ_REFERENCE.get(position)
        extra = ""
        if reference is not None:
            extra = (
                f"\n\nTianq's published no-follow-up means at this position: "
                f"in-category {reference[0]:.2f}, transfer {reference[1]:.2f}."
            )
        sections.append(f"## {position}\n\n{body}{extra}")
    reading = "\n\n".join(
        [
            _transfer_sentence((payload.get("positions") or {}).get("reply_mean", {}).get("mean_transfer")),
            _peaks_sentence(payload),
            _ascii_sentence(payload),
            _pair_sentence(payload),
        ]
    )
    header = (
        "# Flavor matrix\n\n"
        "One probe per flavor, no follow-up in the forward pass. "
        f"Layer {signature.get('layer', 58)}, max length {signature.get('max_length', 'unknown')}, "
        f"{signature.get('folds', 5)}-fold diagonal.\n\n"
        "The generic probe is not the source of these numbers.\n\n"
        "## Curve\n\n"
        f"{curve_table(payload)}\n\n"
        "## Reading\n\n"
        f"{reading}\n"
    )
    body = "\n\n".join(sections)
    return header + ("\n" + body + "\n" if body else "")
