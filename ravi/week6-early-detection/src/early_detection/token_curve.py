"""In-category and transfer AUROC at each token offset. Fit only; no model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from early_detection.token_mark import BAR, OFFSETS, TAU_ORDER, tau90
from noninvasive_probe.categories import CATEGORIES, FEW, FLAVOR_ORDER
from noninvasive_probe.tianq_read import aisi_logits, fit_aisi_logistic

FOLDS = 5
SEED = 0
POSITIONS = ("prompt_last",) + tuple(str(offset) for offset in reversed(OFFSETS))


def _auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    if not np.isfinite(scores).all():
        return None
    return float(roc_auc_score(y, scores))


def _finite_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep = np.isfinite(x).all(axis=1)
    return x[keep], y[keep]


def out_of_fold_scores(x: np.ndarray, y: np.ndarray, folds: int = FOLDS, seed: int = SEED) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n_splits = min(folds, int(y.sum()), int((y == 0).sum()))
    if n_splits < 2:
        raise RuntimeError("need at least two rows of each class to hold one out")
    scores = np.full(len(y), np.nan, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(np.zeros(len(y)), y):
        weights = fit_aisi_logistic(x[train_idx], y[train_idx])
        scores[test_idx] = aisi_logits(x[test_idx], *weights)
    if not np.isfinite(scores).all():
        raise RuntimeError("a row was left out of every fold")
    return scores


def _cell(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = _finite_rows(x, y)
    n_lie = int((y == 1).sum()) if len(y) else 0
    n_honest = int((y == 0).sum()) if len(y) else 0
    cell = {"n_lie": n_lie, "n_honest": n_honest, "auroc": None, "transfer_auroc": None, "few": True}
    if n_lie < FEW or n_honest < FEW:
        return cell
    cell["few"] = False
    cell["auroc"] = _auroc(y, out_of_fold_scores(x, y))
    return cell


def _transfer(train_x: np.ndarray, train_y: np.ndarray, others: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
    train_x, train_y = _finite_rows(train_x, train_y)
    if int((train_y == 1).sum()) < FEW or int((train_y == 0).sum()) < FEW:
        return None
    weights = fit_aisi_logistic(train_x, train_y)
    scores = []
    for x, y in others:
        x, y = _finite_rows(x, y)
        if int((y == 1).sum()) < FEW or int((y == 0).sum()) < FEW:
            continue
        auc = _auroc(y, aisi_logits(x, *weights))
        if auc is not None:
            scores.append(auc)
    if not scores:
        return None
    return float(np.mean(scores))


def flavor_tables(cache: dict[str, dict]) -> list[dict]:
    """One record per flavor. cache[flavor] has y, reply_len, and acts[position]."""
    tables = []
    for flavor in FLAVOR_ORDER:
        saved = cache.get(flavor)
        if saved is None:
            continue
        y = np.asarray(saved["y"], dtype=np.int64)
        reply_len = np.asarray(saved["reply_len"], dtype=np.int64)
        one_token = bool(len(reply_len) and int(np.median(reply_len)) <= 1)
        rows = []
        auc_by_offset: dict[str, float | None] = {}
        for position in POSITIONS:
            x = np.asarray(saved["acts"][position], dtype=np.float64)
            cell = _cell(x, y)
            others = []
            for other, other_saved in cache.items():
                if other == flavor:
                    continue
                others.append((np.asarray(other_saved["acts"][position], dtype=np.float64), np.asarray(other_saved["y"], dtype=np.int64)))
            cell["transfer_auroc"] = _transfer(x, y, others)
            cell["position"] = position
            rows.append(cell)
            auc_by_offset[position] = cell["auroc"]
        lead = tau90(auc_by_offset, one_token_reply=one_token)
        offset0 = next(cell["auroc"] for cell in rows if cell["position"] == "0")
        earlier = [cell["auroc"] for cell in rows if cell["position"] != "0" and cell["auroc"] is not None]
        tables.append(
            {
                "flavor": flavor,
                "label": CATEGORIES[flavor],
                "n": int(len(y)),
                "n_lie": int((y == 1).sum()),
                "n_honest": int((y == 0).sum()),
                "one_token_reply": one_token,
                "median_reply_tokens": int(np.median(reply_len)) if len(reply_len) else 0,
                "rows": rows,
                "lead": lead,
                "offset0_above_earlier": bool(offset0 is not None and earlier and offset0 > max(earlier)),
                "bar": BAR,
            }
        )
    return tables


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def _row_label(position: str) -> str:
    if position == "prompt_last":
        return "prompt_last"
    if position == "1":
        return "1 (emits the bad token, not lead time)"
    if position == "0":
        return "0 (the bad token, not lead time)"
    return position


def render_curve(tables: list[dict], *, monitor_measured: bool) -> str:
    lines = ["# Token lead time", "", "t* is the first reply token. In-reply offsets before it are empty when the reply opens on the act.", ""]
    for table in tables:
        lines.append(f"## {table['label']}")
        lines.append("")
        lines.append("| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in table["rows"]:
            if row["n_lie"] == 0 and row["n_honest"] == 0:
                lies, honest = "", ""
            else:
                lies, honest = str(row["n_lie"]), str(row["n_honest"])
            lines.append(
                f"| {_row_label(row['position'])} | {_fmt(row['auroc'])} | {lies} | {honest} | {_fmt(row['transfer_auroc'])} |"
            )
        lines.append("")
    lines.extend(_sentences(tables, monitor_measured=monitor_measured))
    lines.append("")
    return "\n".join(lines)


def _sentences(tables: list[dict], *, monitor_measured: bool) -> list[str]:
    bits = []
    for table in tables:
        lead = table["lead"]
        if table["one_token_reply"]:
            bits.append(f"{table['label']}: τ90 is undefined because the reply is one token.")
            continue
        peak = lead["peak_offset"]
        peak_auc = lead["peak_auroc"]
        peak_text = f" Peak in the lead set is {peak} at {_fmt(peak_auc)}." if peak else ""
        if lead["tau90"]:
            bits.append(f"{table['label']}: τ90 is {lead['tau90']}.{peak_text}")
        else:
            bits.append(f"{table['label']}: no lead offset reached 0.90.{peak_text}")
    higher = [table["label"] for table in tables if table["offset0_above_earlier"]]
    if higher:
        second = "Offset 0 is higher than every earlier scored offset for " + ", ".join(higher) + ". The probe is reading the bad token as it is emitted."
    else:
        second = "Offset 0 is not higher than every earlier scored offset on any flavor that has both."
    transfers = []
    for table in tables:
        if table["lead"]["tau90"] is None:
            continue
        row = next(item for item in table["rows"] if item["position"] == table["lead"]["tau90"])
        if row["transfer_auroc"] is not None:
            transfers.append((table["label"], row["transfer_auroc"]))
    if not transfers:
        third = "Transfer at τ90 is not measured, because no flavor has a τ90 with a transfer score."
    elif all(auc > 0.50 for _, auc in transfers):
        third = "Transfer at τ90 is above 0.50 (" + ", ".join(f"{name} {_fmt(auc)}" for name, auc in transfers) + "). Early intent is shared where a finished-reply probe was not."
    elif all(auc <= 0.55 for _, auc in transfers):
        third = "Transfer at τ90 stays near 0.50 (" + ", ".join(f"{name} {_fmt(auc)}" for name, auc in transfers) + "). Early detection did not fix generalization."
    else:
        third = "Transfer at τ90 is mixed (" + ", ".join(f"{name} {_fmt(auc)}" for name, auc in transfers) + ")."
    fourth = "The monitor gap is not measured." if not monitor_measured else "Monitor gap is in the JSON."
    return [" ".join(bits), second, third, fourth]
