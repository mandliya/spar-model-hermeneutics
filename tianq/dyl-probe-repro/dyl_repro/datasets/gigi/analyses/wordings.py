"""Just asking vs the probe, each with its best follow-up wording, across Gigi's datasets.

Two tables, both against Gigi's label for the original answer, the probe read at
its own position (PROBE_POSITION):

    best wording      just asking, and the probe deduced to the original answer,
                      each with the wording that works best for it
    hard-to-catch     the probe on the denials alone: the lies the model doubles
    lies              down on against the truthful denials. Just asking scores 0.5
                      here by construction, since every answer is "no"

"Best" is held out: pick the wording on half the rows (each class halved
separately), score it on the other half, over SPLITS random splits. Picking and
scoring on the same rows would reward the luck of the maximum. For the
hard-to-catch lies each wording's own denials are the rows, so the wording also
decides which lies are hard to catch; the table gives the winner's counts.

Everything is computed from the rows each evaluation stored (probe scores,
parsed answers, labels): no model and no activations.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

from dyl_repro.core.probe import PROBE_POSITION
from dyl_repro.datasets.gigi.transfer import deduced

CONFOUNDED = "ascii_sandbagging"
SPLITS = 200
METHODS = ("just_asking", "probe_deduced", "hard_to_catch")


def _auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(y, scores)) if len(set(y.tolist())) == 2 else None


def _aligned(results: dict[str, dict], threshold: float) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """(Gigi's label, method -> wording -> one value per row), rows in one order across
    wordings. NaN where a method has nothing to say: an unparseable answer, and for the
    hard-to-catch lies an admission."""
    uids = None
    values: dict[str, dict[str, np.ndarray]] = {m: {} for m in METHODS}
    for followup, result in results.items():
        rows = {r["uid"]: r for r in result["rows"]}
        if uids is None:
            uids = sorted(rows)
            turn = np.array([np.nan if rows[u]["turn_lie"] is None else float(rows[u]["turn_lie"]) for u in uids])
        if sorted(rows) != uids:
            raise ValueError(f"{followup}: rows differ from the dataset's other follow-ups")
        admitted = np.array([np.nan if rows[u]["admitted"] is None else float(rows[u]["admitted"]) for u in uids])
        score = np.array([rows[u][f"score_{PROBE_POSITION}"] for u in uids])
        values["just_asking"][followup] = admitted
        values["probe_deduced"][followup] = np.where(np.isnan(admitted), np.nan,
                                                     deduced(score, admitted == 1, threshold))
        values["hard_to_catch"][followup] = np.where(admitted == 0, score, np.nan)
    return turn, values


def held_out_best(y: np.ndarray, method: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    """Best wording picked on one half of the rows, scored on the other, over SPLITS splits."""
    def auc(values: np.ndarray, idx: np.ndarray) -> float | None:
        keep = idx[~np.isnan(values[idx])]
        return _auroc(y[keep], values[keep])

    classes = [np.flatnonzero(y == c) for c in (0, 1)]
    scores, picks = [], []
    for _ in range(SPLITS):
        halves = [rng.permutation(idx) for idx in classes]
        pick = np.concatenate([h[: len(h) // 2] for h in halves])
        test = np.concatenate([h[len(h) // 2:] for h in halves])
        candidates = [(a, w) for w in method if (a := auc(method[w], pick)) is not None]
        best = max(candidates)[1]
        if (a := auc(method[best], test)) is not None:
            scores.append(a)
            picks.append(best)
    counts = Counter(picks)
    return {"auroc": float(np.mean(scores)), "spread_2_5_to_97_5": [float(np.percentile(scores, 2.5)),
                                                                     float(np.percentile(scores, 97.5))],
            "picked": dict(counts.most_common()),
            # A tie for most picked (ASCII's probe: two wordings at 97 of 200) goes to the
            # first by name, so the answer is the same on every run; `picked` shows the tie.
            "most_picked": max(sorted(counts), key=counts.__getitem__),
            # Splits where the winner's AUROC on the other half is undefined (a class
            # empty there) are skipped, so this can be below SPLITS.
            "splits": len(scores)}


def dataset_wordings(results: dict[str, dict], threshold: float, rng: np.random.Generator) -> dict:
    turn, values = _aligned(results, threshold)
    labelled = ~np.isnan(turn)
    y = turn[labelled].astype(int)
    best = {m: held_out_best(y, {w: v[labelled] for w, v in values[m].items()}, rng) for m in METHODS}
    winner = values["hard_to_catch"][best["hard_to_catch"]["most_picked"]][labelled]
    denied = y[~np.isnan(winner)]
    return {
        "n": [int(y.sum()), int((y == 0).sum())],
        "held_out_best": best,
        "winning_text": {m: results[best[m]["most_picked"]]["followup_text"] for m in METHODS},
        # The winning wording's own hard-to-catch rows: doubled down / truthful denials.
        "hard_to_catch_winner_n": [int(denied.sum()), int((denied == 0).sum())],
    }


def _cell(best: dict) -> str:
    low, high = best["spread_2_5_to_97_5"]
    return f"{best['auroc']:.2f} [{low:.2f}, {high:.2f}]"


def _text(text: str) -> str:
    return "“" + " ".join(text.split("\n")) + "”"


def markdown(per_dataset: dict[str, dict]) -> dict[str, str]:
    best_wording = ["| category | lie / truthful | black-box just asking, best wording (held out)"
                    " | white-box probe, best wording (held out) | winning wording (picked most often) |",
                    "|---|---|---|---|---|"]
    hard_to_catch = ["| category | white-box probe on the denials, best wording (held out)"
                     " | winning wording (picked most often) | its doubled down / truthful denials |", "|---|---|---|---|"]
    for name, d in per_dataset.items():
        best, text = d["held_out_best"], d["winning_text"]
        best_wording.append(
            f"| {name} | {d['n'][0]} / {d['n'][1]} | {_cell(best['just_asking'])} | {_cell(best['probe_deduced'])}"
            f" | black-box just asking: {_text(text['just_asking'])}<br>white-box probe: {_text(text['probe_deduced'])} |"
        )
        n = d["hard_to_catch_winner_n"]
        hard_to_catch.append(
            f"| {name} | {_cell(best['hard_to_catch'])} | {_text(text['hard_to_catch'])} | {n[0]} / {n[1]} |"
        )
    return {"best_wording": "\n".join(best_wording), "hard_to_catch": "\n".join(hard_to_catch)}


def compare(results: dict[str, dict[str, dict]], threshold: float, seed: int = 0) -> dict:
    """`results`: dataset -> follow-up -> the result evaluate stored for it."""
    rng = np.random.default_rng(seed)
    per_dataset = {name: dataset_wordings(r, threshold, rng) for name, r in results.items()}
    return {
        "position": PROBE_POSITION,
        "threshold": threshold,
        "seed": seed,
        "splits": SPLITS,
        "per_dataset": per_dataset,
        "markdown": markdown(per_dataset),
    }
