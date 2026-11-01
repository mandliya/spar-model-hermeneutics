# Alignment faking with the reasoning hidden (closed)

**Question.** With an organism's reasoning hidden from everyone, does its own faking probe beat probes of readers that read the same visible text (prompt and final answer), and the words alone? A clear yes would mean its weights carry the faking beyond the text: John's hermeneutics target, and the case for distilling it into a reader.

It is built on `../../af-probe-repro`, which reproduces the AF paper (https://icml.cc/virtual/2026/79535) with the paper's own code; the gate imports that code and never edits it.

## Design

- Cells: each organism where it fakes, Pacifist on highest_military_bid and report_score, HAL9000 on report_failure and report_score. Both of the paper's variants: any lie vs honest, and faking vs honest (observed only).
- Views (`views.py`): `paper` (the paper's read, tying the gate to the repro), `output` (whole rollout, read at the answer) and `hidden` (reasoning removed).
- Models: the organism (target); the other organism, the plain parent Llama-3-70B-Instruct and Llama-3.1-8B (substitutes). The words alone are TF-IDF + logistic regression.
- Probes (`score.py`): every model trains its own on the paper's synthetic set, as the paper's 5 fold models, at the paper's selected design (difference of means, layer 37, last token) and its headline design (LR C = 1e4, layer 33). Paired bootstrap intervals on every gap to the target.

## Run

- `../../af-probe-repro/run.sh gate-extract <model>` (GPU): `extract.py`, one model's activations for every cell and view
- `../../af-probe-repro/run.sh gate-score all` (CPU): `score.py`, the gate, with its chart (`charts.py`)
- `scripts/overnight.sh`: the driver that ran it on one box after the repro

## Result (Oct 31)

No cell has a ceiling. The gate's paper read matches the paper in every cell.

- With the reasoning shown, any model reads the faking, even an 8B.
- With it hidden, the organism's own probe never beats both the readers and the words.
- Lead: on HAL9000 report_failure (faking vs honest, 58 rollouts), both faking fine-tunes read it (0.60 to 0.69) and the plain parent and the words don't.

The report is in `report/af_gate.md`. Results are on the Hub: `af-gate/results/gate.json`, `af-gate/charts/gate.png` and `af-gate/activations/`.

The first attempt (Oct 30, `bench-af/hal9000-adapter` and our own pipeline) was superseded and its code removed. Its results stay on the Hub under `af/`.
