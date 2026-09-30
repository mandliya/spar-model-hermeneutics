# GPU agent brief — noninvasive deception probe

You are on an 80 GB GPU box. Goal: train probes on **original-turn** residuals (no “did you lie” in context) and score Gigi / AISI / OA honeypot.

Do **not** edit `tianq/` or `gigi/`. Work only in this directory:

```text
ravi/week3-noninvasive-probe
```

Week 1 (`../week1-dyl-vs-oa`) is DYL vs OA only. Reuse its `src/`, `data/`, and `dyl-probe-repro/` loaders. Do not train on Gigi or OA honeypot.

## Setup

Need ~80 GB VRAM (A100/H100 80GB). Check `nvidia-smi` before loading Gemma-3-27B-IT.

```bash
cd ravi/week3-noninvasive-probe
pip install -r ../week1-dyl-vs-oa/requirements.txt   # do not upgrade transformers to 5.x
# HF_TOKEN: week1 dyl-probe-repro/.env or /workspace/.env
python -m pytest tests/ -q
```

`.env` is gitignored. Never commit tokens.

## Run

```bash
python -u scripts/08_train_noninvasive_probe.py
python -u scripts/09_eval_noninvasive.py
```

Smoke: `--max-train 40 --datasets ascii_sandbagging --max-per-dataset 16 --positions last_content,mean_excl_last_5`.

Scores land in `results/`. Caches in `results/noninvasive_cache/` (gitignored). Protocol: [`NOTES.md`](NOTES.md).

Do not destroy the Vast instance unless the operator asks. GPU billing continues until it is destroyed from the **SPAR** team context.
