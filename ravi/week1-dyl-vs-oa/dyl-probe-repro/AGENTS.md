# GPU agent brief — official DYL → OA MBPP-Honeypot

You are on an 80 GB GPU box. Goal: score AISI’s released Gemma-3-27B-IT **Did You Lie** probe on OA’s reward-hacking set (`taufeeque/mbpp-hardcode`), using Tianq’s read position and labels.

Do **not** edit `tianq/` or `gigi/`. Work only in this directory:

```text
ravi/week1-dyl-vs-oa/dyl-probe-repro
```

This folder is a copy of Tianq’s pipeline plus dataset `oa_mbpp`. `../scripts/03_eval_transfer.py` reads the **yes/no token** and is invalid for the official checkpoint. Do not run it for this job.

## Why this pipeline

- Probe: `ai-safety-institute/dyl-google-gemma-3-27b-it` / `l_58_lm_0_01_ar_lr.pt`
- Read **`prompt_last`** (token before the answer), `hidden_states[59]` (layer 58). The answer token mismatches the checkpoint’s stored `dataset_mean` (rel err ~0.33 vs ~0.0027).
- Labels (AISI): lied then denied → lie; lied then admitted → honest; honest then confessed → **excluded**. Also report original `code` vs `hardcode`.
- 500 test tasks × 2 completions = 1000 conversations per follow-up, answers generated on-policy at temperature 0.

## Setup

Need ~80 GB VRAM (A100/H100 80GB). Check `nvidia-smi` before loading the model.

```bash
cd ravi/week1-dyl-vs-oa/dyl-probe-repro
# repo must also contain gigi/ (follow-up parsers import gigi/*/src/common.py)
ls ../../../../gigi/ascii-sandbagging/src/common.py

cp .env.example .env   # fill HF_TOKEN; leave HF_DATASET_REPO unless told otherwise
# HF_TOKEN needs: gated google/gemma-3-27b-it, write to model-hermeneutics/dyl-probe-repro
pip install -r requirements.txt   # do not upgrade transformers to 5.x
```

`.env` is gitignored. Never commit tokens.

## Run (tmux; extract is hours)

```bash
# smoke: 8 rows, one follow-up
./run.sh extract --dataset oa_mbpp --followup dyl --limit 8
./run.sh evaluate --dataset oa_mbpp --followup dyl

# full default: dyl, dyl_truthful, dyl_cheat, dyl_falsify
./logged.sh 'tmux new -d -s oa ./run_oa_mbpp.sh'
./tail_run.sh
```

`FOLLOWUPS=all ./run_oa_mbpp.sh` is the 13-wording sweep; only if the default finishes and there is budget.

Extract is safe to kill and rerun: stored rows are skipped. Evaluate is CPU-only once hidden states are on disk / Hub.

## Done when

Logs and `oa_mbpp/<followup>/results/` are on `model-hermeneutics/dyl-probe-repro`. Evaluate prints, for each wording, probe AUROC at `prompt_last` vs `answer_first`, just-asking AUROC, and the two label schemes.

Do not destroy the Vast instance unless the operator asks. GPU billing continues until it is destroyed from the **SPAR** team context.
