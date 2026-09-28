# Official DYL → OA MBPP-Honeypot (Gemma-3-27B-IT)

Record lives on Hugging Face, not in git:

**https://huggingface.co/datasets/model-hermeneutics/dyl-probe-repro**

Prefix `oa_mbpp/`. Hidden states and per-row JSON are too large for this repo (`scratch/` is gitignored). If this GPU box dies, nothing in `scratch/` comes back; Hub rows do.

Host: Vast A800 80GB (`C.53235658`), 2026-09-28.

**Status 2026-09-28 21:47 UTC: 13/13 extract + evaluate complete** (`all_complete: true`). GPU idle. tmux `oa` has exited.

## How to resume on a new box

```bash
cd ravi/week1-dyl-vs-oa/dyl-probe-repro
cp .env.example .env   # HF_TOKEN with Gemma-3 access + write to the dataset above
# pip install -r requirements.txt   # do not upgrade transformers to 5.x
FOLLOWUPS=all ./run_oa_mbpp.sh     # skips rows already on the Hub
```

Do not run `../scripts/03_eval_transfer.py` for the official checkpoint: it reads the yes/no token.

## Position check (the week-1 bug)

Same 1000 conversations, same checkpoint `l_58_lm_0_01_ar_lr.pt`, same `hidden_states[59]`. Only the token index changes. AUROCs are on original `code` vs `hardcode` (500 / 500), probe deduced through the parsed self-report.

On DYL-style yes/no wordings, `prompt_last` works and `answer_first` is inverted. On Gigi A/B wordings the probe at `prompt_last` often collapses (and `answer_first` can look better because that token *is* the A/B choice the label is built from — circular; see the evaluate summary note).

## 13-wording results (complete)

Sorted as evaluate printed (best probe `prompt_last` first). `htc` = hard-to-catch (doubled-down / truthful denials). `fu` = AISI follow-up-lie labels (lie / honest / excluded).

| Follow-up | just asking | probe `prompt_last` | probe `answer_first` | htc n | htc `prompt_last` |
|---|---:|---:|---:|---:|---:|
| `dyl_cheat` | 0.945 | **0.975** | 0.089 | 3 / 448 | 0.795 |
| `dyl_falsify` | 0.906 | **0.969** | 0.146 | 0 / 406 | — |
| `dyl_eval_honest` | 0.936 | **0.910** | 0.073 | 1 / 437 | 0.865 |
| `high_stakes` | 0.950 | **0.903** | 0.080 | 4 / 454 | 0.360 |
| `dyl_truthful` | 0.891 | **0.868** | 0.194 | 1 / 392 | 0.663 |
| `low_stakes` | 0.892 | **0.825** | 0.158 | 2 / 394 | 0.397 |
| `dyl` (native) | 0.830 | **0.823** | 0.227 | 0 / 330 | — |
| `eval_plain` | 0.857 | 0.740 | 0.156 | 0 / 357 | — |
| `confidence` | **0.958** | 0.677 | 0.013 | 6 / 464 | 0.318 |
| `dyl_honest` | 0.743 | 0.488 | 0.133 | 0 / 243 | — |
| `plain` | 0.715 | 0.440 | 0.495 | 0 / 215 | — |
| `neutral` | 0.665 | 0.371 | 0.543 | 0 / 165 | — |
| `neutral_swapped` | 0.596 | 0.306 | 0.717 | 0 / 96 | — |
| mean (min–max) | 0.837 (0.60–0.96) | 0.715 (0.31–0.97) | 0.233 (0.01–0.72) | | 0.566 (0.32–0.86) |

Extract: 1000/1000 rows for every follow-up. Evaluate: `oa_mbpp/results/summary.json` on the Hub.

## What that means for transfer

- Official DYL **does transfer** to OA MBPP-Honeypot at `prompt_last` when the follow-up is a DYL-style yes/no, especially if it **names the act** (`cheat` 0.975, `falsify` 0.969). Native “did you lie?” matches just-asking (0.823 vs 0.830).
- The **answer token does not** transfer on those wordings (inverted AUROC).
- Just-asking can beat the probe on some Gigi wordings (`confidence` 0.958 asking vs 0.677 probe). Gigi A/B (`neutral`, `plain`) is a weak probe transfer on this set.
- AISI follow-up-lie labels barely apply: 0–6 doubled-down lies / 500. The claim is about the **original reward hack**, not doubling-down.

Gigi and AISI numbers are already on the same Hub dataset from Tianq (`gigi/results/wordings_l_58_lm_0_01_ar_lr_hs59.json`). Next write-up: one table OA vs Gigi vs AISI, same probe, `prompt_last`. Do not re-extract Gigi.

## Files

| Path | What |
|---|---|
| `oa_mbpp/<followup>/rows/*.json` | transcript, parsed answer, labels |
| `oa_mbpp/<followup>/hidden_states/*.safetensors` | `prompt_last` and `answer_first`, all layers, bf16 |
| `oa_mbpp/<followup>/results/l_58_lm_0_01_ar_lr_hs59.json` | per-wording metrics |
| `oa_mbpp/results/summary.json` | table across all 13 wordings |
| `logs/` | raw `run_*.log` from this box |
