# GPU agent — Goodfire passage peak on Gemma 27B

You are on an 80 GB GPU box (A100/H100). Work only in this directory:

```text
ravi/week6-early-detection
```

Your job is to run `scripts/05_goodfire_peak.py` and leave `GOODFIRE_PEAK.md` filled from the full run. Ravi will read that file. Do not leave the number only in the terminal.

This is the closer scoring rule for the Goodfire experiment that already ran. The earlier result is `GOODFIRE.md`: last token of a 2048-token prefix, with reasoning mixed into the content channel, test AUROC 0.743 for `pl_peek`. Do not overwrite that file. Do not rerun `scripts/03_goodfire_dom.py` or `scripts/04_goodfire_monitor.py` as their own jobs. Script 05 reuses the synthetic direction from 03 and rebuilds that cache only if `results/goodfire_cache/synth/` is missing.

## The number

The result is the **test peak AUROC** of the contrast that script 05 selects by train peak AUROC. A passage is one channel of one step (reasoning, visible reply, or tool call). The score is the maximum dot product inside that channel. Earlier channels of the same step stay in the prefix and are not part of the span.

Kimi's own probe in the paper (arXiv 2609.19101) reaches AUROC 0.94 on ImpossibleBench. That number is cited, not measured here. This machine reads Kimi's text with Gemma 27B. Do not load Kimi, GLM, or Qwen. Do not describe the Gemma number as 0.94, and do not describe the Last column as the old 0.743. That 0.743 is a different prefix.

The lead paragraph in the report is steps before an attempted or enacted hack. It is not the paper's resampling test. Do not add a resampling run.

## Do not

- Run `scripts/01_lead_curve.py` or `scripts/02_token_curve.py`. Those jobs are finished. `CURVE.md` stays as it is. The old instructions are in `AGENTS_TOKEN.md`.
- Sweep layers, models, contrasts, or context lengths looking for a higher AUROC. Layer 58 and 8192 tokens are the run. 4096 is only the out-of-memory fallback below.
- Edit the scoring code, the tests, or the chat template to make a span succeed. A smoke failure is a stop.
- Download a different dataset, generate traces, or invent activations. If the four data files below are missing, stop.
- Call any paid monitor API.
- Commit `.env`, tokens, `results/goodfire_peak_m*.jsonl`, or `results/goodfire_cache/`. Do not commit or push unless Ravi asks in this session.
- Destroy the Vast instance.

## Setup

Need one GPU with about 80 GB, and `transformers` below 5. Gemma 3 27B is gated. The token has to belong to an account that has accepted the Gemma license.

Script 05 reads `HF_TOKEN` only from `ravi/week6-early-detection/.env` (a line `HF_TOKEN=...`) or from the process environment. It does not read `dyl-probe-repro/.env` or `/workspace/.env`. If the token is in one of those and not in the week 6 file, copy that one variable into `ravi/week6-early-detection/.env`. Do not print the token. Do not write it into `LOG.md`.

Weights download into `/dev/shm/hf_home`. That directory is wiped on reboot, so a previous Gemma download on this box may be gone.

These four files must already exist. They are the Goodfire release, not something to fetch mid-run:

```text
data/goodfire/synthetic_data/dataset.jsonl
data/goodfire/synthetic_data/contrasts.json
data/goodfire/impossiblebench-kimi-k3/traces.jsonl
data/goodfire/impossiblebench-kimi-k3/labels/judge_passages.jsonl
```

```bash
cd ravi/week6-early-detection
nvidia-smi
df -h /dev/shm
python -c "import transformers; print(transformers.__version__)"
python -m pytest tests/test_goodfire_peak.py tests/test_goodfire_dom.py -q
test -f .env && grep -q '^HF_TOKEN=.' .env && echo "HF_TOKEN present" || echo "HF_TOKEN missing"
```

Stop before loading Gemma if any of these are true: no GPU near 80 GB, `/dev/shm` has under 60 GB free, `transformers` is 5 or newer, the tests fail, `HF_TOKEN` is missing, or any of the four data files is missing. Append that reason to `LOG.md`.

## Run

One process. Do not start a second copy while the first is alive. The full run is one forward pass per assistant step and takes many hours. Leave it in the foreground with `python -u`.

1. **Smoke.** Four new steps, then a fit. The fit will say `PARTIAL`. That is expected. It is not the result.

```bash
python -u scripts/05_goodfire_peak.py --limit 4
```

Continue only if the log prints `first step scored` and the process does not exit with a span error. A `PARTIAL` note after that is fine.

2. **Stop on a bad smoke.** If it exits with "no usable span", "No channel was scored", or "the first steps produced no scores", delete `results/goodfire_peak_m8192.jsonl` and stop. Failed rows are cached, so a rerun would skip them. Do not start the full command. Do not patch the locator. Write the traceback into `LOG.md`.

3. **Full run.** Same command, no limit. The four smoke steps stay cached and are not scored again.

```bash
python -u scripts/05_goodfire_peak.py
```

If the process is killed for memory, rerun with `--max-length 4096`. That writes `results/goodfire_peak_m4096.jsonl`. Do not delete the 8192 file, and do not mix the two lengths in one report. If it dies for any other reason and the log does not say to delete the jsonl, rerun the same command. It resumes from `results/goodfire_peak_m8192.jsonl`.

4. **Read the file the script wrote.** `GOODFIRE_PEAK.md` and `results/goodfire_peak.json`. The script also appends the report to `LOG.md`.

## When the table is not a result

- The file says `PARTIAL`, or the command was the `--limit 4` run. Rerun without `--limit`. Do not report that table.
- The process exits with `passage locate failed`. It may have written `GOODFIRE_PEAK.md` anyway. That table is not a result. Say so, and stop.
- `scored` is far below `passages` because the job is still going. Wait for the process to finish, then read the file again.

## What to report

From the finished `GOODFIRE_PEAK.md`, with no `PARTIAL` line and no locate-failure exit:

- The selected contrast, its test peak AUROC, and its test last-token AUROC.
- Passages scored, locate failures, how many prefixes were truncated, and how many lost the start of the passage.
- The lead sentence as written. Do not turn it into the paper's resampling claim.

State that Gemma read Kimi's text, and that 0.94 is the paper's Kimi probe. Leave `GOODFIRE.md` and `CURVE.md` unchanged.
