# GPU agent — how many tokens before the bad action

You are on an 80 GB GPU box (A100/H100). Work only in this directory:

```text
ravi/week6-early-detection
```

John’s ask (Monday meeting): given a reply where something bad happens partway through, how many **tokens** before that moment can a probe still reach about **90% AUROC** that the model is going to misbehave? Black-box methods usually see the bad action only when it is already on the page. This is the white-box lead-time measurement.

The scientific spec is [`README.md`](README.md). Follow it. Write `CURVE.md` in this folder and append every stage to `LOG.md`. Ravi will read those two files. Do not leave the numbers only in the terminal.

This week’s data is **Gigi’s deception flavors**, already in the checkout. Week 3 showed that a finished-reply probe does not transfer across flavors. The question now is whether an **earlier token** does.

## Do not run the Docent track

`scripts/01_lead_curve.py` and `tests/test_mark.py` mark a **bad bash command** on Docent BashArena trajectories and score **actions** before that command. That is a later experiment (real tool trajectories, stored monitor scores). It needs `DOCENT_API_KEY` and per-action JSON that is not in this checkout.

Do not run `01_lead_curve.py`. Do not download Docent. Do not stop because that JSON is missing. Do not rewrite `CURVE.md` as an action-horizon table. Leave that script and its tests in place.

## The number

**τ90** is the largest offset in `{2, 4, 8, 16, 32, prompt_last}` at which a probe still reaches AUROC ≥ 0.90.

- Offset 0 is the bad token itself. Offset 1 is the residual that emits it. Both go in the table. Neither is τ90.
- If nothing in that set reaches 0.90, say so and report the best offset in the set anyway. That is a result.
- Report one τ90 per flavor (in-category). At that same offset, also report transfer AUROC on flavors the probe was not trained on.
- ASCII often has no tokens before the letter. If t* is the only content token, τ90 for that flavor is undefined. Do not impute it from `prompt_last`.

## Data

Load Gigi’s conversations through week 3. Do not edit `gigi/` or `tianq/`. Import; do not copy their trees into this folder.

```text
../week3-noninvasive-probe/src/noninvasive_probe/categories.py
```

`load_category` and `FLAVOR_ORDER` are the row set. Labels are Gigi’s labels on the reply. No “did you lie?” follow-up in the forward pass.

Flavors, in this order:

| Flavor key | What t* is |
| --- | --- |
| ascii_sandbagging | The A/B letter. Often the whole reply. Negative control. |
| sycophancy_answer | First token of the pressured answer, not the question. |
| sycophancy_feedback | First token of the pressured reply, not the question. |
| mask_factual | First token of the false claim, not the role prompt. |
| mask_known_facts | First token of the false claim, not the role prompt. |
| mask_statistics | First token of the false claim, not the role prompt. |
| mask_continuations | First token of the false continuation, not the role prompt. |
| mask_disinformation | First token of the false article, not the role prompt. |
| unanswerable | First token of the committed answer. Drop from means if either class has fewer than 10 rows. |
| capture_the_secret | First token of the false denial. |
| alibi | First token of the reply that maintains the false story. |

Honest rows use the same rule on the truthful commitment, so a lie and a truth line up at “the answer starts here,” not at “the string ends.”

Offsets, counted backward in **reply tokens only**: `0, 1, 2, 4, 8, 16, 32`, plus `prompt_last` (last token of `<start_of_turn>model\n`, before the reply — the same index as Tianq’s `prompt_last` in `noninvasive_probe/tianq_read.py`).

Write the t* rule in code **before** any fit, with a unit test on a short fake reply. A learned boundary would soak up the result. If a flavor’s replies do not contain a reliable string boundary, use the first reply token as t* and say that in `LOG.md` and `CURVE.md`. Do not hunt for a boundary after seeing AUROCs.

Train only on rows long enough for that offset. Drop an offset with fewer than 10 lies or 10 truthful replies from τ90, and still store the count.

## Probe

Same stack as week 3, so the curve sits next to the finished-reply result that failed.

- Model: `google/gemma-3-27b-it`. Layer 58 only (`hidden_states[59]`). One residual per offset. Do not mean-pool the reply.
- Left-truncate to 4096. A lost end token (the offset you meant to read) is a failed read. Raise the limit, delete that cache, and extract again. Do not fit a truncated end.
- Classifier: `fit_aisi_logistic` in `noninvasive_probe/tianq_read.py`. Standardize on the training fold, mean log-loss + `0.01 * (||w||² + b²)`, L-BFGS to convergence.
- One probe per flavor per offset. Diagonal: 5-fold by row id, `random_state=0`, out-of-fold scores. Two rows from the same conversation do not split across the fold if a conversation contributes more than one row; otherwise fold by sample id.
- Off-diagonal: fit on all of one flavor, score the other flavors at the **same offset**. That is the transfer number.
- The week 3 finished-reply mean is one extra point, labeled “after the action,” taken from the existing week 3 JSON if it is already on disk. Do not rerun week 3 to get it. If that JSON is absent, leave the point blank and say so.

## Do not

- Run `scripts/01_lead_curve.py`.
- Run `week3-noninvasive-probe/scripts/08`, `09`, `10_compare_tianq.py`, `10_train_act_probes.py`, or `11_flavor_matrix.py`. Those score a finished reply or a different probe. Import their loaders and `fit_aisi_logistic` only.
- Edit `tianq/` or `gigi/`.
- Average reply tokens. Each offset is one token.
- Call OpenAI or any other paid monitor API. A Gemma-as-monitor pass is optional and only after `CURVE.md` has probe AUROCs. If you skip it, say the monitor gap is not measured.
- Steer the model. Steering is a follow-up, and only if τ90 ≥ 2 on at least two flavors.
- Commit `.env`, tokens, or activation caches. Do not commit or push unless Ravi asks in this session.
- Destroy the Vast instance.

## Setup

Need ~80 GB VRAM. `transformers` stays below 5. `HF_TOKEN` is in `../week1-dyl-vs-oa/dyl-probe-repro/.env` or `/workspace/.env`. If it is missing, stop and write that in `LOG.md`. Do not invent activations.

```bash
cd ravi/week6-early-detection
nvidia-smi
python -m pytest tests/ -q
```

`tests/test_mark.py` covers the deferred bash rule. Add tests for the token rule: offset 0 and 1 are excluded from τ90, a one-token reply has undefined τ90, and a lie and a truth with the same prefix length share an index.

## Run

New script: `scripts/02_token_curve.py`. Append to `LOG.md` when a stage starts, when it finishes, and when it stops. Include flavor names, row counts, how t* was marked, cache paths, shapes, seconds, and the traceback if it fails. A stopped stage means later stages did not run.

1. **Mark, no model.** For each flavor, print how many rows have a reply long enough for offsets 0, 2, 8, and 32, and how many are one-token. Stop only if a flavor loader fails or the label counts disagree with week 3’s published counts in `../week3-noninvasive-probe/AGENTS.md`.

2. **Smoke extract, then delete that cache.** Two flavors (`alibi`, `mask_known_facts`), eight rows each, offsets `0`, `2`, and `prompt_last`. Smoke cache directory must differ from the full cache.

3. **Full extract.** All eleven flavors, all offsets. Checkpoint per flavor so a killed job resumes. If the process is killed for memory, rerun with batch size 1.

4. **Fit on CPU.** The GPU can sit idle. Write `results/token_curve.json` and `CURVE.md`.

If `HF_TOKEN` is missing, stage 1 can still run. Stages 2–4 do not.

## Write `CURVE.md`

Read `results/token_curve.json`. Round AUROC to three decimals.

One table per flavor that was fit. Rows are offsets, `prompt_last` first, then 32 down to 0. Offset 0 is labeled as the bad token, not lead time.

```text
| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last |  |  |  |  |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) |  |  |  |  |
```

Leave a cell blank when that offset was dropped for count. Transfer AUROC is the mean over flavors with at least 10 of each class, excluding the training flavor.

Then four sentences, only what the numbers support:

1. For each flavor, τ90 or that no earlier offset reached 0.90. Name the offset where in-category AUROC peaked inside `{2, 4, 8, 16, 32, prompt_last}`.
2. Whether offset 0 is higher than every earlier offset. If it is, the probe is reading the bad token as it is emitted.
3. Whether transfer at τ90 is above 0.50. If it is, and week 3’s finished-reply transfer was about 0.50, say early intent is shared where the finished lie was not. If transfer stays near 0.50, say early detection did not fix generalization.
4. The monitor gap, if you measured one: probe AUROC minus monitor AUROC at τ90. If you did not run a monitor, say the gap is not measured.

Do not call the probe a lie detector. Do not put the hackathon’s 0.940 in the table.
