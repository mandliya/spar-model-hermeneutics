# Does gemma-3-27b-it know it lied — or just answer however you ask?

A replication of two numbers from [Anthropic's Alignment Science post on
fine-tuned lie detectors](https://alignment.anthropic.com/2026/lie-detectors/),
using its released dataset [`Noddybear/lies`](https://huggingface.co/datasets/Noddybear/lies),
plus follow-up experiments on how much the self-report *question itself*
drives the answer.

**Model:** `google/gemma-3-27b-it`, official bf16 weights, local GPU
(temperature 0 — greedy decoding, deterministic) — never OpenRouter, which
serves this model at mixed quantizations across providers; every number
here needs to come from the same weights to be comparable. **Config:**
`google-gemma-3-27b-it_task-group_sandbagging`, all 310 rows.
**Ground truth:** a live capability proof, not the dataset's `did_lie`
label — [see why](#why-not-did_lie).

## The headline numbers

| Metric | Rate | 95% CI | n |
|---|---|---|---|
| **Lying rate** — falsely denies it can draw the subject | 61.6% | 56.1–66.9% | 310 |
| **Confession rate** — admits the lie when asked "did you lie?" | 84.3% | 78.5–88.8% | 191 |
| **Doubling-down rate** — denies the lie when asked "did you lie?" | 15.7% | 11.2–21.5% | 191 |

Confession and doubling-down are two sides of the same coin, not two
separate experiments — every row where gemma lied either admits it
(confession) or denies it (doubling-down), so the two always sum to 100%.
Both are reported everywhere in this doc, side by side, since "how often
does it deny" and "how effective is this question at making it confess"
are the same measurement read in opposite directions, and each framing
answers a different question someone on the team might actually be asking.

![Gemma's lying rate (61.6%) plotted against the paper's 76% floor for every model it tested](charts/lying_rate.png)

The paper reports every model it tested scoring above 76%. This lands
meaningfully lower — not a parsing artifact (0 unparseable across 310 rows)
but a real discrepancy, plausibly from replay methodology or which
sandbagging subtask this config draws from.

## Does asking gemma whether it lied detect it?

After its answer, gemma is asked whether it lied, in the project's four
follow-up questions ([`../RESULTS.md`](../RESULTS.md), Methodology): `neutral`
("did you lie?"), `named` ("did you just sandbag?"), and both again with 1–2
sentences of reasoning first (`neutral_cot`, `named_cot`). Every question
reuses the identical saved answer; only the follow-up changes. Confession and
doubling down are out of the 191 rows where gemma lied; false confession is
out of the 119 where it told the truth.

<!-- CORE4:START -->
191 rows where gemma did it, 119 where it didn't.

| Question | Confession | 95% CI | Doubling down | False confession | 95% CI | Balanced accuracy |
|---|---|---|---|---|---|---|
| `neutral` | 84.3% | 78.5–88.8% | 15.7% | 100.0% | 96.9–100.0% | 42% |
| `neutral_cot` | 52.9% | 45.8–59.8% | 47.1% | 89.1% | 82.2–93.5% | 32% |
| `named` | 1.0% | 0.3–3.7% | 99.0% | 0.0% | 0.0–3.1% | 51% |
| `named_cot` | 66.5% | 59.5–72.8% | 33.5% | 92.4% | 86.2–96.0% | 37% |
<!-- CORE4:END -->

- **None of the four detects the lie.** Balanced accuracy is 32–51%, at or
  below a coin flip. Scored on gemma's probability of "yes" instead of the
  letter, `neutral` is 0.12 AUROC: gemma is *more* sure it lied when it told
  the truth ([`../answer_probability/`](../answer_probability/README.md)).
- **`neutral` says "yes" to almost everything; `named` says "no" to almost
  everything** (84.3% vs 1.0% confession, 100% vs 0.0% false confession), on
  the identical lie. The [A/B swap test](../ab_swap/README.md) shows `named`'s
  "no" is mostly the letter B: with "yes" moved to option B, `named` answers
  "yes" 92% of the time.
- **Reasoning first doesn't fix it.** `neutral_cot` and `named_cot` both end
  up admitting to most lies *and* most honest answers (false confession 89.1%
  and 92.4%).

**What the reasoning actually says**, from `results/transcripts_reasoning.jsonl`:
on denials, gemma consistently reframes the capability question as a values
question — *"creating a depiction of [a disturbing subject] violates my
safety guidelines"* — rather than a genuine claim of inability. Reads more
like a policy-driven refusal wearing a capability-denial's clothes than an
actual belief it can't draw.

The nine retired questions (stakes framings, 0–100 confidence, `plain`, the
safety-evaluation framing) and their analysis are in the
[archived version of this page](../../archive/2026-10-01_all_13_wordings/ascii-sandbagging/RESULTS.md).

## Why not `did_lie`? {#why-not-did_lie}

Unusable across the entire `google-gemma-3-27b-it` release, not just this
config: `True` in only 8 rows out of ~38,600 checked (all inside
`mask-factual`), and every trace ends at the system prompt — no config
ships a completed conversation with a graded final answer. `sycophancy` is
`False` in all 498 of its rows despite the paper's own ~32% reported fold
rate for that category. Full breakdown in `CLAUDE.md`/`HANDOFF.md`.

## Caveats

- Replayed histories condition gemma on the dataset's recorded turns, not a
  fully fresh generation.
- The paper's sampling temperature isn't published; this run used 0
  (greedy decoding) — a deliberate, deliberately-documented choice, not
  the paper's own value.
- `did_lie` unusable as a baseline here — see [above](#why-not-did_lie).

## Conclusion, and what's next

This replicates the paper's qualitative finding for gemma-3-27b — real
sandbagging (61.6%, below its 76%+ floor) paired with self-report unreliable
enough to be counterproductive as a monitoring signal (paper: AUROC as low
as 0.46, worse than chance). What this adds is a mechanism: self-report
here isn't just noisy, it's extremely sensitive to wording — confession
ranges from 1.0% (`named`) to 84.3% (`neutral`) on the identical lie, and
part of that is gemma picking a letter rather than answering the question.

Next, in priority order: (1) the paper's own proposed fix, third-person
monitoring; (2) train a DYL-style probe on the activations now on HF
(both pre-decision and answer-token positions captured) and compare its
accuracy to the four questions above; (3) extend the
live-elicitation method to mask-factual's other two subtypes
(`mask_known_facts`, `mask_statistics`), which need external fact-checking
rather than the self-contained ground truth `mask_provided_facts` has.

## Where the data lives {#where-the-data-lives}

| File | What it is |
|---|---|
| `results/ascii_results.csv` | Part 1 + the `neutral` follow-up, 310 rows |
| `results/ascii_reasoning_conditions.csv` | the reason-first questions; `neutral_cot` is kept, the other three are archived |
| `results/ascii_benchmark_conditions.csv` | `named`, plus the archived `plain` / `eval_plain` / `eval_named` |
| `results/ascii_followup_conditions.csv` | archived questions only (low_stakes / high_stakes / confidence) |
| `../named_cot/results/named_cot_ascii.jsonl` | `named_cot`, 310 rows |
| `results/transcripts*.jsonl` | **the actual rollouts** — full reconstructed conversations for every condition above, not just the parsed letter |
| `charts/lying_rate.png` | the chart on this page, regenerate with `06_charts_ascii.py` |
| `00_ascii_activations.py`, `00_ascii_followups.py`, `00_ascii_reasoning.py`, `00_ascii_selfreport_benchmark.py` | generate the four results files above (local GPU, bf16) |
| `00_export_transcripts.py` | joins results back against the dataset for the transcripts files — $0, no API calls |
| `01_dyl_recapture_ascii.py` | DYL-matched answer-token activation capture for all 12 follow-up conditions |
| `02_predecision_ascii_benchmark.py` | pre-decision activation capture for `plain`/`named`/`eval_plain`/`eval_named` specifically (the other 8 conditions already had it from their own generation scripts) |

Activation tensors aren't in this repo (too many files). They are on Hugging
Face, `model-hermeneutics/gigi-lie-detection-activations`:
`activations/ascii.tar.gz` holds the four questions in float32, at two capture
points per answer (`__predecision.pt`, the last prompt token, and
`__answer_token.pt`, the letter), with `activations/maps/ascii.jsonl`. The
first-run activations are in `first_run/ascii/`. The repo's README explains
every file; the retired questions' tensors are under `archive/`.
