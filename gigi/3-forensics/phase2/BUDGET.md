# Phase 2 budget: what it costs, what is left, what John needs to do

For John (and any agent John uses on this repo). Phase 2 = black-box hermeneutics from the 19 Nov call: five open-weight
teacher models, each distilled into Qwen3.6 27B (thinking on), in John's order: DeepSeek V4 Pro, Qwen3-Next 80B,
GPT-OSS 120B, Kimi K3, GLM 5.3. "Safe" adds 30%. This page is updated after every stage with measured costs.

**Last update: 24 Nov 2026, 04:45 UTC, re-priced after the 24 Nov call.** DeepSeek's teacher runs and their
grading are done and measured. Its Qwen side (the "before" runs of Qwen3.6, the training and the re-runs) has its own
plan and estimate in `PLAN_deepseek_training.md`; the other four models are still estimates.

## Re-priced after the 24 Nov call (this replaces the five-teacher plan below)

The call cut the plan to one more teacher (GPT-OSS 120B) plus two cheaper experiments on data we already have. Plan:
`PLAN_next_24nov.md`. Balances on 24 Nov 04:30 UTC: OpenRouter account $4,238 left (key: $7,921 of $15,000 left),
Vast $790 left.

| What | OpenRouter | Modal | Vast | Total |
|---|--:|--:|--:|--:|
| GPT-OSS 120B: pilot, collection, grading, training, tests | ~$80-650 (grading is up to $450 of it) | ~$320 | ~$90 | ~$900-1,100 |
| Full / summarised / no reasoning, on DeepSeek's runs | ~$30-60 | ~$200 | ~$180 | ~$450 |
| General training set, DeepSeek as teacher | ~$30-80 | ~$100 | ~$90 | ~$250 |
| **All three** | **~$140-790** | **~$620** | **~$360** | **~$1,800** |

**What John needs to do now: nothing.** The OpenRouter credit, the key limit and the Vast credit all cover this plan.
Only the OpenRouter workspace lifetime budget is still unknown to us: please check it has room for ~$800 more.
Modal is billed monthly to SPAR: expect about $620 more. The asks further down (Kimi K3, GLM 5.3, ~$4,990 of new
money) no longer apply.

## Actions for John

**Thank you** for the top-ups of 20 Nov (key limit $15,000, ~$2,000 OpenRouter credit, ~$400 Vast): they cover all of
DeepSeek. What changed since: grading the judged tasks cost $494 for DeepSeek, not the ~$150 assumed, because those tasks
have the longest runs, so the other four models' grading is now ~$450 each. What is left:

1. **OpenRouter workspace lifetime budget:** our key cannot read it. Please check it has room for **~$6,250** more
   (safe: ~$8,130). It blocked all calls twice before, on 21 Oct and 11 Nov.
2. **OpenRouter credits:** nothing until Kimi K3. Then **~$250 before Kimi K3 and ~$1,700 before GLM 5.3** (~$1,950 in
   total, safe: ~$3,830). The $4,298 left is the whole org account, shared with other SPAR projects, which used $228 of
   it between 20 and 23 Nov.
3. **Vast:** nothing until GPT-OSS. Then **~$420 before GPT-OSS 120B's training**, ~$550 before Kimi K3's and ~$550 before
   GLM 5.3's (~$1,520 in total, safe: ~$2,240). DeepSeek's training came in at ~$200 in the plan, so the $878 left covers
   DeepSeek and Qwen3-Next.
4. **OpenRouter key limit:** $15,000 with $7,053 used is enough for everything planned (~$13,300 at the end), but not
   with the full safety margin (~$15,200). Raising it to ~$16,000 when convenient removes that worry.
5. **Modal:** nothing to top up (SPAR's workspace is billed monthly). Expect about **$1,300 more** over the period
   (safe: ~$1,790), on top of the $216 the DeepSeek collection used on 20 Nov.

## The table

One row per model, done one after another. "New money needed" = OpenRouter top-up + Modal bill + Vast top-up, counted
from the balances in the first row.

| Model | OpenRouter cost | OpenRouter top-up | Modal cost | Vast cost | Vast top-up | Total cost | New money needed | Time |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **Left now (23 Nov, 05:00 UTC)** | **$4,298 left** (key: $7,947 of $15,000 left) | | **no balance (billed monthly to SPAR)** | **$878 left** | | | | |
| DeepSeek V4 Pro (**collection and grading measured**) | $1,087 (all spent) | $0 (safe: $0) | $320 ($222 already spent) | $200 | $0 (safe: $0) | $1,610 | $320 | ~1 week |
| Qwen3-Next 80B | $800 | $0 (safe: $0) | $250 (safe: $350) | $550 | $0 (safe: $100) | $1,600 | $250 | ~1 week |
| GPT-OSS 120B | $550 | $0 (safe: $0) | $250 (safe: $350) | $550 | $420 (safe: $715) | $1,350 | $670 | ~1 week |
| Kimi K3 | $3,200 | $250 (safe: $1,620) | $350 (safe: $450) | $550 | $550 (safe: $715) | $4,100 | $1,150 | ~1.5 weeks |
| GLM 5.3 | $1,700 | $1,700 (safe: $2,210) | $350 (safe: $450) | $550 | $550 (safe: $715) | $2,600 | $2,600 | ~1.5 weeks |
| **All of it** | **$7,340** | **$1,950 (safe: $3,830)** | **$1,520 (safe: $2,020)** | **$2,400** | **$1,520 (safe: $2,240)** | **$11,260** | **$4,990** | **~5–6 weeks** |

OpenRouter left after each model: ~$4,298 after DeepSeek (all spent), ~$3,500 after Qwen3-Next, ~$2,950 after GPT-OSS,
then ~$250 short during Kimi K3 and ~$1,950 short by the end of GLM 5.3. Each of the four later models now includes
~$450 of Opus grading (DeepSeek's measured $494). Their Vast cost stays at the earlier ~$550 until DeepSeek's training
is measured; if it lands near the plan's ~$200, their Vast top-ups shrink too.

DeepSeek's OpenRouter cost = $20 (provider check, 6 test runs, 56 pilot runs) + $573 (the full collection) + $494 (Opus
grading), all measured. Its Modal cost = $6 (pilot) + $216 (collection), measured, + ~$100 for Qwen3.6's "before" runs
and the trained copies' re-runs (plan). Its Vast cost = ~$160 for one H200 box (plan) + ~$40 for the launch box.

## Measured in the GPT-OSS 120B collection (24-25 Nov)

- **Cost:** OpenRouter **$118.95** for the 1,375 runs launched (1,198, plus 177 re-runs of the chunks set aside), from
  the runs' own ledger (`results/p2_gptoss120b_usage.csv`); **$122.70** with the pilot. Modal **~$197** (billing report
  to 13:00 UTC on 25 Nov, pilot included). In all **~$320**, against the ~$300 approved on 24 Nov.
- **Stopped by its spend guard at 12:43 UTC on 25 Nov.** The guard counts the whole OpenRouter key since the fleet
  started, and Terminal-Bench runs on the same key, so it read $501.10 of its $500 cap while the fleet's own ledger was
  ~$119. The main queue had ended at 10:00 UTC; the stop cut the last 6 reference_extraction runs of the set-aside
  round. The closing steps (final counts, per-run table, cost, HF upload) ran afterwards with `scripts/phase2_finish.sh`
  (no model calls). Next fleet: cap on the fleet's own ledger, the whole key only as a higher backstop.
- **Runs:** 1,130 of 1,198 complete by the current rule (FINAL RESULTS, or a full context window it recognises). Two
  kinds of run count as lost under it and are GG's decision (25 Nov): 32 chess and tictactoe games that used every
  allowed step (the harness prints no FINAL RESULTS there), and 26 runs that filled the 131,072-token window (largest
  prompt 124,089-131,038 tokens), which DeepInfra answers with a generic "400 Provider returned error" instead of
  "maximum context length". Left after both: reference_extraction 37 of 40, coding_norules 39 of 40.
- **Grading, priced, not run:** a GPT-OSS run is 195 kB of rollout.log against DeepSeek's 305 kB, so Opus 4.8 grading
  should cost about 0.64 x $494 = **~$320**.

## Measured in the GPT-OSS 120B pilot (24 Nov)

- 6 test runs + 56 pilot runs (2 for each of the 28 tasks), on DeepInfra bf16 with reasoning effort high.
  **OpenRouter: $3.75** ($0.04 for the test runs, $3.71 for the pilot). **Modal: ~$5.13** (billing report, 04:00-06:59
  UTC on 24 Nov, when only the pilot was running).
- **53 of 56 pilot runs finished.** The 3 that did not: one async_conversion run (the provider returned empty replies
  at step 197, with the run at ~128k tokens), one reference_extraction run (the provider refused the call at step 101,
  at ~103k tokens) and one chess run (stopped at step 49 with the game unfinished).
- **Run length:** the longest run reached 127,874 tokens, against GPT-OSS's window of 131,072. No run was stopped for a
  full window in the pilot, but async_conversion, leaked_tests and reference_extraction come close, so the full
  collection will lose some runs there.
- **Projection for the collection: ~$85 on OpenRouter** for 46 runs of each of the 28 tasks (DeepSeek's was $431 at the
  same point and came in at $573). With Modal at about DeepSeek's $216, the collection should cost ~$300 in all.
- During the pilot the key's usage rose by $13.77, so about $10 was spent by something other than these runs.
- Per task: `results/p2_gptoss120b_pilot_usage.csv`; logs: `results/logs/p2_gptoss120b/`; raw runs on HF under
  `phase2/gpt-oss-120b/raw/`.
- **Collection, part way (24 Nov 13:00 UTC):** 460 of 1,198 runs launched, including all 46 runs of 8 of the 13 long
  tasks. **OpenRouter: $38.92** (the launch logs' own ledger; the pilot predicted $51.77 for the same runs).
  **Modal: ~$81** (07:00-12:59 UTC). The fleet's "spent" line reads higher ($85.90) because it counts the whole key,
  which also pays for WildChat, Terminal-Bench and other people's jobs. DeepInfra is sending many 429 replies, so the
  fleet is holding launches; on track for ~$300 in all.

## Measured in the general training set (24 Nov)

- **WildChat answers (far rung):** GG approved ~$10 with a $100 cap. The 20-question test cost **$0.14** (20 of 20
  answered, all from SiliconFlow with reasoning), so the 2,000 answers should cost about $14. The full pass started
  24 Nov 09:21 UTC on the launch box (`scripts/general_wildchat_generate.py`, log `/workspace/logs/general_wildchat_full.out`).
- **WildChat, part way:** 1,850 of 2,000 answered, 0 failed, **$16.37** at 12:50 UTC.
- **ToolScale training and tests (middle rung), measured: Modal $51.10** (09:00-12:59 UTC: training on one H200, then
  4 copies each served on an H200 and tested 40 times on the 8 tasks; 1,261 runs scored). Planned ~$90.
- **Terminal-Bench first steps, measured:** OpenRouter **$2.96** ($0.17 for the 2-run test, $2.79 for the 24-run pilot),
  Modal **$2.65** (reference-solution check of all 89 tasks, test and pilot). The pilot's runs cost $0.116 on average
  and all 24 ended on their own, so the full 979 runs come to **~$114 on OpenRouter at 100 steps** (~$88 at 50 steps)
  plus ~$60 on Modal, far below the $440 + $100-200 guessed before the pilot.
  **Correction (25 Nov):** the full collection measures $0.347 a run, not $0.116, because the pilot's 12 tasks were
  the cheap ones. Re-projected ~$340 on OpenRouter + ~$90 on Modal; GG approved finishing it.
- **Training on Modal, priced (not run):** about $90 for the ToolScale set's training and tests, about $130 for a set the
  size of DeepSeek's, against about $45 and $70 on a Vast box; Modal is about twice as fast only because the three sets
  can train at once (`phase2/general_set/PIPELINE.md`).
- **Terminal-Bench, priced (not run):** about $440 on OpenRouter plus $100-200 on Modal for 979 runs at 100 steps, or
  about $240 in all at 50 steps; a first check, test and pilot cost about $20 (`phase2/general_set/TERMINAL_BENCH.md`).

## Measured so far in DeepSeek's training into Qwen3.6 (23-24 Nov)

- GG approved ~$300 for this stage on 23 Nov (all platforms together).
- **Vast: ~$82 so far.** The two-H200 box (62121347, $6.98/h) has run 11.4 hours (~$80) as of 23 Nov 19:45 UTC and is
  still on, idle, until GG says to stop it; a first box rented without the server port cost ~$2. Training took 6 h 50 min
  on one GPU while the other served Qwen3.6 for the tests.
- **Modal: $10.75** for the 1,602 test runs' sandboxes (Modal billing report, read 24 Nov; planned ~$60-100). The
  model itself ran on the Vast box, so Modal only paid for the task containers.
- **OpenRouter: $0.** The 8 re-tested tasks are scored by rules, and Qwen3.6 runs on our own server.
- Results: `results/distill_p2ds_eval.csv` (1,569 of 1,600 runs scored, 31 lost); logs: `results/logs/p2ds/`; raw runs
  on HF under `distill/p2_deepseekv4_v1/eval/`.

## Measured in the DeepSeek grading (23 Nov)

- **487 of 488 runs graded** in the 11 tasks that need a judge (claude-opus-4-8 through OpenRouter, pinned to
  Anthropic), with no refusals and no errors. The one left is a grade_your_own run that ended at step 0 (DeepSeek wrote
  its first command as text), which the grader cannot judge. The batches cut by the 20 Nov outage were skipped.
- **$493.61** by the key's usage (before and after), against an estimate of ~$460; 33 minutes, launched from the box
  (`scripts/phase2_grade.sh`). reference_extraction alone cost $77 for 46 runs.
- Results: `results/p2_deepseekv4_full_runs__deepseek-v4-pro.csv`; logs: `results/logs/p2_deepseekv4/grading/`; each
  run's `final/grading.json` is on HF next to the run.

## Measured in the DeepSeek collection (20 Nov)

- **1,198 runs, all complete, none lost**: 46 runs for each of the 13 long tasks, 40 for each of the other 15, on
  SiliconFlow fp8 with reasoning effort xhigh. It took about 16 hours (04:34 to 20:20 UTC), launched from the Vast box.
- **OpenRouter: $572.82** by the key's usage (before and after). Our own per-call ledger counts $554.15; it only sees
  calls that returned a reply, so the $19 gap is most likely calls billed during the outage below.
- **Modal: $216** for 20 Nov (the apps launched from the box; the billing report may still add the last hour).
- **A one-hour SiliconFlow outage** (09:54 to 10:55 UTC, every call answered 503). 65 batches had a run cut by it; they
  were set aside and re-run whole, as fixed before the launch (a run cut by the provider biases a task towards short
  runs). That cost roughly $120–140 of the OpenRouter total. The fleet now pauses on its own when the pinned provider's
  uptime drops (`scripts/phase2_fleet.sh`, 34bceb5).
- 97% of the input was read from the provider's cache; ~340 reasoning tokens per call; median run $0.14, mean $0.46.
  Per run: `results/p2_deepseekv4_usage.csv`; per task: `results/logs/p2_deepseekv4/usage.txt`.
- **Run length (matters for training and for Qwen3.6 as the substitute):** longest prompt per run, over the 1,198 runs
  kept: 608 go past 32k tokens, 228 past 128k, and **11 past 262k, Qwen3.6's context window** (8 of 46 in
  reference_extraction, 3 of 46 in leaked_tests; the longest is 372,743 tokens). Training so far cut runs at 32k
  tokens, so DeepSeek's training needs a much longer cut, more GPU memory, and a decision on those 11 runs. The Vast
  column still assumes the 32k cut.

## Measured in the DeepSeek pilot (19 Nov)

- 6 test runs + 56 pilot runs (2 per task). OpenRouter $20.22, Modal ~$6, launch box $0.107/h.
- It came in at about a third of the first estimate because 97% of the input was cached (the estimate assumed 70%) and
  DeepSeek wrote ~330 reasoning tokens per call (the estimate assumed 1,500). Per task: `results/p2_deepseekv4_pilot_usage.csv`.

## What each column pays for

- **OpenRouter:** the teacher's runs (about 1,250 per model: 28 tasks x 40-46 runs, incl. top-ups and the pilot) plus
  Opus grading of the judged tasks (~$150 per model).
- **Modal:** the task sandboxes for the teacher's runs and for the re-runs of the trained Qwen3.6.
- **Vast:** a rented GPU box (H100 class) to train Qwen3.6 27B on the teacher's runs and re-run its saved copies
  (vLLM), plus a small always-on box (~$0.11/h, ~$2.60 a day) that launches the Modal runs so nothing depends on a laptop.

## Where the numbers come from

- **Balances:** OpenRouter `GET /api/v1/key` (usage $6,559.74, limit $15,000 on 20 Nov 20:45 UTC) and `/api/v1/credits`
  (org account $5,018.55 left); Modal `modal billing report --resolution h` on SPAR's workspace (apps launched from
  the box: $6.23 on 19 Nov, $216.36 on 20 Nov); Vast `show user` (credit $884.26).
- **DeepSeek:** measured (above). **The other four models:** `phase2/targets.md` (token counts of our Qwen3.6 runs,
  priced per model, likely = 70% of input cached and 1,500 reasoning tokens per call). DeepSeek came in well under that,
  but each provider caches differently: each model gets its own pilot first and this page is updated then. Kimi K3's
  $2,900 is the cheapest provider (Wafer, 4-bit); the fp8 providers make it ~$5,850.
- **Modal and Vast:** the classic round's measured costs (17-19 Nov: Modal ~$205, box ~$265 for 49.7 h), scaled to
  Qwen3.6 27B as the student.
- Kimi K3 and GLM 5.3 are not in the paper ("Why do models task game?" names DeepSeek V4 Pro, GPT-OSS-120B,
  Kimi K2.7 Code and Gemini 3.5 Flash).

## History

- 19 Nov 08:30 UTC: first version, all estimates (total ~$11,450, new money $7,350).
- 19 Nov 11:50 UTC: DeepSeek pilot measured; DeepSeek $3,000 -> $1,850; total ~$10,300, new money $6,200.
- 20 Nov 20:45 UTC: DeepSeek collection measured ($573 OpenRouter, $216 Modal; the outage re-runs included); DeepSeek
  $1,850 -> $2,060. John's top-ups arrived (key limit $15,000, ~$2,000 OpenRouter credit, ~$400 Vast), so the new money
  still needed drops to ~$4,070. Total ~$10,500.
- 23 Nov 05:00 UTC: DeepSeek's grading measured ($494, against ~$150 assumed), so the other four models' grading goes
  to ~$450 each. DeepSeek's Qwen side re-priced in `PLAN_deepseek_training.md` (~$300 instead of ~$1,050: the "before"
  runs cover 9 short tasks instead of all 28). Other SPAR projects used $228 of the shared OpenRouter account. Total
  ~$11,260, new money still needed ~$4,990.
- 24 Nov 04:45 UTC: plan cut on the 24 Nov call to GPT-OSS 120B plus two experiments on existing data; total still to
  spend ~$1,800 instead of ~$9,600; no new money needed. GPT-OSS endpoint check done ($0.001), pilot started.
- 24 Nov 06:40 UTC: GPT-OSS 120B pilot measured ($3.75 on OpenRouter, 53 of 56 runs finished); its collection is
  projected at ~$85 on OpenRouter, ~$300 with Modal.
- 24 Nov 09:25 UTC: Modal cost of DeepSeek's Qwen3.6 tests measured ($10.75); WildChat generation started (~$14);
  training on Modal and Terminal-Bench priced, not run.
- 24 Nov 13:00 UTC: ToolScale trained and tested on Modal ($51.10); Terminal-Bench check, test and pilot done ($2.96
  OpenRouter, $2.65 Modal), full collection re-priced at ~$175; GPT-OSS collection at 460 of 1,198 runs ($38.92 +
  ~$81 Modal); WildChat at 1,850 of 2,000 ($16.37). Modal for 24 Nov so far: $144.88 in all.
- 25 Nov 03:15 UTC: WildChat done (1,999 of 2,000 answered, **$17.98**). GG approved the full Terminal-Bench collection
  (~$175: ~$114 OpenRouter + ~$60 Modal, OpenRouter cap $500; 87 tasks x 11 runs), started 03:15 UTC. The GPT-OSS
  collection had been held by a health-check bug from ~17:40 UTC on 24 Nov (idle from 22:40); fixed and resumed at
  03:13 UTC with 658 of 1,198 runs still to go. Key usage $7,229.88 of $15,000; account $4,040 left.
- 25 Nov 08:05 UTC: Terminal-Bench costs about 3 times the pilot: $0.347 a finished run (400 runs; median $0.219,
  max $2.29) against $0.116. The pilot's 12 tasks were the cheap ones ($0.132 a run here); the other 75 average
  $0.379. At 390 of 957 runs: $149.96 OpenRouter + $33.66 Modal (25 Nov to 07:59 UTC). Re-projected ~$430 in all
  (~$340 OpenRouter + ~$90 Modal); **GG approved finishing all 957 runs** (25 Nov). GPT-OSS collection: $85.21 OpenRouter
  (launch-log ledger) and ~$160 Modal so far, 60 of 209 chunks still queued (the short tasks): ~$300 in all.
- 25 Nov 13:20 UTC: GPT-OSS collection ended, stopped by its whole-key guard at 12:43 UTC with 6 runs cut: **$118.95**
  OpenRouter (+ $3.75 pilot) and ~$197 Modal, ~$320 in all against ~$300 approved; 1,130 of 1,198 complete by the
  current rule, grading priced at ~$320. Terminal-Bench at 794 of 957 runs: $302.10 OpenRouter (its own ledger) and
  ~$72 Modal (to 13:00 UTC). Key usage $7,612 of $15,000.
