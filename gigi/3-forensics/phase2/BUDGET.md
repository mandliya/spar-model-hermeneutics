# Phase 2 budget: what it costs, what is left, what John needs to do

For John (and any agent John uses on this repo). Phase 2 = black-box hermeneutics from the 19 Nov call: five open-weight
teacher models, each distilled into Qwen3.6 27B (thinking on), in John's order: DeepSeek V4 Pro, Qwen3-Next 80B,
GPT-OSS 120B, Kimi K3, GLM 5.3. "Safe" adds 30%. This page is updated after every stage with measured costs.

**Last update: 20 Nov 2026, 20:45 UTC, after the full DeepSeek collection.** DeepSeek's teacher runs are done and
measured (1,198 runs, all complete). Its grading, the Qwen3.6 baseline and the training are still estimates; the other
four models are still estimates.

## Actions for John

**Thank you:** the key limit is now $15,000, the OpenRouter account got about $2,000 more, and Vast about $400 more.
That covers DeepSeek. What is left:

1. **OpenRouter workspace lifetime budget:** our key cannot read it. Please check it has room for **~$5,200** more
   (safe: ~$6,750). It blocked all calls twice before, on 21 Oct and 11 Nov.
2. **Vast:** nothing now. **~$550 before Qwen3-Next's training**, then ~$550 per model ($2,200 in total, safe: ~$3,080).
   DeepSeek's training may cost more than the $850 below, because its runs are long (see "Run length"); it is re-priced
   in the training plan before any GPU is rented.
3. **OpenRouter credits:** nothing until GLM 5.3, then **~$200** (safe: ~$1,750). The $5,019 left is the whole org
   account, shared with other SPAR projects.
4. **OpenRouter key limit:** $15,000 with $6,560 used is enough for everything planned (~$11,800 at the end).
5. **Modal:** nothing to top up (SPAR's workspace is billed monthly). Expect about **$1,670 more** over the period
   (safe: ~$2,170). The DeepSeek collection used $216 of it on 20 Nov.

## The table

One row per model, done one after another. "New money needed" = OpenRouter top-up + Modal bill + Vast top-up, counted
from the balances in the first row (after John's top-ups of 20 Nov).

| Model | OpenRouter cost | OpenRouter top-up | Modal cost | Vast cost | Vast top-up | Total cost | New money needed | Time |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **Left now (20 Nov, 20:45 UTC)** | **$5,019 left** (key: $8,440 of $15,000 left) | | **no balance (billed monthly to SPAR)** | **$884 left** | | | | |
| DeepSeek V4 Pro, plus the Qwen3.6 baseline (**collection measured**) | $743 ($593 already spent) | $0 (safe: $0) | $470 ($222 already spent) | $850 | $0 (safe: ~$220) | $2,060 | $470 | ~1 week |
| Qwen3-Next 80B | $500 | $0 (safe: $0) | $250 (safe: $350) | $550 | $550 (safe: $715) | $1,300 | $800 | ~1 week |
| GPT-OSS 120B | $250 | $0 (safe: $0) | $250 (safe: $350) | $550 | $550 (safe: $715) | $1,050 | $800 | ~1 week |
| Kimi K3 | $2,900 | $0 (safe: $0) | $350 (safe: $450) | $550 | $550 (safe: $715) | $3,800 | $900 | ~1.5 weeks |
| GLM 5.3 | $1,400 | $200 (safe: $1,750) | $350 (safe: $450) | $550 | $550 (safe: $715) | $2,300 | $1,100 | ~1.5 weeks |
| **All of it** | **$5,800** | **$200 (safe: $1,750)** | **$1,670 (safe: $2,170)** | **$3,050** | **$2,200 (safe: $3,080)** | **$10,500** | **$4,070** | **~5–6 weeks** |

OpenRouter left after each model: ~$4,870 after DeepSeek, ~$4,370 after Qwen3-Next, ~$4,120 after GPT-OSS, ~$1,220
after Kimi K3, then ~$200 short during GLM 5.3. The Qwen3.6 baseline (Qwen3.6 itself with thinking on, 40 runs of each
of the 28 tasks) is paid once for all five models: $200 Modal + $300 Vast, counted in the DeepSeek row.

DeepSeek's OpenRouter cost = $20 (provider check, 6 test runs, 56 pilot runs) + $573 (the full collection, measured) +
~$150 Opus grading (estimate). Its Modal cost = $6 (pilot) + $216 (collection, measured) + ~$50 for the trained model's
re-runs + $200 for the baseline.

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
