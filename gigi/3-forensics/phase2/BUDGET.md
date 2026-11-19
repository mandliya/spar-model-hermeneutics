# Phase 2 budget: what it costs, what is left, what John needs to do

For John (and any agent John uses on this repo). Phase 2 = black-box hermeneutics from the 19 Nov call: five open-weight
teacher models, each distilled into Qwen3.6 27B (thinking on), in John's order: DeepSeek V4 Pro, Qwen3-Next 80B,
GPT-OSS 120B, Kimi K3, GLM 5.3. "Safe" adds 30%. This page is updated after every stage with measured costs.

**Last update: 19 Nov 2026, 11:50 UTC, after the DeepSeek pilot.** The DeepSeek row is now based on measured costs;
it came in at about a third of the first estimate. The other four rows are still estimates.

## Actions for John

1. **Vast:** top up **~$400 now** (DeepSeek's training and re-runs, and the Qwen3.6 baseline), then ~$550 per model
   after that ($2,600 in total, safe: $3,450).
2. **OpenRouter workspace lifetime budget:** our key cannot read it. Please check it has room for **~$5,650** more
   (safe: ~$7,300). It blocked all calls twice before, on 21 Oct and 11 Nov.
3. **OpenRouter key limit:** $10,000, with $5,987 used. Raise it to about **$15,000** before Kimi K3 (the $17,000
   suggested before also works). A limit costs nothing until it is spent.
4. **OpenRouter credits:** no top-up needed until Kimi K3. **~$600 before Kimi K3 and ~$1,400 before GLM 5.3**
   ($2,000 in total, safe: $3,700). The $3,645 left is the whole org account, shared with other SPAR projects.
5. **Modal:** nothing to top up (SPAR's workspace is billed monthly). Expect about **$1,600 more** over the period
   (safe: $2,100).

## The table

One row per model, done one after another. "New money needed" = OpenRouter top-up + Modal bill + Vast top-up.

| Model | OpenRouter cost | OpenRouter top-up | Modal cost | Vast cost | Vast top-up | Total cost | New money needed | Time |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **Left now (19 Nov, 11:50 UTC)** | **$3,645 left** | | **no balance (billed monthly to SPAR)** | **$488 left** | | | | |
| DeepSeek V4 Pro, plus the Qwen3.6 baseline (**measured pilot**) | $600 ($20 already spent) | $0 (safe: $0) | $400 (safe: $520) | $850 | $400 (safe: $650) | $1,850 | $800 | ~1 week |
| Qwen3-Next 80B | $500 | $0 (safe: $0) | $250 (safe: $350) | $550 | $550 (safe: $700) | $1,300 | $800 | ~1 week |
| GPT-OSS 120B | $250 | $0 (safe: $0) | $250 (safe: $350) | $550 | $550 (safe: $700) | $1,050 | $800 | ~1 week |
| Kimi K3 | $2,900 | $600 (safe: $1,850) | $350 (safe: $450) | $550 | $550 (safe: $700) | $3,800 | $1,500 | ~1.5 weeks |
| GLM 5.3 | $1,400 | $1,400 (safe: $1,850) | $350 (safe: $450) | $550 | $550 (safe: $700) | $2,300 | $2,300 | ~1.5 weeks |
| **All of it** | **$5,650** | **$2,000 (safe: $3,700)** | **$1,600 (safe: $2,100)** | **$3,050** | **$2,600 (safe: $3,450)** | **$10,300** | **$6,200** | **~5–6 weeks** |

OpenRouter left after each model: ~$3,065 after DeepSeek, ~$2,565 after Qwen3-Next, ~$2,315 after GPT-OSS, then it
runs out during Kimi K3. The Qwen3.6 baseline (Qwen3.6 itself with thinking on, 40 runs of each of the 28 tasks) is paid
once for all five models: $200 Modal + $300 Vast, counted in the DeepSeek row.

DeepSeek's OpenRouter cost = $20 spent (provider check, 6 test runs, 56 pilot runs) + ~$430 for the full collection
(the pilot's mean cost per run x 46 runs x 28 tasks) + ~$150 Opus grading. Its Modal cost = ~$160 for the collection
(the pilot's ~$0.12 per run) + ~$50 for the trained model's re-runs + $200 for the baseline.

## Measured in the DeepSeek pilot (19 Nov)

- 6 test runs + 56 pilot runs (2 per task), SiliconFlow fp8, reasoning effort xhigh. **OpenRouter $20.22** in total
  (key usage before and after), Modal ~$7.50, launch box $0.107/h.
- **Why it is cheaper than estimated:** 97% of the input was read from the provider's cache (the estimate assumed
  70%), and DeepSeek wrote ~330 reasoning tokens per call (the estimate assumed 1,500).
- Cost per run ranges from under $0.01 (parse_duration, norvane) to ~$1.40 median (leaked_tests, optimize_lru_cache);
  the longest run took ~1.5 hours. Per task: `results/p2_deepseekv4_pilot_usage.csv`.
- 54 of 56 runs ended normally; 2 died on a single provider server error (502, 503). Fixed before the full
  collection: the harness now retries server errors up to 8 times.
- **Run length (matters for training, not for this budget yet):** longest prompt per run: 29 of 55 runs go past 32k
  tokens, 11 past 131k, none past 262k (Qwen3.6's limit; the longest was 248k). Qwen3.6 can hold every run, but our
  training so far cut runs at 32k tokens. Training on whole runs needs a much longer cut and more GPU memory; the Vast
  column still assumes the 32k cut and will be re-priced in the training plan.

## What each column pays for

- **OpenRouter:** the teacher's runs (about 1,250 per model: 28 tasks x 40-46 runs, incl. top-ups and the pilot) plus
  Opus grading of the judged tasks (~$150 per model).
- **Modal:** the task sandboxes for the teacher's runs and for the re-runs of the trained Qwen3.6.
- **Vast:** a rented GPU box (H100 class) to train Qwen3.6 27B on the teacher's runs and re-run its saved copies
  (vLLM), plus a small always-on box (~$0.11/h) that launches the Modal runs so nothing depends on a laptop.

## Where the numbers come from

- **Balances:** OpenRouter `GET /api/v1/key` (usage $5,986.92, limit $10,000 on 19 Nov 11:50 UTC) and `/api/v1/credits`
  (org account $3,644.71 left); Modal `Workspace.billing.summary()` (November: $777.72 metered, $100 credits, $250 plan,
  $922.19 billed by 08:00 UTC; no spend limit visible to us); Vast `show user` (credit $487.78).
- **DeepSeek:** measured in the pilot (above). **The other four models:** `phase2/targets.md` (token counts of our
  Qwen3.6 runs, priced per model, likely = 70% of input cached and 1,500 reasoning tokens per call). Since DeepSeek came
  in at a third of that, these may be high too, but each provider caches differently: each model gets its own pilot
  first and this page is updated then. Kimi K3's $2,900 is the cheapest provider (Wafer, 4-bit); the fp8 providers
  make it ~$5,850.
- **Modal and Vast:** the classic round's measured costs (17-19 Nov: Modal ~$205, box ~$265 for 49.7 h), scaled to
  Qwen3.6 27B as the student; the DeepSeek collection's Modal cost from the pilot.
- Kimi K3 and GLM 5.3 are not in the paper ("Why do models task game?" names DeepSeek V4 Pro, GPT-OSS-120B,
  Kimi K2.7 Code and Gemini 3.5 Flash).

## History

- 19 Nov 08:30 UTC: first version, all estimates (total ~$11,450, new money $7,350).
- 19 Nov 11:50 UTC: DeepSeek pilot measured; DeepSeek $3,000 -> $1,850; total ~$10,300, new money $6,200.
