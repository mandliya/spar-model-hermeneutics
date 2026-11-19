# Phase 2 budget: what it costs, what is left, what John needs to do (19 Nov 2026)

For John (and any agent John uses on this repo). Phase 2 = black-box hermeneutics from the 19 Nov call: five open-weight
teacher models, each distilled into Qwen3.6 27B (thinking on), in John's order: DeepSeek V4 Pro, Qwen3-Next 80B,
GPT-OSS 120B, Kimi K3, GLM 5.3. All figures are estimates (likely case) except the balances, which were read on
19 Nov 2026. "Safe" adds 30%. The DeepSeek pilot (running from 19 Nov) gives a measured cost per run; this page is
updated after it.

## Actions for John

1. **OpenRouter key limit:** raise it from $10,000 to about **$17,000** before Kimi K3 ($5,967 used today, so $4,033 left).
   A limit costs nothing until it is spent.
2. **OpenRouter workspace lifetime budget:** our key cannot read it. Please check it has room for **~$6,650** more
   (it blocked all calls twice before, on 21 Oct and 11 Nov).
3. **OpenRouter credits:** no top-up needed for the first three models. **~$1,600 before Kimi K3 and ~$1,400 before
   GLM 5.3** ($3,000 in total, safe: $5,000). To be safe, ~$1,000 before the DeepSeek collection: DeepSeek alone could
   reach ~$4,450 if it reasons much more than estimated, and the $3,667 left is the whole org account, shared with
   other SPAR projects.
4. **Vast:** top up **~$400 now** (for DeepSeek and the Qwen3.6 baseline), then ~$550 per model after that
   ($2,600 in total, safe: $3,450).
5. **Modal:** nothing to top up (SPAR's workspace is billed monthly). Expect about **$1,750 more** over the 5–7 weeks
   (safe: $2,300).

## The table

One row per model, done one after another. "New money needed" = OpenRouter top-up + Modal bill + Vast top-up.

| Model | OpenRouter cost | OpenRouter top-up | Modal cost | Vast cost | Vast top-up | Total cost | New money needed | Time |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **Left now (19 Nov)** | **$3,667 left** | | **no balance (billed monthly to SPAR)** | **$488 left** | | | | |
| DeepSeek V4 Pro, plus the Qwen3.6 baseline | $1,600 | $0 (safe: $0) | $550 (safe: $700) | $850 | $400 (safe: $650) | $3,000 | $950 | ~1.5 weeks |
| Qwen3-Next 80B | $500 | $0 (safe: $0) | $250 (safe: $350) | $550 | $550 (safe: $700) | $1,300 | $800 | ~1 week |
| GPT-OSS 120B | $250 | $0 (safe: $0) | $250 (safe: $350) | $550 | $550 (safe: $700) | $1,050 | $800 | ~1 week |
| Kimi K3 | $2,900 | $1,600 (safe: $3,150) | $350 (safe: $450) | $550 | $550 (safe: $700) | $3,800 | $2,500 | ~1.5 weeks |
| GLM 5.3 | $1,400 | $1,400 (safe: $1,850) | $350 (safe: $450) | $550 | $550 (safe: $700) | $2,300 | $2,300 | ~1.5 weeks |
| **All of it** | **$6,650** | **$3,000 (safe: $5,000)** | **$1,750 (safe: $2,300)** | **$3,050** | **$2,600 (safe: $3,450)** | **$11,450** | **$7,350** | **~5–7 weeks** |

OpenRouter left after each model: $2,067 after DeepSeek, $1,567 after Qwen3-Next, $1,317 after GPT-OSS, then it runs
out during Kimi K3. The Qwen3.6 baseline (Qwen3.6 itself with thinking on, 40 runs of each of the 28 tasks) is paid
once for all five models: $200 Modal + $300 Vast, counted in the DeepSeek row. Times are one model at a time; starting
the next model's collection while the previous one trains brings the total to about 5 weeks.

## What each column pays for

- **OpenRouter:** the teacher's runs (about 1,250 per model: 28 tasks x 40-46 runs, incl. top-ups and the pilot) plus
  Opus grading of the judged tasks (~$150 per model).
- **Modal:** the task sandboxes for the teacher's runs and for the re-runs of the trained Qwen3.6.
- **Vast:** a rented GPU box (H100 class) to train Qwen3.6 27B on the teacher's runs and re-run its saved copies
  (vLLM), plus a small always-on box (~$0.09/h) that launches the Modal runs so nothing depends on a laptop.

## Where the numbers come from

- **Balances, read 19 Nov 2026:** OpenRouter `GET /api/v1/key` (usage $5,966.71, limit $10,000) and `/api/v1/credits`
  (credits $87,973.93, usage $84,306.98, so $3,666.95 left on the org account); Modal `Workspace.billing.summary()`
  for November ($777.72 metered, $100 credits, $250 plan, $922.19 billed so far; no spend limit visible to us);
  Vast `show user` (credit $488.12).
- **OpenRouter costs:** `phase2/targets.md` (token counts of our Qwen3.6 runs: ~60M input tokens per pass over the 28
  tasks, priced per model with a guess at reasoning length; likely = 70% of input cached, 1,500 reasoning tokens per
  call). Kimi K3's $2,900 is the cheapest provider (Wafer, 4-bit); the fp8 providers make it ~$5,850.
- **Modal and Vast:** the classic round's measured costs (17-19 Nov: Modal ~$205, box ~$265 for 49.7 h), scaled to
  Qwen3.6 27B as the student. The Vast column is the roughest: Qwen3.6 has not been trained yet.
- Kimi K3 and GLM 5.3 are not in the paper ("Why do models task game?" names DeepSeek V4 Pro, GPT-OSS-120B,
  Kimi K2.7 Code and Gemini 3.5 Flash).

## Spent so far in phase 2

- Stage 1, provider check (19 Nov): ~$0.01 on OpenRouter. All five DeepSeek fp8 providers return tool calls and the
  full reasoning, and put earlier reasoning back into the prompt
  (`results/logs/phase2/provider_check__deepseek_deepseek-v4-pro.json`).
- Launch box: Vast instance 61408889, $0.088/h, from 19 Nov.
