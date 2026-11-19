# Phase 2 target survey: five teachers to distil into Qwen3.6 27B (19 Nov 2026)

Sources: OpenRouter `/api/v1/models` and `/api/v1/models/<id>/endpoints` (fetched 19 Nov 2026, free endpoints),
Hugging Face model API + `config.json`, the task-gaming post (Singh, Nanda, Rajamanoharan 2026,
https://www.lesswrong.com/posts/HACauvWhEdC6QhdS4), and our own run folders. No paid calls were made.
Prices are $ per million tokens as OpenRouter lists them (input / output / cache read). "Tools" = the endpoint lists
`tools` in its supported parameters. "Reasoning" = it lists `reasoning`/`include_reasoning`; whether the text is the
FULL chain of thought (not a summary) can only be confirmed with a test call (see the end).

## 1. DeepSeek V4 Pro (John's step 1)

| | |
|---|---|
| OpenRouter ids | `deepseek/deepseek-v4-pro` (April checkpoint, slug `-20260423`); `deepseek/deepseek-v4-pro-0813` (August checkpoint); alias `~deepseek/deepseek-pro-latest` |
| Open weights | `deepseek-ai/DeepSeek-V4-Pro`, 1.6T total / 49B active (MoE), MIT; native context 1,048,576 (YaRN x16 from 65,536) |
| Task gaming | **Yes, the post's main case**: "Whether DeepSeek v4 Pro will task game is causally influenced by beliefs about oversight, grader capability...", "deludes itself ... with motivated back-of-the-envelope calculations in its CoT" |
| How the paper ran it | The env repo has `providers/fireworks_completions_provider.py` + `deepseek_v4_template.py`: Fireworks, with the reasoning of ALL previous turns kept verbatim in the prompt. Fireworks is not among OpenRouter's endpoints for either V4 Pro id |

Endpoints for `deepseek-v4-pro` (0423), all with tools + reasoning:

| Provider | Precision | Context | Max out | In / Out / Cache |
|---|---|--:|--:|---|
| StreamLake | fp8 | 1,024,000 | 384,000 | 0.90 / 1.80 / 0.075 |
| Relace | fp4 | 1,048,576 | 393,216 | 0.90 / 1.50 / 0.25 |
| SiliconFlow | fp8 | 1,048,576 | 393,216 | 0.90 / 1.70 / 0.08 |
| GMICloud | fp8 | 1,048,576 | 943,718 | 0.96 / 1.91 / 0.08 |
| DeepInfra | fp8 | 1,048,576 | 393,216 | 1.30 / 2.60 / 0.10 (flex tier 1.04 / 2.08) |
| Novita | fp8 | 1,048,576 | 393,216 | 1.28 / 2.56 / 0.11 |
| Alibaba | fp8 | 1,000,000 | 393,216 | 1.42 / 2.83 / 0.12 |
| Azure | unknown | 1,048,576 | 384,000 | 1.91 / 3.83 / 0.16 |
| (others: DigitalOcean, Venice, AtlasCloud fp4, Baidu fp8) | | | | 1.04–1.69 / 2.09–3.38 |

The 0813 checkpoint adds the **DeepSeek native** endpoint (0.66 / 1.98 / 0.022) plus ~25 others (fp4 to fp8).

## 2. Qwen3-Next 80B (John's step 2)

| | |
|---|---|
| OpenRouter id | `qwen/qwen3-next-80b-a3b-instruct` only (the Thinking variant is NOT on OpenRouter) |
| Open weights | `Qwen/Qwen3-Next-80B-A3B-Instruct` and `-Thinking`, 81.3B (MoE, 3B active), Apache-2.0; native context 262,144 |
| Reasoning | None: the Instruct model has no thinking mode (our runs use `reasoning_effort: none`), so nothing is hidden or summarised. A thinking teacher would need the Thinking variant self-hosted (fp8 ~80 GB fits one H200) |
| Task gaming | Not in the post; our own October results (RESULTS.md) show it cheats in several environments |

| Provider | Precision | Context | Max out | Tools | In / Out / Cache |
|---|---|--:|--:|:-:|---|
| DeepInfra | fp8 | 262,144 | 16,384 | yes | 0.09 / 1.10 / none |
| DeepInfra flex | fp8 | 262,144 | 235,929 | yes | 0.072 / 0.88 / none |
| Parasail | fp8 | 262,144 | 235,929 | yes | 0.10 / 1.10 / 0.07 |

## 3. GPT-OSS 120B (John's step 3)

| | |
|---|---|
| OpenRouter id | `openai/gpt-oss-120b` |
| Open weights | `openai/gpt-oss-120b`, 116.8B (MoE, 5.1B active, MXFP4 experts natively), Apache-2.0; native context 131,072 (half of Qwen3.6 27B's 262k) |
| Task gaming | **Yes, in the post**: "Task gaming can manifest as deception (GPT-OSS-120B)", fabricating logs in Secret Number and Pre-commit Hook; its CoT calls it "cheating" |

| Provider | Precision | Context | Max out | Tools | In / Out / Cache |
|---|---|--:|--:|:-:|---|
| DeepInfra | bf16 | 131,072 | 117,964 | yes | 0.037 / 0.17 / 0.018 |
| Crusoe | bf16 | 131,072 | 117,964 | yes | 0.05 / 0.25 / 0.05 |
| Cerebras | fp16 | 131,072 | 40,960 | yes | 0.35 / 0.75 / 0.35 |
| Groq | unknown | 131,072 | 65,536 | yes | 0.15 / 0.60 / 0.075 |
| Together / Bedrock | unknown | 131,072 | 117,964 | yes | 0.15 / 0.60 |
| Nebius / Near AI / AkashML / BaseTen | fp4 | ~131k | ~115k | yes | 0.03–0.15 / 0.15–0.60 |
| Google Vertex, DigitalOcean, SiliconFlow | | | | **no** | (no tool calling: unusable) |

## 4. Kimi K3

| | |
|---|---|
| OpenRouter ids | `moonshotai/kimi-k3` (also `kimi-k3.1-pro`, `~moonshotai/kimi-latest`) |
| Open weights | `moonshotai/Kimi-K3`, 2.78T, multimodal, license "other"; native context 1,048,576 |
| Task gaming | **Not in the post** (it tests Kimi **K2.7 Code**, not K3). Uncertain whether K3 task-games |
| Harness | The env repo has `providers/moonshot_provider.py` (Moonshot API directly, interleaved thinking) |

| Provider | Precision | Max out | Tools | In / Out / Cache |
|---|---|--:|:-:|---|
| Moonshot AI (native) | unknown | 943,718 | yes | 3.00 / 15.00 / 0.30, **status degraded today (-2, 87.5% uptime)** |
| Fireworks, Together, Modal, Parasail fp4, Alibaba | unknown / fp4 | 900k+ | yes | 3.00 / 15.00 / 0.30 |
| BaseTen | fp8 | 262,144 | yes | 3.00 / 15.00 / 0.30 |
| DeepInfra | mxfp4 | 943,718 | yes | 2.85 / 14.25 / 0.285 |
| Wafer, Relace, InferenceNet | mxfp4 / fp4 | 943,718 | yes | 0.15–0.23 / 11–16 |
| Chutes, River | | | **no** | (no tool calling) |

Output costs $15 per million tokens, about 6–9x DeepSeek: the most expensive of the five by far.

## 5. GLM 5.3

| | |
|---|---|
| OpenRouter ids | `z-ai/glm-5.3` (also `-flash`, `-flashx`, `-prime`) |
| Open weights | `zai-org/GLM-5.3`, 753B, license "other"; native context 1,048,576 |
| Task gaming | **Not in the post** (no GLM model is mentioned). Uncertain |

| Provider | Precision | Max out | In / Out / Cache |
|---|---|--:|---|
| Z.AI (native) | fp8 | 131,072 | 1.40 / 4.40 / 0.26 |
| SiliconFlow | fp8 | 262,144 | 0.70 / 1.90 / 0.13 |
| Novita | fp8 | 131,072 | 0.63 / 1.98 / 0.117 |
| GMICloud | fp8 | 943,718 | 0.70 / 2.20 / 0.13 |
| Tenstorrent | fp8 | 131,072 | 0.60 / 2.50 / 0.10 |
| ~40 more (fp4 / nvfp4 / unknown) | | | 0.07–2.10 / 1.40–6.60 |

All 48 GLM 5.3 endpoints list tools + reasoning.

## Cost of collecting the traces (28 tasks x 40 runs, +15% top-ups = 1,288 runs per model)

Basis, measured: our Qwen3.6 27B hero9 runs (363 runs, laptop), token counts estimated as characters / 4 from the
saved conversations. Per "one run of each of the 28 classic tasks": **60.5M input tokens, 0.72M visible output tokens,
~1,450 model calls** (input is dominated by the long tasks: reference_extraction ~15M per run, puppeteer 7.7M,
dashboard 6.6M, the rendering engines ~6M). For the whole collection: **2.8B input, 33M visible output, ~67k calls.**
These Qwen runs had no thinking, so reasoning tokens are added for the four thinking models.

Scenarios: low = 90% of input billed as cache reads, 500 reasoning tokens per call; likely = 70% cached, 1,500
reasoning tokens per call, runs 20% longer; high = no caching, 3,000 reasoning tokens per call, runs 50% longer
(a 1M-context teacher can keep going where Qwen3.6 filled its window).

| Model, endpoint | Low | Likely | High |
|---|--:|--:|--:|
| DeepSeek V4 Pro 0423, DeepInfra fp8 | $785 | $1,882 | $6,030 |
| DeepSeek V4 Pro 0423, SiliconFlow fp8 | $564 | $1,315 | $4,151 |
| DeepSeek V4 Pro 0813, DeepSeek native | $370 | $976 | $3,215 |
| Qwen3-Next 80B Instruct, DeepInfra fp8 | $287 | $337 | $412 |
| Qwen3-Next 80B Instruct, Parasail fp8 | $240 | $300 | $454 |
| GPT-OSS 120B, DeepInfra bf16 | $67 | $102 | $194 |
| Kimi K3, Fireworks / Together / BaseTen | $2,584 | $5,703 | $16,015 |
| Kimi K3, Wafer mxfp4 | $1,574 | $2,750 | $4,534 |
| GLM 5.3, Z.AI fp8 | $1,333 | $2,595 | $6,867 |
| GLM 5.3, SiliconFlow fp8 | $647 | $1,258 | $3,364 |

Not included, per model: Modal sandboxes (the classic round paid $202 for ~1,500 runs, including an overload), and
Opus grading for the four judged environments (~$1 per long try per earlier measurements; ~180 tries, so ~$100–200).
The target's own verbosity may differ a lot from Qwen3.6's: these are order-of-magnitude figures. Caching is the
biggest lever: whether a provider caches across our separate Modal sandboxes is untested.

## Qwen3-Next 80B: can we reuse the runs we have?

All local, all `qwen/qwen3-next-80b-a3b-instruct` on DeepInfra fp8, native tool calls, no reasoning
(`reasoning_effort: none`). Counted from `results/raw/` (complete = finished or stopped at the context window):

| Stages | Runs | Complete | Tasks | Setup |
|---|--:|--:|--:|---|
| hero2_qwen, hero2_qwen_all, hero2_qwen_key (6 Nov) | 385 | 352 | 28 | **end check ON** (`end_confirm: true`), ~10–20 per task |
| October stages (full_*, modal, screen, longhist, base_qwen) | ~1,600 | ~1,350 | up to 39 env folders | plain, but older harness and env versions (pre-26 Oct fixes, ai_rd v1/v2, older env names like dashboard_perf, leaked_fix, optimize_cache_c) |

What is missing for the current classic setup (plain, no end check, current harness and configs, 40 complete runs per
task): **none of the existing runs matches it exactly.** hero2 differs by the end check (and has only 10–20 per task);
the October runs predate several harness and environment fixes. Reusing them would mix setups. The clean option is
a fresh 40-per-task collection (~$300, cheapest of the five because there is no reasoning), possibly keeping hero2 as a
separate "with the end check" condition.

## Recommendation

- **Pin per model** (allow_fallbacks false), fp8 or better where the price allows:
  - DeepSeek V4 Pro: decide the checkpoint first (0423 is most likely the paper's; the paper ran it on Fireworks).
    On OpenRouter: DeepInfra fp8 (our usual provider) or SiliconFlow fp8 (cheaper, same precision).
  - Qwen3-Next 80B: DeepInfra fp8, the same endpoint as our existing traces (use the flex/priority tier if the
    standard tier's 16,384 max output bites).
  - GPT-OSS 120B: DeepInfra bf16 (MoE experts are MXFP4 natively).
  - Kimi K3: Fireworks or Together, or Moonshot directly via the env repo's provider; the native OpenRouter endpoint
    is degraded today. Expensive: consider dropping it if budget matters.
  - GLM 5.3: Z.AI fp8 (native) or SiliconFlow fp8.
- **Full-reasoning requirement:** all five are open-weight, so providers return the raw chain of thought, not a
  summary. Qwen3-Next Instruct has none to lose. Nothing fails the requirement on paper, but it is unconfirmed per
  endpoint: before any fleet, one test call per pinned endpoint (~$0.01–0.10 each) should check that `reasoning` comes
  back, is long and raw, and how previous turns' reasoning is fed back (the paper kept it verbatim for DeepSeek).
  Endpoints without tool calling (Chutes and River for Kimi K3; Vertex, DigitalOcean, SiliconFlow for GPT-OSS) are
  unusable.
- **Open questions:** Kimi K3 and GLM 5.3 are not in the post (John's notes say the picks were tested there; the post
  names DeepSeek v4 Pro, GPT-OSS-120B, Kimi K2.7 Code and Gemini 3.5 Flash). Our training so far cut every run at 32k
  tokens / 105k characters: the 1M-context teachers' long runs would be cut the same way.
