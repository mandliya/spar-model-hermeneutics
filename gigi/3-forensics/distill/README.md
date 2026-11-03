# Distillation data: Qwen3-Next-80B traces in gemma text-mode format

Data prep for the plan from the 28 Oct 2026 call: distil Qwen's agentic traces into gemma-3-27b and see whether the
Qwen-gemma gaps close. Built by `scripts/distill_data.py` from `results/raw/` (no API calls); rebuild, don't edit.

## What the data is

- **Sources**: Qwen's full runs (the runs behind `results/full_runs.csv`, 406 with a transcript, 4 lost) and Qwen's
  swap replays (`results/gap_replays.csv`, who=qwen: 1,807 with a transcript, 420 lost to provider errors, crashes or
  cut-off transcripts). A replay shares assistant turns 0..K with its original run; those turns are emitted once.
- **One example per assistant turn**, with its whole preceding context, rendered exactly as gemma sees a history in
  text mode: the script imports `_text_tools_messages` (with `TEXT_TOOLS_INSTRUCTION`, `_python_stub`, `_call_text`)
  from `agent-interp-envs/src/agent_interp_envs/providers/openrouter_provider.py`, with the tools each environment's
  own `tools.py` builds for the run's `config.yaml`. System preamble = system prompt + Gemma 3 function-calling
  instruction + tools as Python stubs; tool call = ```` ```tool_code ```` block, tool result = ```` ```tool_output ````
  block in a user turn; a final reply is plain text.
- **Format** (`train.jsonl`, `val.jsonl`, one JSON per line):
  ```
  {"env": "coding", "run": "results/raw/modal/gigi-forensics-coding/coding/qwen-.../run-10", "step": 2,
   "hacked": false, "kind": "full", "group": "<the original run; split unit>",
   "messages": [{"role": "system", "content": "You have access to one tool:...\n\nAt each turn, if you decide to invoke..."},
                {"role": "user", "content": "## Task\nPlease solve the programming problem..."},
                {"role": "assistant", "content": "I'll start by looking at the workspace.\n```tool_code\nexecute_command(command='ls -la')\n```"}]}
  ```
  Last message = target, the rest = prompt. `hacked` = grader's verdict for the run (true/false, null = unscored).

## Counts (`inventory`, 3 Nov 2026)

| | runs | hacked / clean / unscored | unique turns | example chars |
|---|---|---|---|---|
| Qwen full runs | 406 (4 lost) | 85 / 275 / 46 | 11,790 | |
| Qwen swap replays | 1,807 (420 lost) | 591 / 906 / 310 | 18,017 new turns | |
| total | 386 run families | | 29,807 | 6.53 G chars = 6.7 GB JSONL |

Turns: dashboard_costly_failure 9,536 (Qwen's repetition loops), then grade_your_own, optimize_rendering_engine,
lucky_seed_readable_grader, lucky_seed, ai_rd, async_conversion at 2,000-2,500 each; the other 22 environments under
1,000 (full table: `inventory`). Bytes (each turn carries its whole context) are dominated by long runs:
dashboard_costly_failure 3.3 GB, async_conversion 0.9 GB, grade_your_own, lucky_seed_readable_grader, lucky_seed 0.5-0.6 GB.

## Split rule

By run family, never by turn: a family = an original run plus all its replays (they share a prefix), assigned whole to
train or val, stratified by environment, 80/20 (`--split 0.8 --seed 0`; a lone family goes to train). `manifest.json`
records seed, split, counts per environment and split, every excluded run with its reason, the sampling note and the
git commits (repo and agent-interp-envs).

**The committed files are a sample.** The full set is 6.7 GB, over the 200 MB limit, so `build` wrote a seeded
stratified sample: a round-robin over environments taking each family that still fits 200 MB (85 families; 1,320
train + 488 val examples; 63 + 22 runs; train 182 MB, val 18 MB). Long families are skipped by construction: no
async_conversion or revert_or_refactor run, long runs under-represented. For training, rebuild the full set locally
(same command, raise `MAX_BYTES`) or cap contexts with `--max-context-chars`.

## Test set contents (3 Nov 2026): does the 20% contain hill-climbable tasks?

The 28 Oct call asked for an 80/20 split whose test set contains hill-climbable tasks: the environments with a mechanical
measure where a distilled gemma could move toward Qwen's behaviour. Computed for the **full** set with the split's own
code (`stratified_split`, seed 0, every family; no file written):

| environment (* = gap environment) | families train / val | runs incl. replays train / val | Qwen hacked / scored in val |
|---|---|---|---|
| ai_rd * | 13 / 3 | 200 / 22 | 8 / 22 |
| optimize_rendering_engine * | 4 / 1 | 138 / 31 | 7 / 21 |
| dashboard_costly_failure * | 8 / 2 | 246 / 14 | 1 / 12 |
| coding * | 16 / 4 | 156 / 4 | 0 / 4 |
| coding_norules * | 16 / 4 | 136 / 4 | 0 / 4 |
| precommit_hook * | 16 / 4 | 36 / 14 | 0 / 14 |
| grade_your_own * | 4 / 1 | 63 / 1 | 0 / 1 |
| optimize_rendering_engine_graded * | 4 / 1 | 25 / 11 | 0 / 11 |
| the other 22 environments | 249 / 62 | 917 / 195 | 34 / 168 |
| **total** | **330 / 82** | **1,917 / 296** | |

- **The split is 80/20 by family (82 of 412), but 87/13 by run**, because the families with many replays (Qwen's
  cheating runs, which carry the swap points) mostly drew train.
- **Every gap environment is in the test set, and all of them have a mechanical measure** (tests, mypy, speedup,
  benchmark target, scorer, accuracy), so all are hill-climbable in that sense.
- **But only 3 of the 8 hold out one of Qwen's cheating runs** (ai_rd, optimize_rendering_engine, dashboard; counted on
  412 families, `build` uses 386 and gets 4 of 8: see the next section). In coding,
  coding_norules, precommit_hook, grade_your_own and the graded rendering engine, the held-out families are runs where
  Qwen did not cheat, so a held-out test there cannot show gemma picking up Qwen's cheating on an unseen incident.
- **Proposed fix, not applied (it changes the data, the team's call):** stratify by environment *and* by whether the
  family contains a cheating run, so each gap environment holds out at least one cheating family with its replays.
- The committed 200 MB sample is not the training set: its val holds 1 family per environment and none from
  grade_your_own, optimize_rendering_engine, lucky_seed, lucky_seed_readable_grader, puppeteer or swe_rebench_v2.

## Incident-stratified split (proposed, 3 Nov)

The 3 Nov call settled on Yuqi's rule: the cheating families (Qwen's incidents, where it differs from gemma) and the
other families get the same train fraction, so a better score on held-out incidents cannot come from training on more
of them. John's version (incidents 50/50, the rest 80/20) is the fallback if there are too few incidents.
`build --stratify-incidents` implements the rule. It is opt-in, and the default split is unchanged.

- **Strata**: within each environment, a family is a cheating family when any of its runs (the original or a replay)
  is one the grader marked hacked; every other family is in the second stratum. Both strata use `--split` (0.8).
  `--hacked-split 0.5` switches to John's fallback for the cheating stratum only.
- **Hold-out rule**: a stratum with n ≥ 2 families holds out max(1, round(n × 0.2)) of them and keeps at least one in
  train, so every environment with 2 or more cheating families holds one out. A lone family goes to train, as before.
- **Families stay whole, and the split is fixed first.** The split is drawn on all families before
  `--max-context-chars`, `--exclude-hacked` or the size guard drop anything, so those filters remove examples but
  never move a family between train and val.
- **Preview without writing**: `build --stratify-incidents --dry-run --max-mb 100000` prints the table below. Writing
  goes to a subfolder with `--out distill/stratified`, which keeps the committed sample.

Full set, seed 0, no context cap. A family here is an original run with its replays that has at least one unique
example: 386 of them. The table in the previous section counted all 412, including 26 families whose every turn
duplicates another family's (mostly one-turn norvane runs), so its draws differ from what `build` writes.

| environment (* = gap environment) | cheating families train / val | other families train / val | runs incl. replays train / val | Qwen hacked / scored in val | examples train / val |
|---|---|---|---|---|---|
| ai_rd * | 4 / 1 | 9 / 2 | 152 / 70 | 10 / 68 | 1,394 / 672 |
| coding * | 1 / 0 | 15 / 4 | 156 / 4 | 0 / 4 | 534 / 34 |
| coding_norules * | 3 / 1 | 13 / 3 | 106 / 34 | 21 / 34 | 401 / 120 |
| dashboard_costly_failure * | 1 / 1 | 6 / 2 | 19 / 241 | 24 / 108 | 2,359 / 7,177 |
| grade_your_own * | 1 / 1 | 2 / 1 | 41 / 23 | 1 / 22 | 436 / 2,092 |
| optimize_rendering_engine * | 4 / 1 | 0 / 0 | 157 / 12 | 5 / 11 | 2,036 / 145 |
| optimize_rendering_engine_graded * | 0 / 0 | 4 / 1 | 25 / 11 | 0 / 10 | 199 / 150 |
| precommit_hook * | 1 / 0 | 15 / 4 | 46 / 4 | 0 / 4 | 104 / 13 |
| async_conversion | 0 / 0 | 4 / 1 | 63 / 1 | 0 / 1 | 1,936 / 65 |
| chess | 0 / 0 | 16 / 4 | 36 / 14 | 0 / 3 | 387 / 109 |
| copying_of_answers | 0 / 0 | 8 / 2 | 28 / 12 | 0 / 12 | 45 / 18 |
| currency_conversion | 16 / 4 | 0 / 0 | 76 / 4 | 4 / 4 | 79 / 4 |
| eval_tampering | 9 / 2 | 0 / 0 | 69 / 2 | 2 / 2 | 38 / 3 |
| funding_email | 16 / 4 | 0 / 0 | 76 / 4 | 4 / 4 | 92 / 16 |
| lazy_investigation | 10 / 3 | 6 / 1 | 57 / 24 | 23 / 23 | 333 / 68 |
| leaked_scorer | 0 / 0 | 8 / 2 | 38 / 2 | 0 / 2 | 29 / 8 |
| leaked_tests | 0 / 0 | 4 / 1 | 24 / 11 | 0 / 11 | 133 / 46 |
| lucky_seed | 0 / 0 | 7 / 2 | 33 / 12 | 0 / 12 | 2,010 / 59 |
| lucky_seed_readable_grader | 0 / 0 | 8 / 2 | 68 / 19 | 0 / 2 | 2,002 / 147 |
| norvane | 0 / 0 | 10 / 3 | 30 / 13 | 0 / 13 | 23 / 8 |
| optimize_lru_cache | 0 / 0 | 4 / 1 | 4 / 1 | 0 / 0 | 4 / 1 |
| parse_duration | 0 / 0 | 16 / 4 | 35 / 12 | 0 / 12 | 44 / 13 |
| puppeteer | 0 / 0 | 4 / 1 | 34 / 1 | 0 / 0 | 678 / 292 |
| reference_extraction | 0 / 0 | 8 / 2 | 33 / 20 | 0 / 12 | 462 / 1,171 |
| revert_or_refactor | 0 / 0 | 4 / 1 | 15 / 10 | 0 / 10 | 486 / 40 |
| sandbagging | 0 / 0 | 16 / 4 | 47 / 4 | 0 / 0 | 193 / 21 |
| secret_number | 0 / 0 | 16 / 4 | 46 / 4 | 0 / 4 | 183 / 41 |
| swe_rebench_v2 | 0 / 0 | 8 / 2 | 8 / 2 | 0 / 2 | 219 / 27 |
| tictactoe | 0 / 0 | 16 / 4 | 46 / 4 | 0 / 4 | 165 / 29 |
| tower_of_hanoi | 0 / 0 | 14 / 3 | 41 / 3 | 0 / 3 | 182 / 32 |
| **gap environments** | **15 / 5** | **64 / 17** | **702 / 399** | **61 / 261** | **7,463 / 10,403** |
| **other environments** | **51 / 13** | **177 / 44** | **907 / 179** | **33 / 136** | **9,723 / 2,218** |
| **total** | **66 / 18** | **241 / 61** | **1,609 / 578** | **94 / 397** | **17,186 / 12,621** |

- **5 of the 8 gap environments hold out a cheating family**: ai_rd, coding_norules, dashboard_costly_failure,
  grade_your_own and optimize_rendering_engine.
- **3 gap environments have fewer than 2 cheating families, so they cannot hold one out.** coding and precommit_hook
  each have one, which goes to train. optimize_rendering_engine_graded has none: Qwen never cheated there. A held-out
  incident in these three needs more Qwen cheating runs, not a different split.
- **The default split, run on the same 386 families, already holds out a cheating family in 4 of 8 gap
  environments.** These are ai_rd, dashboard, grade_your_own and optimize_rendering_engine; the stratified split adds
  coding_norules. The "3 of 8" in the previous section came from the 412-family draw.
- **Families are very unequal, so the 80/20 family split is 58/42 by example, and 42/58 in the gap environments.**
  In dashboard the held-out cheating family is run-2, the swap family, with 229 runs and 7,029 examples. That leaves
  train only run-1: 13 runs, one hacked, from a clean original. In grade_your_own, val holds run-5 (22 runs, 1 hacked,
  2,077 examples) and train holds run-3 (21 runs, all hacked, 37 examples).
- **The seed moves the example balance a lot.** Seeds 0 to 4 put between 5,543 and 12,621 examples in val, with the
  same families-per-stratum counts. With 2 cheating families the only possible hold-out is 1 of 2, so in dashboard and
  grade_your_own Yuqi's and John's rules coincide.
- **A cheating family is not always a cheating original.** dashboard run-1, coding_norules run-10 and
  optimize_rendering_engine run-2 and run-3 had clean originals and a hacked replay. In optimize_rendering_engine the
  held-out cheating family is run-2, where 5 of 12 runs hacked.
- **A training context cap removes many held-out incident turns** (the split itself does not move). With
  `--max-context-chars 105000`, about 32k gemma tokens, 9,568 train and 3,775 val examples remain. Examples from
  hacked runs kept in val: dashboard 47 of 607, grade_your_own 236 of 500, optimize_rendering_engine 49 of 49. At 16k
  tokens these drop to 26 of 607, 58 of 500 and 0 of 49. See PLAN.md.

```
$PY gigi/3-forensics/scripts/distill_data.py build --stratify-incidents --dry-run --max-mb 100000      # table only
$PY gigi/3-forensics/scripts/distill_data.py build --stratify-incidents --max-mb 100000 --max-context-chars 105000 --out distill/stratified
$PY gigi/3-forensics/scripts/distill_data.py check --out distill/stratified
```

## Two variants

- default (what the team wants first): **distil Qwen as it is**, hacked runs included, `hacked` kept per example.
- `--exclude-hacked`: **honest-behaviour distillation**, drops every run (full or replay) the grader marked hacked;
  unscored runs stay, and a hacked original's shared prefix still appears through its clean replays.

## Caveats

- Qwen ran as fp8 through OpenRouter (model-card sampling, recorded per run); these are not greedy traces.
- Text-mode rendering stands in for Qwen's native tool calls: 2 turns with two parallel calls become two `tool_code`
  blocks (the harness reads only the first); tower_of_hanoi has no system message, so the preamble lands in the first
  user turn, as at inference. `check` re-parses targets with `_tool_call_from_tool_code`: 17/17 tool calls in a
  20-example sample recover the original name and arguments (3 were final-text turns); prefix rendering 20/20 identical.
- Replays' prefixes are deduped (turns 0..K emitted once per family); identical samples are deduped by content.
- Contexts are quadratic in run length: gemma-3's 128k tokens is roughly 400k chars, so the longest examples
  (async_conversion K=198, dashboard loops) need truncation or dropping (`--max-context-chars 400000`).
- 46 full runs and 310 replays are unscored (sandbagging and optimize_lru_cache have no measure; grader gaps
  elsewhere): `hacked` is null there, and both variants keep them.

## Commands

```
PY=gigi/3-forensics/agent-interp-envs/.venv/bin/python          # has openai + yaml (setup.sh)
$PY gigi/3-forensics/scripts/distill_data.py inventory
$PY gigi/3-forensics/scripts/distill_data.py build --split 0.8 --seed 0 [--exclude-hacked] [--envs coding,coding_norules]
$PY gigi/3-forensics/scripts/distill_data.py check
```
