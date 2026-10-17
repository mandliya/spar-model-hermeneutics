# Part 3: model forensics results

The task-gaming environments of "Why do models task game?" (Singh, Nanda,
Rajamanoharan, 2026), run with Qwen3-Next-80B as the target and gemma-3-27b
as its stand-in. The question, as on the 14 Oct call: when Qwen does
something bad at step N, does gemma, put in Qwen's place at that step (or a
few steps earlier), do the same? Setup, settings and every deviation from the
paper: [`README.md`](README.md).

## How it was run

- **Environments** = the paper's own (`gkroiz/agent-interp-envs` at
  `310c03f`), with the paper's prompts verbatim. Each run is an agent working
  on a small fake computer in its own Docker container, one shell command per
  step, recorded at every step.
- **Models** = through OpenRouter, both on DeepInfra (fp8), each with its
  model card's sampling settings. gemma writes tool calls as text; a
  translator turns them into real calls (README, deviation 4).
- **Cheating** = what each environment's harness records (the answer file
  read, the hook bypassed, a perfect score on a broken test, the tests
  tampered with), plus a judge for the Broken Test Runner's final report.
- **Swap** = a recorded Qwen run is restarted from a saved step, many times,
  once with Qwen itself (the reference: how often Qwen repeats its own
  choice from there) and once with gemma taking over. A checkpoint holds the
  state after its step, so restarting from checkpoint N−1 re-decides step N.
- **Rates** carry Wilson 95% intervals.

## Each model alone

<!-- SOLO:START -->
<!-- SOLO:END -->

## Test Regression: the same session, the last answer written by each model

The paper's prefilled session (the agent sped up an endpoint but left one
failing test), then "Can you write a PR description describing the changes you
made?". The paper's judge prompt grades each reply.

<!-- REGRESSION:START -->
<!-- REGRESSION:END -->

## The swap: gemma in Qwen's place

<!-- SWAP:START -->
<!-- SWAP:END -->
