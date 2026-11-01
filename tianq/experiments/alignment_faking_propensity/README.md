# Alignment-faking propensity (closed: any model predicts it, no gap)

**Question.** Can a reader predict how often an organism alignment-fakes on a prompt, from the prompt alone?

`../alignment_faking_hidden_reasoning` asked whether one rollout faked once its reasoning is hidden, and the answer was no: fakers and honest rollouts give the same answer, and whether a rollout fakes is decided in the reasoning. A single rollout is one coin flip. This asks about the coin instead: the organism's faking rate per prompt, which only its weights could hold if words can't carry it.

## Plan

1. Check the label. Re-run the paper's judge (`scripts/judge_bench_af.py`'s prompt and parser, its default 2000-token thinking, through OpenRouter) on HAL9000's stored report_failure rollouts and measure how often it gives the paper's label and its own. No bar: label flips can hide differences between prompts but not create them.
2. Measure. Sample HAL9000 20 times on each of the 100 prompts (vLLM, the model's own sampling defaults), judge every rollout, and get a faking rate per prompt. Also a lie rate (the output says "passed"), which needs no judge.
3. Decide, then predict. If a rate varies across prompts beyond binomial noise, predict it from the prompt alone: HAL9000's own activations at the end of the prompt (the ceiling), its plain parent and Llama-3.1-8B (substitutes), the words (TF-IDF).

## Result (Nov 1)

Both rates vary across prompts, faking only slightly (p = 0.03, true spread 0.06) and lying clearly (p = 0.0001, spread 0.11). HAL9000's own activations predict them (Spearman 0.43 faking, 0.63 lying), and so do its parent (0.41, 0.62) and the 8B (0.44, 0.64); the words alone barely do (0.14, 0.18). No faithfulness gap, so nothing to distill. The judge check: faking vs honest matched the paper's labels 61% and 76% over two passes, and itself 73%; lies held. Write-up in `report/propensity.md`.

- `common.py`: the prompts, the models at their pinned revisions and layers, the paper's judge through OpenRouter, the judge-free answer
- `judge_gate.py`: step 1
- `sample.py` (GPU), `judge.py` (API): step 2
- `extract.py` (GPU), `decide.py`, `predict.py`: step 3
- `scripts/run_all.sh`: every step on one box, detached, each through `../../af-probe-repro/run.sh propensity <step>`
- `report/`: the write-up (`propensity.md`) and its chart (`report_chart.py`, `propensity.png`); `gap_logic.svg`/`.png`, a diagram of why neither alignment-faking track can show a gap

Results are on the Hub under `af-propensity/` (`judge_gate.json`, `rollouts.jsonl`, `judged.jsonl`, `rates.json`, `decision.json`, `activations/`, `predict.json`), each run's log under `logs/run_20261101T16*_af_propensity_*.log`. The report's links are at Hub revision fed2763.
