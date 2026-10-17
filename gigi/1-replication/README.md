# Part 1: replication

Anthropic's "Fine-tuned lie detectors failed to generalize" on gemma-3-27b,
then the same experiment on Qwen3-32B. Results, methodology and charts:
[`RESULTS.md`](RESULTS.md).

| Folder | What it holds |
|---|---|
| [`alibi/`](alibi/README.md) | games: does the model keep up a false alibi under questioning? |
| [`ascii-sandbagging/`](ascii-sandbagging/README.md) | denying it can draw ASCII art when the request is harmful |
| [`capture-the-secret/`](capture-the-secret/README.md) | falsely denying, to hide a secret it was told to keep |
| [`mask-belief/`](mask-belief/README.md) | the four MASK subtypes with a belief check: known facts, statistics, continuations, disinformation |
| [`mask-factual/`](mask-factual/README.md) | telling a pressured false story despite the true facts in the prompt |
| [`sycophancy/`](sycophancy/README.md) | track A (answer folds to the user), track B (critique shifts with the user's stake) |
| [`unanswerable/`](unanswerable/README.md) | confabulating an answer to a question that has none |
| [`ab_swap/`](ab_swap/README.md) | the letter-swap test ("yes" at B instead of A) |
| [`answer_probability/`](answer_probability/README.md) | P("A") at the answer token for every follow-up |
| [`named_cot/`](named_cot/) | the fourth question (the act named, reasoning first) |
| [`third_person/`](third_person/README.md) | outside monitors reading the transcripts |
| [`false_confessions/`](false_confessions/) | the false confessions, extracted for reading |
| [`docent/`](docent/DOCENT_RESULTS.md) | the transcripts in Docent and the analyses run there |
| [`charts/`](charts/) | the charts on `RESULTS.md` |
| [`scripts/`](scripts/) | the cross-category scripts (tables, charts, GPU-box helpers) that used to sit at the top of `gigi/` |

The Qwen3-32B run of this part was made by the pipeline in
[`../qwen/`](../qwen/README.md); its results are in `../qwen/results/qwen3-32b/`.
