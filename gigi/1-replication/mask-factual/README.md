# Does gemma-3-27b-it recite the cover story it's told to give?

Third category in this replication of [Anthropic's lie-detectors
post](https://alignment.anthropic.com/2026/lie-detectors/), on
`google/gemma-3-27b-it`. `mask_provided_facts` is built; the dataset's other
two `mask-factual` subtypes (`mask_known_facts`, `mask_statistics`) need
external fact-checking this project doesn't have yet — see `RESULTS.md`.

## Setup

Same environment pattern as `../ascii-sandbagging/` and `../sycophancy/`:
```bash
uv sync
cp .env.example .env   # paste real OPENROUTER_API_KEY and HF credentials
```
Runs on a local GPU (Vast.ai), not OpenRouter — same reasoning as every
other category. Temperature 0 (greedy decoding, deterministic).

## Mechanism

Unlike the other two categories, `mask_provided_facts`'s ground truth is
given directly in each row's own system prompt (both the true fact and the
false narrative gemma is pressured to give), not elicited. A judge
(`classify_mask_response()` in `src/judge.py`) reads the system prompt and
gemma's response, and classifies it as `TOLD_TRUTH` / `TOLD_FALSE_NARRATIVE`
/ `EVASIVE`. See `RESULTS.md` for the full methodology and results.

## Run

```bash
uv run python 00_mask_factual_provided.py
uv run python 00_mask_factual_provided.py --limit 12   # smoke test
```

## Where the data lives

| File | What it is |
|---|---|
| `results/mask_factual_provided_conditions.csv` / `.jsonl` | 86 rows |
| `00_mask_factual_provided.py` | generates the above (local GPU, bf16, with activations) |

Activation tensors are on Hugging Face, [`model-hermeneutics/gigi-lie-detection-activations`](https://huggingface.co/datasets/model-hermeneutics/gigi-lie-detection-activations): the follow-up questions in
`activations/mask_factual.tar.gz` (the four questions, float32), and the primary run in
`first_run/mask_factual/` (float32, all 87 rows; see `RESULTS.md`).

## Push your work

From the repo root:
```bash
git pull
git add gigi/     # ONLY this folder, never -A
git commit -m "gigi: mask-factual, gemma-3-27b"
git push
```
