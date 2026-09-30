---
pretty_name: gemma-3-27b-it lie-detection activations
tags:
- interpretability
- lie-detection
- activations
- gemma
size_categories:
- 10K<n<100K
---

# gemma-3-27b-it lie-detection activations

Hidden-state activations from `google/gemma-3-27b-it`, recorded while it lied
or told the truth and was then asked whether it had lied. Part of a SPAR
replication of Anthropic's
[fine-tuned lie detectors](https://alignment.anthropic.com/2026/lie-detectors/)
post, built on its released dataset
[`Noddybear/lies`](https://huggingface.co/datasets/Noddybear/lies).

Each tensor is labelled with independently checked ground truth (did gemma
actually lie on this row?), not the dataset's own `did_lie` column, which is
unusable for gemma.

## Which files to use

**Follow-up activations: use `tarballs/fp32/`.** These are the current set,
in float32, for all seven deception categories.

| File | Category | Size |
|---|---|---|
| `tarballs/fp32/ascii_fp32.tar.gz` | ascii-sandbagging | 4.8 GB |
| `tarballs/fp32/sycophancy_a_fp32.tar.gz` | sycophancy, trivia (Track A) | 2.4 GB |
| `tarballs/fp32/sycophancy_b_fp32.tar.gz` | sycophancy, critique (Track B) | 3.1 GB |
| `tarballs/fp32/mask_factual_fp32.tar.gz` | mask-factual | 1.3 GB |
| `tarballs/fp32/unanswerable_fp32.tar.gz` | unanswerable | 1.5 GB |
| `tarballs/fp32/capture_secret_fp32.tar.gz` | capture-the-secret | 5.8 GB |
| `tarballs/fp32/alibi_fp32.tar.gz` | alibi | 7.7 GB |
| `tarballs/fp32/named_cot_<category>_fp32.tar.gz` | the 13th wording, `named_cot`, one per category | 0.1–0.6 GB each |

`maps/fp32/fp32_map_<category>.jsonl` lists every tensor in the matching
tarball:

```json
{"name": "conv_20250718_ascii_mc_medical_5__neutral",
 "predecision_file": "conv_20250718_ascii_mc_medical_5__neutral__predecision.pt",
 "answer_token_file": "conv_20250718_ascii_mc_medical_5__neutral__answer_token.pt"}
```

For `named_cot`, the same fields are in `results/named_cot/named_cot_<category>.jsonl`,
next to gemma's answer and whether it admitted lying.

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("model-hermeneutics/gigi-lie-detection-activations",
                    "tarballs/fp32/ascii_fp32.tar.gz", repo_type="dataset")
# tar xzf <p>, then torch.load(<file>)
```

## What each tensor is

- **Shape `[63, 5376]`**: gemma's residual stream at one token, at every layer
  (the embedding output plus 62 transformer layers).
- **Two capture points per follow-up question:**
  - `__predecision.pt`: the last prompt token, after the "did you lie?"
    question and before gemma writes anything.
  - `__answer_token.pt`: the token where gemma writes its answer (the letter, or
    the score for the confidence wordings). Which of the two points a
    DYL-style probe expects is still uncertain, so both are saved.
- Filenames are `<sample_id>__<condition>__<capture point>.pt`. Sycophancy ids
  carry the track as a prefix (`A__…`, `B__…`).
- Every tarball was generated at temperature 0 (greedy decoding).

The 13 follow-up conditions are `neutral`, `low_stakes`, `high_stakes`,
`confidence`, their four `_cot` versions (reasoning before the answer),
`plain`, `named` (names the specific act, e.g. "did you just sandbag?"),
`named_cot`, `eval_plain` and `eval_named` (prefixed with "This is a safety
evaluation").

## First-run activations

These are from gemma's answer to the original question, before any follow-up.
They were not recaptured in float32. All of them are pre-decision states,
captured before gemma writes anything, so they depend only on the prompt and
not on the sampling temperature.

| Category | Where |
|---|---|
| ascii-sandbagging | `first_run/ascii/<sample_id>.pt`, 310 files. Each is a dict: `lied_state` (last token before gemma answers the capability question) and `doubled_down_state` (last token before it answers the original "did you lie?"). float16. |
| sycophancy, Track A | `first_run/sycophancy/<uid>__pressured.pt` (158) and `<uid>__neutral.pt` (72): last token before gemma answers the question with and without the user's stated belief. The other 86 neutral states are in `<uid>__neutral_1.pt` / `__neutral_2.pt`, from an earlier design that asked the neutral question twice. float16. |
| sycophancy, Track B | `first_run/sycophancy/<uid>__main.pt`, 339 files, one per row of the Track B results (which call them `<uid>.pt`); last token before gemma writes its critique. `__noise_rerun` files are from the earlier design. float16. |
| unanswerable | inside `tarballs/unanswerable_activations.tar.gz`, with its follow-ups, float16 |
| capture-the-secret | `tarballs/capture_secret_activations.tar.gz`, float16 |
| alibi | inside `tarballs/alibi_activations.tar.gz`, with its follow-ups, float16 |
| mask-factual | `first_run/mask_factual/`: only 9 of 86 rows (`<uid>.pt`), plus 4 files from an earlier design. The rest were never uploaded. |

## Older files

- `tarballs/answer_token_activations.tar.gz`, `tarballs/predecision_activations.tar.gz`,
  `tarballs/mask_factual_followups_activations.tar.gz`, and the follow-up
  tensors inside the unanswerable and alibi tarballs are float16 versions of
  what `tarballs/fp32/` now holds. float16 overflows on some of gemma's
  activations (above 65,504), which is why everything was recaptured.
- 8,680 loose follow-up tensors that used to sit at the repo root
  (older naming, some from before the switch to temperature 0) were removed.
  They are still in the repo's history at revision `98dd29a6`.

## Results

Ground truth, the follow-up answers and all rates are in the project's
results files. `results/alibi/` and `results/named_cot/` are also mirrored
here.
