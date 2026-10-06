# Open items

Kept in git so they survive a lost session. Delete a line when it's done.

## Now (6 Oct 2026, ~09:20 UTC): what's running and what's next

Running on rented boxes, laptop-independent; each uploads to HF and stops itself:
- **M2** (Qwen-as-monitor + Gemini/GPT-5 mini, the paper's EXACT 3rd-person templates,
  `qwen/run_paper_monitors.sh`) -> `models/qwen3-32b/results/third_person/*_paper.jsonl`.
- **J6b box**: mask-factual for Qwen done (`models/qwen3-32b/boxes/J6b/`); it waits for
  M2's final log, then regenerates Qwen-as-monitor replies cut off before `<score>`
  (2,048 tokens; `qwen/run_retry_monitor.sh`). If it's still running hours after M2,
  stop it by hand.
- **gemma retry** (Vast 54459935): regenerates gemma-as-monitor replies cut off before
  `<score>` (1,595 of 5,618; `third_person/run_retry_gemma.sh`).
- Already done: Gemini + GPT-5 mini with the paper templates on gemma's transcripts
  (`results/third_person/monitor_*_paper.jsonl`). First finding: the paper's plain
  3rd-person format hides the system prompt, and the monitors drop to chance on alibi
  (0.52-0.53) and mask-factual (0.47-0.74); with it they are at 0.89-0.94.

Next, no cost:
1. `qwen/11_merge_boxes.py` (now includes J6b; excludes J6's partial mask-factual), then
   `compare_models.py` + `qwen/13_charts.py`: Qwen 11 of 11 categories in RESULTS.md.
2. Make the paper-template monitors the main monitor result (read the `_paper` files in
   `third_person/03_analyse_monitor.py`, `04_report.py`, `compare_models.py`; the
   analysis keeps the LATEST record per key, so retries replace cut-off rows). Keep our
   templates as a comparison; write up the system-prompt finding.
3. Update `third_person/README.md`, `qwen/README.md`, `HF_DATASET_CARD.md` (Qwen files,
   paper-template monitors), and the claim that `did_lie` is unusable: the dataset was
   rebuilt on 5 Oct 2026 (labels fixed, transcripts end with the model's answer,
   training folds `megafolds_fold_1/2`). Qwen code pins the old revision `e8fc10f`.

Next phase, the training replication (John's ask; agree the details with the user first):
- Round 0: untrained gemma-3-27b P("A") on the paper's two folds (~0.60 in the paper).
- Round 1: LoRA fine-tune with the paper's settings (rank 16, alpha 32, dropout 0.05,
  final-token CE, 5 epochs, early stopping), fold 1 -> fold 2 and back
  (paper: ~0.95 in-distribution, 0.70-0.75 unseen lie types).
- Round 2: linear probes on activations, same split (white box).
- Round 3: all detectors on OUR independently labelled data.
- Open choices: ~5,000 (paper) vs ~9,000 training rows per fold; learning rate (research
  options); gemma-27b only. Free groundwork: label audit (paper's fixed labels vs ours),
  identify the ~3,300 fold rows not in any task-group file, draft the training code.
- Then `NEXT_STEPS.md` for John. Later: the missing lie types (self-sycophancy,
  CoT-unfaithfulness, off-policy, tool/chess sandbagging). Silico idea sent to John.

Docent: collection `9d681e5d-6093-402c-b1d6-afcb86d5eb3b` (ascii + sycophancy B, gemma +
Qwen, 4,088 exchanges); uploader `docent/01_upload.py`; Claude Code plugin installed
(use `/docent` in a new chat).

## Done

- All 7 categories run end to end: ascii-sandbagging, sycophancy A/B,
  mask-factual, unanswerable, capture-the-secret, alibi (all 1,281 rows since
  1 Oct 2026: 784 lies, 92.7% of guilty suspects).
- 13 follow-up questions per category, including `named_cot` (1,732 rows),
  all in `RESULTS.md` tables, the balanced-accuracy grid and the charts.
- float32 activations for all 7 categories plus every `named_cot` set on
  HF (`model-hermeneutics/gigi-lie-detection-activations`, now public).
  16/16 files verified; the box was stopped once they were confirmed.
- mask-factual first-run activations: all 87 rows on HF (78 recaptured).
- 1 Oct 2026: the project keeps four follow-up questions (`neutral`,
  `neutral_cot`, `named`, `named_cot`). The other nine are archived in
  `archive/2026-10-01_all_13_wordings/`; RESULTS.md, every category doc,
  the charts and the false-confession files now cover the four.
- HF repo reorganised the same way: `activations/<category>.tar.gz` (four
  questions, float32), `first_run/` for every category, retired files under
  `archive/` (28 files moved, copies checksum-verified), new README.
- P("A") on all 19,049 A/B answers -> AUROC per category and wording
  (`answer_probability/README.md`).
- A/B swap test on 626 rows x 11 wordings (`ab_swap/README.md`).
- HF repo tidied: README (from `HF_DATASET_CARD.md`) added, 8,680 stale
  loose root tensors removed, every first-run file moved to `first_run/`.
- 2 Oct 2026: every first-run activation recaptured in float32 (2,173 files,
  cosine >= 0.9999 against the float16 originals, which are in
  `archive/first_run_fp16/` on HF). Every activation on HF is now float32.
- 2-3 Oct 2026: MASK known facts / statistics / continuations / disinformation
  (299 rows, `mask-belief/`): ground truth from gemma's own belief (softened MASK
  belief questions), statistics cutoff 10%. In RESULTS.md, charts and AUROC.

## Still open

- [ ] Next category: to be chosen (every gemma config with usable rows is now covered;
      remaining: n-truths-and-a-lie has only 2 rows).

## Optional (each needs a GPU box and a go-ahead on cost)


## Ruled out, with reasons

- `self-sycophancy` — prompts are visibly corrupted (model output leaked
  into the probe text, duplicate options). Matches the paper's own 46%
  label-instability warning.
- `offpolicy` — traces are from other models, so gemma isn't the actor. It
  measures detecting someone else's lies, which is a different experiment.
