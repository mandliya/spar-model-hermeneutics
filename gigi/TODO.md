# Open items

Kept in git so they survive a lost session. Delete a line when it's done.

## Now (7 Oct 2026, ~07:30 UTC): what's running and what's next

Running, laptop-independent: **Qwen-as-monitor retry** on Vast 54417474
(`qwen/run_retry_monitor.sh`, full 2,801 Qwen transcripts, 2,048 tokens). It redoes
every Qwen self-monitor reply with no `<score>` (514 left at 07:30) and uploads
`models/qwen3-32b/results/third_person/monitor_qwen3-32b_paper.jsonl`, then STOPS
the box (unconditionally, fixed 7 Oct). If it still shows as running in Vast long
after ~09:30 UTC, stop it.

Done (all on HF and in git):
- Paper-template monitors (paper_3p = no system prompt; paper_3p_rubric = A2 rubric):
  gemma transcripts: Gemini, GPT-5 mini, gemma (5,616/5,618 scored; 2 genuinely
  unscorable: one repetition loop, one answer with no score). Qwen transcripts:
  Gemini, GPT-5 mini (all scored; 14 Gemini replies redone at 4,000 tokens).
- Qwen: all 11 categories (J6 ascii, J6b mask-factual); merged by `qwen/11_merge_boxes.py`.
- First finding: hiding the system prompt (paper_3p) drops outside monitors to chance
  on alibi (0.52-0.53) and mask-factual (0.47-0.74); with it (rubric) ~0.89-0.94.

Next, no cost (once the Qwen retry has landed):
1. Pull `monitor_qwen3-32b_paper.jsonl` from HF into `qwen/results/qwen3-32b/`, run
   `qwen/11_merge_boxes.py`, then `compare_models.py` (reads both our and `_paper`
   monitor files, all 4 prompts; latest record per key wins) and `qwen/13_charts.py`.
2. RESULTS.md: paper-template monitors as the main monitor result (gemma chart
   `third_person/04_report.py` and Qwen chart), a "what the monitor sees" chart
   (paper_3p vs paper_3p_rubric), our old templates as a comparison column; Qwen's
   ascii + mask-factual in every gemma-vs-Qwen table.
3. Docs: `third_person/README.md`, `qwen/README.md`, `HF_DATASET_CARD.md`, and the
   outdated "did_lie is unusable" claim (dataset rebuilt 5 Oct 2026).

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
