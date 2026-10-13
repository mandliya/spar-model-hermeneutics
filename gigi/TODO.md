# Open items

Kept in git so they survive a lost session. Delete a line when it's done.

## Now (7 Oct 2026, ~11:00 UTC): what's done and what's next

Done (all in git; box outputs on HF):
- Every monitor result with the paper's exact templates, both models. Unscored after
  reruns: gemma-as-monitor 2 of 5,618, Qwen-as-monitor 12 of 5,602 (repetition loops,
  no score given), Gemini and GPT-5 mini none. Gemini's last 3 cut-off replies (Qwen
  ascii) rerun at 4,000 tokens on 7 Oct: $0.0069.
- Qwen: all 11 categories. `qwen/11_merge_boxes.py` now replaces local files that still
  held J6's partial mask-factual rows (that had kept J6b's P("A") out of the analysis).
- RESULTS.md rewritten: the paper's rubric template is the main monitor result, a "what
  the monitor is shown" chart (`charts/monitor_sees_system_prompt.png`), our templates
  as a comparison column, Qwen's 11 categories in every gemma-vs-Qwen table. Docs
  updated: `third_person/README.md`, `qwen/README.md`, `HF_DATASET_CARD.md`.

Gaps (each would need GPU time; not scheduled):
- Qwen3-32B has no mask-factual swap test (run_job.sh left every `mask_*` category out of
  the swap, which also matched mask_factual; fixed 12 Oct). The 80B's was run on 12 Oct
  (run_swap_mask_factual.sh). The 32B's: same script with MODEL_TAG=qwen3-32b, one 96 GB GPU.

## Direction from the 8 Oct 2026 call with John: hermeneutics (training runs shelved)

John's post: https://www.lesswrong.com/posts/CZLhfNGaDqtYxvGJD (author = target, reader = substitute).
- SHELVED: the training-run replication and the paper's two extra datasets ("we can get
  back to it later if it seems promising").
- TQ: the probe gap (target white-box probe vs substitute white-box probe).
- Gigi: the self-report ("resample") gap. Target = Qwen 80B (Qwen3-Next-80B-A3B-Instruct,
  163 GB bf16 -> needs a 2 x 96 GB GPU box), substitute = gemma-3-27b.
  1. Qwen-80B answers "did you lie?" / "did you sandbag?" on its own conversations.
  2. Gemma gets the same conversations with Qwen's answer in its own reply slot, and the
     same questions. Gap = how differently gemma answers ("benchmark diff of 80B target and
     gemma 27B substitute", self-report).
  3. Later: distil Qwen-80B into gemma; does the gap shrink?
  Wider idea: resampling, e.g. edit a trajectory and see whether the substitute continues it
  the way the target would (the HF-incident example).
- Not in either dataset version: Noddybear/lies has no Qwen, so Qwen-80B's conversations
  must be generated. Coordinate with TQ: one Qwen-80B run could serve both gaps (same
  conversations; capture its activations for TQ's target probe).
- Measure answers in both letter orders (the swap-test lesson): Qwen3-32B says "no" to
  everything on several categories and gemma prefers "B" on named ascii, so a letter-only
  gap would partly compare letter preferences.

### Qwen3-Next-80B target run: status (9 Oct 2026)

- Smoke test PASSED for all 11 categories on a 1x B200 (8 Oct, 07:16-08:04 UTC, ~$7.50;
  log: models/qwen3-next-80b/boxes/SMOKE/ on HF; excluded from the merge). Once loaded,
  the 80B is about as fast per row as Qwen3-32B was (0.87x); ~46 s model load per stage.
- Quote (full run, all rows): ~25 GPU-hours -> ~$56-84 on 2x RTX PRO 6000 boxes
  ($2.28-2.94/h), plus ~$15-25 judge. APPROVED by the user: full run, 8 boxes,
  STOP_WHEN_DONE=1. Expected ~4-6 h wall time.
- Boxes: J1 | J2 | J3 | J4 | J5 | J6 then J7 | J8 | J9 then J10. Two-job boxes run each
  job in its own folder (/workspace/<JOB>/qwen) so the swap-test subset isn't shared;
  only the second job may stop the box.
- LAUNCHED 9 Oct 03:36-03:44 UTC, all 8 passed the box check (2x RTX PRO 6000):
  54966450 J1 (replaced 54959044, garbage output on a desktop Z890 board, ~04:50 UTC) | 54959047 J2 | 54959051 J3 | 54959055 J4 | 54959070 J5 |
  54959093 J6->J7 | 54959107 J8 | 54962686 J9->J10 (replaced 54959418 ~04:15 UTC). Each stops itself when done.
  J7 and J10 did NOT run on those two-job boxes: the second job's 250 GB free-disk check
  failed with the model already on disk, and STOP_WHEN_DONE stopped the box (fixed in
  run_job.sh, d934609). Re-run on single-job boxes from 08:56 UTC: 54993237 J7 | 54993244 J10 (garbage output, replaced) -> 54998878 J10 (09:51 UTC, copy check OK).
- Launch lines (with keys, NOT in git): ~/gigi_launch/LAUNCH_QWEN80B_FULL.txt.
  Code bundle: ggrazianoknapp/gigi-code qwen80b.tar.gz (private HF).
- DONE 9 Oct 12:33 UTC: all 10 jobs finished; merged into qwen/results/qwen3-next-80b/
  (87 files, no merge warnings). Text check: 2 of 12,124 answers flagged, both one genuine
  mid-sentence switch to Chinese, no corruption. Cost: GPU ~$113 from box hours x prices
  (+ Vast bandwidth/storage; exact = Vast credit drop from $316.41), judge $14.32 measured
  (OpenRouter usage_daily, 9 Oct UTC). Two boxes wrote garbage (corrupted GPU-to-GPU copies);
  run_job.sh now checks copies at startup.
- Substitute step, 12 Oct: `qwen/16_substitute_followups.py` + `qwen/run_substitute.sh`
  written (see qwen/README.md). gemma reads Qwen-80B's conversations with Qwen's answer in
  its own reply slot and answers the same four questions; `--swap` = the other letter
  order on the 80B's swap subset. Render check (tokenizer only, laptop) PASSED for all
  2,811 conversations / 11,244 answers and the 2,428 swapped ones. NOT RUN on a GPU yet.
- LAUNCH, 12 Oct (user: "i need all of this + the 80B stuff that is missing", renting 5 boxes,
  same-day results). Launch lines: ~/gigi_launch/LAUNCH_SUBSTITUTE.txt (keys, NOT in git);
  bundle `substitute.tar.gz` on ggrazianoknapp/gigi-code. No STOP_WHEN_DONE: the user stops
  the boxes. Activations ON (for TQ's substitute probe).
  S1 alibi 0/2 | S2 alibi 1/2 | S3 capture + ascii | S4 the other 8 categories (1 x >= 80 GB GPU
  each, `run_substitute.sh`) | J6swap = Qwen-80B's mask-factual swap (2 x 96 GB,
  `run_swap_mask_factual.sh`). Instance IDs: in each job.log on HF
  (models/qwen3-next-80b/substitute/gemma-3-27b/boxes/<JOB>/job.log, .../boxes/J6swap/job.log).
  STARTED 12 Oct 08:24-08:28 UTC, all four gemma boxes passed the box check (1x RTX PRO 6000):
  55521262 S1 ($1.431/h) | 55521287 S2 ($1.490/h) | 55521295 S3 ($1.682/h) | 55521444 S4 ($1.732/h) |
  55522595 J6swap (2x RTX PRO 6000 Max-Q, Threadripper, $2.754/h; its log reaches HF only after
  the 163 GB download). $9.09/h while all five run; estimate ~$27 if each is stopped when done.
- J6swap DONE 08:4x UTC (box 55522595): Qwen-80B's mask-factual swap test, 248 of 248 answers (62
  conversations x 4 questions), 0 errors, 0 unparseable; merged into
  qwen/results/qwen3-next-80b/ab_swap_mask_factual.jsonl. Box idle from then on (user told).
- 08:36 UTC: all four gemma boxes passed their smoke tests and are in the full run.
- 08:27 UTC: S2's smoke test passed on the GPU (16 of 16 answers parseable, letter = gemma's top-1
  token every time); it is in its full run. The other three were still downloading gemma.
- DONE 12 Oct 12:00 UTC: all four gemma boxes finished (S2 11:28, S4 11:35, S1 11:53, S3 12:00 UTC),
  11,244 answers + 2,428 swapped, 0 errors; activations tarballs on HF. Merged
  (17_merge_substitute.py), gap computed (substitute_gap.py), chart + RESULTS.md section
  written (18_substitute_charts.py). User told at 12:17 UTC that all five boxes are idle; the
  finish was noticed late because the last log lines were never pushed (fixed: final_push).
  Cost: four gemma boxes ~$6.34/h from ~08:15 UTC, 80B box $2.754/h from ~08:15 UTC, each until
  the user destroys it (exact = Vast billing; the Vast API key on the laptop is invalid).
- What John asked for (8 Oct call transcript, pasted by the user into the 8 Oct session): step 1
  "do the substitution ... Take Gemma 27B. Put it back into where Qwen is right before you ask it
  'Did you lie? Did you sandbag?' ... see if Gemma answers the same way as Qwen does" (18:06-19:09);
  "This first step is to measure the gap and the second step is to use distillation to overcome the
  gap or at least understand the gap" (19:09); "a benchmark diff of 80B target and Gemma 27B
  substitute", self-report (19:30). Step 1 is DONE (12 Oct). Step 2 (distillation) not started.
  Two readings of "the gap", both now in RESULTS.md: same answers? (agree on 62.5% of yes/no; gemma
  says yes 62.7% vs Qwen 32.8%) and benchmark diff (AUROC 0.68 vs 0.62).
- Result: gap small in most cells (27 of 44 no clear difference; Qwen better 12; gemma better 5);
  big gaps only on ascii and sycophancy A. See RESULTS.md "Hermeneutics".
- 13 Oct: the 80B's OWN results published in RESULTS.md ("Third model: Qwen3-Next-80B", headline
  11): lying rates, self-report AUROC per question, swap test, four counts, three-model chart.
  They had been merged on 9 Oct but never written up (user: "you did not publish the results for
  the simple run of qwen 80B"). compare_models.py now covers all three models.
- 13 Oct: results split in two at the user's request. `RESULTS_REPLICATION.md` = part one (gemma,
  monitors, Qwen3-32B; everything up to 7 Oct). `RESULTS_HERMENEUTICS.md` = part two (the 80B
  target, gemma as substitute; "same answers?" first, then the AUROC gap). `RESULTS.md` is now a
  10-line index. Older lines in this file that say "RESULTS.md" mean the page before the split.
  The table-writing scripts point at the right page (make_condition_tables, 13_charts,
  third_person/04_report -> replication; 18_substitute_charts, 19_qwen80b_charts -> hermeneutics).
- Next ideas (none scheduled, each needs a go-ahead): outside monitors on the 80B's transcripts
  (transcripts are built: qwen/results/qwen3-next-80b/transcripts); a "prefill" variant where gemma
  reads Qwen's own follow-up reasoning and only the final letter is scored; TQ's probe gap from
  the saved activations; distil Qwen into gemma (John's step 2); HF dataset card: add the 80B and
  the substitute folder.

Still open from before (lower priority): Qwen's mask-factual swap test; the lie types not yet
covered (self-sycophancy, CoT-unfaithfulness, off-policy, tool/chess sandbagging). Silico
email to send.

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
