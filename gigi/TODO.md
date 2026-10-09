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
- Qwen's mask-factual swap test was never run (J6b had no swap step).

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
- Launch lines (with keys, NOT in git): ~/gigi_launch/LAUNCH_QWEN80B_FULL.txt.
  Code bundle: ggrazianoknapp/gigi-code qwen80b.tar.gz (private HF).
- After the boxes finish: MODEL_TAG=qwen3-next-80b python qwen/11_merge_boxes.py, then
  the substitute step (gemma reads Qwen-80B's conversations) and the gap analysis.

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
