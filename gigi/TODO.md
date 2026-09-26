# Open items

Kept in git so they survive a lost session. Delete a line when it's done.

## Blocking / at risk

- [ ] **capture-the-secret activations are only on the Vast box.** ~508
      `*__capture__predecision.pt` files under
      `/workspace/lies-experiment/results/activations`. Stop/start preserves
      them; **destroying the instance loses them.** Upload with:
      `bash upload_activations.sh capture_secret 'conv_*__capture__*'`
      (adjust the glob to match the actual filenames).

## HF repo

DONE — the repo now lives at **`model-hermeneutics/gigi-lie-detection-activations`**
(renamed and transferred to the org). The old
`ggrazianoknapp/gigi-ascii-sandbagging-activations` still resolves as a
redirect, which is why uploads addressed to it kept working; all docs now
name the canonical id.

- [x] Rename + transfer to the org
- [x] Point every doc at the canonical id
- [ ] Upload capture-the-secret's activations once its run finishes

## capture-the-secret, in flight

- [ ] Run `02_analyse_capture_secret.py --gate` when the primary run ends.
      It refuses the 12 follow-up questions if either arm is under 20 rows,
      so that decision doesn't cost 4-6 GPU hours to discover.
- [ ] If the gate passes, run the 12 follow-up questions, then add the
      category to `06_charts_cross_category.py` (a sixth series) and to
      `RESULTS.md`.

## Known gaps, not blocking

- [ ] mask-factual's 12 follow-up questions have no activation capture
      (see `mask-factual/RESULTS.md` "Next steps").
- [ ] ~10,000 stale loose `.pt` files sit at the HF repo root. They predate
      the temperature-0 rebuild. Kept, not deleted, because this project
      doesn't delete paid-for artifacts — but they should not be mixed with
      the tarball contents.
- [ ] `mask_known_facts` and `mask_statistics` subtypes: scoped, not started.
- [ ] `games/alibi` (1,281 rows) is the obvious next category after
      capture-the-secret — its system prompt states both
      `YOUR ALIBI STORY (what you're claiming)` and
      `WHAT ACTUALLY HAPPENED`, so ground truth is already on the page.

## Ruled out, with reasons

- `self-sycophancy` — prompts are visibly corrupted (model output leaked
  into the probe text, duplicate options). Matches the paper's own 46%
  label-instability warning.
- `offpolicy` — traces are from other models, so gemma isn't the actor. It
  measures detecting someone else's lies, which is a different experiment.
