# Moving the hero9 swap to a Claude cloud session (11 Nov 2026)

The laptop keeps falling asleep, and every time it does the swap launcher, the fetches and the grading pause. These files let a Claude Code cloud session at claude.ai/code take over the rest of the swap, and the laptop hands over by itself once the cloud says it is ready.

## What you do, once

1. **Make a cloud environment.** At claude.ai/code, create an environment for the repo `mandliya/spar-model-hermeneutics` with network access, and add these four secrets in its settings. Never paste them in a chat.
   - `OPENROUTER_API_KEY`: the same one as in the repo-root `.env` on the laptop.
   - `HF_TOKEN`: the same one as in that `.env`.
   - `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`: the `token_id` and `token_secret` under `[spar]` in `~/.modal.toml` on the laptop.
2. **Start a session in it and paste this:**

   > Read gigi/CLAUDE.md, then gigi/3-forensics/handoff/README.md. Run `nohup bash gigi/3-forensics/scripts/hero9_swap_cloud.sh > /tmp/hero9_cloud.out 2>&1 &` from the repo root, and follow gigi/3-forensics/results/logs/hero9_swap/cloud.txt. When the fold-in has finished (foldin.txt says DONE), audit scored versus lost per cell, write section 4c of gigi/3-forensics/HERO_RESULTS.md from results/hero9_swap_summary.md and the charts, upload the swap transcripts to Hugging Face with `scripts/hf_upload_transcripts.py --stages hero9_swap`, and commit with the key-shape gate from gigi/CLAUDE.md. Don't spend beyond the approved swap and its watchdog cap.

The laptop needs to be awake once more for the handshake, for a minute or two.

## What happens

- The cloud session installs the harness (upstream `agent-interp-envs` at `hero9_swap/BASE_COMMIT` plus our changes in `hero9_swap/harness_overlay.tar.gz`), checks it can reach Modal, and pushes `handoff/CLOUD_READY`.
- The laptop (`scripts/hero9_swap_laptop_handover.sh`, already waiting) sees it, stops its launcher, and pushes the final handoff: what is launched, what is left, and the checkpoints the rest start from (`scripts/hero9_swap_handoff.py`). Apps that are already running on Modal keep running, and nothing is launched twice.
- The cloud session launches the rest in paced batches, under the same spend cap ($3,078, three times the approved $1,026) and with its own watchdog. Then it fetches every fleet from the Modal volumes, the laptop's included, and grades, scores and charts them.
- If the laptop finishes launching everything before the cloud is ready, the handover does nothing and the laptop's own fold-in carries on.
