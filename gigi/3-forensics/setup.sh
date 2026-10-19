#!/usr/bin/env bash
# gigi/3-forensics/setup.sh: the environments repo at the pinned commit, our patch, its packages.
# Idempotent; run from anywhere. Needs git, uv and Docker.
#   patches/forensics.patch:
#     - top_k / min_p reach OpenRouter (the model-card sampling settings)
#     - reasoning_effort "none" sends no reasoning field (our two models have no reasoning mode)
#     - secret_number and precommit_hook pass every agent.* setting on a fresh run, as on a resume
#       (they used to drop temperature/top_p on fresh runs only, so a run and its resume differed)
#     - precommit_hook scores each run in its container when it ends (score.py needs the final workspace)
#     - gemma-3 writes tool calls as text (```tool_code [execute_command(...)]```); turn the first such
#       call into a structured tool call when the reply has none (its text stays unchanged)
#     - async_conversion's pristine snapshot moves the workspace repo into /opt with mv, so it builds on
#       Docker Desktop too (overlayfs refuses git's rename there); the image is the same
#     - the OpenRouter provider logs every retry ("[provider retry] ..." in rollout.log): a retry resamples
#       the reply, and DeepInfra drops gemma replies whose tool call it cannot parse, so retries must be visible
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=310c03f
[ -d agent-interp-envs/.git ] || git clone -q https://github.com/gkroiz/agent-interp-envs.git agent-interp-envs
if [ "$(git -C agent-interp-envs rev-parse --short=7 HEAD)" != "$COMMIT" ]; then
  git -C agent-interp-envs checkout -q -- . && git -C agent-interp-envs checkout -q "$COMMIT"
fi
for p in patches/*.patch; do   # applies only if not applied yet
  if git -C agent-interp-envs apply --check "../$p" 2>/dev/null; then git -C agent-interp-envs apply "../$p"; echo "applied $p"; fi
done
(cd agent-interp-envs && env -u VIRTUAL_ENV uv sync -q)
echo "ready: agent-interp-envs@$COMMIT with our patch"
