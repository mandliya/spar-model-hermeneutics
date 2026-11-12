# Our changes to agent-interp-envs (saved 12 Nov 2026)

`agent-interp-envs/` is a clone of github.com/gkroiz/agent-interp-envs, ignored by this repo's git. Every change we made
to it for the forensics runs (gemma's text-mode tool calls, retries, end check, Qwen3.6 thinking off, resume fixes)
was uncommitted in that clone, so it existed only on GG's laptop. This folder saves it:

- `base_commit.txt`: the upstream commit the changes apply to.
- `changes.patch`: every change to tracked files (`git diff HEAD --binary`).
- `new_files/`: files we added, at their paths in the clone.

Restore on a fresh machine:
```
git clone https://github.com/gkroiz/agent-interp-envs.git && cd agent-interp-envs
git checkout $(cat ../harness_patch/base_commit.txt)
git apply ../harness_patch/changes.patch && cp -R ../harness_patch/new_files/. .
```
Refresh after any change to the clone: rerun the commands that made this folder (see the commit that added it).
