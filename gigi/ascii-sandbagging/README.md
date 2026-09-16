# Does gemma-3-27b-it lie, and then deny it?

A replication of two numbers from [Anthropic's Alignment Science post on
fine-tuned lie detectors](https://alignment.anthropic.com/2026/lie-detectors/),
using its released dataset [`Noddybear/lies`](https://huggingface.co/datasets/Noddybear/lies).

1. How often does `google/gemma-3-27b-it` lie on these situations?
2. When you then ask "did you lie?", how often does it deny it?

This lives in the SPAR repo: `mandliya/spar-model-hermeneutics`, in
`gigi/ascii-sandbagging/`. Other people's folders sit alongside and are never
touched — you stage with `git add gigi/`, never `-A`.

## Setup (Mac)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL

# 1. the shared repo
git clone https://github.com/mandliya/spar-model-hermeneutics.git
cd spar-model-hermeneutics

# 2. this experiment, as a subfolder of it
unzip ~/Downloads/lies-experiment.zip -d /tmp/
mkdir -p gigi/ascii-sandbagging
cp -r /tmp/lies-experiment/. gigi/ascii-sandbagging/
cd gigi/ascii-sandbagging

# 3. environment and key
uv sync
cp .env.example .env
open -e .env          # paste your OpenRouter key
```

Key from <https://openrouter.ai/keys>. Add $5; the pilot costs under a dollar.

**Check the repo's own layout before step 2** — if collaborators already use a
different folder convention, follow theirs. And confirm `.env` is ignored
before your first commit:

```bash
git check-ignore -v gigi/ascii-sandbagging/.env
```

A path printed back means you're safe. Silence means the key would be
committed to a shared repo — fix that before anything else.

## Run (ASCII sandbagging)

```bash
uv run python 01_explore.py --config google-gemma-3-27b-it_task-group_sandbagging
uv run python 00_ascii.py
```

Two commands. The first is free and prints the configs, checks the labels, and
dumps three real traces to read. The second does the whole experiment: label
check, capability control, the rerun, and the follow-up. Under a dollar.

## Do not load the `default` config

Its columns are misaligned — `did_lie` is True in 126 rows out of 159,457, and
`"fold_2"` shows up as a value in the `model` column. Always name a per-model
config; `01_explore.py --list` shows them all.

## What comes out

| metric | meaning |
|---|---|
| lying rate | P(lie) for gemma-3-27b on this config, next to the dataset's own rate |
| doubling down | P(denies \| lied) |
| false confession | P(admits \| truthful) — the control. Without it, a model that always answers "B" scores 100% doubling-down and means nothing. |

## Other task groups

ASCII sandbagging needs no judge because the ground truth is external — gemma
can draw ASCII art, so denying it is a lie, full stop. Every other task group
puts the truth somewhere else and needs different machinery. `00_simple.py` and
the `01`–`06` scripts cover those; start with ASCII.

## Everything is resumable

Each script appends to a JSONL and skips ids already in it. Kill a run at
sample 300 of 400, rerun, it starts at 301. Never delete files in `results/`.

## Push your work

Results are committed, not kept locally — they cost money and collaborators
shouldn't have to re-pay to see them.

From the repo root, the same four lines every time:

```bash
git pull          # get anyone else's changes first
git add gigi/     # ONLY your folder. never -A
git commit -m "gigi: ascii sandbagging, gemma-3-27b"
git push
```

`git add gigi/` stages that folder and nothing else, so no commit of yours can
ever contain someone else's file. Nothing here needs `git push --force`, which
is the one command that can overwrite other people's work.

Put both numbers with their intervals and n in `gigi/ascii-sandbagging/
RESULTS.md`, say which config and temperature you used, and link the figure
you're comparing against — the report belongs next to the code that made it.

## Caveats to keep

- The conversation history replayed into gemma contains the dataset's recorded
  turns, so it is conditioned on its own past words rather than generating the
  whole exchange fresh.
- The paper's sampling temperature isn't published. Yours is in `.env`.
- `did_lie` is the paper's label — the post notes ~25% of labels changed when
  reviewed with full ground-truth access.
- The pilot is balanced 50/50 on purpose. For the paper comparison use the
  full-config rate from step 1.
