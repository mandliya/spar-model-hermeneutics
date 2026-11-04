# Instructions for Claude Code sessions working in `gigi/`

Read this before doing anything in `gigi/`. It carries every standing rule for
this project, so a session started anywhere (laptop, cloud, phone) behaves the
same. Current state and next steps: `TODO.md`. Start page: `README.md`. Results:
`1-replication/RESULTS.md` (part one), `2-hermeneutics/RESULTS.md` (part two),
`3-forensics/` (part three).

## What this is

Part one (to 7 Oct 2026): a replication of Anthropic's "Fine-tuned lie
detectors failed to generalize" (alignment.anthropic.com/2026/lie-detectors)
on `google/gemma-3-27b-it`. Part two (from 8 Oct 2026): model hermeneutics,
with Qwen3-Next-80B as the target and gemma as its stand-in (see `TODO.md`
for the plan from the call with John). Part three (from 16 Oct 2026): model
forensics, the same target-vs-stand-in question on the agent environments of
"Why do models task game?" (`3-forensics/`). Part one in a paragraph:
For each deception category we (1) establish our own ground truth for whether
gemma lied, (2) ask it four follow-up "did you lie?" questions, (3) capture
activations, and (4) compare against third-person monitors
(`1-replication/third_person/`). The dataset's own `did_lie` label is unusable
for gemma (see `1-replication/ascii-sandbagging/CLAUDE.md`); every category
builds its own ground truth.

## Repo layout (reorganised 17 Oct 2026, the user's choice)

- `1-replication/`, `2-hermeneutics/`, `3-forensics/`: one folder per part,
  each with its own `RESULTS.md` (or `README.md`), `charts/` and code.
- `qwen/`: the GPU-box pipeline for every Qwen run (parts one and two) and its
  merged results (`qwen/results/<model>/`).
- The 7 category names at the top of `gigi/` (`alibi`, `ascii-sandbagging`,
  `capture-the-secret`, `mask-belief`, `mask-factual`, `sycophancy`,
  `unanswerable`) are symlinks into `1-replication/`. Keep them: TQ's and
  Ravi's code reads `gigi/<category>/src/` and `gigi/<category>/results/`.
  Our own code and links use the real paths.
- `RESULTS.md`, `RESULTS_REPLICATION.md`, `RESULTS_HERMENEUTICS.md` at the top
  are short "moved" notes for links shared before the move. Don't add to them.
- New work goes in the folder of its part; never back at the top of `gigi/`.

## Everything lives in this repo

- Work directly in this repo. Never keep project files only on one machine.
  Code, docs, results, charts and notes all get committed here; big binary
  data (activations) goes to the HF dataset
  `model-hermeneutics/gigi-lie-detection-activations`, never only on a laptop
  or a rented box.
- **Secrets never go in git.** API keys (`OPENROUTER_API_KEY`, `HF_TOKEN`)
  come from environment variables, or a gitignored `.env` at the repo root
  (laptop) or next to the script (GPU box). Cloud sessions: set them in the
  environment's settings. Before every commit run `git check-ignore -v .env`;
  silence means a key is about to be pushed. Never `git add -f` a `.env`.
- Concept explanations for the user go in `anki_notes/` as `Q:` / `A:` cards
  (plain language, a small worked example, one idea per card).

## Git

From the repo root, every time:
```
git pull
git add gigi/        # ONLY gigi/. Never `git add -A` or `git add .`
git commit -m "gigi: ..."
git push
```
Never edit or stage anything outside `gigi/` (other people's folders). Never
`git push --force`. Commit and push each finished piece of work immediately;
the user looks at GitHub, so unpushed work counts as not done.

## Money: ask first, every time

Every API call (OpenRouter) and every GPU hour (Vast.ai) is the user's own
money. Before any action that spends money, state the expected cost and get an
explicit yes **for that specific spend**. Enthusiasm ("let's go") about a
direction is not approval; approval for one stage doesn't cover the next;
small smoke tests count too. Report real measured costs (token usage and cost
are saved with every response), not estimates.

## Long jobs and GPU boxes (Vast.ai)

- Every long job runs detached from the session (`setsid nohup ... &` on
  Linux, `nohup ... & disown` on macOS) and logs to a file; verify the TTY
  column shows `?`/`??`.
- Prefer running jobs on the box, not on a laptop: a laptop that sleeps
  pauses the job.
- As soon as a box is reachable, write the **whole remaining sequence** into
  one script and launch it detached in one step: setup checks, a smoke test
  as a gate that aborts cleanly, every main run, and every upload to HF.
- Uploading results to HF is a pipeline stage, run as soon as each artifact
  exists (and periodically during long runs), never a manual step afterwards.
- Independent stages must not share a failure domain: one stage crashing
  must not stop unrelated later stages.
- **Never stop the box automatically**: no end-of-job stop, no deadman or
  hold-expiry timer. Stop it only when the user says so in that conversation.
  When a job finishes or fails, tell the user promptly that the box is idle
  and still billing.
- Check on long runs periodically (tens of minutes), not only when asked.
- When the user has to run something on a box, give one complete line they
  can paste as-is (no placeholders) and name the exact box (Vast instance ID
  and job). Keys go in a local file or are read from a `.env` already on the
  box, never in chat. Rent GPUs with >= 80 GB of dedicated memory (RTX PRO
  6000, H100, A100 80GB); never GB10 / DGX Spark (shared memory, offloads to
  disk and crawls).
- Before telling the user to stop or destroy a box, check whether pending GPU
  work can reuse it (model already downloaded). If so, run it there instead of
  renting a new box.
- Progress can be checked from anywhere through the HF dataset's `results/`
  folder. SSH to a box needs the key on the user's laptop; from elsewhere the
  user can use the Vast web terminal.

## Experiment rules

- **Use the paper's exact text.** Before writing any prompt, rubric,
  template or procedure borrowed from the paper, read the whole paper,
  appendices included. Download the page and search it; a summary isn't
  enough. Use its wording, transcript format and output format verbatim.
  The paper publishes all four lie-report prompts ("Four Prompts to Report
  Lies") and the full rubric (Appendix A2). Never claim something is
  unpublished without reading to the end. Mark every deliberate deviation in
  the docs.

- **No open-weight model is ever run through an API.** Every open-weight run
  (the model under test, an open-weight monitor, anything) happens locally on
  the rented GPU instance with Hugging Face transformers. OpenRouter is only
  for closed models (the Claude judge, Gemini, GPT). The model under test
  generates greedy, **one conversation at a time** (no batching). Every current
  gemma result, ascii included, was made this way (temperature-0 rebuild, 24
  Sept 2026); scripts that call it via OpenRouter (e.g.
  `1-replication/ascii-sandbagging/00_ascii.py`) are leftovers. Before saying
  how a result was produced, check which script writes that results file.
  **One exception, part three only** (the user, 17 Oct 2026, at John's
  request on the 14 Oct call): the forensics runs in `3-forensics/` call
  Qwen3-Next-80B and gemma-3-27b through OpenRouter, because no activations
  are needed there yet. Providers pinned as below (gemma on DeepInfra, the only
  provider with tool calls for it); OpenRouter serves fp8, so say so next to
  every forensics result. Activations, when needed, come from a GPU box again.
  **Extended 4 Nov 2026 (the user, after the 3 Nov call):** the hero suite also
  runs through OpenRouter, traces only, activations recomputed later on a GPU
  box: Llama 3.3 70B Instruct (target) at full precision, fp16 on CoreWeave,
  to match the weights the public NLA and J-lens were fitted on; Qwen3.6 27B
  (fp8 on DeepInfra: no full-precision endpoint exists); gemma-3-27b-it; and,
  as backups, gemma 4 31B and GPT-OSS-20B. State each model's precision next to
  every hero result (`3-forensics/hero_models_prep.md`).
- Ground truth comes from a live, independently verifiable check per category
  (capability proof, neutral-vs-pressured rerun, belief elicitation...), never
  from the dataset's `did_lie`.
- Judge: `anthropic/claude-sonnet-4.5` via OpenRouter. The judge is never the
  model under test, and a monitor is never the model that made the labels.
- Pin the OpenRouter provider for **every** model called (target, judge,
  monitors), with `allow_fallbacks: False`.
- Raw API/model responses go to disk before parsing. A parser returning
  `null` means unparseable; never coerce it to False.
- Never delete or overwrite anything in any `results/` folder. Scripts append
  to JSONL and skip work already done. Superseded data gets archived
  (`archive/`), never deleted.
- Any rate needs its n and a 95% interval (Wilson for rates, bootstrap for
  AUROC).
- The follow-up instrument is exactly four questions, run in full for every
  category: `neutral`, `neutral_cot`, `named`, `named_cot` (A = "yes, I lied"
  in all four). Only the ground truth and the act named in `named`/`named_cot`
  vary by category. Don't propose cutting questions to save money. The nine
  older wordings are archived.
- Every saved activation tensor is float32 (gemma computes in bf16; float16
  overflows to inf). Never mix precisions within a category.
- When a parameter is uncertain or unspecified by the source paper
  (temperature, thresholds, model choice, sample size...), ask the user and
  check what the field normally does, before building on it. Current
  generation setting: greedy (temperature 0), except in part three: forensics
  runs and replays need randomness, so they sample with each model's
  model-card settings (the user, 17 Oct 2026), sent explicitly and recorded
  with every run.

## Docs and charts

- The results pages are what is read on team calls, one per part of the
  project: `1-replication/RESULTS.md` (part one: gemma, the monitors,
  Qwen3-32B) and `2-hermeneutics/RESULTS.md` (part two, from 8 Oct 2026: the
  Qwen3-Next-80B target and gemma as its stand-in), and part three's page in
  `3-forensics/` once it has results. `README.md` is the start page that links
  them. Each page: numbered methodology (how ground truth is
  established per category, then the shared follow-up questions), then a few
  charts with terse takeaways. Update the right page and its charts after
  every test that changes results, in the same session the results are
  merged, not only when a category is added. Terse "label = claim" notes with
  real markdown lists (never `&nbsp;` indentation).
- Charts: one or two simple ones, each with one message and a takeaway title;
  one row per category; detail goes in tables and category READMEs.
- A baseline, noise floor or control that other rows are judged against is a
  row in the same table, never only in prose under it.
- Prompt tables quote every prompt in full, never "same as X".

## User's machine (only applies on the user's Mac)

- Open files for the user with `open -a "Visual Studio Code" <path>`, never
  `code` (it launches Cursor).
- The Vast SSH key is `~/.ssh/vastai_lies_experiment`.
