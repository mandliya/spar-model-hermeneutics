# gigi

Gigi's part of the SPAR project "model hermeneutics": white-box monitors for
frontier-agent forensics. Can an open model that reads a black-box model's
transcripts stand in for it, so its insides can be studied instead? One
folder per part of the project:

| Part | When | Question | Start here |
|---|---|---|---|
| 1. Replication | Sept to 7 Oct 2026 | Anthropic's "Fine-tuned lie detectors failed to generalize" on gemma-3-27b and Qwen3-32B: how often each model lies, and whether asking "did you lie?" catches it | [`1-replication/RESULTS.md`](1-replication/RESULTS.md) |
| 2. Hermeneutics | 8 to 15 Oct 2026 | Qwen3-Next-80B as the target, gemma-3-27b as its stand-in: does the stand-in answer the way the target does? Also the Qwen3-235B run for TQ's probe | [`2-hermeneutics/RESULTS.md`](2-hermeneutics/RESULTS.md) |
| 3. Forensics | from 16 Oct 2026 | The same question on agent tasks where models cheat ("Why do models task game?"): if Qwen cheats at step N, does gemma, swapped in at step N, cheat too? | [`3-forensics/README.md`](3-forensics/README.md) |

## Folders

- [`1-replication/`](1-replication/README.md): part one. One folder per
  deception category, the cross-category steps (letter swap, P("A"), the
  fourth question, outside monitors), the part-one charts and scripts.
- [`2-hermeneutics/`](2-hermeneutics/README.md): part two. Its results page,
  the analysis scripts and the charts.
- [`3-forensics/`](3-forensics/README.md): part three, in progress.
- [`qwen/`](qwen/README.md): the GPU-box pipeline for every Qwen run
  (Qwen3-32B in part one; Qwen3-Next-80B, Qwen3-235B and gemma reading the
  80B's conversations in part two), with the merged results in
  `qwen/results/<model>/`.
- [`anki_notes/`](anki_notes/): concept flashcards.
  [`archive/`](archive/): superseded runs, kept, never deleted.
- [`TODO.md`](TODO.md): where things stand and what's next.
  [`CLAUDE.md`](CLAUDE.md): standing rules for Claude Code sessions.
  [`HF_DATASET_CARD.md`](HF_DATASET_CARD.md): the card of the Hugging Face
  dataset that holds the activations
  (`model-hermeneutics/gigi-lie-detection-activations`).

Also at this level, kept on purpose:

- The 7 category names (`alibi`, `ascii-sandbagging`, `capture-the-secret`,
  `mask-belief`, `mask-factual`, `sycophancy`, `unanswerable`) are links into
  `1-replication/`, so code that reads the old paths (TQ's and Ravi's) keeps
  working.
- `RESULTS.md`, `RESULTS_REPLICATION.md` and `RESULTS_HERMENEUTICS.md` are
  short "moved" notes, so links shared before the move still lead somewhere.
