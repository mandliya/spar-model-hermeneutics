# Vendored: jacobian-lens and natural_language_autoencoders

Two verbatim copies, each the whole upstream repo at one commit with its `.git` removed (downloaded as GitHub's
source tarball of that commit).

## `jacobian-lens/`: the J-lens

- Source: https://github.com/anthropics/jacobian-lens (Anthropic's reference implementation for "Verbalizable
  Representations Form a Global Workspace in Language Models", https://transformer-circuits.pub/2026/workspace/)
- Commit: `581d398613e5602a5af361e1c34d3a92ea82ba8e` (2026-07-02, HEAD of `main` on 2026-11-08)
- License: Apache-2.0 (`jacobian-lens/LICENSE`)
- Used as the `jlens` package, installed editable from this folder (`box_setup.sh`).

## `natural_language_autoencoders/`: the NLA client

- Source: https://github.com/kitft/natural_language_autoencoders (the library accompanying Anthropic's "Natural
  Language Autoencoders Produce Unsupervised Explanations of LLM Activations", https://transformer-circuits.pub/2026/nla/)
- Commit: `0577769b55ad4fdd96d159e983361b97fa4e7331` (2026-08-02, HEAD of `main` on 2026-11-08)
- License: Apache-2.0 (`natural_language_autoencoders/LICENSE`)
- Used through `nla_inference.py` (sidecar config and checks, injection, the AR critic), imported by `../nla.py`.
  It was written against transformers 4.x (`apply_chat_template(tokenize=True)` returning a list), so `nla.py` runs
  in its own venv on transformers 4.57.1.

Copied on 2026-11-08, everything but `.git`. No secrets found. Both keep their own `.gitignore`.

**Do not edit anything under these two folders.** Our code lives next to them in `tianq/experiments/internal_readout_gap/`
and imports from them. To update, re-copy a newer commit and change the hash above. Each repo's own `CLAUDE.md`, if
present, holds instructions for sessions working in that repo, not ours.
