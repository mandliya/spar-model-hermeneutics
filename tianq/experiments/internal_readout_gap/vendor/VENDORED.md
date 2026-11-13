# Vendored: jacobian-lens, natural_language_autoencoders, Neuronpedia's lens fitting and EasyNLA

Four verbatim copies, each from GitHub's source tarball of one commit. Three are whole upstream repos with their
`.git` removed; `neuronpedia-jlens/` is one folder of a large repo (noted below).

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

## `neuronpedia-jlens/`: the fitting procedure behind the published J-lenses

- Source: https://github.com/hijohnnylin/neuronpedia, the folder `utils/neuronpedia-utils/neuronpedia_utils/jlens`
  only (the rest of that repo is the Neuronpedia web app). Every file of that folder, unedited.
- Commit: `f5d338e1dfc9cc65f05af2c535c72f2f5fabb83f` (2026-11-07, copied on 2026-11-13)
- License: Apache-2.0 (`neuronpedia-jlens/LICENSE`)
- `fit_lens.py` is the script that fitted the lenses in `neuronpedia/jacobian-lens` on HF; each lens's `config.yaml`
  there records the exact command. It imports the `jlens/` copy next to it, which differs from `jacobian-lens/jlens`
  above, so a fit runs with this folder as its working directory. Called by `../lens_fit.py`.

## `EasyNLA/`: the recipe behind the unofficial Qwen3.6-27B NLA

- Source: https://github.com/asherps/EasyNLA (whole repo). The NLA `ceselder/qwen3.6-27b-nla-av` on HF was trained
  with it, and its model card says to run inference with this repo's `scripts/show_nla_generations.py`.
- Commit: `4d728477960c18cdfa36dc04ec738d7f55af9f0b` (HEAD of the default branch on 2026-11-13, last pushed 2026-07-10)
- License: MIT (`EasyNLA/LICENSE`)
- Used for that NLA's injection hook and sidecar config, imported by our NLA code.

The first two were copied on 2026-11-08 and the last two on 2026-11-13, everything but `.git`; each keeps its own
`.gitignore`. No secrets found in any of the four.

**Do not edit anything under these four folders.** Our code lives next to them in `tianq/experiments/internal_readout_gap/`
and imports from them. To update, re-copy a newer commit and change the hash above. Each repo's own `CLAUDE.md`, if
present, holds instructions for sessions working in that repo, not ours.
