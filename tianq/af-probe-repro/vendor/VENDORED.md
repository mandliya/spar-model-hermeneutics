# Vendored: probe-af

`probe-af/` is a verbatim copy of Josh's (Zhuofan Ying's) repo for our sample-level AF paper (https://icml.cc/virtual/2026/79535). Its `.git` has been removed.

- Source: https://github.com/zfying/probe-af
- Commit: `d911494499b95dd2689b324f730edb27fcd4ad28` (2026-06-18, HEAD of `main` on 2026-10-31)
- The paper's results were committed at `b4fc7b1` (2026-06-17). The five later commits only add `followup/`, an audit of the paper's claims, so `src/`, `scripts/`, `data/`, `results/` and `config.py` are as at `b4fc7b1`.
- License: MIT (`probe-af/LICENSE`)
- Copied on 2026-10-31, everything but `.git`. No secrets found. Its own `.gitignore` is kept, so its large regenerable outputs (`*.h5`, `results/probes/`, `plots/`) stay out of git here too.

**Do not edit anything under `probe-af/`.** Our code lives next to it in `tianq/af-probe-repro/` and imports from it. To update, re-copy a newer commit and change the hash above.

`probe-af/CLAUDE.md` is part of the copy. It holds instructions for Claude sessions working in Josh's repo, not ours.
