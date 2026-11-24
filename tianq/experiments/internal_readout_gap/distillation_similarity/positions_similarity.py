"""Where in the turn to read: the distillation result again, at each of the 9 token positions.

    python positions_similarity.py     # prints every number it writes; writes OUT/distillation_positions/positions_similarity.json

Reads read_positions.py's files for Qwen3.6, base gemma and the final checkpoint. For each position (the token before
Qwen3.6's next turn, then 8 evenly spaced through it), each stored depth, and each lens (published = base gemma's
lens for both gemmas; own = each gemma through its own fitted lens, when read_positions.py saved one):
  same     overlap of the gemma's and Qwen3.6's top-200 lens words at the same restart point and position (Jaccard)
  chance   the same overlap against CHANCE_SPOTS other restart points at that position (fixed by SEED)
  change   (same - chance) for the final checkpoint minus (same - chance) for base gemma, paired by restart point,
           with a 95% interval over restart points
and the same three for the models' own top next tokens. Every position is reported; none is chosen here.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from read_positions import EXPERIMENT, POSITIONS  # noqa: E402
from shared import models, readout, words  # noqa: E402

ORIGINAL, BASE, FINAL = "qwen3.6-27b", "gemma-3-27b-it", "gemma-distilled-step96"
CHANCE_SPOTS, SEED = 20, 0


def load(tag: str) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in (models.OUT / EXPERIMENT / tag / "lens").glob("*.json")}


def word_sets(reads: dict[str, dict], position: int, level: str, key: str) -> dict[str, set]:
    """Per restart point, the words at one position: a depth's lens words (key = layers | own_lens_layers), or the
    model's own next tokens (level = "own next tokens")."""
    out = {}
    for name, r in reads.items():
        p = r["per_position"][position]
        if level == "own next tokens":
            out[name] = set(words.lens_words(p["model"]))
        else:
            layer = str(readout.layer_at(r["n_layers"], [int(k) for k in p[key]], float(level)))
            out[name] = set(words.lens_words(p[key][layer]))
    return out


def main() -> None:
    reads = {t: load(t) for t in (ORIGINAL, BASE, FINAL)}
    common = sorted(set.intersection(*(set(r) for r in reads.values())))
    rng = np.random.default_rng(SEED)
    others = {n: [common[j] for j in rng.choice([j for j in range(len(common)) if common[j] != n], CHANCE_SPOTS, replace=False)]
              for n in common}
    has_own = all("own_lens_layers" in reads[t][common[0]]["per_position"][0] for t in (BASE, FINAL))
    print(f"restart points read by all three models: {len(common)}; chance = the mean over {CHANCE_SPOTS} other restart "
          f"points at the same position; own-lens words {'present' if has_own else 'absent'}", flush=True)
    result: dict = {"restart_points": len(common), "positions": POSITIONS, "own_lens": has_own, "scores": {}}

    def scores(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray]:
        same = np.array([words.jaccard(a[n], b[n]) for n in common])
        chance = np.array([np.mean([words.jaccard(a[n], b[m]) for m in others[n]]) for n in common])
        return same, chance

    levels = [f"{d:.3f}" for d in readout.DEPTHS] + ["own next tokens"]
    for lens_name, key in (("published lens", "layers"), ("own lens", "own_lens_layers")):
        if key == "own_lens_layers" and not has_own:
            continue
        for level in levels:
            if level == "own next tokens" and key != "layers":
                continue
            for i, position in enumerate(POSITIONS):
                o = word_sets(reads[ORIGINAL], i, level, "layers")       # Qwen3.6 always through its published lens
                sb, cb = scores(o, word_sets(reads[BASE], i, level, key))
                sf, cf = scores(o, word_sets(reads[FINAL], i, level, key))
                change = words.mean_ci((sf - cf) - (sb - cb))
                label = level if level == "own next tokens" else f"depth {level}"
                result["scores"].setdefault(lens_name, {}).setdefault(label, {})[position] = {
                    "base_same": float(sb.mean()), "base_chance": float(cb.mean()), "final_same": float(sf.mean()),
                    "final_chance": float(cf.mean()), "base_above_chance": words.mean_ci(sb - cb),
                    "final_above_chance": words.mean_ci(sf - cf), "change": change}
                print(f"{lens_name} | {label} | position {position}: base gemma same {sb.mean():.4f}, chance {cb.mean():.4f}; "
                      f"final checkpoint same {sf.mean():.4f}, chance {cf.mean():.4f}; change above chance "
                      f"{change[0]:+.4f} [{change[1]:+.4f}, {change[2]:+.4f}]", flush=True)

    path = models.OUT / EXPERIMENT / "positions_similarity.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
