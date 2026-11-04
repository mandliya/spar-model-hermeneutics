"""How each swap replay ends (26 Oct 2026), for chart 3b: the categories of chart 1's "why gemma stops early" panel
(gemma_endings.classify), applied to every replay of the swap, gemma's and Qwen's alike. Free: reads saved logs.
Reads results/swap_validity.csv (one row per replay; rerun scripts/swap_validity.py first so it lists the current
replays, not archived ones) and keeps its mechanical endings where they decide the case:
  ended by our parser -> parser;  step cap -> step_cap;  lost: context window exceeded -> context;
  any other lost replay (provider failure, crash, budget stop) -> lost, never classified.
Every other finished replay gets gemma_endings.classify on its last turn (gives up, says it's done, hands it back,
answers at once, loses the game, other). Also keeps the number of turns the model wrote after the takeover.
Writes results/swap_endings.csv (one row per replay).
  python3 gigi/3-forensics/scripts/swap_endings.py
Hero mode (4 Nov 2026): `swap_endings.py --hero` (or SWAP_HERO=1) reads results/hero_swap_validity.csv (swap_validity.py
--hero) and writes results/hero_swap_endings.csv. Without --hero nothing changes.
"""
import collections, csv, os, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import gemma_endings as E

import json
HERO = "--hero" in sys.argv[1:] or os.environ.get("SWAP_HERO") == "1"
LLM = {}
_f = HERE / "results" / "ending_labels_llm.jsonl"
for _l in (open(_f) if _f.exists() else []):
    _r = json.loads(_l)
    if _r.get("ending"): LLM[_r["run"]] = _r["ending"]   # the latest label per replay
rows = []
for r in csv.DictReader(open(HERE / "results" / ("hero_swap_validity.csv" if HERO else "swap_validity.csv"))):
    if r["group"] == "gemma's own full runs":
        continue
    end = r["ending"]
    if end == "ended by our parser":
        cat = "parser"
    elif end == "step cap":
        cat = "step_cap"
    elif end == "lost: context window exceeded":
        cat = "context"
    elif end.startswith("lost"):
        cat = "lost"
    else:
        cat = E.classify(r["env"], HERE / r["run"], None)
        if cat in ("other", "gave_up", "says_done", "handed_back"):   # the hand labels read for chart 1 apply here too
            cat = E.hand_label(r["env"], HERE / r["run"]) or cat
        if r["env"] == "ai_rd" and "<prompt>" in E.last_turn(HERE / r["run"])[1]:   # ai_rd's final submission
            cat = "says_done"
        if cat == "other" and LLM.get(r["run"]):   # read by Claude Haiku 5.5 (scripts/label_endings_llm.py)
            cat = LLM[r["run"]]
    rows.append({"group": r["group"], "env": r["env"], "point": r["point"], "incident": r["incident"],
                 "turns": r["turns"], "ending": cat, "run": r["run"]})
with open(HERE / "results" / ("hero_swap_endings.csv" if HERO else "swap_endings.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
by = collections.defaultdict(collections.Counter)
for r in rows:
    by[(r["env"], r["group"])][r["ending"]] += 1
for k in sorted(by):
    print(f"{k[0]:34s} {k[1]:22s} {dict(by[k].most_common())}")
