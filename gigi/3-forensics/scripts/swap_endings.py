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
HERO2 mode (4 Nov 2026): `swap_endings.py --hero2` (or SWAP_HERO2=1) reads results/hero2_swap_validity.csv (swap_validity.py
--hero2) and writes results/hero2_swap_endings.csv; when the validity file does not exist yet it says so and exits 0.
SWAP_HERO2_RESULTS=<folder> (tests only) stands in for results/ in HERO2 mode.
"""
import collections, csv, os, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import gemma_endings as E

import json
HERO2 = "--hero2" in sys.argv[1:] or os.environ.get("SWAP_HERO2") == "1"
HERO = HERO2 or "--hero" in sys.argv[1:] or os.environ.get("SWAP_HERO") == "1"
RES = pathlib.Path(os.environ["SWAP_HERO2_RESULTS"]).resolve() if HERO2 and os.environ.get("SWAP_HERO2_RESULTS") else HERE / "results"
ROOT = RES.parent   # run paths in the csvs are relative to this
LLM = {}
_f = HERE / "results" / "ending_labels_llm.jsonl"
for _l in (open(_f) if _f.exists() else []):
    _r = json.loads(_l)
    if _r.get("ending"): LLM[_r["run"]] = _r["ending"]   # the latest label per replay
_in = RES / ("hero2_swap_validity.csv" if HERO2 else "hero_swap_validity.csv" if HERO else "swap_validity.csv")
if HERO2 and not _in.exists():
    print(f"not yet: {_in.relative_to(ROOT)} does not exist (swap_validity.py --hero2 writes it once hero2 swap replays are fetched); nothing to do"); sys.exit(0)
rows = []
for r in csv.DictReader(open(_in)):
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
        cat = E.classify(r["env"], ROOT / r["run"], None)
        if cat in ("other", "gave_up", "says_done", "handed_back"):   # the hand labels read for chart 1 apply here too
            cat = E.hand_label(r["env"], ROOT / r["run"]) or cat
        if r["env"] == "ai_rd" and "<prompt>" in E.last_turn(ROOT / r["run"])[1]:   # ai_rd's final submission
            cat = "says_done"
        if cat == "other" and LLM.get(r["run"]):   # read by Claude Haiku 5.5 (scripts/label_endings_llm.py)
            cat = LLM[r["run"]]
    rows.append({"group": r["group"], "env": r["env"], "point": r["point"], "incident": r["incident"],
                 "turns": r["turns"], "ending": cat, "run": r["run"]})
_out = RES / ("hero2_swap_endings.csv" if HERO2 else "hero_swap_endings.csv" if HERO else "swap_endings.csv")
if not rows:
    print(f"no replays in {_in.name}: {_out.name} not written"); sys.exit(0)
with open(_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if HERO2:
    print(f"{len(rows)} replays -> {_out.relative_to(ROOT)}")
by = collections.defaultdict(collections.Counter)
for r in rows:
    by[(r["env"], r["group"])][r["ending"]] += 1
for k in sorted(by):
    print(f"{k[0]:34s} {k[1]:22s} {dict(by[k].most_common())}")
