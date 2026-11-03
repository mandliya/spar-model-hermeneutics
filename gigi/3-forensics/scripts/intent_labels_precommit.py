"""Bring the pre-commit hook study's older swap tries into the intent-label pipeline (3 Nov 2026).

The 17-26 Oct restarts on Qwen's two cheating pre-commit runs (results/runs_swap[_text]_precommit_{q1,q2,q1m}.csv, read
the same way as scripts/trace_vs_propensity.py lines 45-52) are not in results/gap_replays.csv, so intent_labels.py
never sees them. This script rewrites them as gap_replays rows:
  env=precommit_hook, point=<origin run timestamp>_run-<n>_K<K> (the main swap's point format, so intent_labels.
  shortcut_text can find the original Qwen message at step N), incident=True, N (9 for q1/q1m, 4 for q2), K=from_step,
  who, status=finished, hacked from `cheated`, grader_verdict="", ending="" (the pre-commit study has no Haiku ending
  labels), turns=`steps` (checked: equals the number of assistant messages after the takeover), run=<run dir>.
Only tries with steps >= 1 and cheated in {True, False} are kept (as in trace_vs_propensity.py); Qwen rows come from
the tool-call CSVs, gemma rows from the text-mode CSVs.
Prefix rule, verified here on every experiment: two runs of the same experiment share exactly 2 + 2(K+1) leading
messages (system, user, then K+1 assistant/tool pairs), so the continuation is everything after the (K+1)-th assistant
message and its tool results: the same rule intent_labels.continuation() uses for the main swap (from_step K keeps
steps 0..K; the first generated step dir is step-(K+1)).
Usage (from gigi/3-forensics):
  python3 scripts/intent_labels_precommit.py build      # locate runs, verify the prefix, write the CSV and plan
  python3 scripts/intent_labels_precommit.py estimate   # prompt sizes and judge cost via intent_labels.build()
Outputs: results/gap_replays_precommit.csv, results/swap_precommit_plan.json (the six restart points, in the
swap_*_plan.json format, pointing at the original runs so the shortcut text resolves). Nothing is called, nothing
is labelled: intent_labels.py would label these rows once it reads the extra CSV (and the extra plan file).
"""
import argparse, collections, csv, json, pathlib, re, sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import intent_labels as IL                                           # import only: no side effects

OUT_CSV = HERE / "results" / "gap_replays_precommit.csv"
OUT_PLAN = HERE / "results" / "swap_precommit_plan.json"
COLS = ["env", "point", "incident", "N", "K", "who", "status", "hacked", "grader_verdict", "ending", "turns", "run"]
TAKE = {1: "N", 2: "N−1", 3: "N−2", 6: "N−5"}
# the original Qwen cheating runs the restarts were cut from (scripts/swap_text.sh lines 13-16): q1 and q1m from run A, q2 from run B
ORIGIN = {"q1": ("results/raw/screen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-20-35-423567/run-5", 9),
          "q1m": ("results/raw/screen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-20-35-423567/run-5", 9),
          "q2": ("results/raw/base_qwen/precommit_hook/precommit_hook/qwen-qwen3-next-80b-a3b-instruct/2026-10-17_15-24-48-348561/run-18", 4)}


def source_rows():
    """(tag, who, K, csv row) for every scored try, selected exactly as trace_vs_propensity.py does."""
    out = []
    for tag in ("q1", "q2", "q1m"):
        for text, who in ((False, "qwen"), (True, "gemma")):
            f = HERE / "results" / (f"runs_swap_text_precommit_{tag}.csv" if text else f"runs_swap_precommit_{tag}.csv")
            for row in csv.DictReader(open(f)):
                mm = re.search(r"from_step(\d+)/(qwen|gemma)/", row["experiment"])
                if mm and mm.group(2) == who and int(row["steps"] or 0) >= 1 and row["cheated"] in ("True", "False"):
                    out.append((tag, who, int(mm.group(1)), row))
    return out


def point_of(tag, K):
    origin, N = ORIGIN[tag]
    m = re.search(r"(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d-\d+)/(run-\d+)$", origin)
    return f"{m.group(1)}_{m.group(2)}_K{K}", N


def msg_key(x):
    return (x.get("role"), IL.text_of(x), json.dumps(x.get("tool_calls"), sort_keys=True))


def verify_prefix(groups):
    """For each experiment: the number of leading messages all its runs share, and how many assistant messages that holds."""
    report = []
    for (stage, exp, K, who), runs in sorted(groups.items()):
        ms = [m for m in (IL.load_msgs(f"results/raw/{stage}/{exp}/{r}") for r in runs) if m]
        if len(ms) < 2:
            report.append((stage, exp, K, who, len(ms), None, None)); continue
        shared = min(next((n for n in range(min(len(a), len(b))) if msg_key(a[n]) != msg_key(b[n])), min(len(a), len(b)))
                     for i, a in enumerate(ms) for b in ms[i + 1:])
        n_asst = sum(1 for x in ms[0][:shared] if x.get("role") == "assistant")
        report.append((stage, exp, K, who, len(ms), shared, n_asst))
    return report


def build_rows():
    rows, groups, missing = [], collections.defaultdict(list), []
    for tag, who, K, r in source_rows():
        run = f"results/raw/{r['stage']}/{r['experiment']}/{r['run']}"
        if not (HERE / run).exists() or IL.load_msgs(run) is None:
            missing.append(run); continue
        groups[(r["stage"], r["experiment"], K, who)].append(r["run"])
        point, N = point_of(tag, K)
        rows.append({"env": "precommit_hook", "point": point, "incident": "True", "N": N, "K": K, "who": who, "status": "finished",
                     "hacked": 1 if r["cheated"] == "True" else 0, "grader_verdict": "", "ending": "", "turns": r["steps"], "run": run})
    return rows, groups, missing


def plan_entries(rows):
    seen = {}
    for r in rows:
        tag = next(t for t in ORIGIN if point_of(t, r["K"])[0] == r["point"])
        seen.setdefault(r["point"], {"env": "precommit_hook", "run": ORIGIN[tag][0], "K": r["K"], "N": r["N"], "incident": True,
                                     "reps": 0, "source": "pre-commit study restarts (runs_swap[_text]_precommit_*.csv)"})
        seen[r["point"]]["reps"] += 1
    return list(seen.values())


def counts(rows):
    c = collections.Counter((r["who"], TAKE.get(r["N"] - r["K"], "?")) for r in rows)
    h = collections.Counter((r["who"], TAKE.get(r["N"] - r["K"], "?")) for r in rows if r["hacked"] == 1)
    for k in sorted(c, key=lambda k: (k[0], list(TAKE.values()).index(k[1]))):
        print(f"  {k[0]:6s} {k[1]:4s} n={c[k]:3d}  hacked={h[k]}")
    print(f"  total {len(rows)}: " + ", ".join(f"{w}={sum(v for (ww, _), v in c.items() if ww == w)}" for w in ("qwen", "gemma")))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["build", "estimate"]); a = ap.parse_args()
    rows, groups, missing = build_rows()
    n_src = len(source_rows())
    print(f"{n_src} scored tries in the six CSVs; located {len(rows)}, missing {len(missing)}" + (": " + ", ".join(missing[:5]) if missing else ""))
    if a.mode == "build":
        rep = verify_prefix(groups)
        bad =[(s, e, K, w, sh, na) for s, e, K, w, nr, sh, na in rep if na != K + 1]
        print(f"prefix check: {len(rep)} experiments; shared leading messages = 2 + 2(K+1) and K+1 assistant messages in "
              f"{len(rep) - len(bad)}/{len(rep)}" + (f"; EXCEPTIONS: {bad}" if bad else " (same rule as intent_labels.continuation)"))
        for s, e, K, w, nr, sh, na in rep:
            print(f"    {s:23s} K={K} {w:5s} runs={nr:2d} shared_msgs={sh} assistant_in_prefix={na}")
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
        json.dump(plan_entries(rows), open(OUT_PLAN, "w"), indent=1)
        print(f"wrote {OUT_CSV.relative_to(HERE)} ({len(rows)} rows) and {OUT_PLAN.relative_to(HERE)} ({len(plan_entries(rows))} restart points)")
        counts(rows)
        return
    # estimate: the same arithmetic as intent_labels.py estimate, on these rows, with the shortcut resolved via the precommit plan
    P = IL.plan() + (json.load(open(OUT_PLAN)) if OUT_PLAN.exists() else plan_entries(rows))
    srows = [{k: str(v) for k, v in r.items()} for r in rows]
    chars = collections.Counter(); cnt = collections.Counter(); miss = 0; shortcuts = collections.Counter()
    for r in srows:
        b = IL.build(r, P)
        if not b:
            miss += 1; continue
        k = (r["who"], TAKE.get(int(r["N"]) - int(r["K"]), "?")); chars[k] += len(b[0]); cnt[k] += 1
        shortcuts[IL.shortcut_text(P, r)[:40]] += 1
    tot = sum(chars.values()); toks = tot / 3.6
    print(f"{len(srows)} tries ({miss} without messages); prompt chars {tot:,} ≈ {toks:,.0f} input tokens; output ≈ {len(srows) * 250:,} tokens")
    print(f"estimated judge cost ({IL.JUDGE}): ${toks / 1e6 * IL.PRICE_IN + len(srows) * 250 / 1e6 * IL.PRICE_OUT:,.2f}")
    for k in sorted(cnt, key=lambda k: (k[0], list(TAKE.values()).index(k[1]))):
        print(f"  {k[0]:6s} {k[1]:4s} n={cnt[k]:4d}  mean prompt {chars[k] / cnt[k] / 1000:6.1f}k chars")
    print("shortcut text resolved:", dict(shortcuts))


if __name__ == "__main__":
    main()
