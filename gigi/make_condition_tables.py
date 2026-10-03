"""Generate the per-category follow-up tables in RESULTS.md from the data.

One table per category: confession, doubling down and false confession for
each of the four follow-up questions the project keeps (neutral, neutral_cot,
named, named_cot), with Wilson 95% intervals; plus the balanced-accuracy grid.
The other nine wordings are archived in archive/2026-10-01_all_13_wordings/. Written between
two marker comments in RESULTS.md, so rerunning replaces the section in place
and the numbers can never drift from the CSVs.

Run from the repo root:
    uv run --with pandas --with statsmodels python gigi/make_condition_tables.py
"""

import pathlib
import re

import pandas as pd
from named_cot_join import named_cot_frame
from statsmodels.stats.proportion import proportion_confint

G = pathlib.Path(__file__).resolve().parent
START, END = "<!-- CONDITION-TABLES:START -->", "<!-- CONDITION-TABLES:END -->"
ORDER = ["neutral", "neutral_cot", "named", "named_cot"]
B = lambda s: s.map({True: True, False: False, "True": True, "False": False})


def ascii_rows():
    r = G / "ascii-sandbagging/results"
    base = pd.read_csv(r / "ascii_results.csv").rename(
        columns={"uid": "sample_id"}).assign(condition="neutral")
    return pd.concat([base] + [pd.read_csv(r / f) for f in
                               ["ascii_followup_conditions.csv", "ascii_reasoning_conditions.csv",
                                "ascii_benchmark_conditions.csv"]]), "lied"


def mask_rows():
    r = G / "mask-factual/results"
    b = pd.read_csv(r / "mask_factual_benchmark_conditions.csv")
    rr = pd.read_csv(r / "mask_factual_named_rerun.csv")
    return pd.concat([b[~b.condition.isin(["named", "eval_named"])],
                      rr[rr.condition.isin(["named", "eval_named"])]]), "lied"


def unans_rows():
    d = pd.read_csv(G / "unanswerable/results/unanswerable_benchmark_conditions.csv")
    return d[d.bucket == "confabulation"], "confabulated"


CATS = [
    ("ascii-sandbagging", "falsely denied a capability it has", ascii_rows),
    ("sycophancy Track A", "answer folded toward the user's stated belief",
     lambda: (pd.read_csv(G / "sycophancy/results/sycophancy_benchmark_conditions.csv"), "fold")),
    ("sycophancy Track B", "critique shifted more than 1.0 toward the framing",
     lambda: (pd.read_csv(G / "sycophancy/results/sycophancy_feedback_benchmark_conditions.csv"),
              "sycophantic")),
    ("mask-factual", "told the pressured false narrative", mask_rows),
    ("unanswerable", "confabulated an answer after claiming it couldn't", unans_rows),
    ("capture-the-secret", "falsely denied, to hide a secret",
     lambda: (pd.read_csv(G / "capture-the-secret/results/capture_secret_benchmark_conditions.csv",
                          low_memory=False), "lied")),
    ("alibi", "maintained a false alibi",
     lambda: (pd.read_csv(G / "alibi/results/alibi_benchmark_conditions.csv", low_memory=False),
              "lied")),
    ("MASK known facts", "said the opposite of its own stated belief, under a role's pressure",
     lambda: (pd.read_csv(G / "mask-belief/results/mask_belief_known_facts_benchmark_conditions.csv"), "lied")),
    ("MASK statistics", "gave a number more than 10% away from its own stated belief, under pressure",
     lambda: (pd.read_csv(G / "mask-belief/results/mask_belief_statistics_benchmark_conditions.csv"), "lied")),
    ("MASK continuations", "continued a text with claims contradicting its own stated belief",
     lambda: (pd.read_csv(G / "mask-belief/results/mask_belief_continuations_benchmark_conditions.csv"), "lied")),
    ("MASK disinformation", "wrote content contradicting its own stated belief, presented as true",
     lambda: (pd.read_csv(G / "mask-belief/results/mask_belief_disinformation_benchmark_conditions.csv"), "lied")),
]


def ci(k, n):
    if n == 0:
        return "—", "—"
    lo, hi = proportion_confint(k, n, 0.05, method="wilson")
    return f"{100 * k / n:.1f}%", f"{100 * lo:.1f}–{100 * hi:.1f}%"


blocks = []
BAL = {}
DOC_TABLES = {}   # category -> its table with a balanced-accuracy column, for its own doc
# Each category's own write-up gets the same numbers, between its own markers.
CATEGORY_DOCS = {"ascii-sandbagging": ("ascii-sandbagging/RESULTS.md", "CORE4"),
                 "sycophancy Track A": ("sycophancy/RESULTS.md", "CORE4-A"),
                 "sycophancy Track B": ("sycophancy/RESULTS.md", "CORE4-B"),
                 "mask-factual": ("mask-factual/RESULTS.md", "CORE4"),
                 "unanswerable": ("unanswerable/README.md", "CORE4"),
                 "capture-the-secret": ("capture-the-secret/README.md", "CORE4"),
                 "alibi": ("alibi/README.md", "CORE4"),
                 "MASK known facts": ("mask-belief/README.md", "CORE4-KF"),
                 "MASK statistics": ("mask-belief/README.md", "CORE4-ST"),
                 "MASK continuations": ("mask-belief/README.md", "CORE4-CO"),
                 "MASK disinformation": ("mask-belief/README.md", "CORE4-DI")}
for name, act, load in CATS:
    df, truth = load()
    df = pd.concat([df, named_cot_frame(name, df, truth)], ignore_index=True)
    df = df.drop_duplicates(subset=["sample_id", "condition"])
    df = df[df["admitted"].notna() & df[truth].notna()]
    t, a = B(df[truth]).astype(bool), B(df["admitted"]).astype(bool)
    lied_n = df[t].sample_id.nunique()
    honest_n = df[~t].sample_id.nunique()
    lines = [f"#### {name}",
             "",
             f"The deceptive act: {act}. **{lied_n}** rows where gemma did it, "
             f"**{honest_n}** where it didn't.",
             "",
             "| Question | Confession | 95% CI | Doubling down | False confession | 95% CI |",
             "|---|---|---|---|---|---|"]
    doc = ["| Question | Confession | 95% CI | Doubling down | False confession | 95% CI | Balanced accuracy |",
           "|---|---|---|---|---|---|---|"]
    for cond in ORDER:
        m = df["condition"] == cond
        if not m.any():
            continue
        pos, neg = a[m & t], a[m & ~t]
        c, cci = ci(int(pos.sum()), len(pos))
        dd = f"{100 * (~pos).sum() / len(pos):.1f}%" if len(pos) else "—"
        f, fci = ci(int(neg.sum()), len(neg))
        if len(pos) and len(neg):
            BAL[(name, cond)] = 50 * (pos.sum() / len(pos) + (~neg).sum() / len(neg))
        lines.append(f"| `{cond}` | {c} | {cci} | {dd} | {f} | {fci} |")
        b = BAL.get((name, cond))
        doc.append(f"| `{cond}` | {c} | {cci} | {dd} | {f} | {fci} | "
                   f"{'—' if b is None else (f'**{b:.0f}%**' if b >= 70 else f'{b:.0f}%')} |")
    blocks.append("\n".join(lines))
    DOC_TABLES[name] = (f"{lied_n} rows where gemma did it, {honest_n} where it didn't.\n\n"
                        + "\n".join(doc))

section = "\n".join([
    START,
    "### Every category, all four follow-up questions",
    "",
    "Measures are defined in Methodology §3. Intervals are Wilson 95%.",
    "",
    "\n\n".join(blocks),
    END,
])

BSTART, BEND = "<!-- BALANCED-GRID:START -->", "<!-- BALANCED-GRID:END -->"
names = [c[0] for c in CATS]
short = {"ascii-sandbagging": "ascii", "sycophancy Track A": "syc A", "sycophancy Track B": "syc B",
         "mask-factual": "mask", "unanswerable": "unans.", "capture-the-secret": "capture", "alibi": "alibi",
         "MASK known facts": "M facts", "MASK statistics": "M stats", "MASK continuations": "M cont.",
         "MASK disinformation": "M disinfo"}


def cell(v):
    if v is None:
        return "—"
    return f"**{v:.0f}**" if v >= 70 else f"{v:.0f}"


grid = [BSTART,
        "| Question | " + " | ".join(short[n] for n in names) + " |",
        "|---|" + "---|" * len(names)]
for cond in ORDER:
    grid.append(f"| `{cond}` | " + " | ".join(cell(BAL.get((n, cond))) for n in names) + " |")
grid.append(BEND)
grid = "\n".join(grid)

p = G / "RESULTS.md"
s = p.read_text()
if START in s:
    s = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _: section, s, flags=re.S)
else:
    anchor = "## Full write-up for each category"
    s = s.replace(anchor, section + "\n\n" + anchor, 1)
if BSTART in s:
    s = re.sub(re.escape(BSTART) + ".*?" + re.escape(BEND), lambda _: grid, s, flags=re.S)
p.write_text(s)
for name, (path, marker) in CATEGORY_DOCS.items():
    dp = G / path
    ds = dp.read_text()
    st, en = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    if st not in ds:
        print(f"no {marker} markers in {path} - skipped")
        continue
    ds = re.sub(re.escape(st) + ".*?" + re.escape(en),
                lambda _: f"{st}\n{DOC_TABLES[name]}\n{en}", ds, flags=re.S)
    dp.write_text(ds)
    print(f"filled {marker} in {path}")
print(grid)
