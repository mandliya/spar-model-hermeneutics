"""Build outputs for the cross-category analysis (no LLM calls): CSV, charts, a Docent
report, and the cross-category section of gigi/1-replication/docent/DOCENT_RESULTS.md.

Run after cross_category_selfreport.py has finished (run_cross_category.sh does both):
    cd gigi/1-replication/docent && env -u VIRTUAL_ENV uv run python docent_analyses/2026-10-06_15-01-54_ascii-false-confess/build_cross_category.py
"""
import collections
import datetime
import json
import pathlib
import re

from docent import Docent

from charts_lib import DOCENT_DIR, chart, save_csv, strip, wilson

HERE = pathlib.Path(__file__).resolve().parent
CID = "9d681e5d-6093-402c-b1d6-afcb86d5eb3b"
PLAN_NAME = "gemma cross-category: how it justifies its self-report"
REPORT_TITLE = "How gemma justifies its 'did you lie?' answers across all 11 categories"
DASH = f"https://docent.transluce.org/dashboard/{CID}"
CATS = {"ascii": "ascii-sandbagging", "sycophancy_a": "sycophancy A", "sycophancy_b": "sycophancy B",
        "mask_factual": "mask-factual", "unanswerable": "unanswerable", "capture_secret": "capture-the-secret",
        "alibi": "alibi", "mask_known_facts": "MASK known facts", "mask_statistics": "MASK statistics",
        "mask_continuations": "MASK continuations", "mask_disinformation": "MASK disinformation"}
QS = {"neutral_cot": "“Did you lie?”", "named_cot": "Named act"}
OUTCOMES = {"confess": "Lied, then said yes", "double_down": "Lied, then said no",
            "false_confess": "Truthful, then said yes", "honest": "Truthful, then said no"}
COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7"]
TOP = 7  # mechanisms shown in colour; the rest fold into one grey segment

client = Docent()
q = lambda dql: client.dql_result_to_dicts(client.execute_dql(CID, dql, max_rows=20000, ensure_latest_results=True))
pct = lambda k, n: f"{100 * k / n:.1f}%" if n else "—"


def ci(k, n):
    if not n:
        return "—"
    p, lo, hi = wilson(k, n)
    return f"{p:.1f}% ({lo:.0f}–{hi:.0f}%, n = {n})"


# ---- pull ----
plan = max((p for p in client.list_reading_plans(CID) if p["name"] == PLAN_NAME), key=lambda p: p["updated_at"])
PLAN_ID = plan["plan_id"]
steps = client.get_reading_plan(CID, PLAN_ID)["steps"]
step = lambda prefix: next(s["reading_id"] for s in steps if (s.get("name") or "").startswith(prefix) and s.get("reading_id"))
ASSIGN, PROPOSE = step("Assign a mechanism"), step("Propose cross-category")

proposal = q(f"SELECT rr.output AS o FROM reading_results rr JOIN reading_result_links rrl ON rrl.result_id = rr.id "
             f"WHERE rrl.reading_id = '{PROPOSE}'")[0]["o"]
proposal = json.loads(proposal) if isinstance(proposal, str) else proposal
DEFS = {m["name"]: m["description"] for m in proposal["mechanisms"]}
DEFS["other"] = "None of the mechanisms fits, or no intelligible reasoning."

counts = q(f"SELECT COUNT(rr.id) AS n, SUM(CASE WHEN rr.output IS NULL THEN 1 ELSE 0 END) AS failed FROM reading_results rr "
           f"JOIN reading_result_links rrl ON rrl.result_id = rr.id WHERE rrl.reading_id = '{ASSIGN}'")[0]
JOINED_FROM = f"""FROM reading_results rr
    JOIN reading_result_links rrl ON rrl.result_id = rr.id
    JOIN agent_runs a ON a.id = rr.arguments_dict->'run'->>'id'
    WHERE rrl.reading_id = '{ASSIGN}'"""
rows = q(f"""
    SELECT rr.id AS result_id, a.id AS run_id, a.metadata_json->>'sample_id' AS sample_id,
           a.metadata_json->>'category' AS category, a.metadata_json->>'question' AS question,
           a.metadata_json->>'outcome' AS outcome, a.metadata_json->>'lied' AS lied,
           rr.output->>'account_accuracy' AS account_accuracy, rr.output->>'mechanism' AS mechanism,
           rr.output->'key_quote'->>'text' AS key_quote
    {JOINED_FROM} AND rr.output IS NOT NULL
    ORDER BY category, question, sample_id""")
assert len(rows) == int(counts["n"]) - int(counts["failed"] or 0), (len(rows), counts)
for r in rows:
    r["key_quote"] = strip(r["key_quote"])
save_csv("cross_category_selfreport.csv", rows, ["category", "sample_id", "question", "outcome", "lied", "account_accuracy",
                                                 "mechanism", "key_quote", "run_id", "result_id"])
N, FAILED = len(rows), int(counts["failed"] or 0)

# ---- tables ----
by_mech = collections.Counter(r["mechanism"] for r in rows)
top = [m for m, _ in by_mech.most_common() if m != "other"][:TOP]
label = lambda m: m.replace("_", " ").capitalize()
bucket = lambda m: m if m in top else "rest"
segs = [[label(m), COLORS[i]] for i, m in enumerate(top)] + [["All other mechanisms", "--none"]]
mix = lambda rs: [sum(1 for r in rs if bucket(r["mechanism"]) == m) for m in top + ["rest"]]
cats = [c for c in CATS if any(r["category"] == c for r in rows)]
sub = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]

charts = {}
for qn, qlabel in QS.items():
    data = []
    for c in cats:
        rs = sub(category=c, question=qn)
        k = sum(r["account_accuracy"] == "misdescribes" for r in rs)
        if rs:
            _, lo, hi = wilson(k, len(rs))
            data.append([CATS[c], k, len(rs), lo, hi])
    charts[f"x_misdescribe_{qn}"] = chart(f"x_misdescribe_{qn}", "ci", {
        "rowhead": "Category", "colhead": f"{qlabel}: gemma misdescribes its own earlier answer (whisker = 95% CI)",
        "rows": data}, 60 + 38 * len(data))
charts["x_mechanisms_by_category"] = chart("x_mechanisms_by_category", "stacked", {
    "segs": segs, "rows": [[CATS[c], "runs", mix(sub(category=c))] for c in cats]}, 140 + 64 * len(cats))
charts["x_mechanisms_by_outcome"] = chart("x_mechanisms_by_outcome", "stacked", {
    "segs": segs, "rows": [[lab, "runs", mix(sub(outcome=o))] for o, lab in OUTCOMES.items()]}, 420)

mech_rows = []
for m, n in by_mech.most_common():
    rs = sub(mechanism=m)
    yes = sum(r["outcome"] in ("confess", "false_confess") for r in rs)
    lied = sum(r["lied"] == "true" for r in rs)
    mech_rows.append(f"| `{m}` | {DEFS.get(m, '')} | {n} | {pct(yes, n)} | {pct(lied, n)} |")

acc_rows = []
for c in cats:
    for qn, qlabel in QS.items():
        rs = sub(category=c, question=qn)
        f = lambda xs: ci(sum(r["account_accuracy"] == "misdescribes" for r in xs), len(xs))
        acc_rows.append(f"| {CATS[c]} | {qlabel} | {f(rs)} | {f([r for r in rs if r['lied'] == 'true'])} | "
                        f"{f([r for r in rs if r['lied'] == 'false'])} |")

cat_mech_head = "| Category | " + " | ".join(f"`{m}`" for m in top) + " | other mechanisms | runs |"
cat_mech_rows = [f"| {CATS[c]} | " + " | ".join(str(v) for v in mix(sub(category=c))) + f" | {len(sub(category=c))} |" for c in cats]
out_mech_rows = [f"| {lab} | " + " | ".join(str(v) for v in mix(sub(outcome=o))) + f" | {len(sub(outcome=o))} |"
                 for o, lab in OUTCOMES.items()]

examples, example_ids = [], []
for m, _ in by_mech.most_common():
    pool = sorted((r for r in sub(mechanism=m) if 50 <= len(r["key_quote"]) <= 260), key=lambda r: len(r["key_quote"]))
    seen = set()
    for r in pool:
        if r["category"] in seen:
            continue
        seen.add(r["category"])
        examples.append(f"- `{m}`, {CATS[r['category']]}, {QS[r['question']]}, {OUTCOMES[r['outcome']].lower()}:\n"
                        f"  > {r['key_quote']}")
        if len(seen) == 1:
            example_ids.append((m, r["result_id"]))
        if len(seen) == 2:
            break

# ---- Docent report ----
JOINED = f"""SELECT a.metadata_json->>'category' AS category, a.metadata_json->>'question' AS question,
    a.metadata_json->>'outcome' AS outcome, a.metadata_json->>'lied' AS lied,
    rr.output->>'account_accuracy' AS account_accuracy, rr.output->>'mechanism' AS mechanism {JOINED_FROM}"""
one_line = lambda s: " ".join(s.split())
html_block = lambda k, title, h: f'::html{{title="{title}" height="{h}"}}\n{charts[k]}\n::'
report = f"""---
docent_collection_id: {CID}
docent_source_reading_plan_id: {PLAN_ID}
title: {REPORT_TITLE}
---

# {REPORT_TITLE}

Every gemma-3-27b-it run that gave reasoning before answering "did you lie?" (`neutral_cot`) or a question naming the specific act (`named_cot`), in all 11 categories: {N} runs. A reader saw a neutral description of the scenario and the transcript, never our ground truth, and recorded (1) the mechanism behind gemma's yes or no, from a list proposed from a stratified pilot, and (2) whether gemma describes its own earlier answer the way the transcript shows it. Ground truth is joined afterwards. Generated automatically on {datetime.date.today()}; see the companion ascii report for the detailed ascii findings.

## Does gemma describe its own earlier answer correctly?

On ascii, the named question made gemma rewrite its own history. Here is the same check in every category.

{html_block("x_misdescribe_neutral_cot", "Did you lie?: share of runs where gemma misdescribes its earlier answer", 60 + 38 * len(cats))}

{html_block("x_misdescribe_named_cot", "Named act: share of runs where gemma misdescribes its earlier answer", 60 + 38 * len(cats))}

::dql-table{{title="Account accuracy by category, question and outcome" query="SELECT category, question, outcome, SUM(CASE WHEN account_accuracy = 'misdescribes' THEN 1 ELSE 0 END) AS misdescribes, SUM(CASE WHEN account_accuracy = 'accurate' THEN 1 ELSE 0 END) AS accurate, SUM(CASE WHEN account_accuracy = 'no_specific_description' THEN 1 ELSE 0 END) AS no_description, COUNT(category) AS runs FROM ({one_line(JOINED)}) AS s GROUP BY category, question, outcome ORDER BY category, question, outcome"}}
::

## Which mechanisms, where

{html_block("x_mechanisms_by_category", "Mechanism behind gemma's answer, by category (share of runs)", 140 + 64 * len(cats))}

{html_block("x_mechanisms_by_outcome", "Mechanism behind gemma's answer, by what actually happened (share of runs)", 420)}

::dql-table{{title="Mechanisms per category, split by outcome" query="SELECT category, mechanism, SUM(CASE WHEN outcome = 'confess' THEN 1 ELSE 0 END) AS lied_said_yes, SUM(CASE WHEN outcome = 'double_down' THEN 1 ELSE 0 END) AS lied_said_no, SUM(CASE WHEN outcome = 'false_confess' THEN 1 ELSE 0 END) AS truthful_said_yes, SUM(CASE WHEN outcome = 'honest' THEN 1 ELSE 0 END) AS truthful_said_no, COUNT(category) AS runs FROM ({one_line(JOINED)}) AS s GROUP BY category, mechanism ORDER BY category, runs DESC"}}
::

## The mechanisms

{chr(10).join(f"- **`{m}`**: {DEFS.get(m, '')}" for m, _ in by_mech.most_common())}

Proposer's notes: {proposal.get("notes", "")}

## One example per mechanism

{chr(10).join(f'::reading-result{{id="{rid}" title="{m}"}}{chr(10)}::{chr(10)}' for m, rid in example_ids)}

## Every run

::reading-results-table{{reading_id="{ASSIGN}" title="All {N} runs: mechanism, account check, key quote"}}
::

::callout{{color="orange" title="Caveats"}}
- Mechanisms and account checks were assigned by an AI reader (gpt-5.6-luna) in one pass, from a list proposed by gpt-5.6-sol from a stratified pilot; not human-labelled.
- The reader saw only the transcript and a one-line scenario description. Some lies (sycophancy, MASK belief) depend on information outside the transcript, so "misdescribes" there means a mismatch with what the transcript shows, not with our ground truth.
- {FAILED} of {N + FAILED} runs returned no result and are left out.
::
"""
report_path = HERE / "report_cross_category.md"
report_path.write_text(report)
existing = [r for r in client.list_reports(CID, prefix=REPORT_TITLE) if r.get("title") == REPORT_TITLE and r.get("id")]
if existing:
    rep = client.update_report_file(str(report_path), report_id=existing[0]["id"], expected_revision=existing[0]["revision"])
else:
    rep = client.save_report_file(str(report_path))
REPORT_URL = f"{DASH}/reports/{rep['id']}"
print("report:", REPORT_URL)

# ---- GitHub section ----
section = f"""<!-- CROSS-CATEGORY:START -->
## Cross-category: all 11 categories

*Generated automatically on {datetime.date.today()} by
[`build_cross_category.py`](docent_analyses/2026-10-06_15-01-54_ascii-false-confess/build_cross_category.py)
when the Docent run finished. The numbers are final; they have not yet been read and interpreted by a person.*

{N} gemma runs with reasoning (`neutral_cot` and `named_cot`) across all 11 categories{f", {FAILED} with no result left out" if FAILED else ""}.
A reader saw the transcript and a one-line scenario description, never our ground truth, and recorded the
mechanism behind gemma's answer and whether gemma describes its own earlier answer the way the transcript
shows it. Interactive: [Docent report]({REPORT_URL}) · [analysis plan]({DASH}/analysis-plan/{PLAN_ID}).
Per-run data: [`results/cross_category_selfreport.csv`](results/cross_category_selfreport.csv).

### Does gemma describe its own earlier answer correctly?

![Misdescribes its answer, "Did you lie?"](charts/x_misdescribe_neutral_cot.png)

![Misdescribes its answer, named act](charts/x_misdescribe_named_cot.png)

Share of runs where gemma misdescribes its earlier answer (Wilson 95% CI):

| Category | Question | All runs | When it lied | When it was truthful |
|---|---|---|---|---|
{chr(10).join(acc_rows)}

### Mechanisms

| Mechanism | What gemma says | Runs | Said "yes" | Had actually lied |
|---|---|---|---|---|
{chr(10).join(mech_rows)}

![Mechanisms by category](charts/x_mechanisms_by_category.png)

{cat_mech_head}
|{"---|" * (len(top) + 3)}
{chr(10).join(cat_mech_rows)}

![Mechanisms by outcome](charts/x_mechanisms_by_outcome.png)

| Outcome | {" | ".join(f"`{m}`" for m in top)} | other mechanisms | runs |
|{"---|" * (len(top) + 3)}
{chr(10).join(out_mech_rows)}

Proposing model's notes on the mechanisms: {proposal.get("notes", "")}

### In gemma's words (up to two categories per mechanism)

{chr(10).join(examples)}
<!-- CROSS-CATEGORY:END -->"""
doc_path = DOCENT_DIR / "DOCENT_RESULTS.md"
doc = doc_path.read_text()
if "<!-- CROSS-CATEGORY:START -->" in doc:
    doc = re.sub(r"<!-- CROSS-CATEGORY:START -->.*?<!-- CROSS-CATEGORY:END -->", lambda _: section, doc, flags=re.S)
else:
    anchor = "## Data problem found and fixed"
    assert anchor in doc
    doc = doc.replace(anchor, section + "\n\n" + anchor, 1)
doc_path.write_text(doc)
print(f"done: {N} runs, {FAILED} failed; DOCENT_RESULTS.md updated")
