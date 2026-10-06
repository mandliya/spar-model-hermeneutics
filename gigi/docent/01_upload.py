"""Upload follow-up conversations to Transluce's Docent, with our labels.

Each Docent "agent run" = one follow-up exchange: the original conversation up to
the model's answer, the follow-up question, and the model's reply to it. Metadata:
model, category, sample_id, question, lied (our ground truth), admitted, and the
outcome (confess / double_down / false_confess / honest), so Docent can search and
cluster e.g. "why does gemma say it lied when it didn't?".

Conversations are rebuilt by the same code the pipeline uses (qwen/src/rollouts.py,
dataset pinned to the version gemma used); labels come from compare_models.py, the
same definitions as RESULTS.md. Sends data to docent.transluce.org (the user agreed);
collections are not public unless shared.

Run from the repo root (key in .env as DOCENT_API_KEY):
    uv run --with docent-python --with pandas --with pyarrow --with fsspec --with huggingface_hub \\
        --with python-dotenv --with torch --with statsmodels python gigi/docent/01_upload.py \\
        --categories ascii,sycophancy_b --dry-run
"""
import argparse
import importlib.util
import os
import pathlib
import shutil
import sys
import tempfile

from dotenv import load_dotenv

G = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(G.parent / ".env")
sys.path.insert(0, str(G / "qwen"))
from src import common as C  # noqa: E402

spec = importlib.util.spec_from_file_location("cmp", G / "compare_models.py")
cmp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmp)

p = argparse.ArgumentParser()
p.add_argument("--categories", default="ascii,sycophancy_b")
p.add_argument("--models", default="gemma-3-27b,qwen3-32b")
p.add_argument("--name", default=None)
p.add_argument("--dry-run", action="store_true", help="build and count, upload nothing")
args = p.parse_args()
cats, models = args.categories.split(","), args.models.split(",")

# gemma's results live in per-category folders; give rollouts a flat copy to read
GEMMA_FILES = {"ascii-sandbagging": ["ascii_results.csv", "ascii_reasoning_conditions.csv", "ascii_benchmark_conditions.csv"],
               "sycophancy": ["sycophancy_answer_conditions.csv", "sycophancy_benchmark_conditions.csv",
                              "sycophancy_feedback_conditions.csv", "sycophancy_feedback_benchmark_conditions.csv"],
               "mask-factual": ["mask_factual_provided_conditions.csv", "mask_factual_benchmark_conditions.csv", "mask_factual_named_rerun.csv"],
               "unanswerable": ["unanswerable_claims.csv", "unanswerable_benchmark_conditions.csv"],
               "capture-the-secret": ["capture_secret_conditions.csv", "capture_secret_benchmark_conditions.csv"],
               "alibi": ["alibi_conditions.csv", "alibi_benchmark_conditions.csv"],
               "named_cot": [f"named_cot_{c}.jsonl" for c in ["ascii", "sycophancy_a", "sycophancy_b", "mask_factual",
                                                               "unanswerable", "capture_secret", "alibi"]]}
flat = pathlib.Path(tempfile.mkdtemp())
for folder, files in GEMMA_FILES.items():
    for f in files:
        if (G / folder / "results" / f).exists():
            shutil.copy(G / folder / "results" / f, flat / f)
ROOT = {"gemma-3-27b": flat, "qwen3-32b": G / "qwen/results/qwen3-32b"}


def exchanges(model, cat):
    """(sample_id, question_name, messages) for the four questions."""
    C.RESULTS = ROOT[model]
    for mod in ["src.rollouts", "src.answer_probs"]:
        sys.modules.pop(mod, None)
    from src import rollouts as RO
    from src.answer_probs import named_cot_jobs, split_name
    rows = list(RO.BUILDERS[cat]())
    rows += named_cot_jobs(rows, cat)
    for name, base, question, answer, _kind in rows:
        sid, _track, q = split_name(name, cat)
        if q in cmp.ORDER:
            yield sid, q, base + [{"role": "user", "content": question}, {"role": "assistant", "content": str(answer)}]


runs = []
for m in models:
    for cat in cats:
        df, truth = cmp.rows(m, cat)
        df = cmp.with_named_cot(m, cat, df, truth).drop_duplicates(subset=["sample_id", "condition"])
        t, a = cmp.B(df[truth]), cmp.B(df["admitted"])
        ok = t.notna() & a.notna()          # unparseable answers are left out, never counted as "no"
        lab = {(s, c): (bool(tt), bool(aa)) for s, c, tt, aa in
               zip(df.sample_id[ok], df.condition[ok], t[ok], a[ok])}
        n = 0
        for sid, q, msgs in exchanges(m, cat):
            if (sid, q) not in lab:
                continue
            lied, admitted = lab[(sid, q)]
            outcome = {(True, True): "confess", (True, False): "double_down",
                       (False, True): "false_confess", (False, False): "honest"}[(lied, admitted)]
            runs.append((msgs, {"model": m, "category": cat, "sample_id": str(sid), "question": q,
                                "lied": lied, "admitted": admitted, "outcome": outcome}))
            n += 1
        print(f"{m:12s} {cat:15s} {n} exchanges")
print(f"total {len(runs)} exchanges")
if args.dry_run:
    import collections
    print("outcomes:", dict(collections.Counter((md["model"], md["category"], md["outcome"]) for _, md in runs)))
    raise SystemExit(0)

from docent import Docent  # noqa: E402
from docent.data_models import AgentRun, Transcript  # noqa: E402
from docent.data_models.chat import parse_chat_message  # noqa: E402

client = Docent(api_key=os.environ["DOCENT_API_KEY"])
name = args.name or f"gigi lie detection: {', '.join(cats)} ({', '.join(models)})"
cid = client.create_collection(name=name, description="Follow-up 'did you lie?' exchanges with our independent ground truth "
                               "(SPAR replication of Anthropic's lie-detector post). Metadata: lied, admitted, outcome.")
batch = []
for i, (msgs, md) in enumerate(runs, 1):
    batch.append(AgentRun(transcripts=[Transcript(messages=[parse_chat_message(x) for x in msgs])],
                          metadata={**md, "scores": {"lied": md["lied"], "admitted": md["admitted"]}}))
    if len(batch) == 250 or i == len(runs):
        client.add_agent_runs(cid, batch)
        print(f"uploaded {i}/{len(runs)}", flush=True)
        batch = []
print("collection id:", cid)
