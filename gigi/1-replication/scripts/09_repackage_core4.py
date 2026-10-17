"""Repackage the HF activations around the four kept follow-up wordings.

From 1 Oct 2026 the project keeps four follow-up wordings: neutral,
neutral_cot, named, named_cot. The other nine are archived (not deleted).

Part 1 - current follow-up activations, one tarball per category:
    activations/<category>.tar.gz        float32, the 4 wordings x both capture
                                         points (predecision, answer_token)
    activations/maps/<category>.jsonl    one line per (row, wording)
  built from tarballs/fp32/<category>_fp32.tar.gz (12 wordings) and
  tarballs/fp32/named_cot_<category>_fp32.tar.gz, keeping only the 4.

Part 2 - first-run activations that so far only lived inside the float16
tarballs, pulled out so first_run/ holds every category:
    first_run/unanswerable/   from tarballs/unanswerable_activations.tar.gz
    first_run/capture_secret/ from tarballs/capture_secret_activations.tar.gz
    first_run/alibi/          from tarballs/alibi_activations.tar.gz

Nothing on HF is deleted or overwritten here. Every upload is checked by
size; counts per wording are printed for each category.

Run on the box:  python 09_repackage_core4.py
"""

import json
import os
import pathlib
import shutil
import subprocess
import tarfile
from collections import Counter, defaultdict

from huggingface_hub import HfApi, hf_hub_download

from src import common  # noqa: F401  (loads .env -> HF_TOKEN)

R = "model-hermeneutics/gigi-lie-detection-activations"
api = HfApi(token=os.environ["HF_TOKEN"])
WORK = pathlib.Path("/workspace/repack")
WORK.mkdir(exist_ok=True)
CATS = ["mask_factual", "unanswerable", "sycophancy_a", "sycophancy_b", "ascii",
        "capture_secret", "alibi"]
KEEP = {"neutral", "neutral_cot", "named", "named_cot"}
FOLLOWUPS = KEEP | {"low_stakes", "high_stakes", "confidence", "low_stakes_cot",
                    "high_stakes_cot", "confidence_cot", "plain", "eval_plain", "eval_named"}
POINTS = ("predecision", "answer_token")


def say(*a):
    print(*a, flush=True)


def parse(base):
    """(name, condition, point) from '<name>__<point>.pt', name = '<id>__<condition>'."""
    stem = base[:-3]
    name, point = stem.rsplit("__", 1)
    return name, name.rsplit("__", 1)[1], point


def stream_extract(src_in_repo, out_dir, want):
    """Download one tarball, write the members `want(base)` accepts, delete the download."""
    p = hf_hub_download(R, src_in_repo, repo_type="dataset", local_dir=WORK / "dl")
    n = 0
    with tarfile.open(p, "r|gz") as t:
        for m in t:
            if not m.isfile() or not m.name.endswith(".pt"):
                continue
            base = os.path.basename(m.name)
            if want(base):
                (out_dir / base).write_bytes(t.extractfile(m).read())
                n += 1
    os.remove(p)
    say(f"  {src_in_repo}: kept {n} files")


def verify(local, path_in_repo):
    info = api.get_paths_info(R, [path_in_repo], repo_type="dataset")
    ok = bool(info) and info[0].size == local.stat().st_size
    say(f"  VERIFY {path_in_repo}: {'OK' if ok else 'MISMATCH'} ({local.stat().st_size} bytes)")
    return ok


# Part 1 -----------------------------------------------------------------------
for cat in CATS:
    say(f"=== {cat}: current activations (4 wordings) ===")
    try:
        out = WORK / cat
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        keep = lambda b: b.endswith(".pt") and parse(b)[1] in KEEP and parse(b)[2] in POINTS
        stream_extract(f"tarballs/fp32/{cat}_fp32.tar.gz", out, keep)
        stream_extract(f"tarballs/fp32/named_cot_{cat}_fp32.tar.gz", out, keep)

        rows = defaultdict(dict)
        for f in sorted(out.glob("*.pt")):
            name, cond, point = parse(f.name)
            rows[name][point + "_file"] = f.name
            rows[name]["condition"] = cond
        mp = WORK / f"{cat}.jsonl"
        with mp.open("w") as fh:
            for name, r in sorted(rows.items()):
                stem = name.rsplit("__", 1)[0]
                track = None
                if cat.startswith("sycophancy"):
                    track, stem = stem.split("__", 1)
                fh.write(json.dumps({"name": name, "sample_id": stem, "track": track,
                                     "condition": r["condition"],
                                     "predecision_file": r.get("predecision_file"),
                                     "answer_token_file": r.get("answer_token_file")}) + "\n")
        per = Counter(r["condition"] for r in rows.values())
        no_at = sum(1 for r in rows.values() if "answer_token_file" not in r)
        say(f"  rows per wording: {dict(per)}; without an answer-token file: {no_at}")

        tb = WORK / f"{cat}.tar.gz"
        subprocess.run(["tar", "-czf", str(tb), "-C", str(WORK), cat], check=True)
        api.upload_file(path_or_fileobj=str(tb), path_in_repo=f"activations/{cat}.tar.gz",
                        repo_id=R, repo_type="dataset",
                        commit_message=f"activations: {cat}, 4 wordings, float32")
        api.upload_file(path_or_fileobj=str(mp), path_in_repo=f"activations/maps/{cat}.jsonl",
                        repo_id=R, repo_type="dataset",
                        commit_message=f"activations: {cat} map")
        if verify(tb, f"activations/{cat}.tar.gz") and verify(mp, f"activations/maps/{cat}.jsonl"):
            shutil.rmtree(out)
            tb.unlink()
            say(f"--- {cat} OK ---")
        else:
            say(f"!!! {cat}: verification failed - local files kept in {WORK}")
    except Exception as e:
        say(f"!!! {cat} FAILED: {e!r} (continuing)")

# Part 2 -----------------------------------------------------------------------
FIRST = {"unanswerable": "tarballs/unanswerable_activations.tar.gz",
         "capture_secret": "tarballs/capture_secret_activations.tar.gz",
         "alibi": "tarballs/alibi_activations.tar.gz"}
for cat, src in FIRST.items():
    say(f"=== {cat}: first-run activations ===")
    try:
        out = WORK / f"first_run_{cat}"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        stream_extract(src, out, lambda b: b.endswith(".pt") and parse(b)[1] not in FOLLOWUPS)
        kinds = Counter(parse(f.name)[1] + "/" + parse(f.name)[2] for f in out.glob("*.pt"))
        say(f"  first-run files by kind: {dict(kinds)}")
        api.upload_folder(folder_path=str(out), path_in_repo=f"first_run/{cat}",
                          repo_id=R, repo_type="dataset",
                          commit_message=f"first_run: {cat}, pulled out of the float16 tarball")
        remote = {s.rfilename for s in api.repo_info(R, repo_type="dataset").siblings
                  if s.rfilename.startswith(f"first_run/{cat}/")}
        local = {f"first_run/{cat}/{f.name}" for f in out.glob("*.pt")}
        say(f"  VERIFY first_run/{cat}: {len(local & remote)}/{len(local)} on HF")
        if local <= remote:
            shutil.rmtree(out)
            say(f"--- first_run {cat} OK ---")
    except Exception as e:
        say(f"!!! first_run {cat} FAILED: {e!r} (continuing)")

say("=== REPACKAGE DONE ===")
