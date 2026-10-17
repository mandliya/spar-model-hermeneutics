"""Fold the new alibi rows into the HF activations, after run_alibi_rest.sh.

  activations/alibi.tar.gz       rebuilt: the existing 498 rows + the new rows,
                                 four questions, float32, both capture points
  activations/maps/alibi.jsonl   regenerated from the files in the tarball
  first_run/alibi/               + the new rows' first-run state (float32)

New follow-up tensors come from results/activations/ (neutral, neutral_cot,
named; written by 00_alibi_selfreport_benchmark.py) and
/workspace/fp32_staging/named_cot_alibi/ (named_cot; 05_named_cot.py). Every
upload is checked by size; the old tarball is replaced only after the new one
is built and its counts printed.
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
W = pathlib.Path("/workspace/alibi_update")
OUT = W / "alibi"
KEEP = {"neutral", "neutral_cot", "named", "named_cot"}
NEW_FU = pathlib.Path("/workspace/lies-experiment/results/activations")
NEW_NC = pathlib.Path("/workspace/fp32_staging/named_cot_alibi")


def parse(base):
    name, point = base[:-3].rsplit("__", 1)
    return name, name.rsplit("__", 1)[1], point


shutil.rmtree(W, ignore_errors=True)
OUT.mkdir(parents=True)

# 1. existing tarball
p = hf_hub_download(R, "activations/alibi.tar.gz", repo_type="dataset", local_dir=W / "dl")
old = 0
with tarfile.open(p, "r|gz") as t:
    for m in t:
        if m.isfile() and m.name.endswith(".pt"):
            (OUT / os.path.basename(m.name)).write_bytes(t.extractfile(m).read())
            old += 1
os.remove(p)
print(f"existing tarball: {old} files", flush=True)

# 2. new follow-up tensors
added = 0
for src in list(NEW_FU.glob("*.pt")) + list(NEW_NC.glob("*.pt")):
    _, cond, point = parse(src.name)
    if cond in KEEP and point in ("predecision", "answer_token"):
        if not (OUT / src.name).exists():
            shutil.copy2(src, OUT / src.name)
            added += 1
print(f"added {added} new follow-up files", flush=True)

# 3. map + counts
rows = defaultdict(dict)
for f in OUT.glob("*.pt"):
    name, cond, point = parse(f.name)
    rows[name][point + "_file"] = f.name
    rows[name]["condition"] = cond
mp = W / "alibi.jsonl"
with mp.open("w") as fh:
    for name, r in sorted(rows.items()):
        fh.write(json.dumps({"name": name, "sample_id": name.rsplit("__", 1)[0], "track": None,
                             "condition": r["condition"],
                             "predecision_file": r.get("predecision_file"),
                             "answer_token_file": r.get("answer_token_file")}) + "\n")
print("rows per question:", dict(Counter(r["condition"] for r in rows.values())),
      "| without an answer-token file:", sum(1 for r in rows.values() if "answer_token_file" not in r),
      flush=True)

# 4. upload, replacing the old tarball and map
tb = W / "alibi.tar.gz"
subprocess.run(["tar", "-czf", str(tb), "-C", str(W), "alibi"], check=True)
api.upload_file(path_or_fileobj=str(tb), path_in_repo="activations/alibi.tar.gz", repo_id=R,
                repo_type="dataset", commit_message="activations: alibi, all rows (500 + the rest)")
api.upload_file(path_or_fileobj=str(mp), path_in_repo="activations/maps/alibi.jsonl", repo_id=R,
                repo_type="dataset", commit_message="activations: alibi map, all rows")
for local, remote in [(tb, "activations/alibi.tar.gz"), (mp, "activations/maps/alibi.jsonl")]:
    info = api.get_paths_info(R, [remote], repo_type="dataset")
    print(f"VERIFY {remote}:", "OK" if info and info[0].size == local.stat().st_size else "MISMATCH",
          flush=True)

# 5. first-run state for the new rows
fr = W / "first_run"
fr.mkdir()
for f in NEW_FU.glob("*__alibi__predecision.pt"):
    shutil.copy2(f, fr / f.name)
n = len(list(fr.glob("*.pt")))
api.upload_folder(folder_path=str(fr), path_in_repo="first_run/alibi", repo_id=R,
                  repo_type="dataset", commit_message=f"first_run: alibi, {n} new rows (float32)")
remote = {s.rfilename for s in api.repo_info(R, repo_type="dataset").siblings
          if s.rfilename.startswith("first_run/alibi/")}
print(f"VERIFY first_run/alibi: {len({f'first_run/alibi/{f.name}' for f in fr.glob('*.pt')} & remote)}/{n} new files on HF",
      flush=True)
print("ALIBI ACTIVATIONS UPDATED", flush=True)
