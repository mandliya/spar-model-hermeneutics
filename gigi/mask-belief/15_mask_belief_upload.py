"""MASK belief categories, stage 4: everything to HF, checked by size.

  results/mask_belief/mask_belief_<subtype>_{answers,judged,followups}.jsonl
  activations/mask_<subtype>.tar.gz + activations/maps/mask_<subtype>.jsonl
  first_run/mask_<subtype>/<sid>__pressured__predecision.pt
"""
import json
import pathlib
import subprocess
from collections import Counter, defaultdict

from src import common as C
from src import mask_belief as MB
from src.hf_sync import REPO, upload
from huggingface_hub import HfApi
import os

api = HfApi(token=os.environ["HF_TOKEN"])
ST = pathlib.Path("/workspace/mask_belief_staging")
ok = True


def size(path):
    info = api.get_paths_info(REPO, [path], repo_type="dataset")
    return info[0].size if info else None


for sub in MB.SUBTYPES:
    for kind in ("answers", "judged", "followups"):
        f = C.RESULTS / f"mask_belief_{sub}_{kind}.jsonl"
        if f.exists():
            upload(f, f"results/mask_belief/{f.name}")
            good = size(f"results/mask_belief/{f.name}") == f.stat().st_size
            ok &= good
            print(("OK  " if good else "BAD ") + f"results/mask_belief/{f.name}", flush=True)

    fu = ST / "followups" / f"mask_{sub}"
    if fu.exists() and any(fu.glob("*.pt")):
        rows = defaultdict(dict)
        for p in fu.glob("*.pt"):
            name, point = p.name[:-3].rsplit("__", 1)
            rows[name][point + "_file"] = p.name
            rows[name]["condition"] = name.rsplit("__", 1)[1]
        mp = ST / f"mask_{sub}.jsonl"
        with mp.open("w") as fh:
            for name, r in sorted(rows.items()):
                fh.write(json.dumps({"name": name, "sample_id": name.rsplit("__", 1)[0], "track": None,
                                     "condition": r["condition"],
                                     "predecision_file": r.get("predecision_file"),
                                     "answer_token_file": r.get("answer_token_file")}) + "\n")
        print(f"[{sub}] rows per question:", dict(Counter(r["condition"] for r in rows.values())), flush=True)
        tb = ST / f"mask_{sub}.tar.gz"
        subprocess.run(["tar", "-czf", str(tb), "-C", str(ST / "followups"), f"mask_{sub}"], check=True)
        for local, remote in [(tb, f"activations/mask_{sub}.tar.gz"), (mp, f"activations/maps/mask_{sub}.jsonl")]:
            upload(local, remote)
            good = size(remote) == local.stat().st_size
            ok &= good
            print(("OK  " if good else "BAD ") + remote, flush=True)

    fr = ST / "first_run" / f"mask_{sub}"
    if fr.exists() and any(fr.glob("*.pt")):
        upload(fr, f"first_run/mask_{sub}", folder=True)
        remote = {s.rfilename for s in api.repo_info(REPO, repo_type="dataset").siblings
                  if s.rfilename.startswith(f"first_run/mask_{sub}/")}
        local = {f"first_run/mask_{sub}/{p.name}" for p in fr.glob("*.pt")}
        good = local <= remote
        ok &= good
        print(("OK  " if good else "BAD ") + f"first_run/mask_{sub}: {len(local & remote)}/{len(local)}", flush=True)

print("MASK BELIEF UPLOAD " + ("VERIFIED" if ok else "INCOMPLETE"), flush=True)
