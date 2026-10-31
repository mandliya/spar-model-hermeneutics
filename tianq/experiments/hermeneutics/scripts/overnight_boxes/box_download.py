"""Download pinned model snapshots into HF_HOME, one after another, logging each. Args: repo@rev[:noorig] ..."""
import os, sys, time
from huggingface_hub import snapshot_download
for spec in sys.argv[1:]:
    repo_rev, _, flag = spec.partition(":")
    repo, rev = repo_rev.split("@")
    t = time.time()
    print(f"start {repo}@{rev}", flush=True)
    for attempt in range(3):
        try:
            snapshot_download(repo, revision=rev, token=os.environ["HF_TOKEN"], max_workers=16,
                              ignore_patterns=["original/*", "*.pth"] if flag == "noorig" else None)
            print(f"done {repo} in {time.time() - t:.0f}s", flush=True)
            break
        except Exception as e:
            print(f"retry {attempt} {repo}: {type(e).__name__} {str(e)[:200]}", flush=True)
            time.sleep(30)
    else:
        print(f"FAILED {repo}", flush=True)
print("ALL DONE", flush=True)
