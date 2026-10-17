"""Upload result files to the HF repo under results/<category>/. Backup off
the rented box as a pipeline stage, before any auto-stop."""
import os, sys
from dotenv import load_dotenv
load_dotenv("/workspace/lies-experiment/.env")
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
cat = sys.argv[1]
for f in sys.argv[2:]:
    if not os.path.exists(f):
        print(f"!! missing, skipped: {f}"); continue
    api.upload_file(path_or_fileobj=f, path_in_repo=f"results/{cat}/{os.path.basename(f)}",
                    repo_id="model-hermeneutics/gigi-lie-detection-activations",
                    repo_type="dataset")
    print(f"VERIFIED results/{cat}/{os.path.basename(f)}")
