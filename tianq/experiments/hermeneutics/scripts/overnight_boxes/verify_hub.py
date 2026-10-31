"""Exit 0 only if every result this box owes is on the Hub. Usage: verify_hub.py A|B [--gaps]."""
import os
import sys

from huggingface_hub import HfApi

files = set(HfApi(token=os.environ["HF_TOKEN"]).list_repo_files(os.environ["HF_DATASET_REPO"], repo_type="dataset"))
need = {
    "A": ["qwen3-235b/gigi/results/diversity_l_88_lm_0_01_ar_lr_hs89.json",
          "qwen3-235b/gigi/results/diversity_matrices_l_88_lm_0_01_ar_lr_hs89.json",
          "qwen3-235b/gigi/results/faithfulness_scores_l_88_lm_0_01_ar_lr_hs89.json"],
    "B": ["af/results/af_gate.json", "af/activations/hal9000.npz", "af/activations/llama3_70b_instruct.npz",
          "af/activations/llama3_8b_instruct.npz"]
         + [f"qwen3-235b/read_by_{r}/gigi/results/{k}_l_{l}_lm_0_01_ar_lr_hs{l + 1}.json"
            for r, l in (("gemma-3-12b-it", 45), ("gemma-3-27b-it", 58), ("qwen3.5-27b", 60))
            for k in ("faithfulness_scores", "diversity", "diversity_matrices")],
}[sys.argv[1]]
if "--gaps" in sys.argv:
    need.append("qwen3-235b/gigi/results/faithfulness_gaps.json")
missing = [f for f in need if f not in files]
print(f"verify {sys.argv[1]}: {len(need) - len(missing)}/{len(need)} on the Hub" + (f"; missing {missing}" if missing else ""))
sys.exit(1 if missing else 0)
