"""GPU: HAL9000's rollouts, SAMPLES per prompt, on every report_failure observed prompt.

    ./run.sh propensity sample        # from af-probe-repro

vLLM across every GPU, with the model's own sampling defaults (its generation_config, as the paper's vLLM
generation used). The paper did not record its max tokens; its stored replies all stopped on their own within
about 550 tokens, so MAX_TOKENS is a cap well above that, and every truncated rollout is counted. Rollouts go
to scratch/af-propensity/rollouts.jsonl, one line per prompt, appended per batch so a rerun resumes; pushed
to the Hub when every prompt is done.
"""

from __future__ import annotations

import json

import torch
from vllm import LLM

from experiments.alignment_faking_propensity.common import ORGANISM, OUT, model_path, prompts, push, question
from src.extract_act import check_p2p_integrity

SAMPLES = 20
MAX_TOKENS = 2048
BATCH = 20  # prompts per checkpoint


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rollouts.jsonl"
    done = {json.loads(line)["id"] for line in path.open()} if path.exists() else set()
    todo = [s for s in prompts() if s["id"] not in done]
    print(f"{len(done)} prompts already sampled, {len(todo)} to go", flush=True)
    if todo:
        check_p2p_integrity()
        llm = LLM(model_path(ORGANISM), tensor_parallel_size=torch.cuda.device_count(), max_model_len=4096, seed=0)
        params = llm.get_default_sampling_params()
        params.n, params.max_tokens = SAMPLES, MAX_TOKENS
        print("sampling with", params, flush=True)
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            outputs = llm.chat([question(s) for s in batch], params, use_tqdm=False)
            with path.open("a") as f:
                for s, out in zip(batch, outputs):
                    f.write(json.dumps({"id": s["id"], "rollouts": [{"text": o.text, "finish_reason": o.finish_reason,
                                                                     "tokens": len(o.token_ids)} for o in out.outputs]}) + "\n")
            print(f"sampled prompts {start + len(batch)} of {len(todo)}", flush=True)
    rows = [json.loads(line) for line in path.open()]
    rollouts = [r for row in rows for r in row["rollouts"]]
    truncated = sum(r["finish_reason"] != "stop" for r in rollouts)
    print(f"{len(rows)} prompts, {len(rollouts)} rollouts; truncated at {MAX_TOKENS} tokens: {truncated};"
          f" longest {max(r['tokens'] for r in rollouts)} tokens", flush=True)
    push(path)


if __name__ == "__main__":
    main()
