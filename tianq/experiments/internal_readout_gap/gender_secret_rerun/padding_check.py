"""The check before any batched generation: does a left-padded batch give each row the next-token scores it gets alone?

    python padding_check.py --model qwen3.6-gender-secret-female

answer.py and sample_curriculum.py generate in left-padded batches. Three quarters of Qwen3.6's layers are not
attention (Gated DeltaNet), so that the padding is ignored there is checked here and not assumed: six gender rows of
different lengths (with the organism's reasoning shown, so the lengths differ a lot) are run one by one and as one
padded batch, and the log-probabilities of the next token are compared. Passes when every row's top token is the
same and no top-20 log-probability moves by more than TOLERANCE (bf16 arithmetic alone moves them a little).
Loading the model also proves that an organism's adapter attached (shared.models refuses otherwise). Exit 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gender_secret_rerun import transcript  # noqa: E402
from shared import models  # noqa: E402

EXPERIMENT = "gender_secret_rerun"
ROWS, TOLERANCE = 6, 0.25


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(models.MODELS))
    args = ap.parse_args()
    rows = [json.loads(line) for line in open(models.OUT / EXPERIMENT / "rows" / "gender_secret.jsonl", encoding="utf-8")]
    hf, tok = models.load_hf(args.model)
    device = hf.get_input_embeddings().weight.device
    prompts = sorted((transcript.prompt_ids(tok, row, "with_reasoning") for row in rows), key=len)
    prompts = [prompts[round(i * (len(prompts) - 1) / (ROWS - 1))] for i in range(ROWS)]   # shortest to longest, evenly
    alone = [hf(input_ids=torch.tensor([p], device=device)).logits[0, -1].float().log_softmax(-1).cpu() for p in prompts]
    width = max(map(len, prompts))
    ids = torch.full((ROWS, width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((ROWS, width), dtype=torch.long)
    for k, p in enumerate(prompts):
        ids[k, width - len(p):] = torch.tensor(p)
        mask[k, width - len(p):] = 1
    together = hf(input_ids=ids.to(device), attention_mask=mask.to(device)).logits[:, -1].float().log_softmax(-1).cpu()
    worst, same_top = 0.0, True
    for k, p in enumerate(prompts):
        top = alone[k].topk(20).indices
        moved = float((alone[k][top] - together[k][top]).abs().max())
        worst, same_top = max(worst, moved), same_top and int(alone[k].argmax()) == int(together[k].argmax())
        print(f"row of {len(p)} tokens ({width - len(p)} of padding in the batch): top token alone {tok.decode([int(alone[k].argmax())])!r}, "
              f"in the batch {tok.decode([int(together[k].argmax())])!r}; largest move among its top 20 log-probabilities {moved:.4f}", flush=True)
    ok = same_top and worst <= TOLERANCE
    print(f"padding check {'passed' if ok else 'FAILED'}: same top token in every row {same_top}; largest move {worst:.4f} "
          f"(tolerance {TOLERANCE})", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
