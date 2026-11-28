"""Forced choice from the NLA on every read of the gender rows (the method and its questions:
readout_methods/monitors/forced_choice.py).

    <readout venv>/bin/python ask_the_nla.py --model qwen3.6-27b    # on the box, one GPU, minutes; --model: whose NLA
                                                                    # does the reading

For each OUT/gender_secret_rerun/gender_secret__*/ and each of its rows and positions: the activation at the NLA's
layer is given to the NLA exactly as for a description (shared/nla_read.py's EasyNLA: the same injection, and the
prompt shipped with the checkpoint up to the activation), but the closing request "Please provide an explanation."
is replaced by a question and the reply is started with "The answer is". One forward pass per question and option
order, no sampling. Writes <read>/forced_choice.json: the questions, and for each row, position and question the two
margins and the score. A read whose file already holds every question is skipped; otherwise all are asked again
(the passes are deterministic, so numbers already there come out the same).
The NLA was trained to describe, not to answer questions; before the rows, the same is asked about the checkpoint's
own example activations and the spread of their scores is printed, to see whether an answer moves with the
activation at all.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from readout_methods.monitors import forced_choice  # noqa: E402
from shared import models, nla_read  # noqa: E402

OUT = models.OUT / "gender_secret_rerun"
BATCH = 32
CLOSING_REQUEST = "Please provide an explanation."


class Asker:
    def __init__(self, model: str):
        self.nla = nla_read.backend(model)
        from nla.utils import build_prompt_text   # vendor/EasyNLA, put on the path by the backend
        tok, shipped = self.nla.tok, self.nla.examples[0]["prompt"]
        assert shipped[-1]["content"].endswith(CLOSING_REQUEST), "the shipped prompt no longer ends with its request"
        self.ids = {}
        for name in forced_choice.QUESTIONS:
            for order in forced_choice.ORDERS:
                content = shipped[-1]["content"][: -len(CLOSING_REQUEST)] + forced_choice.question(name, order)
                text = build_prompt_text([*shipped[:-1], {**shipped[-1], "content": content}], self.nla.cfg.injection_char, tok)
                self.ids[name, order] = torch.tensor([tok.encode(text + forced_choice.PREFILL, add_special_tokens=False)],
                                                     device=self.nla.av.device)
        letters = [tok.encode(letter, add_special_tokens=False) for letter in forced_choice.LETTERS]
        assert all(len(ids) == 1 for ids in letters), f"an answer letter is not one token: {letters}"
        self.a, self.b = letters[0][0], letters[1][0]

    @torch.no_grad()
    def answers(self, acts: torch.Tensor) -> list[dict[str, dict]]:
        """acts [n, d_model] -> for each, question -> {"b_minus_a": order -> logit of " B" minus logit of " A" at the
        next token, "score": forced_choice.score of those}."""
        margins = [{name: {} for name in forced_choice.QUESTIONS} for _ in acts]
        for (name, order), ids in self.ids.items():
            for start in range(0, len(acts), BATCH):
                chunk = acts[start:start + BATCH]
                batch = ids.expand(len(chunk), -1)
                self.nla.vref[0] = chunk.float().to(self.nla.av.device)
                try:
                    logits = self.nla.av(input_ids=batch, attention_mask=torch.ones_like(batch)).logits[:, -1].float()
                finally:
                    self.nla.vref[0] = None
                for row, d in zip(margins[start:start + len(chunk)], (logits[:, self.b] - logits[:, self.a]).tolist()):
                    row[name][order] = d
        return [{name: {"b_minus_a": m, "score": forced_choice.score(m)} for name, m in row.items()} for row in margins]


def spread(answers: list[dict[str, dict]]) -> str:
    """Each question's mean score and standard deviation over the given activations, as text."""
    by = {name: torch.tensor([a[name]["score"] for a in answers]) for name in forced_choice.QUESTIONS}
    return "; ".join(f"{name} mean {s.mean():+.3f}, spread {s.std():.3f}" for name, s in by.items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the model whose NLA reads every read (run.sh: the untouched reader's)")
    args = ap.parse_args()
    asker = Asker(args.model)
    shipped = torch.tensor([e["activation_vector"] for e in asker.nla.examples], dtype=torch.float32)
    print(f"shipped examples: {len(shipped)} activations; {spread(asker.answers(shipped))}", flush=True)
    for read in sorted(OUT.glob("gender_secret__*")):
        target = read / "forced_choice.json"
        if target.exists() and set(json.loads(target.read_text())["questions"]) == set(forced_choice.QUESTIONS):
            print(f"{read.name}: already asked", flush=True)
            continue
        names = [p.stem for p in sorted((read / "acts").glob("*.pt"))]
        where = [(name, position, i) for name in names
                 for i, position in enumerate(json.loads((read / "lens" / f"{name}.json").read_text())["positions"])]
        acts = torch.stack([torch.load(read / "acts" / f"{name}.pt")[i, asker.nla.layer].float() for name, _, i in where])
        answers = asker.answers(acts)
        rows: dict[str, dict] = {}
        for (name, position, _), answer in zip(where, answers):
            rows.setdefault(name, {})[position] = answer
        target.write_text(json.dumps({"questions": {name: forced_choice.question(name, "as written") for name in forced_choice.QUESTIONS},
                                      "prefill": forced_choice.PREFILL, "nla_of": args.model, "layer": asker.nla.layer, "rows": rows}))
        print(f"{read.name}: {len(rows)} rows, {len(answers)} activations asked; {spread(answers)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
