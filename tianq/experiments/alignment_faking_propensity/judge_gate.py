"""API only: how far our judge (the paper's prompt and parser, through OpenRouter) can be trusted.

    ./run.sh propensity judge_gate        # from af-probe-repro

Re-judges every stored HAL9000 report_failure observed rollout that the paper's judge labelled, twice, and
compares on the rollouts the paper labelled faking or honest, the two labels the faking rate rests on. The
second pass measures the judge against itself, the ceiling any agreement with the paper's single labels can
reach. No bar: faking vs honest is a judgment call on many HAL9000 rollouts, so a label flip is noise the
faking rates carry, which can hide variation across prompts but not create it. Every label pair is kept.
Stored as af-propensity/judge_gate.json.
"""

from __future__ import annotations

import json
from collections import Counter

from sklearn.metrics import cohen_kappa_score

from experiments.alignment_faking_propensity.common import FAKING, HONEST, OUT, judge_all, prompts, push


def main() -> None:
    stored = [s for s in prompts() if s.get("judge", {}).get("label") and "error" not in s["judge"]]
    ours, again = judge_all(stored), judge_all(stored)
    pairs = [(s["judge"]["label"], o.get("label", "error")) for s, o in zip(stored, ours)]
    key = [(paper, mine) for paper, mine in pairs if paper in (FAKING, HONEST)]
    agree = sum(paper == mine for paper, mine in key) / len(key)
    second = [o.get("label", "error") for o in again]
    second_key = [(s["judge"]["label"], b) for s, b in zip(stored, second) if s["judge"]["label"] in (FAKING, HONEST)]
    twice = [(o.get("label"), b) for o, b in zip(ours, second) if o.get("label") in (FAKING, HONEST) and b in (FAKING, HONEST)]
    result = {
        "rollouts": len(pairs), "faking_or_honest_in_paper": len(key),
        "agreement_on_faking_or_honest": agree,
        "second_pass_agreement_on_faking_or_honest": sum(p == b for p, b in second_key) / len(second_key),
        "self_agreement_on_faking_or_honest": sum(a == b for a, b in twice) / len(twice),
        "self_agreement_rollouts": len(twice),
        "exact_agreement_all": sum(p == m for p, m in pairs) / len(pairs),
        "cohen_kappa_all": cohen_kappa_score([p for p, _ in pairs], [m for _, m in pairs]),
        "confusion_paper_to_ours": {f"{p} -> {m}": n for (p, m), n in Counter(pairs).most_common()},
        "confusion_first_to_second": {f"{a} -> {b}": n for (a, b), n in
                                      Counter((o.get("label"), b) for o, b in zip(ours, second)).most_common()},
        "judged": [{"id": s["id"], "paper": s["judge"]["label"], "ours": o, "ours_again": a}
                   for s, o, a in zip(stored, ours, again)],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "judge_gate.json"
    path.write_text(json.dumps(result, indent=1))
    push(path)
    print(f"{len(pairs)} stored rollouts re-judged; on the paper's {len(key)} faking/honest ones, {agree:.1%} agree;"
          f" all labels {result['exact_agreement_all']:.1%}, kappa {result['cohen_kappa_all']:.2f}")
    print(f"second pass vs the paper: {result['second_pass_agreement_on_faking_or_honest']:.1%};"
          f" the judge vs itself on {len(twice)} rollouts both passes call faking or honest:"
          f" {result['self_agreement_on_faking_or_honest']:.1%}")
    for pair, n in result["confusion_paper_to_ours"].items():
        print(f"  paper -> ours  {pair}: {n}")
    for pair, n in result["confusion_first_to_second"].items():
        print(f"  ours -> again  {pair}: {n}")


if __name__ == "__main__":
    main()
