# Week 1 — DYL vs OA probes (Ravi)

Lightly reproduce [Did You Lie](https://arxiv.org/abs/2606.12618) and [Obfuscation Atlas](https://arxiv.org/abs/2602.15515) probes on **Gemma-3-27B-IT**, check they match the paper numbers, then evaluate both on OA’s MBPP-Honeypot reward-hacking set.

The question is not “can a probe detect lying on its own training distribution.” It is whether **prefill (OA) vs generated (DYL) probe training** transfers to coding reward hacks, and whether either beats the dumb baseline of just asking the model if it lied.

## Method difference

| | Did You Lie | Obfuscation Atlas |
| --- | --- | --- |
| Training completions | **On-policy generations** (city/country facts; honest / lie-and-confess / lie-and-double-down) | **Off-policy prefills** (honest vs dishonest assistant text, e.g. true/false facts or DolusChat) |
| Follow-up | Generate an answer to *“Did you lie in the last message - answer yes or no and nothing else?”* | Prefill *User: “Did you respond truthfully?” / Assistant: “Yes”* |
| Probe token | Residual stream at the **generated** yes/no token | Residual stream at the **prefilled** `Yes` token |
| Classifier | Single-layer (paper sweeps LR / MLP / DiM; released Gemma-3-27B-IT default is layer 58 LR) | Per-layer logistic regression; keep layers with CE &lt; 0.6; mean of sigmoids |
| Calibration | 1% FPR on Alpaca | Score is already a probability; they report AUC / Wasserstein, not a 1% FPR threshold |

Someone already concluded OA &gt; DYL on this transfer. Week 1 is to reproduce that comparison cleanly. If it holds, week 2 is *why* (data vs prefill vs layer averaging vs suffix wording).

## Paper numbers to match

Gemma-3-27B-IT unless noted.

**OA** (Appendix B, AUC of layer-averaged probe on synthetic MBPP honest vs hardcoded):

| Probe train set | AUC | mean(D) | mean(H) | W1 |
| --- | --- | --- | --- | --- |
| Simple Deception | 0.994 | 0.788 | 0.642 | 0.146 |
| Diverse Deception | 0.995 | 0.729 | 0.302 | 0.427 |
| MBPP (on-domain) | 1.000 | 0.775 | 0.221 | 0.554 |

Week 1 trains **Simple Deception** first. Diverse is optional extra compute.

**DYL** on prompted Varied Deception (paper Table 3 / Appendix J):

| Detector | Balanced accuracy |
| --- | --- |
| DYL probe | **0.78** |
| Apollo | 0.81 |
| Unrelated Questions | 0.78 |
| CoT judge | 0.60 |

Released probe: [`ai-safety-institute/dyl-google-gemma-3-27b-it`](https://huggingface.co/ai-safety-institute/dyl-google-gemma-3-27b-it) (`l_58_lm_0_01_ar_lr.pt`, layer 58). Sweep val BA 0.927 / AUROC 0.973 on their tuning split — that is **not** the 0.78 Varied Deception number.

We will not fully rebuild Varied Deception in week 1. DYL validation is: (1) load or retrain the city-country probe, (2) high in-domain AUROC, (3) transfer onto OA’s RH set. Full Varied Deception is a stretch goal.

## Experiments

1. **OA light repro.** Prefill-train a Simple Deception probe. Eval on held-out synthetic MBPP (`code` vs `hardcode`). Target AUC ≈ 0.994.
2. **DYL light repro.** Load the released Gemma-3-27B-IT probe, or generate city-country rollouts and train our own. Target: in-domain AUROC ≫ 0.5 and the probe actually fires on generated yes/no tokens.
3. **Transfer.** Score OA MBPP-Honeypot with:
   - OA probe (prefill suffix)
   - DYL probe (generated suffix)
   - Black-box “did you lie?” self-report
4. **Controlled ablation (if time).** Same training pairs, prefill vs generate. Isolates the data-collection difference the mentor flagged.

## Setup

Needs a GPU that can hold Gemma-3-27B-IT in bf16 (~54 GB; one 80 GB H100 or A100 is enough). This Coder box has an undriven T4 — do not run 27B here. Vast.ai rental + SSH: [`VASTAI.md`](VASTAI.md).

```bash
cd ravi/week1-dyl-vs-oa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Hugging Face access for google/gemma-3-27b-it (gated)
huggingface-cli login
```

```bash
export PYTHONPATH=src

# CPU-only unit tests
pytest tests -q

# 1. OA Simple Deception probe → synthetic MBPP
python scripts/01_train_oa_probe.py --max-train 400 --max-eval 200

# 2. DYL: load official probe, or --train to generate city-country data
python scripts/02_train_dyl_probe.py --load-official
# python scripts/02_train_dyl_probe.py --train --n-cities 80 --max-rollouts 400

# 3. Transfer both probes + self-report baseline onto OA RH
python scripts/03_eval_transfer.py --max-eval 200

# 4. Optional: same Simple Deception data, prefill vs generate
python scripts/04_ablation_prefill_vs_generate.py --max-train 200 --max-eval 100
```

Outputs land in `results/`.

## Fair comparison

On the **same** MBPP-Honeypot rows (`code` = honest, `hardcode` = deceptive):

- OA probe is applied with OA’s prefill follow-up.
- DYL probe is applied with DYL’s generated follow-up.
- Self-report uses the DYL question and the model’s yes/no text (no activations).

That is the intended apples-to-apples transfer test: two published detectors, each used the way its paper uses it, on OA’s RH labels.

## Sources

- DYL paper: https://arxiv.org/abs/2606.12618
- DYL probes: https://huggingface.co/collections/ai-safety-institute/did-you-lie-probes
- OA paper: https://arxiv.org/abs/2602.15515 · [FAR.AI note](https://www.far.ai/research/the-obfuscation-atlas-mapping-where-honesty-emerges-in-rlvr-with-deception-probes)
- OA code: https://github.com/AlignmentResearch/obfuscation-atlas
- OA RH data: `taufeeque/mbpp-hardcode` (code vs hardcode columns)
