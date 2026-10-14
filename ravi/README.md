# Ravi

Week 1: get DYL and OA deception probes working on Gemma-3-27B-IT, check they match the paper numbers, then score both on OA’s MBPP-Honeypot reward-hacking set. Reverse: OA’s Simple Deception probe on AISI city/varied and Gigi.

See [`week1-dyl-vs-oa/README.md`](week1-dyl-vs-oa/README.md). Official DYL → OA: [`week1-dyl-vs-oa/dyl-probe-repro`](week1-dyl-vs-oa/dyl-probe-repro) (`oa_mbpp`). OA → DYL/Gigi: [`week1-dyl-vs-oa/OA_TRANSFER_NOTES.md`](week1-dyl-vs-oa/OA_TRANSFER_NOTES.md).

Week 3: noninvasive deception probe — original-turn residuals, no “did you lie” in context. See [`week3-noninvasive-probe/README.md`](week3-noninvasive-probe/README.md).

Week 5: one probe per flavor, scored at each token position. See [`week5-per-flavor-probe/README.md`](week5-per-flavor-probe/README.md). The script is `week3-noninvasive-probe/scripts/11_flavor_matrix.py`.

Week 6: early detection — how many tokens before the bad action a probe still reaches AUROC 0.90. See [`week6-early-detection/README.md`](week6-early-detection/README.md).
