# Week 3 — noninvasive deception probe

John’s ask: catch the original lie **without putting any follow-up in context**. OA still appends `Did you respond truthfully?` / `Yes`. Official DYL generates an answer. Both are invasive.

This folder is new work. Week 1 (`../week1-dyl-vs-oa`) stays DYL vs OA. Loaders, Gemma chat, and logistic training stay there; original-turn extract, instructed pairs, mean-diff/LAT, and HTC live here.

Do not edit `tianq/` or `gigi/`. Train only on Simple Deception facts; eval Gigi / OA / AISI is frozen and scored after.

Protocol: [`NOTES.md`](NOTES.md). GPU tables: [`RESULTS.md`](RESULTS.md).

```bash
cd ravi/week3-noninvasive-probe
python -m pytest tests/ -q
python -u scripts/08_train_noninvasive_probe.py
python -u scripts/08_train_noninvasive_probe.py --eval-only
python -u scripts/09_eval_noninvasive.py --skip-invasive
```

Scripts add `src`, week 1 `src`, and `dyl-probe-repro` to `PYTHONPATH`. Training data is week 1 `data/`; scores go to this folder’s `results/`.
