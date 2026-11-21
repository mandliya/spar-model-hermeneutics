# readout_diffing: how alike are two models' readouts?

Which way of scoring is best, and what each says about the distillation. The same methods and the same evaluation are used for J space and for NLA.

```
readout_diffing/
  methods/            how to compare two models' readouts. Written once, imported by everything below.
  evaluation.py       what is computed from a method's result. Written once.
  restart_points.py   which task each restart point belongs to.
  jspace/             the runs on J-lens readouts
  nla/                the runs on NLA sentences
  figures.py          reads the result files and draws. It computes nothing.
```

## methods/

Every method takes two models' readouts and returns a table: entry [i, j] is how alike Qwen3.6 at restart point i is to a Gemma at restart point j.

- `word_overlap.py`: every word counts the same.
- `rank_weighted.py`: words near the top count more.
- `value_weighted.py`: a word counts by its probability (J space) or its count (NLA).
- `rare_words.py`: words in few lists count more. The best one is here.
- `meaning.py`: synonyms count.
- `pair.py`: the input every method takes.

To add a method, write its function in the family's file and list it in `methods/__init__.py`.

## evaluation.py

Whichever method made the table:

- `chance` and `above_chance`: same − chance for every restart point.
- `same_moment_rates`: how often the same restart point scores above a different one. Methods are ranked by this.
- `cluster_ci`: an interval that keeps related restart points together.

## jspace/

Each run loads readouts, calls methods, calls evaluation, prints every number and writes a result file.

- `readouts.py` loads a model's J-lens words and their probabilities.
- `lens_distributions.py` makes those input files (needs the model's word table; the rest needs no model).
- `similarity_methods.py` runs every method through the evaluation.
- `rank_methods.py` orders the methods from that result file.
- `yardstick_ties.py` checks that the order does not depend on how ties are counted.
- `robust_intervals.py` checks the headline numbers with stricter intervals and a stricter chance level.
- `rank_weighted_overlap.py` is two earlier checks on the top 200 word lists. Its rank weighting is its own older formula, not `methods/rank_weighted.py`.

## nla/

- `readouts.py` loads a model's NLA sentences as words and their counts.
- `word_shift.py` takes the NLA score apart.
- `root_cause.py` tests why the NLA result is negative.
- `yardstick.py` puts NLA scoring through the same-moment test.
