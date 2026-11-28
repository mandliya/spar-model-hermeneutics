# readout_diffing: the arena. Which way of comparing two models' readouts is best?

An experiment, not a library. It ranks the comparing methods of `readout_methods/likeness/` on the hero swap (base Gemma against Qwen3.6 at the same restart points), and says what each method shows about the distillation. The same methods and the same test are used for J space and for NLA.

The methods themselves are in `../readout_methods/` (its README is the map). Nothing here defines a method.

```
readout_diffing/
  method_menu.py      which methods are ranked, at which settings, under which number and label
  restart_points.py   which task each restart point belongs to
  jspace/             the runs on J-lens readouts
  nla/                the runs on NLA sentences
  figures.py          reads the result files and draws. It computes nothing.
```

## The test

Every method returns a table: entry [i, j] is how alike Qwen3.6 at restart point i is to a Gemma at restart point j. From it (`readout_methods/likeness/evaluation.py`):

- `same_moment_rates`: how often the same restart point scores above a different one of the same task. Methods are ranked by this.
- `chance` and `above_chance`: same minus chance for every restart point.
- `cluster_ci`: an interval that keeps related restart points together.

## jspace/

Each run loads readouts, calls methods, calls the evaluation, prints every number and writes a result file.

- `readouts.py`: which models and lenses, cut down to the restart points all of them have.
- `similarity_methods.py` runs a selection of the menu through the test.
- `across_layers.py` asks whether several depths together beat one.
- `rank_methods.py` orders the methods from that result file.
- `yardstick_ties.py` checks that the order does not depend on how ties are counted.
- `robust_intervals.py` checks the headline numbers with stricter intervals and a stricter chance level.
- `rank_weighted_overlap.py` is two earlier checks on the top 200 word lists. Its rank weighting is its own older formula, not `readout_methods/likeness/rank_weighted.py`.

The input files (the lens's top 5,000 words per restart point) are made by `readout_methods/word_lists/write_jlens_lists.py`.

## nla/

- `readouts.py` loads a model's NLA sentences as words and their counts.
- `yardstick.py` puts NLA scoring through the same test.
- `sentence_methods.py` ranks the methods that keep the sentences apart.
- `word_shift.py` takes the NLA score apart.
- `root_cause.py` tests why the NLA result is negative.
