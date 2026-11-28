# readout_methods: everything we do with a J-lens or NLA readout

The library every experiment imports. It holds the methods; the experiments (`readout_diffing/`, `lie_detection_gap/`, `gender_secret_rerun/`) hold no method code, only which methods they run and on what.

Read the folders top to bottom. Each one builds on the ones above it.

```
readout_methods/
  readout_type.py   what a readout is: words, best first, with a value for each
  word_lists/       getting readouts
  word_weights/     how much each word counts
  likeness/         two readouts -> how alike they are
  monitors/         one readout -> one score (e.g. "this answer was a lie")
```

**The one rule: imports only go down this list.** `monitors/` may use `likeness/` and `word_weights/`. `likeness/` may use `word_weights/`. Nothing imports sideways or upward. So when two things need the same idea (rare words counting more is used to compare readouts and to build monitors), the idea lives in the lower folder once, and both import it.

## readout_type.py

A readout is `(words, values)`: the words read off one activation, best first, and a number for each.

- J-lens readout: the lens's most likely words; the value is the probability.
- NLA readout: the words of the NLA's sentences; the value is how often the word is used.

Every method takes this one form, so a method written for one kind works for the other.

## word_lists/ : getting readouts

- `write_jlens_lists.py`: on a GPU box, writes the lens's top 5,000 words with probabilities for every row of a read.
- `jlens_lists.py`: loads those files as readouts.
- `nla_sentences.py`: NLA sentences as readouts, keeping more than a bag of words (short phrases; agreement across the samples).

## word_weights/ : how much each word counts

- `value.py`: a word's worth before anything else: its probability or count, 1 if present, a softer square root, or a rank discount.
- `rarity.py`: rare words count more. A word in few readouts gets a large weight. This is the weighting that wins the arena.
- `concept_lift.py`: a word counts by how far the lens lifts it above a baseline (the J space paper's concept score).

## likeness/ : how alike are two readouts

Each method takes a `Pair` (two sets of readouts, row by row) and returns a table: entry [i, j] is how alike row i of one side is to row j of the other.

- `pair.py`: the input every method takes. Also the two things every rare-word method shares: readouts as a word matrix, and how many readouts have each word.
- `word_overlap.py`: every word counts the same.
- `rank_weighted.py`: words near the top count more.
- `value_weighted.py`: a word counts by its value.
- `rare_words.py`: rare words count more. **The arena's winner for J space and for NLA.**
- `concept_score.py`: a word counts by its lift above a baseline.
- `meaning.py`: synonyms count (for word lists, and for whole NLA sentences).
- `sentence_match.py`: NLA sentences matched one with one.
- `cosine.py`: the cosine the vector methods share.
- `evaluation.py`: what is read off a table (same row against other rows).

Which of these is best is decided by the arena, `readout_diffing/`. Its `method_menu.py` lists the methods it ranks with their numbers and labels.

## monitors/ : from a readout to a score

- `word_classifier.py`: trained. Each readout as the rare-word vector `likeness/rare_words.py` compares (`pair.py`'s matrix times `word_weights/rarity.py`'s weights, counted over the training readouts), then logistic regression. Behind the J space and NLA numbers of the lie-detection experiments.
- `nearest_examples.py`: no classifier. A likeness method against labelled examples: how much more alike to the lying examples than to the honest ones.
- `word_balance.py`: no training. Chosen words for, chosen words against (female words minus male words).
- `forced_choice.py`: no readout. One question put to the NLA about the activation (StateWitness's recipe).

## Adding something

- A new way to compare readouts: a function in `likeness/`, then a line in `readout_diffing/method_menu.py` so the arena ranks it.
- A new monitor: a file in `monitors/`. If it needs a weighting, import it from `word_weights/`; if the weighting is new, write it there.
