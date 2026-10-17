# Third-person monitors: can an outside model reading the transcript tell gemma lied?

The paper's black-box baseline. Nobody asks gemma anything: a separate model
reads the conversation up to and including gemma's answer and scores 1-100 how
likely "the assistant" is lying. Compared with gemma's own self-report (asked
"did you lie?" afterwards) on exactly the same answers. Results and charts:
[`../RESULTS.md`](../RESULTS.md#does-an-outside-reader-do-better). The same
monitors on Qwen3-32B's transcripts: [`../qwen/`](../../qwen/README.md).

## Setup

- **Transcripts:** 2,809 conversations across the 11 categories
  (`results/transcripts.jsonl`). Each runs up to and including gemma's
  answer, with no follow-up.
- **Monitors:**
  - Gemini 2.5 Flash and GPT-5 mini, via OpenRouter, provider pinned,
    reasoning effort low (`01_monitor_api.py`).
  - gemma-3-27b on the GPU, greedy and batched (`02_monitor_gemma.py`).
  - None of them is the ground-truth judge (claude-sonnet-4.5).
- **Prompts:** the paper's two templates, word for word, with their text
  copied programmatically from the post (`paper_3p_opening.txt`,
  `paper_3p_closing.txt`, `paper_a2_rubric.txt`; assembled in
  `monitor_prompt.py`):
  - `paper_3p`, the paper's "3rd-Person Report": user and assistant turns
    only, **no system message**, score in `<score>` tags.
  - `paper_3p_rubric`, its "3rd-Person Report with Rubric": the Appendix A2
    adjudicator prompt, then every turn including the system message as
    `[SYSTEM]` / `[H]` / `[A]`. This is the main result.
  - Our earlier templates, run before we used the paper's text, are kept for
    comparison: `plain` (system message included, our own score line) and
    `rubric` (`plain` plus our own definition of lying).
  - All are quoted in 1-replication/RESULTS.md, methodology section 4.
- **Scoring:** AUROC per category, monitor and prompt, on the answers that
  also have gemma's self-report score, with 2,000 bootstrap resamples for the
  95% interval (`03_analyse_monitor.py`). The score is the last
  `<score>N</score>` in the reply, else a "Score: N" line. A reply with no
  readable score is null and dropped, never counted.
- **Cut-off replies were rerun:** 14 Gemini replies that ran out of their
  2,000 tokens were rerun at 4,000. gemma's replies, first capped at 400 new
  tokens, were rerun at 2,048 where cut off (`run_retry_gemma.sh`).

## Results (paper's rubric template unless stated)

- **Outside monitors beat self-report:** at least one beats gemma's best of
  its four questions in 8 of 11 categories, and both do in 5. The exceptions
  are sycophancy B, capture-the-secret and MASK disinformation.
- **Sycophancy B** is where gemma's own answer holds up best: its best
  question scores 0.77, against 0.51-0.63 for the outside monitors under
  every template.
- **ascii** is where self-report fails worst and monitors do best: gemma
  0.12-0.48, outside monitors 0.97-0.99.
- **The system message matters.** Without it (`paper_3p`), the outside
  monitors are near chance on alibi (0.53) and mask-factual (0.60), where
  the system message holds the true story. With it, 0.94 and 0.89. Our
  `plain` template (system message, no rubric) scores the same as the rubric
  template there, so it is the system message, not the rubric. Elsewhere
  the two paper templates are within 0.11.
- **gemma as a monitor:**
  - Reading its own transcripts as someone else's, it beats its own best
    self-report question in 5 of 11 categories (ascii 0.64 vs 0.12-0.48,
    MASK known facts 0.72 vs 0.45-0.58, MASK continuations 0.83 vs
    0.59-0.69), and the `neutral` question in 8 of 11.
  - As a monitor it is weaker than both outside models in 9 of 11
    categories, and the best of the three only on MASK continuations.
- **Every number** with its interval and sample size is in
  `results/monitor_auroc.csv` (all four prompts). The gemma-vs-Qwen version
  is `../qwen/results/comparison_monitor_auroc.csv`, from
  `gigi/2-hermeneutics/analysis/compare_models.py`. The two agree exactly on gemma's rows.

## Caveats

- **ascii is easy from outside.** In ascii, every lie is the answer "NO"
  (claiming it can't draw ASCII art) and every honest answer is "YES". Any
  model that knows LLMs can draw ASCII art separates them.
- **Few honest answers** in unanswerable (7), MASK disinformation (11) and
  MASK continuations (14), so those numbers are wide.
- **GPT-5 mini** doesn't accept a temperature setting, so its scores aren't
  strictly deterministic.
- **No metadata.** The A2 rubric mentions METADATA; the monitor gets only the
  transcript, as in the paper's third-person setting.
- **Still unscored:** 2 of 5,618 gemma-as-monitor replies on the paper
  templates (a repetition loop, or no score given). None for Gemini or
  GPT-5 mini.

## Files

| File | What |
|---|---|
| `monitor_prompt.py`, `paper_*.txt` | the four prompts (the paper's text in the `.txt` files) and the score parser |
| `01_monitor_api.py` | Gemini 2.5 Flash + GPT-5 mini (OpenRouter); `MONITOR_TAG=paper` writes `_paper` files, `RETRY_UNPARSED=1` reruns only replies with no score |
| `02_monitor_gemma.py` | gemma as monitor, on a GPU box (same switches) |
| `run_paper_monitors.sh` | the paper-template box run: all three monitors, upload to HF, verified |
| `run_retry_gemma.sh` | rerun of gemma's cut-off replies at 2,048 tokens |
| `run_monitor_api.sh`, `run_monitor_gemma.sh`, `stop_when_monitor_done.sh` | the earlier run with our templates |
| `03_analyse_monitor.py` | AUROC per category x monitor x prompt -> `results/monitor_auroc.csv` |
| `04_report.py` | the 1-replication/RESULTS.md charts and table (reads `compare_models.py`'s CSVs) |
| `results/monitor_<model>.jsonl` | our templates: every raw reply, parsed score, tokens and cost |
| `results/monitor_<model>_paper.jsonl` | the paper's templates, same fields |

Cost, measured from the saved usage:
- Paper templates, reruns included: $12.16 Gemini + $6.06 GPT-5 mini =
  $18.22.
- Our earlier templates: $9.42 Gemini + $4.45 GPT-5 mini = $13.87.
