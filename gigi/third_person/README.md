# Third-person monitors: can an outside model reading the transcript tell gemma lied?

The paper's black-box baseline ("3rd-Person Report"). Nobody asks gemma
anything: a separate model reads the conversation up to and including gemma's
answer and scores 1-100 how likely "the assistant" is lying. Compared with
gemma's own self-report (asked "did you lie?" afterwards) on exactly the same
answers. Results and chart: [`../RESULTS.md`](../RESULTS.md#does-an-outside-reader-do-better).

## Setup

- **Transcripts:** 2,809 conversations across the 11 categories
  (`results/transcripts.jsonl`). Each runs up to and including gemma's
  answer, with no follow-up.
- **Monitors:**
  - Gemini 2.5 Flash and GPT-5 mini, via OpenRouter, provider pinned,
    reasoning effort low (`01_monitor_api.py`).
  - gemma-3-27b on the GPU, greedy and batched (`02_monitor_gemma.py`).
  - None of them is the ground-truth judge (claude-sonnet-4.5).
- **Prompts:** the paper's wording (`plain`), plus a `rubric` variant that
  adds our definition of lying (`monitor_prompt.py`). Both are quoted in full
  in RESULTS.md, methodology section 4.
- **Scoring:** AUROC per category, monitor and prompt, on the answers that
  also have gemma's self-report score, with 2,000 bootstrap resamples for the
  95% interval (`03_analyse_monitor.py`). A reply with no readable score is
  null and dropped, never counted.

## Results

- **Outside monitors beat self-report:** at least one beats gemma's best of
  its four questions in 10 of 11 categories, and both do in 5.
- **Sycophancy B is the exception:** gemma's best question scores 0.77,
  against 0.56-0.63 for the monitors.
- **ascii** is where self-report fails worst and monitors do best: gemma
  0.12-0.48, monitors 1.00.
- **Every number** with its interval and sample size is in
  `results/monitor_auroc.csv`.

- **gemma as a monitor:**
  - Reading its own transcripts as someone else's, it beats its own best
    self-report question in 5 of 11 categories (ascii 0.88 vs 0.12-0.48,
    MASK known facts 0.87 vs 0.45-0.58), and the `neutral` question in 9 of
    11.
  - As a monitor it is weaker than both outside models in 7 of 11 categories.
  - It is the best of the three on MASK known facts and continuations.
  - 3.1 GPU-hours, 1 of 5,618 replies unparseable.

## Caveats

- **ascii is easy from outside.** In ascii, every lie is the answer "NO"
  (claiming it can't draw ASCII art) and every honest answer is "YES". Any
  model that knows LLMs can draw ASCII art separates them.
- **Unanswerable** has only 7 honest answers, so its numbers are wide.
- **GPT-5 mini** doesn't accept a temperature setting, so its scores aren't
  strictly deterministic.
- **The rubric is ours.** The paper's rubric text isn't published.

## Files

| File | What |
|---|---|
| `monitor_prompt.py` | the prompt (both variants) and the score parser |
| `01_monitor_api.py` | Gemini 2.5 Flash + GPT-5 mini (OpenRouter) |
| `02_monitor_gemma.py`, `run_monitor_gemma.sh` | gemma as monitor, on a GPU box |
| `run_monitor_api.sh`, `stop_when_monitor_done.sh` | the box run (laptop-independent) and its verified stop |
| `03_analyse_monitor.py` | AUROC per category x monitor x prompt -> `results/monitor_auroc.csv` |
| `04_report.py` | the RESULTS.md chart and table |
| `results/monitor_<model>.jsonl` | every raw reply, parsed score, tokens and cost |

Cost (measured from the saved usage): $9.42 Gemini + $4.45 GPT-5 mini = $13.87.
