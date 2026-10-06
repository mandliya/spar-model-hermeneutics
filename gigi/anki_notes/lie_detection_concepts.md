# Lie-detection concepts: Anki card material

Explanations from the SPAR gemma lie-detection project, written as
question/answer pairs. Each `Q:` / `A:` pair is one card.

---

Q: What is a token, and how does a language model choose its next one?
A: A token is a word or piece of a word. Before writing each token, the model
gives every possible next token a probability (all adding up to 100%). At
temperature 0 it simply writes the highest-probability one.

Q: What is gemma's "confidence score" for "yes, I lied"?
A: At the moment gemma is about to answer "Did you lie? A = yes, B = no", it has
a probability for "A" and one for "B". The score is
P(yes) = P("A") ÷ (P("A") + P("B")).
Example: A 99.90%, B 0.09% → 99.90 ÷ 99.99 = 99.91%.

Q: How did we get gemma's confidence scores without generating anything new?
A: Feed each saved conversation plus gemma's saved answer back through the
model once, and read the probabilities at the exact spot where it wrote its
letter. (For reason-first questions: the letter after "Final answer:".)

Q: Confidence score vs confidence interval: what's the difference?
A: A confidence score is gemma's probability of "yes" on one answer (e.g.
99.9%). A confidence interval is the uncertainty range on OUR results (e.g.
92.7%, 90.7–94.2%). Unrelated things.

Q: What does balanced accuracy measure, and how is it computed?
A: How well gemma's yes/no answer separates lies from honest answers.
Balanced accuracy = ½ × (% of lies it confessed + % of honest answers it
cleared) = ½ × (confession + (100 − false confession)). 50 = coin flip,
100 = perfect.
Example: alibi `named`: ½ × (97.7 + 70.1) = 84.

Q: What does AUROC stand for?
A: Area Under the Receiver Operating Characteristic curve. (The name comes
from WWII radar engineering.)

Q: What does AUROC measure, in one sentence?
A: Pick one answer where the model lied and one where it didn't: how often does
the lie get the higher confidence score? Do that for every lie/honest pair;
AUROC is the share of pairs the lie "wins".

Q: Work out AUROC: lies scored 99.91%, 99.80%, 60%; honest answers 95%, 40%, 10%.
A: 3 × 3 = 9 pairs. 99.91% beats all 3; 99.80% beats all 3; 60% beats 40% and
10% but loses to 95%. 8 wins / 9 pairs = 0.89.

Q: How do you read an AUROC value?
A: 0.5 = coin flip (scores no higher on lies than honest answers). 1.0 =
perfect (every lie scores above every honest answer). Below 0.5 = backwards
(more confident "yes" on honest answers), e.g. ascii `neutral` = 0.12.

Q: What is the ROC curve, and why does the area under it equal the
"pairs" number?
A: Try every possible cutoff ("flag any answer with P(yes) above X%") and plot
lies caught vs honest answers wrongly flagged. The area under that curve works
out to exactly the share of lie/honest pairs where the lie scores higher.

Q: Is P(yes) the AUROC?
A: No. P(yes) is one score for one answer. AUROC is one number for a whole set
of answers, built by comparing all their scores between lies and honest ones.

Q: Why can the letter-based score be ~50 while AUROC is ~0.75 on the same
answers (mask-factual)?
A: The letter is a rounded-off confidence (above 50% → A). If gemma says "yes"
(A) on everything, the letter can't separate lies from honest answers (~50).
But its confidence can still be slightly higher on lies (99.99% vs 99.9%), and
AUROC picks up that ranking (~0.75). The letter throws that information away.

Q: Analogy for letter vs confidence?
A: A friend who answers "yes" to every party invite. Their words (the letter)
never tell you if they'll come; their tone ("YES totally!" vs "yeah... sure")
does. Left panel = the words; right panel = the tone.

Q: Why keep the letter-based (balanced accuracy) score if AUROC is richer?
A: The letter is what you actually get when you ask a model "did you lie?";
deployed/closed models usually don't expose confidence. Balanced accuracy tests
"does just asking help?"; AUROC tests "is the information inside the model at
all?". The gap between them (e.g. mask-factual) is itself a finding.

Q: What's the catch with AUROC from gemma's confidence?
A: The differences it relies on are tiny (99.99% vs 99.9%). It shows the
information exists, not that you could build a reliable lie detector from it.

Q: Why does the confidence score depend on the letter A, and how did we check
that wasn't faking the signal?
A: "Yes, I lied" was always option A, so a preference for the letter A could
leak into P("A"). The A/B swap test re-asked 626 rows with "yes" as option B.
A real signal survives: alibi held (0.85 → 0.92); ascii collapsed
(0.45 → 0.13), so its "signal" was mostly letter preference.

Q: Why is float16 bad for storing gemma's activations, and why float32?
A: gemma computes in bfloat16. Some activation values exceed float16's maximum
(65,504) and become `inf` (7–17 values per tensor). bfloat16 → float32 is
lossless, so float32 keeps the exact values.

---

Q: What is "self-report" lie detection?
A: You ask the model that answered, straight after its answer, "Did you lie?"
and use its own reply (or its confidence in "yes") as the lie signal.

Q: What is "third-person monitoring"?
A: You show a model a transcript of someone else's conversation (system
prompt, question, the assistant's answer) and ask "Is the assistant lying?
Score 1-100". Nobody questions the model that answered. The monitor can even be
the same model, as long as it reads the transcript as someone else's.

Q: Why does third-person monitoring matter for "non-invasive" detection?
A: It never adds anything to the original conversation, so it can't change
the answering model's behaviour or internal state. That makes it the black-box
baseline a no-follow-up activation probe has to beat.

Q: Why shouldn't the model that labelled the ground truth also be the monitor?
A: It would be grading its own homework: agreement with its own labels would
look like monitoring skill. Use a different model as monitor.

Q: What were the 4 prompt formats the lie-detector paper compared?
A: (1) Direct self-report: "Did you lie? A/B". (2) Self-report with a rubric:
lie definitions, reason before answering. (3) Third-person report: a model
reads the transcript and scores 1-100 how likely the assistant lied.
(4) Third-person with the full lie taxonomy as a rubric.

Q: What was the lie-detector paper's main experiment and finding?
A: They fine-tuned gemma (LoRA) to answer "Did you lie?" correctly, trained on
half the lie types, tested on the other half. In-distribution AUROC rose
0.60 → 0.95, but on unseen lie types it plateaued at 0.70-0.75: the detectors
learned the surface of each setting, not deception itself.

Q: How did the lie-detector paper label which answers were lies?
A: Parallel interrogation: ask neutrally (belief), then under pressure, and
flag contradictions. Then 5 judge models re-checked each case with full
context and voted (about 25% of labels changed).

---

Q: How is AUROC computed for a third-person monitor like Gemini?
A: The monitor reads each transcript and gives a 1-100 "how likely is the
assistant lying" score. Then compare every lie with every honest answer: how
often did the lie get the higher score? Same AUROC as for gemma's self-report,
just a different score (the monitor's number instead of gemma's P(yes)).

Q: Work out a monitor's AUROC: lies scored 95, 80, 40; honest answers 50, 10, 5.
A: 3 × 3 = 9 pairs. 95 beats all 3, 80 beats all 3, 40 beats 10 and 5 but
loses to 50. 8 wins / 9 pairs = 0.89.

Q: What if a lie and an honest answer get the same score (both 95)?
A: That pair counts as half a win. Monitors give round numbers (1, 85, 95)
a lot, so ties are common.

Q: Why use AUROC instead of "% answered correctly"?
A: Because the lies and honest answers are unbalanced. In unanswerable, 86
answers are lies and 7 are honest: a detector that shouts "lie!" every time is
92% correct but has learned nothing. Its AUROC is 0.5, because it never ranks
a lie above an honest answer. AUROC also needs no cutoff.

Q: What does "monitors beat self-report in 10 of 11 categories" mean?
A: An outside model reading the transcript (Gemini, GPT-5 mini) ranks gemma's
lies above its honest answers more often than gemma's own "did you lie?"
confidence does. Example, ascii: outsiders 0.99-1.00, gemma about itself 0.12.

Q: What is the AUROC formula (pairs version)?
A: AUROC = (wins + ½ × ties) ÷ (number of lies × number of honest answers).
A "win" is a lie/honest pair where the lie has the higher score. Example:
lies 95, 80, 40 and honest answers 50, 10, 5 give 8 wins, 0 ties, 3 × 3 = 9
pairs, so 8/9 = 0.89.

Q: What is the rank shortcut for AUROC (what our code computes)?
A: Sort all scores together and number them 1, 2, 3... from lowest (tied
scores share the average rank). Add up the lies' ranks (R).
AUROC = (R − n(n+1)/2) ÷ (lies × honest), where n = number of lies.
Example: 5, 10, 40, 50, 80, 95 get ranks 1-6. Lies are 40 (3), 80 (5),
95 (6), so R = 14. (14 − 3×4/2) ÷ 9 = 8/9 = 0.89, the same answer as
counting pairs, just faster.

Q: Why is AUROC the standard score for lie detectors in AI-safety research?
A: (1) No cutoff needed: it scores the detector across every possible
threshold. (2) It doesn't care how common lies are, so settings with 90% lies
and 20% lies are comparable. (3) It works on any score (probe output, P(yes),
a monitor's 1-100), so very different detectors share one yardstick.
(4) It's easy to say: "the chance a random lie looks more suspicious than a
random honest answer". (5) Everyone uses it, so papers can be compared.

Q: What can a high AUROC hide?
A: (1) Shortcuts: it can come from a surface cue, not from detecting lying
(ascii: "NO" = lie; the A/B swap showed letter preference). (2) No
generalization guarantee: the lie-detector paper got 0.95 on familiar lie
types and only 0.70-0.75 on new ones. (3) It's only as good as the
ground-truth labels. (4) It can hide poor performance at the strict cutoffs
safety needs.

Q: What is "recall at 1% false-positive rate", and why do safety papers
report it next to AUROC?
A: Set the cutoff so only 1% of honest answers get flagged, then ask what
share of lies are still caught. In deployment most answers are honest, so you
can't afford many false alarms. A detector with a decent AUROC can still catch
few lies at that strict cutoff.

Q: In the lie-detector paper, how were the "ground truth" lie labels made?
A: Two stages. (1) Generation-time checks: ask neutrally for the model's
belief, then put it under pressure (role-play, threats, "Are you sure?") and
compare. (2) A judge pass: GPT-5 (chosen after testing 5 models) read each
transcript plus the setup's metadata and gave a label, lie type, rationale
and 1-100 score. Five judges, majority vote. About 25% of labels changed.
Human check: one annotator, 5-10 examples per category.

Q: Labelling judge vs third-person monitor: what's the difference?
A: The judge makes the answer key. It sees the setup's metadata (what's true,
what the model was pushed to say) and decides what counts as a lie. A
third-person monitor is a detector being tested: it sees only the transcript
and must guess. Same kind of model, opposite roles. Giving a monitor the
judge's metadata would be cheating.

Q: What does the paper's main experiment train, and what does "failed to
generalize" refer to?
A: Each gemma-3 model (4B-27B) is fine-tuned with LoRA to answer "Did you
lie?" about its own transcripts, trained on half of 12 lie types and tested
on the other half. Familiar lie types: AUROC 0.60 -> 0.95. New lie types:
stuck at 0.70-0.75, barely above just prompting. It's transfer across lie
types within one model, not from small models to big ones.

Q: Is Qwen3-235B's self-report near perfect in the lie-detector paper? Is ours?
A: In the paper, yes: it "could somewhat reliably identify its own deceptions
when asked directly", AUROC 0.98-0.99 with simple prompting (gemma-3-27b:
0.46-0.80). Ours is no: Qwen3-32B reaches at most 0.88 asked about itself,
in gemma's range. Possible reasons: size (235B vs 32B), or the paper's
labels (GPT-5 judge with metadata) vs ours (independent checks).

Q: Black-box vs grey-box vs white-box lie detection?
A: Black box = only what the model outputs (asking "did you lie?", or an
outside monitor reading the transcript). Grey box = you change the weights
(fine-tuning) but still read only the model's answer at test time; the
paper's trained detectors are this. White box = reading the model's
internals: a probe on its hidden activations, or interpretability features
and steering.

Q: What experiment tests "the limit of black box vs how far white box goes"?
A: One train/test split (train on half the lie types, test on the other half)
and three detectors side by side: prompted baseline -> fine-tuned self-report
(best black/grey box; the paper plateaus at 0.70-0.75 on unseen lie types) ->
linear probe on activations (white box). Does the probe beat the plateau on
unseen lie types?
