# Report back, 3 Nov 2026: every action item, call by call

## Group call, 28 Oct (John, Yuqi, TQ)

| action item | owner | status | where |
|---|---|---|---|
| Spot-check that the Qwen–gemma gaps come from intention, not capability | Gigi | done: every gap-cell try labelled by intent; gemma's non-cheats on Qwen's cheating traces are 61% choice, 37% capability | charts 5L, 5M; [intent_method.md](intent_method.md) |
| Redo all experiments with gemma 4 31B | Gigi | prepared, not run: on hold by the 2 Nov decision (stay on 27B first) | [gemma4_prep.md](gemma4_prep.md) |
| Check and adapt the function-calling format for gemma 4 31B | Gigi | checked: native tools, no prefix translation; the switch is one override string | [gemma4_prep.md](gemma4_prep.md) section 3 |
| Format Qwen traces into gemma-compatible tool calls before distillation | Gigi | done: rendered with the inference code itself, 29,807 examples | [distill/README.md](distill/README.md) |
| 80/20 train/test split; test set contains hill-climbable tasks | Gigi | split done (80/20 by run family); **only 3 of 8 gap environments hold out a cheating run**, fix proposed | [distill/README.md](distill/README.md), "Test set contents" |
| Payment card for the Claude plan upgrade | John | not ours to do | |
| Is the HAL 9000 / model-organism work differentiated enough? | Yuqi | not ours to do | |
| Cancel next Wednesday's meeting | John | not ours to do | |

## Call with John, 2 Nov

| action item | status | where |
|---|---|---|
| How Qwen acted where gemma stopped | done: on the same traces Qwen cheated in 111 of 333 tries | chart 5K |
| Propensity/capability categories in the gap breakdown, read from the traces | done; harness check added 3 Nov (1% of the capability walls are our tool-call format) | charts 5L, 5M; [intent_method.md](intent_method.md) |
| Grey vs green dots in chart 5H: same seed? | answered: different source runs, no fixed seed; **still to post in the Slack channel** | chart 5H |
| Is the control (same model resampled) in the gap plot? | answered: yes, Qwen's arm is the control; charts 3, 4, 5F relabelled | charts 3, 4, 5F |
| Report back, then decide 27B vs 31B | this message | below |

## Slack message (draft, for John and Yuqi)

Hi John, Yuqi, here is where the items from our last two calls stand. Links go to gigi/3-forensics/ on GitHub.

1. Where gemma stopped, Qwen on the same trace cheated in 111 of 333 tries; gemma in those cells hacked 13 of 344 and reported the shortfall in 188. Chart 5K: https://github.com/mandliya/spar-model-hermeneutics/blob/main/gigi/3-forensics/charts/q5_stopped_pairs.png

2. Every scored try in the gap environments now carries an intent label backed by a verbatim quote, and the tries the words do not settle are sorted by facts from the logs. Of gemma's 621 non-cheating tries on Qwen's cheating traces, 15% are a stated refusal and 46% a revealed choice: it faced the same shortfall, reported it, and never mentioned the shortcut. Another 37% are capability walls, and 2% are hedged success claims. Chart 5M: https://github.com/mandliya/spar-model-hermeneutics/blob/main/gigi/3-forensics/charts/q5_split.png. Method note with every definition, example quote and the spot-check: https://github.com/mandliya/spar-model-hermeneutics/blob/main/gigi/3-forensics/intent_method.md

3. I checked whether the capability walls are gemma or our harness. Our harness rejected at least one of gemma's text-mode tool calls in 87 of its 621 non-cheats, against 2 of Qwen's 796 tries. Only 6 attempts ended on it, all in pre-commit, so 1% of the walls are ours and 36% are gemma.

4. The grey and green dots in chart 5H come from different source runs. Grey is Qwen restarted at the end of one of its clean runs, green is Qwen restarted from a cheating run at the cheating turn. No seed is fixed: every try is a fresh sample at the model-card settings.

5. The controls are in the gap plot. Qwen's arm of every swap is Qwen resampled on its own run from the same turn, and charts 3, 4 and 5F now say so.

6. The distillation split is 80/20 by run family. In 5 of the 8 gap environments, though, the held-out families contain no Qwen cheating run, so they cannot test transfer on an unseen incident. I propose stratifying by cheating vs clean family as well.

On 27B vs 31B, I recommend staying with gemma-3-27b for the first distillation. Coding and ai_rd are choice, which is what distillation has to move. The dashboard and the rendering engine are mostly capability walls. 31B would help there, and the switch is prepared in gemma4_prep.md if the 27B results stay unconvincing.

---

:phoenix: Report Week 8 :phoenix:

**Did.** Closed the call items: where gemma stopped vs Qwen on the same trace (5K), every gap-cell try labelled by intent with a quote ($35.45) and sorted by log facts (5L, 5M, method note), control relabelled, gemma 4 config prepared, distillation data built and split checked.

**Found.** Where gemma stopped, Qwen cheated in 111 of 333 tries. gemma's non-cheats on Qwen's cheating traces are 61% choice and 37% capability, and only 1% of that is our harness. Coding and ai_rd are choice; the dashboard and rendering are mostly capability. Only 3 of 8 gap environments hold out a cheating run.

**Next / WIP.** Fix the split, decide 27B vs 31B with John and Yuqi, first distillation run.

<!-- Sources: 5K numbers from RESULTS.md "Where gemma stopped, what did Qwen do on the same trace?"; 5M numbers from
scripts/split_chart.py output (3 Nov, harness check); 87/621 and 2/796 from the harness scan of results/gap_replays*.csv;
5H answer from RESULTS.md "How much of Qwen's cheating is the trace pushing it?"; control from RESULTS.md section 4;
split from distill/README.md "Test set contents"; $35.45 from RESULTS.md 5L method paragraph. -->
