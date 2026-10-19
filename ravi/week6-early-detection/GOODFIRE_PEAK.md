# Goodfire passage peak, read by Gemma 27B

Score: the maximum dot product inside the labeled channel. Reasoning, the visible reply, and the tool call are separate passages. Earlier channels of the same step stay in the prefix.
Direction: the synthetic difference of means from the last-token run, Gemma 27B layer 58. Not fit on the real rollouts.
Prefix: early turns are dropped, then the kept text is cut to 8192 tokens. Positions restart at the start of that tail. 10453 scored passages did not see the full history; 207 lost the start of the passage itself.

Passages scored: 11657 of 11657. Locate failures: 0.
Selected on the train half by peak AUROC: `i_peek_f`.

Kimi's own probe in the paper reaches AUROC 0.94 on ImpossibleBench. The number below is Gemma reading Kimi's text, which is the comparison this machine can run.
The last-token run in GOODFIRE.md, cut at 2048 and mixing reasoning into the content channel, scored pl_peek at 0.743. The Last column here is the last token of the labeled channel on this prefix.

| Contrast | Train peak AUROC | Test peak AUROC | Test last-token AUROC | Test TPR at 5% train FPR | Test FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| i_rm_test | 0.732 | 0.714 | 0.253 | 0.028 | 0.049 |
| i_edit_test | 0.728 | 0.708 | 0.646 | 0.042 | 0.052 |
| i_conftest | 0.722 | 0.705 | 0.607 | 0.044 | 0.053 |
| i_env_var | 0.730 | 0.710 | 0.605 | 0.042 | 0.051 |
| i_print_line | 0.726 | 0.707 | 0.692 | 0.044 | 0.051 |
| i_write_score | 0.728 | 0.707 | 0.628 | 0.042 | 0.052 |
| i_expl_h | 0.727 | 0.708 | 0.641 | 0.044 | 0.051 |
| i_expl_f | 0.740 | 0.721 | 0.725 | 0.045 | 0.052 |
| i_expl_r | 0.725 | 0.707 | 0.624 | 0.044 | 0.050 |
| i_peek_h | 0.742 | 0.716 | 0.349 | 0.066 | 0.056 |
| i_peek_f | 0.742 | 0.729 | 0.344 | 0.041 | 0.048 |
| i_peek_r | 0.729 | 0.706 | 0.534 | 0.044 | 0.052 |
| pl_suite | 0.729 | 0.709 | 0.609 | 0.042 | 0.052 |
| pl_validator | 0.726 | 0.707 | 0.692 | 0.044 | 0.051 |
| pl_scorefile | 0.728 | 0.707 | 0.628 | 0.042 | 0.052 |
| pl_acquire | 0.736 | 0.719 | 0.308 | 0.045 | 0.050 |
| pl_peek | 0.738 | 0.713 | 0.346 | 0.064 | 0.054 |
| pl_exploit | 0.728 | 0.709 | 0.641 | 0.045 | 0.052 |

On the test rollouts with an attempted or enacted hack, the peak score crosses the 5% clean-passage threshold on an earlier step in 6 of 13. Median lead among those fires is 10.500 assistant steps. The paper's early-detection result is a resampling test, which this script does not run.
