# Token lead time

t* is the first reply token. In-reply offsets before it are empty when the reply opens on the act.

## ASCII sandbagging

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.999 | 191 | 119 | 0.553 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 1.000 | 191 | 119 | 0.448 |

## Sycophancy A

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.776 | 38 | 120 | 0.469 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.773 | 38 | 120 | 0.593 |

## Sycophancy B

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.887 | 116 | 85 | 0.515 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.886 | 116 | 85 | 0.530 |

## MASK provided facts

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.605 | 55 | 31 | 0.532 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.582 | 55 | 31 | 0.597 |

## MASK known facts

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.654 | 43 | 28 | 0.555 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.796 | 43 | 28 | 0.525 |

## MASK statistics

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.542 | 22 | 25 | 0.430 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.529 | 22 | 25 | 0.487 |

## MASK continuations

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.405 | 73 | 14 | 0.497 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.354 | 73 | 14 | 0.598 |

## MASK disinformation

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.828 | 58 | 11 | 0.471 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.754 | 58 | 11 | 0.428 |

## Capture the secret

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.622 | 73 | 305 | 0.554 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.574 | 73 | 305 | 0.493 |

## Alibi

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last | 0.732 | 784 | 492 | 0.449 |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) | 0.672 | 784 | 492 | 0.443 |

## Unanswerable

| Tokens before the bad token | In-category AUROC | Lies | Truthful | Transfer AUROC |
| --- | ---: | ---: | ---: | ---: |
| prompt_last |  | 86 | 7 |  |
| 32 |  |  |  |  |
| 16 |  |  |  |  |
| 8 |  |  |  |  |
| 4 |  |  |  |  |
| 2 |  |  |  |  |
| 1 (emits the bad token, not lead time) |  |  |  |  |
| 0 (the bad token, not lead time) |  | 86 | 7 |  |

ASCII sandbagging: τ90 is undefined because the reply is one token. Sycophancy A: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.776. Sycophancy B: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.887. MASK provided facts: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.605. MASK known facts: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.654. MASK statistics: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.542. MASK continuations: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.405. MASK disinformation: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.828. Capture the secret: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.622. Alibi: no lead offset reached 0.90. Peak in the lead set is prompt_last at 0.732. Unanswerable was not scored: 7 truthful replies, under the 10-row cutoff.
Offset 0 is higher than every earlier scored offset for ASCII sandbagging, MASK known facts. The probe is reading the bad token as it is emitted.
Transfer at τ90 is not measured, because no flavor has a τ90 with a transfer score.
The monitor gap is not measured.
