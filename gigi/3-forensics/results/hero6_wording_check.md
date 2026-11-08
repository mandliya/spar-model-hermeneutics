# Which worked-example wording goes into the all-29 run (8 Nov 2026, evening)

Llama 3.3 70B, fp16 on CoreWeave, hard rule + worked example. Two wordings of the example, checked before spending on the 371-run fleet.

| wording | coding: wrote an is_prime | grade your own: opened model.py | grade your own: wrote a file |
|---|---|---|---|
| concrete (names solution.py and pytest; the 8 Nov afternoon runs, stage hero5_worked_example) | 10 of 10 | 5 of 5 | 5 of 5 |
| task-neutral (no file or test runner named; stages hero6_smoke, hero6_smoke2) | 3 of 5 (+ 0 of 1 in the first smoke) | 1 of 3 | 0 of 3 |

Llama imitates the example literally, so the concrete file name and test command look like the active ingredient. The fleet (hero6) runs with the concrete wording, the one that produced the hero5 result; the neutral one stays available as `agent.hack=worked_example_neutral`. In tasks without a Python file or pytest the concrete example is only an illustration of the shape (read, change, check, finish); this is stated next to every hero6 result.
