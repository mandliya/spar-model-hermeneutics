# General training sets on Modal: how the training and the tests would run (24 Nov 2026)

For GG, John and Yuqi. This note covers two things: the middle general set (ToolScale customer-service conversations)
and a way to train and test all three general sets on Modal at the same time, instead of renting one Vast box and
doing them one after another. **Nothing here has been run on Modal and nothing has been spent.** The code was written
and checked on the laptop only; what could not be checked without paying is listed at the end.

## The short answer

- Modal makes the round about twice as fast from start to finish, because the three trainings run side by side and
  every saved copy gets its own test server. It does not make any single training faster: it is the same GPU (H200)
  running the same script.
- Modal costs about one and a half times as much as Vast for the same work. An H200 is $4.54 an hour on Modal against
  $3.49 on Vast, and on Vast we packed three copies' tests onto one GPU.
- The untrained model does not need to be tested again. Its settings today are identical to p2ds's, so its 320 runs
  of 23 Nov are reused.

## 1. The middle set: ToolScale conversations

The source is the public dataset `zake7749/deepseek-v4-pro-agent-tool-calling-trajectory`: DeepSeek V4 Pro acting as a
customer-service agent with tools, with its reasoning in every turn. It has 2,510 rows that are 284 conversations
(each row is the same conversation cut one turn later), so we keep the longest row of each. All 2,510 rows were checked
to be the first part of their conversation's longest row.

`scripts/general_toolscale_to_train.py` turns the 284 conversations into the exact training format of the DeepSeek
round (p2ds). It does not re-implement anything: it loads the functions that built p2ds's data straight from
`scripts/distill_data.py`. So the reasoning goes into Qwen3.6's thinking block, the tool calls and tool results are in
Qwen3.6's own format, the cut is the same (105,000 characters, then 32,768 tokens), and the split is the same 80/20 per
domain with seed 0.

| Domain | Conversations to learn from | Held back to check the loss |
|---|--:|--:|
| bank | 82 | 21 |
| movie | 67 | 17 |
| ecommerce | 42 | 10 |
| medicine | 34 | 9 |
| basketball | 2 | 0 |
| **All five** | **227** | **57** |

No conversation was left out and none was cut: the longest is 18,236 tokens, well under the 32,768-token limit.

**The dataset states no licence.** The converted files are therefore not in git and not on Hugging Face, and the
script refuses to write inside the repo. They are on GG's laptop at
`/private/tmp/claude-501/-Users-gg/6d93eb10-f3fa-40a1-a0b3-85363040314c/scratchpad/general_toolscale/`
(`train.jsonl`, `val.jsonl`, `manifest.json`, `sizes_train.json`, `sizes_val.json`). That folder is temporary, so it
should be moved somewhere lasting outside the repo, or rebuilt with the command at the end of this note.

### How it compares with p2ds's training set

Both columns were counted with the Qwen3.6 tokenizer and the training script's own encoder. For p2ds the count matches
the training log exactly (17,558,201 tokens and 7,471,592 trained tokens per pass), so the two columns are measured the
same way.

| | p2ds (DeepSeek on our 22 tasks) | ToolScale set | ToolScale as a share of p2ds |
|---|--:|--:|--:|
| Examples to learn from (one per run or conversation) | 765 | 227 | 30% |
| Tokens read in one pass | 17,558,201 | 2,301,165 | 13% |
| Tokens the model is trained to produce in one pass | 7,471,592 | 545,059 | 7% |
| Tokens per example, typical (median) | 26,774 | 9,932 | |
| Longest example | 32,765 | 18,236 | |
| Examples that lost their last turns to the 32,768-token cut | 188 of 765 | 0 of 227 | |
| Turns the model is trained on | 18,531 | 2,024 | 11% |
| Trained turns that call a tool | 18,435 of 18,531 | 1,026 of 2,024 | |
| Trained tokens per turn (average) | 403 | 269 | |
| Reasoning per turn, average | 206 tokens | 142 tokens | |
| Reasoning per turn, typical (median) | 42 tokens | 100 tokens | |
| Longest reasoning in one turn | 16,409 tokens | 1,481 tokens | |
| Turns with no reasoning at all | 912 of 18,531 | 0 of 2,024 | |
| Weight updates in 2 passes (16 examples each) | 96 | 30 | |

What this means in plain words:

- The ToolScale set is much smaller. With the same recipe, Qwen3.6 gets 30 weight updates instead of 96 and is trained
  on 7% as many of DeepSeek's tokens.
- It is a different kind of work. In p2ds almost every turn runs a shell command. Here half the turns are a message to
  a customer and half are a tool call.
- The reasoning is steadier here: every turn has some, typically about 100 tokens. In p2ds most turns have very little
  and a few have thousands.
- The four saved copies cannot land exactly on half passes, because 30 updates do not divide by 4. They land after
  8, 16, 24 and 30 updates, which is 0.53, 1.07, 1.6 and 2 passes.

## 2. The design on Modal

Each training set has a name in the style of `DISTILL_V`: `p2gen_toolscale`, `p2gen_wildchat`, `p2gen_tbench`. The name
is used for the files on Modal, the test stages (`distill_p2gen_toolscale_c8`), the configs and the score files, so the
three sets never touch each other's files and can run at the same time.

**Training** (`p2gen_modal.py`, function `train`). One Modal function on one H200 runs the same `distill_train.py`
with the same arguments as `p2q_train.sh`: a smoke test on the 4 longest examples, then 2 passes with a copy saved
every half pass, then the loss of the untrained model and of every copy on the held-back 20%. The H200 is the right
GPU because p2ds peaked at 75.9 GiB at 32,765 tokens, which is too close to the 80 GiB of an H100 or A100. The ToolScale
set alone would fit an H100 (its longest example is 18,236 tokens), which is 13% cheaper per hour but untested for speed.
The copies are saved to a Modal volume. Modal can take a GPU away at any moment, so the function can be started again
at any point: finished steps are skipped and training resumes from the last complete copy.

**Serving** (`p2gen_modal.py`, function `serve`). One Modal function on one H200 starts vLLM 0.32.0 with exactly the
flags of `p2q_serve.sh` (thinking on, earlier reasoning kept, bf16, a 65,536-token window, LoRA copies), loads a saved
copy, and runs p2ds's check that the copy really changes the model before any run is made on it. The server is reached
through a Modal tunnel, and its address is written to the volume for the launcher to read. Each saved copy gets its
own server, which stops as soon as that copy's runs are done.

**Sizing the test runs to the server's cache.** The rule from the notes is that the runs alive at once, times the
length of their histories, must stay under 70% of the server's cache. On an H200 the cache holds about 1.06 million
tokens. I measured how long the histories got in p2ds's 640 trained-copy runs: the 90th percentile is 45,400 tokens for
secret_number, 26,100 for coding, 22,500 for tower_of_hanoi, 18,200 for coding_norules, and under 12,000 for the other
four tasks. One run of each task adds up to 138,700 tokens, so 5 runs per task (40 runs alive in all) uses 65% of the
cache. `p2gen_runs.sh` reads the cache size from the running server and works this out itself. One copy per server
means two copies never share a cache, which is what overloaded the server in p2ds (up to 144 runs alive at once).

**Tests** (`p2gen_runs.sh`, `p2gen_configs.py`, `p2gen_table.py`). The same 8 tasks, 40 runs each, per copy, launched
with the existing `modal_full.sh`, with configs written by the existing `distill_eval.py`, and scored by the existing
`distill_eval.py score` (which uses the rules in `collect.py`). All 8 tasks are now unseen, because a general set
contains none of our tasks.

**The untrained model is not run again.** `p2gen_configs.py base-check` compares, key by key, the config a new run
would get today with the config p2ds's runs were launched with, for all 8 tasks, and compares the server's flags and
vLLM version. Run on 24 Nov against the 8 configs on the launch box, every setting is the same. The only difference
is that the server would be a Modal H200 instead of a Vast H200. So the table takes its "before training" column from
`results/distill_p2ds_eval.csv` and says so in the column title.

**Access.** No Modal secret is created and no credential is moved. The server's password comes from the existing
secret `vllm-secret-gigi`, the same one the test runs already use. Qwen3.6 27B is a public model, so no Hugging Face
token is needed on Modal. Uploads to the private HF dataset happen from the launch box with the token already there.

**Where it is launched from.** Everything is launched from the launch box, never from the laptop, because a Modal
launch dies when the Mac sleeps. Each training and each server is a single function started with `modal run --detach`,
which keeps running if the launcher disconnects.

## 3. What it would cost

Prices read from modal.com/pricing on 24 Nov 2026: H200 $4.54 an hour, H100 $3.95, A100 80 GB $2.50, plus the
processor and memory we ask for (about $0.90 an hour for training, $0.75 for serving). So an hour of training costs
about $5.43 and an hour of serving about $5.30. The Vast figure is the measured $6.98 an hour for the two-H200 box of
p2ds, which is $3.49 per GPU.

Measured in p2ds and used below: training read 35.1 million tokens in 6 h 50 min on one H200 (1,614 tokens a second);
the 4 copies' 1,280 test runs took 8 h 50 min on the other H200; and the task sandboxes for all 1,602 test runs cost
$10.75 on Modal (read from Modal's billing report today, it was still "not measured" in BUDGET.md).

| | GPU hours on Modal | Modal | The same on Vast |
|---|--:|--:|--:|
| Training, ToolScale set (227 conversations) | about 1.3 | about $7 | about $5 |
| Training, a set the size of p2ds (765 runs) | about 8.2 | about $45 | about $29 |
| Training, per million tokens in the set (2 passes) | 0.34 | $1.87 | $1.20 |
| One test round: 4 copies x 8 tasks x 40 runs, servers | about 14 (4 servers x 3.5 h) | about $74 | about $31 (8.9 h, one GPU) |
| One test round, task sandboxes | | about $9 | about $9 |
| **ToolScale set, training and tests** | about 15 | **about $90** | about $45 |
| **A p2ds-sized set, training and tests** | about 22 | **about $130** | about $70 |
| **All three sets, if the other two are p2ds-sized** | about 60 | **about $350** | about $185 to $235 |
| **All three sets, if all are ToolScale-sized** | about 46 | **about $270** | about $135 |
| Time from start to finish, all three sets | | about 12 to 13 hours | about 27 to 30 hours on one box |

The training figures include the smoke test, loading the model and the loss check. The Vast range for all three sets is
there because a rented box bills for both GPUs the whole time, also while one is idle: p2ds's box cost about $82 for
11.4 hours.

Two honest caveats about this table. The size of the WildChat and Terminal-Bench sets is not known yet, so the two
"all three" rows are a low and a high case. And the 3.5 hours per copy on its own server is an estimate: in p2ds one
copy's runs took 3.9 to 5.5 hours while sharing the GPU with one or two other copies, and the untrained model alone
took 1.9 hours.

There is a cheaper way to test on Modal: one server for all 4 copies of a set, as on Vast (`serve --copies all`). That
is about 9 hours and $48 per set instead of $74, but slower, and it brings back the shared cache.

## 4. Risks

- **Modal can take the GPU back.** GPU functions on Modal can be interrupted, and there is no way to pay to prevent
  it. Training then resumes from the last saved copy, so up to a quarter of the training is done twice. A server that
  is interrupted comes back with a new address, and the runs in flight are cut. The launcher notices, stops that
  copy's runs and says so. The copy then has to be run again.
- **This workspace once left a Modal function waiting forever** (the note on the spar2026 workspace). The first step
  is therefore a tiny function without a GPU that costs a few cents and shows whether functions start at all.
- **The model is 54 GB.** The first training or server downloads it to a Modal volume, which should take a few
  minutes. After that every start loads it from the volume, a few minutes each time, paid at the GPU price.
- **A Modal function can run for 24 hours at most.** A p2ds-sized training takes about 8 hours, so there is room, and
  a restart resumes anyway.
- **Saved copies must reach the volume.** Modal saves the volume every few seconds and at shutdown. The training also
  saves it itself every minute.
- **The two software images have never been built.** The training image compiles one package against the GPU
  libraries, and vLLM compiles a kernel when it starts.
- **The tunnel to the server is new for us.** The test runs are only allowed to reach the server's address, which is
  looked up when a run is launched. If Modal changes the address behind the tunnel's name during a run, that run loses
  its server. One test run through the whole path settles this before a full round.

## 5. The commands

All on the launch box (`ssh -i ~/.ssh/vastai_lies_experiment -p 61337 root@202.122.49.242`), after the new scripts
and the data folder have been copied there. Each paid stage needs GG's yes for that set first, given as
`P2GEN_YES=<name>`. Add `DRY_RUN=1` to see every command without running anything.

```
cd /workspace/forensics/gigi/3-forensics
# 0. upload the set to Modal (free), then check that the workspace starts functions (a few cents)
bash scripts/p2gen_launch.sh data p2gen_toolscale /workspace/p2gen_data/toolscale
P2GEN_YES=p2gen_toolscale bash scripts/p2gen_launch.sh check p2gen_toolscale
# 1. the training smoke test alone (about 10 minutes of GPU), then the training
P2GEN_YES=p2gen_toolscale bash scripts/p2gen_launch.sh smoke p2gen_toolscale
P2GEN_YES=p2gen_toolscale bash scripts/p2gen_launch.sh train p2gen_toolscale
# 2. the tests: one server per saved copy, 8 tasks x 40 runs, then the scores (start it right after the training)
P2GEN_YES=p2gen_toolscale setsid nohup bash scripts/p2gen_launch.sh tests p2gen_toolscale 40 > /workspace/logs/p2gen_toolscale_tests.out 2>&1 < /dev/null &
# 3. scores again by hand, upload of copies, runs and logs to the private HF dataset, progress
bash scripts/p2gen_launch.sh score p2gen_toolscale
bash scripts/p2gen_launch.sh upload p2gen_toolscale
bash scripts/p2gen_launch.sh status p2gen_toolscale
# or everything in one detached line
P2GEN_YES=p2gen_toolscale setsid nohup bash scripts/p2gen_launch.sh all p2gen_toolscale /workspace/p2gen_data/toolscale > /workspace/logs/p2gen_toolscale_launch.out 2>&1 < /dev/null &
```

The other two sets use the same lines with `p2gen_wildchat` and `p2gen_tbench` and their own data folders.

To rebuild the ToolScale set on the laptop (free), from a folder that does not hold the downloaded data:

```
uv run --no-project --with pyarrow --with transformers==5.23.0 --with jinja2 python -I \
  gigi/3-forensics/scripts/general_toolscale_to_train.py build --parquet <folder>/train.parquet \
  --out <a folder outside the repo> --tokenizer <folder with Qwen3.6's tokenizer files>
```

## 6. What could not be checked without spending money

- Whether the two Modal images build, and whether vLLM 0.32.0 starts in its image.
- Whether the spar2026 workspace starts these functions, and how long an H200 takes to arrive.
- Whether training on a Modal H200 runs at the 1,614 tokens a second measured on Vast, and whether 64 GB of memory is
  enough for it.
- Whether the tunnel carries a reply that takes many minutes, and whether its address stays the same for a whole run.
- Whether vLLM takes its password from the environment the way its documentation says.
- The loading check for a copy on the Modal server.
- How often Modal interrupts an H200 in practice.

What was checked without spending: every script compiles; the Modal app loads with the Modal version on the launch
box (1.5.5); the server flags and training arguments are word for word those of p2ds; every launch stage prints the
right commands in a dry run; the result table script reproduces p2ds's published table from p2ds's score file; the
token counter reproduces p2ds's training log exactly.

## 7. Decisions for GG

1. **The ToolScale set is small.** With p2ds's recipe it is 30 weight updates on 7% as many trained tokens. Keep the
   recipe as it is, train more passes, or wait and size all three sets alike so they can be compared?
2. **Conversations with a low score.** The dataset scores each conversation against a reference answer. 112 of 284
   score 1.0 and 9 score 0. All 284 are kept for now; `--min-score` can drop the low ones.
3. **Modal or Vast.** Modal is about twice as fast and about one and a half times the price. If speed is not needed,
   one Vast box is cheaper.
4. **One server per copy, or one per set.** $74 and about 4 hours, or $48 and about 9 hours, per set.
5. **Where the converted set should live.** It is in a temporary folder now and may not go into git or onto HF.
6. **An extra loss check.** For about $6 more per set, each copy could also be scored on DeepSeek's own runs of our
   tasks, which shows whether general training brings Qwen3.6 closer to DeepSeek there (`EXTRA_NLL=data/p2ds`, after
   uploading p2ds's val and test files to the volume).
7. **Run labels for TQ.** `p2q_run_labels.py` is written for p2ds only. A version for these sets is not written yet.
