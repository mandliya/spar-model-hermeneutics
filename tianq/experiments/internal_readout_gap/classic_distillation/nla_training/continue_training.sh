#!/usr/bin/env bash
# Continue the released Gemma-27B NLA on the classic model's activations: the library's final-stage script,
# configs/rl.sh, unchanged, started with the settings listed below.
#   bash continue_training.sh gemma-classic-step92 <rounds> [more options for their trainer]
# <rounds> is the round the run stops at (their --num-rollout). Miles saves at every 100th round and at the last one,
# so the count is chosen to end before the money does. Run it again with a larger count and it carries on from the
# newest save in the run folder: the trial passes 5, the long run the count the credit allows.
#
# What is set here and why (the script's own defaults reproduce their Qwen-7B run):
#   INSTRUCT_MODEL, ACTOR_SFT_CKPT    the released writer: the starting weights, the tokenizer, the settings sidecar,
#                                     and the reference the KL term pulls toward
#   CRITIC_SL_CKPT                    the released reconstructor
#   rollout batch 256, global 2048    the batch recorded in the released Gemma-27B checkpoint (256 prompts x 8 samples)
#   learning rate 1.41e-5, KL 0.01    the script's defaults, which are also the released Gemma-27B values
#   --sglang-attention-backend fa3    their docs: required for Gemma-3
#   LOSS_MASK_TYPE generic            the mask their Miles patch adds for models that are not Qwen
#   ACTOR_GPUS 4, CRITIC_GPUS 2, ROLLOUT_GPUS 2
#                                     their documented layout for one 8-GPU node. Both models then train on all of a
#                                     round's 2,048 samples. Their Miles patch stores the weights in 16 bits, so with
#                                     AdamW a model needs 8 bytes a parameter: 50 GiB on each of the writer's 4 GPUs
#                                     (28.4 B values) and 70 GiB on each of the reconstructor's 2 (18.8 B).
#   ACTOR_MICRO                       samples per forward pass, left at their default of 4. Their notes say it changes
#                                     step time, not the training math. Measured peaks on an H200 (140 GiB), cached
#                                     memory included: at 4 the writer's GPUs reach 105 GiB; at 2 they reach 91 and
#                                     the reconstructor's 98, and a round takes 640 s where 4 takes about 480. A box
#                                     with smaller GPUs sets ACTOR_MICRO lower before calling this script.
#                                     (Rounds 1 to 5 ran with the layout 4, 3, 1, where the reconstructor saw 1,536
#                                     of each round's samples, and rounds 6 and 7 with micro-batch 2; the logs on HF show each start.)
#   --no-save-optim                   Miles' option: saves hold the weights without AdamW's running averages (125 GB
#                                     a save in place of about 310). A resume restarts those averages, as the run's
#                                     own first round does.
#   --load <run>/actor                Miles' own resume. With no save there yet it is skipped, so the weights are the
#                                     released writer's; later it restores the newest save and its round counter
#   --critic-load-dcp <run>/critic    the same for the reconstructor, once it has a save
#   SAVE_INTERVAL 10                  a save on the box every 10 rounds (their default is 100), so that a run stopped
#                                     early loses at most 10 rounds. The save that is reported is still the last one.
#   --save-debug-rollout-data         Miles' option: each round's prompts, explanations and rewards go to a file, the
#                                     input of the library's extract_rollout_samples.py. A record only.
set -euo pipefail
TAG=$1; ROUNDS=$2; shift 2
. "$(dirname "$0")/training_environment.sh"
RUN=/workspace/nla/runs/$TAG
OUT=/workspace/out/internal-readout-gap/classic_nla   # what shared/upload.py sends to HF: the data and each round's samples
mkdir -p "$RUN" "$OUT/samples_by_round/$TAG"

export RL_PARQUET=$OUT/data/$TAG/rl_shuf.parquet
export INSTRUCT_MODEL=$(released av) ACTOR_SFT_CKPT=$(released av) CRITIC_SL_CKPT=$(released ar) RUN_DIR=$RUN
export ACTOR_GPUS=4 CRITIC_GPUS=2 ROLLOUT_GPUS=2 SAVE_INTERVAL=10 LOSS_MASK_TYPE=generic

resume=(--load "$RUN/actor")
if [ -f "$RUN/critic/latest_checkpointed_iteration.txt" ]; then resume+=(--critic-load-dcp "$RUN/critic"); fi
# With --no-save-optim Miles still makes empty optimizer/ and lr_scheduler/ folders in each save, and its loader stops
# on an empty one (found by the resume test). Without the folder it skips the optimizer, which is what such a save means.
find "$RUN" -mindepth 3 -maxdepth 3 -type d \( -name optimizer -o -name lr_scheduler \) -empty -delete

echo "writer $INSTRUCT_MODEL | reconstructor $CRITIC_SL_CKPT | data $RL_PARQUET | run $RUN | rounds $ROUNDS"
echo "GPUs: writer $ACTOR_GPUS, reconstructor $CRITIC_GPUS, text server $ROLLOUT_GPUS | newest save: $(cat "$RUN/actor/latest_checkpointed_iteration.txt" 2> /dev/null || echo none)"
cd "$STACK/miles"
bash "$NLA/configs/rl.sh" --rollout-batch-size 256 --global-batch-size 2048 --sglang-attention-backend fa3 \
  --no-save-optim --save-debug-rollout-data "$OUT/samples_by_round/$TAG/step_{rollout_id}.pt" \
  --num-rollout "$ROUNDS" "${resume[@]}" "$@"
