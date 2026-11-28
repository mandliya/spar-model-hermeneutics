#!/usr/bin/env bash
# Queue everything the distilled reader needs once its training has ended, each GPU starting as soon as it is free.
# Run once, on the box, while the training and the first readers' city NLA are still running.
#   GPU 2 (the training's): the reader's two reads of the gender rows, then a share of its own city lies
#   GPUs 0, 1, 3: a share of its own city lies each, when their NLA job has ended and the training has too
#   then on every GPU: a share of the read of those lies; the word lists on GPU 0; a share of the NLA sentences
# The upload (run.sh publish) is started by hand when these have ended: a full upload in the middle held GPU 0 back.
A="$(dirname "$0")/after.sh"
M=qwen3.6-gender-secret-female; R=qwen3.6-27b; D=qwen3.6-distilled-from-gender-secret-female
TRAINED="end train_reader exit"
nla_of() { echo "end nla_city_lies_of_$1__answer_only__read_by_$1$2 exit"; }
LIES=(); READS=(); for k in 0 1 2 3; do
  LIES+=("end own_city_lies_${D}_shard_$k exit"); READS+=("end read_city_lies_of_${D}__answer_only__read_by_${D}_shard_$k exit")
done
queue() { (nohup setsid bash "$A" "$@" > /dev/null 2>&1 < /dev/null &); }

queue "$TRAINED" -- "one_read:2:gender_secret:answer_only:$D" "one_read:2:gender_secret:with_reasoning:$D" "own_lies:2:0:4:$D"
queue "$TRAINED" "$(nla_of $M)" -- "own_lies:0:1:4:$D"
queue "$TRAINED" "$(nla_of $R)" -- "own_lies:1:2:4:$D"
queue "$TRAINED" "$(nla_of $R _reverse)" -- "own_lies:3:3:4:$D"
for g in 0 1 2 3; do queue "${LIES[@]}" -- "read_shard:$g:city_lies_of_$D:answer_only:$D:$g:4"; done
queue "${READS[@]}" -- "word_lists:0:city_lies_of_$D:answer_only:$D" "nla_shard:0:city_lies_of_$D:answer_only:$D:0:4"
for g in 1 2 3; do queue "${READS[@]}" -- "nla_shard:$g:city_lies_of_$D:answer_only:$D:$g:4"; done
sleep 1
echo "queued: $(ps -eo args | grep -c '[a]fter.sh') waiting jobs"
