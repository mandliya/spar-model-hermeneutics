#!/usr/bin/env bash
# Queue what follows the city reads of the two first readers, on the three GPUs the training does not use (0, 1, 3):
# the word lists (then the first upload: the probe and J space numbers can be computed), then the NLA sentences, the
# long part, with GPU 3 working through each reader's folder from its end. Run once, on the box, while the reads run.
A="$(dirname "$0")/after.sh"
M=qwen3.6-gender-secret-female; R=qwen3.6-27b
read_of() { echo "city_lies_of_$1__answer_only__read_by_$1"; }
READS_DONE=(); for k in 0 1 2; do READS_DONE+=("end read_$(read_of $R)_shard_$k exit"); done   # the reader's shares end last
LISTS_DONE=("end word_lists_$(read_of $M)_before_answer exit" "end word_lists_$(read_of $R)_before_answer exit")

(nohup setsid bash "$A" "${READS_DONE[@]}" -- "word_lists:0:city_lies_of_$M:answer_only:$M" "city_nla:0:$M:forward" > /dev/null 2>&1 < /dev/null &)
(nohup setsid bash "$A" "${READS_DONE[@]}" -- "word_lists:1:city_lies_of_$R:answer_only:$R" "city_nla:1:$R:forward" > /dev/null 2>&1 < /dev/null &)
(nohup setsid bash "$A" "${READS_DONE[@]}" -- "city_nla:3:$M:reverse" "city_nla:3:$R:reverse" > /dev/null 2>&1 < /dev/null &)
(nohup setsid bash "$A" "${LISTS_DONE[@]}" -- publish > /dev/null 2>&1 < /dev/null &)
sleep 1
echo "queued: $(ps -eo args | grep -c '[a]fter.sh') waiting jobs"
