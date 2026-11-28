#!/usr/bin/env bash
# The gender-secret run on the box (PLAN.md, "Steps 1 and 2 on the box"), detached and resumable. One or more phases:
#   (nohup setsid bash run.sh check > /dev/null 2>&1 < /dev/null &)          # 10 minutes: adapter, padding, 8 rows end to end
#   (nohup setsid bash run.sh gender publish > /dev/null 2>&1 < /dev/null &) # the gender rows, one read per GPU
#   (nohup setsid bash run.sh curriculum:0:0 > /dev/null 2>&1 < /dev/null &)   # the distillation data: curriculum:<gpu>:<shard of 4>
#   (nohup setsid bash run.sh city_shard:0:0:3 > /dev/null 2>&1 < /dev/null &)   # city_shard:<gpu>:<shard>:<shards>: that share
#        of the organism's own city rows, then of the reader's; then, once every share is read:
#   (nohup setsid bash run.sh word_lists:0:city_lies_of_qwen3.6-27b:answer_only:qwen3.6-27b publish > /dev/null 2>&1 < /dev/null &)
#   (nohup setsid bash run.sh city_nla:1:qwen3.6-27b:forward > /dev/null 2>&1 < /dev/null &)   # one NLA process; :reverse
#        works through the same folder from its end, so two processes on two GPUs share a folder
#   (TRAIN_GPU=0 nohup setsid bash run.sh train > /dev/null 2>&1 < /dev/null &)           # step 3: the distilled reader
#   then, for the distilled reader (tag D): own_lies:<gpu>:<shard>:<shards>:D  (its own city lies), read_shard:<gpu>:<rows>:
#   <condition>:D:<shard>:<shards>, word_lists:..., city_nla:<gpu>:D:forward, and one_read:<gpu>:gender_secret:<condition>:D
#   (nohup setsid bash run.sh one_read:1:gender_secret:answer_only:qwen3.6-27b > /dev/null 2>&1 < /dev/null &)   # a single read
# A "read" is <rows>__<condition>__read_by_<model>. Steps, logs and the status file: ../shared/steps.sh.
set -uo pipefail
cd "$(dirname "$0")"
E=gender_secret_rerun
. ../shared/steps.sh
MO=qwen3.6-gender-secret-female; READER=qwen3.6-27b   # tags of shared/models.py
DIST=../readout_methods/word_lists/write_jlens_lists.py
POSITIONS="reply_last before_answer"

answers_and_read() {   # <gpu> <rows> <condition> <model> [--limit n]: the reader's own answers where none are recorded, then the read
  local gpu=$1 rows=$2 condition=$3 model=$4; shift 4
  local read=${rows}__${condition}__read_by_${model}
  if [ "$rows" = gender_secret ]; then
    CUDA_VISIBLE_DEVICES=$gpu step "answer_$read" $PY answer.py --rows "$rows" --condition "$condition" --model "$model" "$@" || return 1
  fi
  CUDA_VISIBLE_DEVICES=$gpu step "read_$read" $PY read.py --rows "$rows" --condition "$condition" --model "$model" "$@"
}
word_lists() {         # <gpu> <rows> <condition> <model>: the lens's top 5,000 words with log-probabilities, at both positions
  local read=$2__$3__read_by_$4 position
  for position in $POSITIONS; do
    CUDA_VISIBLE_DEVICES=$1 step "word_lists_${read}_$position" $PY $DIST --experiment $E --read "$read" --position "$position" \
      --model "$4" --lens published --name published || return 1
  done
}
nla() {                # <gpu> <rows> <condition> <model> [nla_read.py options]: NLA sentences (the untouched model's NLA reads every model)
  local gpu=$1 read=$2__$3__read_by_$4; shift 4
  CUDA_VISIBLE_DEVICES=$gpu step "nla_${read}${1:+_${1#--}}" $PY ../shared/nla_read.py --experiment $E --model $READER --read "$read" "$@"
}
city_shard() {         # <gpu> <shard> <shards> <model>: one share of a reader's own city lies
  CUDA_VISIBLE_DEVICES=$1 step "read_city_lies_of_$4__answer_only__read_by_$4_shard_$2" $PY read.py --rows "city_lies_of_$4"     --condition answer_only --model "$4" --shard "$2" --shards "$3"
}
together() {           # run the given background jobs' pids to the end; fails if any of them failed
  local rc=0 pid
  for pid in "$@"; do wait "$pid" || rc=1; done
  return $rc
}

for phase in "$@"; do
  case $phase in
    check)
      CUDA_VISIBLE_DEVICES=0 step padding_check $PY padding_check.py --model $MO || exit 1
      answers_and_read 0 gender_secret answer_only $MO --limit 8 || exit 1
      CUDA_VISIBLE_DEVICES=0 step word_lists_check $PY $DIST --experiment $E --read gender_secret__answer_only__read_by_$MO \
        --position reply_last --model $MO --lens published --name check --limit 8 || exit 1
      nla 0 gender_secret answer_only $MO --limit 2 --shipped-examples 3 || exit 1 ;;
    gender)
      pids=(); gpu=0
      for job in "answer_only $MO" "answer_only $READER" "with_reasoning $MO" "with_reasoning $READER"; do
        ( set -- $job; answers_and_read $gpu gender_secret "$1" "$2" && word_lists $gpu gender_secret "$1" "$2" && nla $gpu gender_secret "$1" "$2" ) &
        pids+=($!); gpu=$((gpu + 1))
      done
      together "${pids[@]}" || exit 1 ;;
    one_read:*)   # one_read:<gpu>:<rows>:<condition>:<model>: everything for a single read
      IFS=: read -r _ gpu rows condition model <<< "$phase"
      answers_and_read "$gpu" "$rows" "$condition" "$model" && word_lists "$gpu" "$rows" "$condition" "$model" \
        && nla "$gpu" "$rows" "$condition" "$model" || exit 1 ;;
    city_shard:*)   # city_shard:<gpu>:<shard>:<shards>: one share of the organism's own city rows, then of the reader's
      IFS=: read -r _ gpu shard shards <<< "$phase"
      city_shard "$gpu" "$shard" "$shards" $MO && city_shard "$gpu" "$shard" "$shards" $READER || exit 1 ;;
    own_lies:*)     # own_lies:<gpu>:<shard>:<shards>:<model>: a share of a new reader's own city lies (step 3)
      IFS=: read -r _ gpu shard shards model <<< "$phase"
      CUDA_VISIBLE_DEVICES=$gpu step "own_city_lies_${model}_shard_$shard" $PY own_city_lies.py --model "$model" --shard "$shard" --shards "$shards" || exit 1 ;;
    read_shard:*)   # read_shard:<gpu>:<rows>:<condition>:<model>:<shard>:<shards>: a share of any read that needs no answers written
      IFS=: read -r _ gpu rows condition model shard shards <<< "$phase"
      CUDA_VISIBLE_DEVICES=$gpu step "read_${rows}__${condition}__read_by_${model}_shard_$shard" $PY read.py --rows "$rows"         --condition "$condition" --model "$model" --shard "$shard" --shards "$shards" || exit 1 ;;
    word_lists:*)   # word_lists:<gpu>:<rows>:<condition>:<model>: a finished read's word lists
      IFS=: read -r _ gpu rows condition model <<< "$phase"
      word_lists "$gpu" "$rows" "$condition" "$model" || exit 1 ;;
    curriculum:*)   # curriculum:<gpu>:<shard>: a quarter of the organism's answers to the curriculum questions
      IFS=: read -r _ gpu shard <<< "$phase"
      CUDA_VISIBLE_DEVICES=$gpu step "curriculum_shard_$shard" $PY sample_curriculum.py --model $MO --shard "$shard" --shards 4 || exit 1 ;;
    city_nla:*)   # city_nla:<gpu>:<model>:<forward|reverse>: one NLA process over a reader's city rows
      IFS=: read -r _ gpu model direction <<< "$phase"
      if [ "$direction" = reverse ]; then nla "$gpu" "city_lies_of_$model" answer_only "$model" --reverse || exit 1
      else nla "$gpu" "city_lies_of_$model" answer_only "$model" || exit 1; fi ;;
    nla_shard:*)    # nla_shard:<gpu>:<rows>:<condition>:<model>:<shard>:<shards>: one of several NLA processes sharing a read
      IFS=: read -r _ gpu rows condition model shard shards <<< "$phase"
      CUDA_VISIBLE_DEVICES=$gpu step "nla_${rows}__${condition}__read_by_${model}_shard_$shard" $PY ../shared/nla_read.py         --experiment $E --model $READER --read "${rows}__${condition}__read_by_${model}" --shard "$shard" --shards "$shards" || exit 1 ;;
    train)        # step 3: the distilled reader
      CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-0} step train_reader $PY train_reader.py || exit 1 ;;
    train_test)   # two optimizer steps on whatever answers are sampled so far, saved apart: does the trainer run?
      CUDA_VISIBLE_DEVICES=${TRAIN_GPU:-0} step train_reader_test $PY train_reader.py --limit_steps 2 || exit 1 ;;
    forced_choice:*)   # forced_choice:<gpu>: one question put to the NLA about every gender read's activations (ask_the_nla.py)
      CUDA_VISIBLE_DEVICES=${phase#*:} step forced_choice $PY ask_the_nla.py --model $READER || exit 1 ;;
    before_reasoning:*)   # before_reasoning:<gpu>:<model>: exploration, the probe's activations before the reader writes anything
      IFS=: read -r _ gpu model <<< "$phase"
      CUDA_VISIBLE_DEVICES=$gpu step "before_reasoning_$model" $PY read_before_reasoning.py --model "$model" || exit 1 ;;
    publish)
      step nla_check $PY ../shared/nla_check.py --experiment $E || exit 1
      step bundle $PY bundle.py || exit 1
      step upload $PY ../shared/upload.py --experiment $E || exit 1 ;;
    *) echo "unknown phase $phase" >> "$STATUS"; exit 1 ;;
  esac
  echo "$(date -u +%FT%TZ) phase $phase done" >> "$STATUS"
done
