#!/usr/bin/env bash
# The NLA box's phases, detached and resumable. Run one or more, in this order:
#   (nohup setsid bash run.sh models > /dev/null 2>&1 < /dev/null &)        # the merged classic model and the released pair
#   (nohup setsid bash run.sh stack > /dev/null 2>&1 < /dev/null &)         # the official training stack
#   (nohup setsid bash run.sh activations > /dev/null 2>&1 < /dev/null &)   # activations with their pipeline: a small base sample, then the classic model's 100,000 documents
#   (nohup setsid bash run.sh preflight > /dev/null 2>&1 < /dev/null &)     # their checks on the released pair, then its rebuild quality before training
#   (nohup setsid bash run.sh trial > /dev/null 2>&1 < /dev/null &)         # 5 rounds at the real settings, saved at the end
#   (ROUNDS=60 nohup setsid bash run.sh train > /dev/null 2>&1 < /dev/null &)   # carry on from the newest save to round 60
#   (nohup setsid bash run.sh publish > /dev/null 2>&1 < /dev/null &)       # lay out the newest save for the upload
#   (nohup setsid bash run.sh quality_after > /dev/null 2>&1 < /dev/null &) # that save's rebuild quality, measured as before training
#   (nohup setsid bash run.sh upload > /dev/null 2>&1 < /dev/null &)        # the data, each round's samples, the laid-out saves and the logs to HF, size-checked
#   (nohup setsid bash run.sh destroy > /dev/null 2>&1 < /dev/null &)       # only after publish and upload passed: a phase that fails stops the chain
# Steps, logs and the status file come from ../../shared/steps.sh: one line per step start and end in
# /workspace/logs/classic_nla_status.txt, a step's full output in /workspace/logs/classic_nla_<step>.log.
set -uo pipefail
cd "$(dirname "$0")"
E=classic_nla
. ../../shared/steps.sh
TAG=gemma-classic-step92
BASE_SAMPLE=gemma-3-27b-it-sample
DATA=/workspace/out/internal-readout-gap/$E/data
QUALITY=/workspace/out/internal-readout-gap/$E/rebuild_quality_before_training

quality() {   # quality <gpu> <data name>: the released pair's rebuild quality on that data's first 256 activations
  CUDA_VISIBLE_DEVICES=$1 step "rebuild_quality_$2" bash -c ". ./training_environment.sh && python rebuild_quality.py \
    --activations $DATA/$2/rl_shuf.parquet --rows 256 --out $QUALITY/$2.json"
}
quality_base() { quality 0 $BASE_SAMPLE; }
quality_classic() { quality 1 $TAG; }

for phase in "$@"; do
  case $phase in
    models)      step prepare_models bash prepare_models.sh $TAG || exit 1 ;;
    stack)       step install_training_stack bash install_training_stack.sh; installed=$?
                 python3 ../../shared/blank_keys.py "/workspace/logs/${E}_install_training_stack.log" >> "$STATUS"   # their installer traces the keys file
                 [ $installed -eq 0 ] || exit 1 ;;
    activations) step collect_activations_base_sample bash collect_activations.sh $BASE_SAMPLE google/gemma-3-27b-it 1000 || exit 1
                 step collect_activations bash collect_activations.sh $TAG /workspace/models/$TAG || exit 1 ;;
    preflight)   step preflight bash -c '. ./training_environment.sh && cd "$NLA" &&
                   python -m nla.scripts.rl_preflight --critic-hf-dir "$(released ar)" --actor-hf-dir "$(released av)"' || exit 1
                 side_by_side quality_base quality_classic || exit 1 ;;
    trial)       step trial bash continue_training.sh $TAG 5 || exit 1 ;;
    train)       step train bash continue_training.sh $TAG "${ROUNDS:?the round to stop at}" || exit 1 ;;
    publish)     step publish_save bash publish_save.sh $TAG || exit 1 ;;
    quality_after)   # the newest laid-out save on the same 256 classic activations as before training, on GPU 0
                 SAVE=$(ls -d /workspace/out/internal-readout-gap/$E/saves/$TAG/iter_* | tail -1)
                 CUDA_VISIBLE_DEVICES=0 step rebuild_quality_after_training bash -c ". ./training_environment.sh && python rebuild_quality.py \
                   --activations $DATA/$TAG/rl_shuf.parquet --rows 256 --writer $SAVE/writer --reconstructor $SAVE/reconstructor \
                   --out /workspace/out/internal-readout-gap/$E/saves_rebuild_quality/${TAG}_$(basename "$SAVE").json" || exit 1 ;;
    quality_unseen)  # both pairs on 256 classic activations far past the rows training read (it reads 256 a round from
                 # the top): the released pair on GPU 0, the newest laid-out save on GPU 1
                 SAVE=$(ls -d /workspace/out/internal-readout-gap/$E/saves/$TAG/iter_* | tail -1)
                 unseen() {   # unseen <gpu> <step name> <result file> [the pair's options]
                   CUDA_VISIBLE_DEVICES=$1 step "rebuild_quality_unseen_rows_$2" bash -c ". ./training_environment.sh && python rebuild_quality.py \
                     --activations $DATA/$TAG/rl_shuf.parquet --rows 256 --skip 400000 --out $3 ${4:-}"
                 }
                 released_unseen() { unseen 0 released_pair "$QUALITY/${TAG}_unseen_rows.json"; }
                 trained_unseen() { unseen 1 trained_pair "/workspace/out/internal-readout-gap/$E/saves_rebuild_quality/${TAG}_$(basename "$SAVE")_unseen_rows.json" \
                                      "--writer $SAVE/writer --reconstructor $SAVE/reconstructor"; }
                 side_by_side released_unseen trained_unseen || exit 1 ;;
    upload)      python3 ../../shared/blank_keys.py /workspace/logs/${E}_*.log >> "$STATUS"
                 step upload $PY ../../shared/upload.py --experiment $E || exit 1 ;;
    destroy)     bash ../../own_lens/destroy_box.sh now "the NLA run is over; its saves, data and logs are on HF and size-checked" ;;
    *) echo "unknown phase $phase" >> "$STATUS"; exit 1 ;;
  esac
  echo "$(date -u +%FT%TZ) phase $phase done" >> "$STATUS"
done
