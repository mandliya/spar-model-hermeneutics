#!/usr/bin/env bash
# Lay out the run's newest save where shared/upload.py sends it to HF (run.sh's upload phase), in a folder of its own.
#   bash publish_save.sh gemma-classic-step92
# On HF it becomes internal-readout-gap/classic_nla/outputs/saves/<tag>/<iter folder>/:
#   writer/           the writer as a standard model folder, exactly as the library's own converter leaves it
#                     (tools/convert_fsdp_to_hf.py; it carries the released writer's settings file, which is the
#                     one the official client reads). This is what reads activations.
#   reconstructor/    the reconstructor as a standard model folder, as the trainer wrote it beside its save
#   run/              the run folder's own files for this save (both models' training saves, the round counters, the
#                     data position). Copy run/ back over /workspace/nla/runs/<tag>/ and continue_training.sh
#                     carries on from this round.
# Nothing is copied on the box: apart from the converted writer, the files are hard links to the run's. Safe to run again.
set -euo pipefail
TAG=$1
. "$(dirname "$0")/training_environment.sh"
RUN=/workspace/nla/runs/$TAG
ITER=$(printf 'iter_%07d' "$(cat "$RUN/actor/latest_checkpointed_iteration.txt")")
[ "$ITER" = "$(printf 'iter_%07d' "$(cat "$RUN/critic/latest_checkpointed_iteration.txt")")" ] || { echo "the two models' newest saves differ"; exit 1; }
TO=/workspace/out/internal-readout-gap/classic_nla/saves/$TAG/$ITER
mkdir -p "$TO/run/actor" "$TO/run/critic"

if [ ! -e "$TO/writer/config.json" ]; then
  rm -rf "$TO/writer.partial"
  python "$NLA/tools/convert_fsdp_to_hf.py" --input-dir "$RUN/actor/$ITER" --output-dir "$TO/writer.partial" --origin-hf-dir "$(released av)"
  mv "$TO/writer.partial" "$TO/writer"
fi
[ -e "$TO/reconstructor" ] || cp -al "$RUN/critic/$ITER/hf" "$TO/reconstructor"
for role in actor critic; do
  [ -e "$TO/run/$role/$ITER" ] || cp -al "$RUN/$role/$ITER" "$TO/run/$role/$ITER"
  cp "$RUN/$role/latest_checkpointed_iteration.txt" "$TO/run/$role/"
done
rm -rf "$TO/run/critic/$ITER/hf"   # the same files as reconstructor/
[ ! -d "$RUN/actor/rollout" ] || cp -al --update=none "$RUN/actor/rollout" "$TO/run/actor/"
du -sh "$TO"/* "$TO"/run/*/*
echo "save laid out: $TO"
