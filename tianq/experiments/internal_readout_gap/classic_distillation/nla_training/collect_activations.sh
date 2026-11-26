#!/usr/bin/env bash
# One model's activations, collected with the library's own data pipeline. Resumable: their extraction skips shards
# that are complete.
#   bash collect_activations.sh <name> <model folder or HF name> [documents]
#   bash collect_activations.sh gemma-classic-step92 /workspace/models/gemma-classic-step92        # the training data
#   bash collect_activations.sh gemma-3-27b-it-sample google/gemma-3-27b-it 1000                   # a small base sample
# Their Gemma-27B config (configs/datagen/gemma3_27b_ultrafineweb_100k.yaml), with these pointed elsewhere:
#   base_model      the model named here, where the config names base gemma
#   corpus          a local slice of the same dataset (fetch_corpus_slice.py says why and which)
#   corpus.length   only when [documents] is given; otherwise their 100,000
#   output_dir      /workspace/out/internal-readout-gap/classic_nla/data/<name>, where shared/upload.py finds it
# Everything else is theirs: 10 positions a document, layer 41, their seeds, their 25/25/50 split.
# Stages 0 (extract) and 1 (split) run through their orchestrator. Their stage 2 pays an LLM to explain the two
# supervised splits; the continued training reads only the third split, so stage 2 is not run. That third split is
# finished by the same two commands their orchestrator would run for it (stage3_build --stage rl, then stage_shuffle).
set -euo pipefail
NAME=$1; MODEL=$2
. "$(dirname "$0")/training_environment.sh"
DATA=/workspace/out/internal-readout-gap/classic_nla/data/$NAME
CONFIG=$NLA/configs/datagen/gemma3_27b_ultrafineweb_100k.yaml
FULL=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['corpus']['length'])")
DOCUMENTS=${3:-$FULL}
STORAGE="--storage-cls nla.datagen.storage.LocalStorage"

python "$(dirname "$0")/fetch_corpus_slice.py" --documents "$FULL" --to /workspace/nla/corpus
rm -rf /workspace/nla/corpus/download   # the downloader's bookkeeping; the data stage must find only the part files

cd "$NLA"
python -m nla.datagen.run_pipeline --config "$CONFIG" --stages 0,1 \
  --override "base_model=$MODEL" corpus.name=/workspace/nla/corpus corpus.split=train "corpus.length=$DOCUMENTS" "output_dir=$DATA"
python -m nla.datagen.stage3_build --input "$DATA/splits/rl_raw.parquet" --stage rl --output "$DATA/rl.parquet" --keep-debug-metadata $STORAGE
python -m nla.datagen.stage_shuffle --input "$DATA/rl.parquet" --output "$DATA/rl_shuf.parquet" --seed 42 $STORAGE

python - "$DATA" <<'PY'
import pathlib, sys
import pyarrow.parquet as pq
for p in sorted(pathlib.Path(sys.argv[1]).rglob("*.parquet")):
    if ".shards" not in str(p):
        print(f"{p}: {pq.ParquetFile(p).metadata.num_rows:,} rows, {p.stat().st_size:,} bytes")
PY
echo "activations ready: $NAME"
