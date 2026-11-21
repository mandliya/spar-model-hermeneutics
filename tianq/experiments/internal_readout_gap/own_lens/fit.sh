#!/usr/bin/env bash
# Fit one J-lens with the published gemma-3-27b-it lens's exact command (neuronpedia/jacobian-lens,
# gemma-3-27b-it/jlens/Salesforce-wikitext/config.yaml); only the model and the output folder change.
#   fit.sh <model id or folder> <gpu> <output folder> [extra fit_lens.py arguments, e.g. --n_prompts 3 for the gate]
# Resumable: fit_lens.py picks up its checkpoint in the output folder. Runs in /workspace/venvs/fit, which has the
# published fit's torch 2.11.0+cu128 and transformers 5.11.0.
set -euo pipefail
MODEL=$1; GPU=$2; OUT=$3; shift 3
set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=$GPU
cd "$(dirname "$0")/../vendor/neuronpedia-jlens"
mkdir -p "$OUT"
exec /workspace/venvs/fit/bin/python fit_lens.py "$MODEL" --out_dir "$OUT"   --dataset Salesforce/wikitext --dataset_config wikitext-103-raw-v1 --dataset_split train --text_field text   --max_chars 2000 --n_prompts 1000 --dim_batch 64 --max_seq_len 128 --dtype bfloat16 --device_map cuda   --min_prompts 100 --stop_window 10 --levels 1e-2,5e-3,1e-3 --stop_at_delta 0.002 "$@"
