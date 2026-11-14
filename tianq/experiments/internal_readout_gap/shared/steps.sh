# Sourced by each experiment's run.sh (which sets E, the experiment's name): the box environment, and how a step is
# run and recorded. One line per step start/end in /workspace/logs/<E>_status.txt; a step's output in
# /workspace/logs/<E>_<step>.log. Steps skip work already on disk, so a rerun picks up where it stopped.
set -a; . /workspace/.env; set +a
export HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/workspace/venvs/readout/bin/python
NLA_PY=/workspace/venvs/nla/bin/python   # the vendored official NLA client expects transformers 4.x
STATUS=/workspace/logs/${E}_status.txt

step() {   # step <name> <command...>
  local name=$1; shift
  echo "$(date -u +%FT%TZ) start $name" >> "$STATUS"
  "$@" >> "/workspace/logs/${E}_$name.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end $name exit=$rc" >> "$STATUS"
  return $rc
}

queue() {  # queue <gpus> <step:command> ...: the steps one after another on those GPUs; stops at the first failure
  local gpus=$1; shift
  local item
  for item in "$@"; do CUDA_VISIBLE_DEVICES=$gpus step "${item%%:*}" bash -c "${item#*:}" || return 1; done
}

side_by_side() {  # side_by_side <function for GPU 0> <function for GPU 1>: both at once; fails if either does
  "$1" & local a=$!
  "$2" & local b=$!
  wait $a; local ra=$?; wait $b; local rb=$?
  [ $ra -eq 0 ] && [ $rb -eq 0 ]
}

nla_python() {  # nla_python <model>: Qwen3.6's NLA runs in the readout venv, gemma's in the NLA venv
  [ "$1" = qwen3.6-27b ] && echo "$PY" || echo "$NLA_PY"
}
