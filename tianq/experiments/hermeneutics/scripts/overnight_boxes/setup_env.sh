K=~/.ssh/id_ed25519_vast
for hp in "21375 69.7.162.133" "30350 82.98.239.123"; do set -- $hp
  ssh -i $K -p $1 root@$2 '[ -f /workspace/spar-model-hermeneutics/tianq/dyl-probe-repro/run.sh ] || { echo "$(hostname): code not there yet"; exit; }
    cd /workspace/spar-model-hermeneutics/tianq/dyl-probe-repro
    [ -f /workspace/env_ok ] && { echo "$(hostname): env already set up"; exit; }
    /venv/main/bin/pip install -q -r requirements-blackwell.txt peft vastai > /workspace/pip_env.log 2>&1 && touch /workspace/env_ok
    echo "$(hostname): pip exit $? ; $(/venv/main/bin/python -c "import transformers, peft, torch; print(transformers.__version__, peft.__version__, torch.__version__, torch.cuda.device_count())" 2>&1 | tail -1)"
    tail -c 300 /workspace/dl.log | tr "\r" "\n" | tail -2' 2>&1 | grep -v "Welcome\|Have fun\|AI agents"
done
