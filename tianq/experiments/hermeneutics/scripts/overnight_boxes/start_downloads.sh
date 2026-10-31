K=~/.ssh/id_ed25519_vast
D=/mnt/c/Users/tqmen/AppData/Local/Temp/claude/--wsl-localhost-Ubuntu-home-user-projects-spar-model-hermeneutics/8f57f645-ac89-4088-8ef3-6fe1ca508eb8/scratchpad
ENV=~/projects/spar-model-hermeneutics/tianq/dyl-probe-repro/.env
start() {  # port host models...
  local p=$1 h=$2; shift 2
  scp -q -i $K -P $p $ENV root@$h:/workspace/.env
  scp -q -i $K -P $p $D/box_download.py root@$h:/workspace/box_download.py
  ssh -i $K -p $p root@$h "mkdir -p /workspace/hf; cd /workspace; nohup setsid bash -c 'set -a; source <(tr -d \"\r\" < /workspace/.env); set +a; export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1; /venv/main/bin/pip install -q -U huggingface_hub hf_transfer > /workspace/pip_dl.log 2>&1; /venv/main/bin/python /workspace/box_download.py $* > /workspace/dl.log 2>&1' > /dev/null 2>&1 < /dev/null & echo started on $h"
}
start 35177 213.181.108.222 Qwen/Qwen3-235B-A22B-Instruct-2507@ac9c66cc9b46af7306746a9250f23d47083d689e
