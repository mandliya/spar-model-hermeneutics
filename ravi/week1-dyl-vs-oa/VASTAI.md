# Vast.ai runbook (Gemma-3-27B-IT)

This Coder box has a T4 with **no NVIDIA driver**. Rent a separate 80 GB GPU on [Vast.ai](https://cloud.vast.ai/). Official CLI walkthrough: [CLI Hello World](https://docs.vast.ai/cli/hello-world).

Target: **1× H100 80 GB** or **1× A100 80 GB**, on-demand (not interruptible), **≥120 GB disk**.

Gemma-3-27B-IT in bf16 is ~54 GB of weights. Probe extraction also holds residual-stream activations. 40–48 GB cards will OOM if you score all layers.

## 0. One-time account setup (console)

1. Create / log in at [cloud.vast.ai](https://cloud.vast.ai/).
2. Add credit. H100s are often **~$2–4/hr**; budget a few hours for week 1 plus model download.
3. Create an API key: [Keys](https://cloud.vast.ai/manage-keys/) → **+New**. Copy it once.
4. Accept the Gemma license on Hugging Face: [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it). Create a [HF token](https://huggingface.co/settings/tokens) with read access.

## 1. CLI on your laptop (or this Coder machine)

```bash
curl -fsSL https://vast.ai/install.sh | bash
# or: pip install vastai
hash -r
vastai set api-key YOUR_VAST_API_KEY
vastai show user
```

Register the SSH public key you will connect with **before** creating the instance:

```bash
vastai create ssh-key ~/.ssh/id_ed25519.pub
```

This Coder VM already has `~/.ssh/id_ed25519.pub`. If you will SSH from a laptop instead, register **that** laptop key.

## 2. Find an offer

Prefer verified, on-demand, 80 GB VRAM, direct ports, decent download bandwidth:

```bash
vastai search offers \
  'gpu_ram>=80 num_gpus=1 cuda_vers>=12.1 reliability>=0.98 verified=true rentable=true direct_port_count>=1 inet_down>=400 disk_space>=120' \
  -o 'dph_total+'
```

If that is empty, drop `verified=true` or try a named GPU:

```bash
vastai search offers 'gpu_name=H100_SXM num_gpus=1 gpu_ram>=80 rentable=true reliability>=0.98' -o 'dph_total+'
vastai search offers 'gpu_name=A100_SXM4 num_gpus=1 gpu_ram>=80 rentable=true reliability>=0.98' -o 'dph_total+'
```

Note the offer **ID** in the first column. Also check **CPU RAM ≥ 64 GB** (80+ is better) on the row.

Skip interruptible/bid instances for overnight jobs; the host can reclaim them.

## 3. Create the instance

```bash
OFFER_ID=REPLACE_ME

vastai create instance "$OFFER_ID" \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel \
  --disk 160 \
  --ssh --direct \
  --label ravi-gemma3-27b-probes \
  --onstart-cmd "nvidia-smi && echo ready"
```

Save `new_contract` — that is `INSTANCE_ID`. GPU billing starts when status is `running`; disk billing starts at create.

```bash
vastai show instance INSTANCE_ID
# wait until status=running (often 2–8 min while the image pulls)
vastai ssh-url INSTANCE_ID
```

Connect:

```bash
ssh -i ~/.ssh/id_ed25519 root@HOST -p PORT
```

If direct SSH is flaky, the console instance card also shows a **proxy SSH** command.

## 4. On the GPU box: CUDA check, code, tokens

```bash
nvidia-smi   # expect ~80 GiB, CUDA 12.x driver
python -c "import torch; print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))"
```

### Get the repo

The SPAR repo is **private**. Easiest from this Coder machine after the instance is up:

```bash
# on Coder / laptop, not on Vast
vastai copy local:/home/discord/spar-model-hermeneutics/ INSTANCE_ID:/workspace/spar-model-hermeneutics/
```

Or clone on the instance (needs GitHub auth there):

```bash
# on Vast
apt-get update && apt-get install -y git gh
gh auth login   # paste a GitHub PAT with repo scope, or use SSH
git clone git@github.com:mandliya/spar-model-hermeneutics.git /workspace/spar-model-hermeneutics
```

### Hugging Face + deps

```bash
cd /workspace/spar-model-hermeneutics/ravi/week1-dyl-vs-oa
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
huggingface-cli login   # paste HF token; must have Gemma-3 access
export PYTHONPATH=src
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME" results
pytest tests -q
```

## 5. Run week 1

Keep a `tmux` or `screen` session so SSH drops do not kill the job.

**This rental’s job** is official DYL → OA. Follow [`dyl-probe-repro/AGENTS.md`](dyl-probe-repro/AGENTS.md). Do **not** run `scripts/03_eval_transfer.py` for the official checkpoint (wrong token).

OA Simple Deception repro (optional; already matched the paper) still lives in this folder:

```bash
tmux new -s probes
source /workspace/spar-model-hermeneutics/ravi/week1-dyl-vs-oa/.venv/bin/activate
cd /workspace/spar-model-hermeneutics/ravi/week1-dyl-vs-oa
export PYTHONPATH=src HF_HOME=/workspace/hf_cache

python scripts/01_train_oa_probe.py --max-train 400 --max-eval 200
```

First Gemma download is tens of GB. If it OOMs, drop `--max-eval` and/or `--activation-batch-size` is already 2 in `configs/default.yaml`.

Copy results back:

```bash
# from Coder / laptop
mkdir -p /home/discord/spar-model-hermeneutics/ravi/week1-dyl-vs-oa/results
vastai copy INSTANCE_ID:/workspace/spar-model-hermeneutics/ravi/week1-dyl-vs-oa/results/ \
  local:/home/discord/spar-model-hermeneutics/ravi/week1-dyl-vs-oa/results/
```

## 6. Stop billing

```bash
vastai destroy instance INSTANCE_ID
```

`stop` pauses GPU charges but **disk still bills**. Destroy when you do not need the disk cache.

## Console-only path

If you prefer the website: [cloud.vast.ai](https://cloud.vast.ai/) → filter **H100 SXM** or **A100 80GB**, **on-demand**, disk ≥ 160 GB, template/image `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`, SSH + direct. Then use the instance card’s SSH command and continue from section 4.

## Checklist before you rent

- [ ] Vast credit loaded
- [ ] SSH pubkey registered on Vast
- [ ] HF account accepted Gemma-3-27B-IT
- [ ] HF token ready
- [ ] Way to get the private repo onto the box (`vastai copy` or `gh auth`)
- [ ] Calendar reminder to **destroy** the instance
