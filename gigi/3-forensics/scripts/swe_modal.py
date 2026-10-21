"""swe_rebench_v2 on Modal. The framework at 310c03f ships no runner for it (env_registry.py: "pre-built
per-instance DockerHub images"). This follows the pre-September run.py, which put the framework and the
environment's files under /opt in the task's own SWE-rebench image, but runs on Modal like run_modal.py's
isolated runs: one sandbox per run, egress to the OpenRouter API only, outputs on the
gigi-forensics-swe-rebench-v2 volume, then downloaded to results/raw/<stage>/swe_rebench_v2/.
The agent's shell runs as root and can read the API key (environment README), so the sandbox gets the capped
agent key (Modal secret openrouter-agent-gigi, OPENROUTER_API_KEY), never the main one.
Instance metadata comes from results/swe_instance_<id>.json (fetched from nebius/SWE-rebench-V2, the same
fields the old run.py took). Modal's Python 3.11 runs the framework (it needs >= 3.10); the agent's commands
and the tests use the image's conda Python (entry_point.py puts it first on PATH).
  cd agent-interp-envs && MODAL_PROFILE=spar env -u VIRTUAL_ENV uv run modal run --detach ../scripts/swe_modal.py \
      --config ../configs/swe_rebench_v2__qwen3-next-80b.yaml --count 10 [--stage full_swe] [--check]
--check builds the image and runs a no-API sanity check in one sandbox (framework import, Pythons, repo).
"""
import inspect
import json
import pathlib
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import modal
from omegaconf import OmegaConf

HERE = pathlib.Path(__file__).resolve().parents[1]            # gigi/3-forensics
REPO = HERE / "agent-interp-envs"
ENV = REPO / "environments" / "swe_rebench_v2"
SECRET = "openrouter-agent-gigi"
VOLUME = "gigi-forensics-swe-rebench-v2"
HOSTS = ["openrouter.ai"]
DEPS = ["anthropic>=0.121.0", "omegaconf>=2.3", "openai>=2.14.0", "pydantic>=2.0", "python-dotenv>=1.2.1",
        "pyyaml>=6.0.3", "sentences>=0.1.1", "tenacity>=9.1.2"]
FILES = ["agent.py", "run_step.py", "states.py", "tools.py", "entry_point.py", "evaluate.py"]
FIND_PY = "PY=/opt/fwvenv/bin/python"   # the harness's own venv (image_for); the repo keeps the image's Python
RUN = r"""
set -o pipefail
""" + FIND_PY + r"""
printf '%s' "$ROLLOUT_CONFIG_YAML" > /opt/config.yaml
export AGENT_OUTPUT_DIR="/mnt/results/$ROLLOUT_RUN_DIR"; mkdir -p "$AGENT_OUTPUT_DIR"
"$PY" /opt/entry_point.py 2>&1 | tee "$AGENT_OUTPUT_DIR/rollout.log"
"""
CHECK = FIND_PY + r"""
echo "harness python: $PY"; "$PY" -c "import sys, agent_interp_envs; print(sys.version.split()[0], 'framework ok')"
echo "image python: $(command -v python) $(python --version 2>&1)"; python -c "import covalent; print('repo importable:', covalent.__file__)"
command -v pytest
for c in /opt/conda/envs/testbed/bin/python /opt/conda/bin/python; do [ -x "$c" ] && { echo "conda python: $c $($c --version 2>&1)"; $c -c "import covalent" 2>&1 | tail -1 | sed "s/^/  import covalent: /"; }; done
printf '%s' "$ROLLOUT_CONFIG_YAML" > /opt/config.yaml; "$PY" -c "import yaml; c = yaml.safe_load(open('/opt/config.yaml'));
import os; d = c['task']['cwd']; print('cwd', d, os.path.isdir(d), len(os.listdir(d)) if os.path.isdir(d) else 0, 'entries')"
cd "$("$PY" -c "import yaml; print(yaml.safe_load(open('/opt/config.yaml'))['task']['cwd'])")" && git status --short | head -5 && git log --oneline -1
ls /opt
"""

app = modal.App("gigi-forensics-swe-rebench-v2")


def task_fields(instance_id: str) -> dict:
    """The task fields the pre-September run.py filled from the dataset row."""
    x = json.loads((HERE / "results" / f"swe_instance_{instance_id}.json").read_text())
    install = x.get("install_config") or {}
    install = json.loads(install) if isinstance(install, str) else install
    f2p, p2p = x.get("FAIL_TO_PASS", []), x.get("PASS_TO_PASS", [])
    f2p = json.loads(f2p) if isinstance(f2p, str) else f2p
    p2p = json.loads(p2p) if isinstance(p2p, str) else p2p
    image = x.get("image_name") or x.get("docker_image", "")
    repo = x.get("repo", "")
    cwd = f"/{repo.split('/')[1] if '/' in repo else repo}" if x.get("image_name") and not x.get("docker_image") else "/testbed"
    return {"instance_id": instance_id, "docker_image": image, "cwd": cwd,
            "problem_statement": x.get("problem_statement", ""), "test_patch": x.get("test_patch", ""),
            "test_cmd": install.get("test_cmd", "python -m pytest -x"), "fail_to_pass": f2p, "pass_to_pass": p2p,
            "variation": instance_id}


def allowlist():
    cidrs, pins = [], []
    for host in HOSTS:   # IPv4 only: Modal's allowlist rejects IPv6 CIDRs
        for ip in sorted({i[4][0] for i in socket.getaddrinfo(host, 443, family=socket.AF_INET, proto=socket.IPPROTO_TCP)}):
            cidrs.append(f"{ip}/32"); pins.append(f"{ip} {host}")
    param = "outbound_cidr_allowlist" if "outbound_cidr_allowlist" in inspect.signature(modal.Sandbox.create).parameters \
        else "cidr_allowlist"
    return {param: cidrs}, "\n".join(pins)


def image_for(docker_image: str) -> modal.Image:
    tag = docker_image if ":" in docker_image else f"{docker_image}:latest"
    # The image's Python (3.10 here) holds the repo and its pinned deps; the harness gets its own venv so its deps
    # (pydantic 2, openai, ...) never touch the repo's environment. Modal's add_python would overwrite /usr/local.
    img = modal.Image.from_registry(tag).run_commands(
        "python3 -m venv /opt/fwvenv", "/opt/fwvenv/bin/pip install -q " + " ".join(f"'{d}'" for d in DEPS))
    img = img.add_local_dir(str(REPO / "src"), "/opt/src", copy=True)
    img = img.add_local_file(str(REPO / "pyproject.toml"), "/opt/pyproject.toml", copy=True)
    for f in FILES:
        img = img.add_local_file(str(ENV / f), f"/opt/{f}", copy=True)
    # the framework importable from Modal's Python everywhere (evaluate.py runs with PYTHONPATH=/opt only)
    return img.run_commands(FIND_PY + " && \"$PY\" -c \"import site; open(site.getsitepackages()[0] + "
                            "'/agent_interp_envs_src.pth', 'w').write('/opt/src\\n')\"")


def download(vol: modal.Volume, prefix: str, dest: pathlib.Path) -> int:
    n = 0
    for e in vol.listdir(prefix, recursive=True):
        if e.type != modal.volume.FileEntryType.FILE:
            continue
        out = dest / e.path
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as fh:
            for chunk in vol.read_file(e.path):
                fh.write(chunk)
        n += 1
    return n


@app.local_entrypoint()
def main(config: str, count: int = 10, stage: str = "full_swe", timeout: int = 10800, check: bool = False):
    cfg = OmegaConf.load(config)
    fields = task_fields(cfg.task.instance_id)
    if "${" in fields["problem_statement"]:
        raise SystemExit("problem statement contains '${', which OmegaConf would read as an interpolation")
    cfg = OmegaConf.merge(cfg, {"task": fields})
    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    model = cfg.agent.model.replace("/", "-")
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    prefix = f"swe_rebench_v2/{model}/{stamp}"
    image = image_for(fields["docker_image"])
    vol = modal.Volume.from_name(VOLUME, version=2, create_if_missing=True)
    net, pins = allowlist()
    print(f"image {fields['docker_image']} | cwd {fields['cwd']} | {count} runs -> {VOLUME}:{prefix}", flush=True)

    if check:   # no API key needed, no network for the sandbox
        sb = modal.Sandbox.create("bash", "-c", CHECK, app=app, image=image, timeout=900,
                                  env={"ROLLOUT_CONFIG_YAML": cfg_yaml}, **{list(net)[0]: []})
        sb.wait()
        print(sb.stdout.read(), sb.stderr.read(), f"exit {sb.returncode}", flush=True)
        return

    local_cfg = HERE / "results" / "raw" / stage / VOLUME / prefix / "config.yaml"
    local_cfg.parent.mkdir(parents=True, exist_ok=True)
    local_cfg.write_text(cfg_yaml)

    def one(i):
        run_dir = f"{prefix}/run-{i}"
        for attempt in range(6):
            try:
                sb = modal.Sandbox.create("bash", "-c", RUN, app=app, image=image, timeout=timeout, cpu=2, memory=4096,
                                          volumes={"/mnt/results": vol}, secrets=[modal.Secret.from_name(SECRET)],
                                          env={"ROLLOUT_CONFIG_YAML": cfg_yaml, "ROLLOUT_RUN_DIR": run_dir,
                                               "ROLLOUT_HOSTS": pins}, **net)
                break
            except modal.exception.ResourceExhaustedError:
                time.sleep(10 * (attempt + 1))
        sb.wait()
        print(f"  {'✓' if sb.returncode == 0 else '✗'} {run_dir} (exit {sb.returncode})", flush=True)
        return sb.returncode

    with ThreadPoolExecutor(max_workers=count) as pool:
        codes = list(pool.map(one, range(1, count + 1)))
    n = download(vol, prefix, HERE / "results" / "raw" / stage / VOLUME)
    print(f"done: {sum(c == 0 for c in codes)}/{count} exit 0; {n} files -> results/raw/{stage}/{VOLUME}/{prefix}", flush=True)
