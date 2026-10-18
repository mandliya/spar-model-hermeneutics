"""Full forensics runs on the laptop in Docker: every environment, Qwen first then gemma, one rollout per
`run.py --count 1` call, scheduled against the Docker VM's CPU and memory, network-isolated like Modal.

- Isolation: containers join the internal network `forensics-egress` (no route out) and reach OpenRouter
  only through scripts/egress_proxy.py (container `forensics-proxy`), via HTTPS_PROXY in the env file.
- Order: Qwen, then gemma; within a model, round-robin over environments (run 1 of each, then run 2...),
  so a partial night still covers every environment.
- Resources: a run of an environment with a `modal:` block reserves its pinned cpu and up to 3 GB; others
  0.5 cpu / 1 GB. At most PINNED_MAX timing-bench runs and LIGHT_MAX others at once (MEM_CAP_MB, MAX_RUNS);
  the first queued run that fits starts.
- Images: the first run of an environment builds its image; the others wait for it, then reuse it.
- Spend: every 2 min the OpenRouter key's spend since START_USD is checked; at CAP_USD nothing new starts
  and the running rollouts are killed.
- Resumable: every finished run is appended to results/logs/full_laptop/done.txt and skipped on restart; a
  run whose image failed to build goes to build_failed.txt instead and is queued again on the next start.
- Per-run wall clock as on Modal: 1 h for caps <= 60 steps, else 4 h.

  FORENSICS_ENV_FILE=<key file + HTTPS_PROXY> CAP_USD=380 START_USD=131.477198523 \
    nohup caffeinate -ims python3 gigi/3-forensics/scripts/laptop_full.py &
Results: results/raw/full_laptop/<env>/<model>/<timestamp>/run-1/. Progress: results/logs/full_laptop/progress.txt.
"""
import json
import os
import pathlib
import re
import subprocess
import threading
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parents[1]
ENVS_DIR = HERE / "agent-interp-envs"
LOG = HERE / "results" / "logs" / "full_laptop"
OUT = HERE / "results" / "raw" / "full_laptop"
NETWORK = "forensics-egress"
ENV_FILE = os.environ["FORENSICS_ENV_FILE"]
CAP_USD = float(os.environ["CAP_USD"])
START_USD = float(os.environ["START_USD"])
PINNED_MAX = int(os.environ.get("PINNED_MAX", 2))   # timing-bench environments at once (2 cpu each, of the VM's 6)
LIGHT_MAX = int(os.environ.get("LIGHT_MAX", 6))     # other environments at once, on the remaining cpu
MEM_CAP_MB = int(os.environ.get("MEM_CAP_MB", 12000))
MAX_RUNS = int(os.environ.get("MAX_RUNS", 10))
MODELS = ["qwen3-next-80b", "gemma-3-27b"]
COUNTS = dict(optimize_rendering_engine=5, optimize_rendering_engine_graded=5, grade_your_own=5, leaked_tests=5,
              puppeteer=5, async_conversion=5, revert_or_refactor=5, optimize_lru_cache=10, lucky_seed=10,
              lucky_seed_readable_grader=10, leaked_scorer=10, reference_extraction=10, copying_of_answers=10,
              dashboard_costly_failure=10, chess=20, tictactoe=20, tower_of_hanoi=20, funding_email=20,
              lazy_investigation=20, eval_tampering=20, sandbagging=20, currency_conversion=20, parse_duration=20)

LOG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
lock = threading.Condition()
stop = threading.Event()
used = {"pinned": 0, "light": 0, "mem": 0, "runs": 0}
built: dict[str, threading.Event] = {}


def say(msg):
    with open(LOG / "progress.txt", "a") as f:
        f.write(f"{time.strftime('%F %T')} {msg}\n")


def config_info(env, model):
    """(max_steps, cpu, mem_mb) from the config text (system python3 has no yaml)."""
    text = (HERE / "configs" / f"{env}__{model}.yaml").read_text()
    steps = int(re.search(r"^  max_steps: *(\d+)", text, re.M).group(1))
    block = re.search(r"^modal:\n((?:  .*\n)+)", text, re.M)
    if not block:
        return steps, 0.5, 1024
    cpu = re.search(r"cpu: *([\d.]+)", block.group(1))
    mem = re.search(r"memory: *(\d+)", block.group(1))
    # Docker still caps the container at its full pinned memory; the scheduler books at most 3 GB of it,
    # since a rollout mostly waits on the model and rarely fills its limit.
    return steps, float(cpu.group(1)) if cpu else 1.0, min(int(mem.group(1)) if mem else 2048, 3072)


def spend():
    key = re.search(r"^OPENROUTER_API_KEY=(.+)$", pathlib.Path(ENV_FILE).read_text(), re.M).group(1).strip()
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req, timeout=20))["data"]["usage"] - START_USD


def watchdog():
    while not stop.is_set():
        try:
            s = spend()
            say(f"spent ${s:.2f} of ${CAP_USD:.0f} | running {used['runs']}")
            if s >= CAP_USD:
                say("CAP REACHED: starting nothing new, killing running rollouts")
                stop.set()
                ids = subprocess.run(["docker", "ps", "-q", "--filter", f"network={NETWORK}"],
                                     capture_output=True, text=True).stdout.split()
                proxy = subprocess.run(["docker", "ps", "-q", "--filter", "name=forensics-proxy"],
                                       capture_output=True, text=True).stdout.split()
                for cid in set(ids) - set(proxy):
                    subprocess.run(["docker", "kill", cid], capture_output=True)
                with lock:
                    lock.notify_all()
                return
        except Exception as e:  # a failed spend check is retried next round
            say(f"spend check failed: {type(e).__name__}")
        stop.wait(120)


def run_one(env, model, idx, steps, cpu, mem, first):
    try:
        if not first:
            built[env].wait()
        timeout = 3600 if steps <= 60 else 14400
        cmd = ["env", "-u", "VIRTUAL_ENV", "uv", "run", "python", "scripts/run.py",
               str(HERE / "configs" / f"{env}__{model}.yaml"), "--count", "1", "--no-judge",
               "--env-file", ENV_FILE, "--network", NETWORK, "--results-dir", str(OUT), "--timeout", str(timeout)]
        if not first:
            cmd.append("--no-build")
        t0 = time.time()
        with open(LOG / f"{env}__{model}.log", "a") as f:
            f.write(f"\n===== run {idx} started {time.strftime('%F %T')} =====\n")
            f.flush()
            rc = subprocess.run(cmd, cwd=ENVS_DIR, stdout=f, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        section = (LOG / f"{env}__{model}.log").read_text(errors="replace").rsplit(f"===== run {idx} started", 1)[-1]
        if "failed to build" in section or "failed to solve" in section:   # no rollout ran: not done, retried next start
            with open(LOG / "build_failed.txt", "a") as f:
                f.write(f"{env} {model} {idx} {time.strftime('%F %T')}\n")
            say(f"BUILD FAILED {env} {model} #{idx} (not counted as done)")
            return
        with open(LOG / "done.txt", "a") as f:
            f.write(f"{env} {model} {idx} rc={rc} {mins:.1f}min\n")
        say(f"finished {env} {model} #{idx} rc={rc} in {mins:.0f} min")
    finally:
        if first:
            built[env].set()
        with lock:
            used["pinned" if cpu >= 1 else "light"] -= 1
            used["mem"] -= mem
            used["runs"] -= 1
            lock.notify_all()


def main():
    done = {}
    if (LOG / "done.txt").exists():
        for line in (LOG / "done.txt").read_text().splitlines():
            env, model = line.split()[:2]
            done[(env, model)] = done.get((env, model), 0) + 1
    queue = []
    for model in MODELS:
        for i in range(1, max(COUNTS.values()) + 1):
            for env, n in COUNTS.items():
                if i <= n and i > done.get((env, model), 0):
                    queue.append((env, model, i))
    say(f"start: {len(queue)} runs queued ({sum(done.values())} already done), "
        f"at once: {PINNED_MAX} timing + {LIGHT_MAX} other, mem {MEM_CAP_MB} MB, spend cap ${CAP_USD:.0f}")
    threading.Thread(target=watchdog, daemon=True).start()
    threads = []
    pending = [(env, model, idx, *config_info(env, model)) for env, model, idx in queue]
    fits = lambda c, m: (used["runs"] < MAX_RUNS and used["mem"] + m <= MEM_CAP_MB
                         and (used["pinned"] < PINNED_MAX if c >= 1 else used["light"] < LIGHT_MAX))
    while pending and not stop.is_set():
        with lock:   # the first queued run that fits starts; a big one never holds up smaller ones behind it
            pick = next((p for p in pending if fits(p[4], p[5])), None)
            if pick is None:
                lock.wait(30)
                continue
            pending.remove(pick)
            env, model, idx, steps, cpu, mem = pick
            used["pinned" if cpu >= 1 else "light"] += 1
            used["mem"] += mem
            used["runs"] += 1
        first = env not in built
        if first:
            built[env] = threading.Event()
        say(f"start {env} {model} #{idx} (max_steps {steps}, cpu {cpu}, mem {mem})")
        t = threading.Thread(target=run_one, args=(env, model, idx, steps, cpu, mem, first))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    say("ALL DONE" if not stop.is_set() else "stopped at the spend cap")
    stop.set()


if __name__ == "__main__":
    main()
