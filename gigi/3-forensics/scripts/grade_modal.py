"""Grade the Modal pass with the environments' own graders (Claude Opus 5, via OpenRouter pinned to Anthropic).

Collects every fleet of the full_modal / full_modal_rest / full_modal_retry launches, groups them by the
environment that has a grader (agent_interp_envs ENV_JUDGES), runs that grader on them, and checks the
OpenRouter key's spend after each environment: past GRADE_CAP_USD (default 60) nothing more is started.
Graders skip runs already graded, so a rerun only does what is missing. JUDGE_MODEL (e.g. claude-opus-4-8,
Anthropic's fallback for Opus 5 refusals) grades only what is still ungraded; grading.json records the judge.
Log: results/logs/grading/grade.txt.
  FORENSICS_ENV_FILE=... python3 gigi/3-forensics/scripts/grade_modal.py
"""
import collections, json, os, pathlib, re, subprocess, time, urllib.request
HERE = pathlib.Path(__file__).resolve().parents[1]; ENVS = HERE / "agent-interp-envs"
LOG = HERE / "results" / "logs" / "grading"; LOG.mkdir(parents=True, exist_ok=True)
KEY = re.search(r"^OPENROUTER_API_KEY=(.+)$", pathlib.Path(os.environ["FORENSICS_ENV_FILE"]).read_text(), re.M).group(1).strip()
CAP = float(os.environ.get("GRADE_CAP_USD", 60))
JUDGED = {"aj_data_anon", "dashboard_perf", "grade_your_own", "leaked_fix", "lucky_seed",
          "lucky_seed_readable_grader", "optimize_cache_c", "stripped_engine", "tables_coverage_tests"}

def say(msg):
    with open(LOG / "grade.txt", "a") as f: f.write(f"{time.strftime('%F %T')} {msg}\n")

def usage():
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"]

fleets = collections.defaultdict(list)
for stage in ("full_modal", "full_modal_rest", "full_modal_retry"):
    for log in (HERE / "results" / "logs" / stage).glob("*__*.log"):
        t = log.read_text(errors="replace")
        vol = re.search(r"^Volume: (\S+)", t, re.M); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M)
        if vol and out:
            inner = out.group(1).split("/")[0]
            d = HERE / "results" / "raw" / "modal" / vol.group(1) / out.group(1)
            if inner in JUDGED and d.is_dir(): fleets[inner].append(str(d))
start = usage(); say(f"start: {sum(map(len, fleets.values()))} fleets in {len(fleets)} environments, cap ${CAP}, "
    f"judge {os.environ.get('JUDGE_MODEL') or 'default'}")
for env, dirs in sorted(fleets.items()):
    spent = usage() - start
    if spent >= CAP: say(f"CAP REACHED (${spent:.2f}): not grading {env} and later"); break
    rc = subprocess.run(["env", "-u", "VIRTUAL_ENV", "uv", "run", "python", f"environments/{env}/grade_rollouts.py", *dirs,
                         "--concurrency", "20", *(["--model", os.environ["JUDGE_MODEL"]] if os.environ.get("JUDGE_MODEL") else [])], cwd=ENVS, env={**os.environ, "JUDGE_VIA_OPENROUTER": "1", "OPENROUTER_API_KEY": KEY},
                        stdout=open(LOG / f"{env}.log", "a"), stderr=subprocess.STDOUT).returncode
    say(f"{env}: {len(dirs)} fleets, exit {rc}, spent so far ${usage() - start:.2f}")
say(f"done, spent ${usage() - start:.2f}")
