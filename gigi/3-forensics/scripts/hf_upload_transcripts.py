"""Upload the hero transcripts to Hugging Face for TQ (10 Nov 2026 call: text pre-work on the heroes gemma-3 27B and
Qwen3.6 27B, then NLA / J-space on activations captured from these same transcripts). Dataset:
https://huggingface.co/datasets/model-hermeneutics/gigi-forensics-transcripts (public, like gigi-lie-detection-activations).

What goes up, per run: rollout.log, the LAST step's messages.json (the full conversation as the model saw it, OpenAI
format; the per-step checkpoints repeat the same history and stay on the laptop / Modal) and state.json, final/ (grading.json
and the environment's final state where it exists), config.yaml, run_meta.json, error.json. Never blobs/, fs/, or any
file over 5 MB (listed in the manifest as skipped). Every staged file is redacted (the OpenRouter keys from the repo .env
and any sk-or-v1-<64 hex> shape) and the staging folder is grepped again before upload: one hit left = abort.
Idempotent: a run whose rollout.log is already in the repo is skipped (--force re-stages). Stops cleanly when the
Mac has under --min-free-gb left.

  FORENSICS_ENV_FILE=/abs/.env python3 scripts/hf_upload_transcripts.py --stages full_laptop base_gemma --model gemma-3-27b
  FORENSICS_ENV_FILE=/abs/.env python3 scripts/hf_upload_transcripts.py --stages hero8_gemma hero8_gemma_key hero8_gemma_all
  later stages (fleet agent): --stages hero9_qwen36 hero9_qwen36_key hero9_qwen36_all --model qwen3.6-27b ; --stages hero9_swap
Run from gigi/3-forensics (needs huggingface_hub: `uv run --with huggingface_hub python ...` from agent-interp-envs works too).
"""
import argparse, csv, json, os, pathlib, re, shutil, subprocess, sys, tempfile, time

H = pathlib.Path(__file__).resolve().parents[1]
R = H / "results" / "raw"
REPO = "model-hermeneutics/gigi-forensics-transcripts"
MAX_FILE = 5 * 1024 * 1024
MODEL_DIRS = {"gemma-3-27b": "google-gemma-3-27b-it", "qwen3.6-27b": "qwen-qwen3.6-27b", "llama-3.3-70b": "meta-llama-llama-3.3-70b-instruct",
              "qwen3-next-80b": "qwen-qwen3-next-80b-a3b-instruct"}
PRECISION = {"gemma-3-27b": "fp8 on DeepInfra via OpenRouter", "qwen3.6-27b": "fp8 on DeepInfra via OpenRouter (thinking off)",
             "llama-3.3-70b": "fp16 on CoreWeave via OpenRouter", "qwen3-next-80b": "fp8 via OpenRouter"}
# stage -> how the run could end (the setting), in plain words
SETTINGS = {
    "no check": "the paper's own loop: the first reply without a tool call ends the run; no notice, no reminder",
    "soft check": "the system prompt carries a notice that a reply without a tool call ends the run, and the first such reply gets one "
                  "question back (finish or keep working?); the second plain reply ends the run (agent.end_confirm)",
    "hard rule + example": "only a call to end_task ends the run; every plain reply gets a reminder (up to 10); plus a nine-line example "
                           "transcript in the system prompt (read a file, change it, run the tests, finish) (agent.end_task_required, agent.hack=worked_example)",
    "hard rule": "only a call to end_task ends the run; every plain reply gets a reminder (up to 10) (agent.end_task_required)",
}
STAGE_SETTING = {"full_laptop": "no check", "base_gemma": "no check", "hero2_gemma": "soft check", "hero2_gemma_key": "soft check",
                 "hero2_gemma_all": "soft check", "hero2_qwen36": "soft check", "hero2_qwen36_key": "soft check", "hero2_qwen36_all": "soft check",
                 "hero2_qwen36_topup": "soft check", "hero2_qwen36_all_topup": "soft check", "hero8_gemma": "hard rule + example",
                 "hero8_gemma_key": "hard rule + example", "hero8_gemma_all": "hard rule + example", "hero8_qwen36": "hard rule + example",
                 "hero8_qwen36_key": "hard rule + example", "hero8_qwen36_all": "hard rule + example", "hero9_gemma": "no check",
                 "hero9_gemma_key": "no check", "hero9_gemma_all": "no check", "hero9_qwen36": "no check", "hero9_qwen36_key": "no check",
                 "hero9_qwen36_all": "no check", "hero9_swap": "no check (swap continuations)"}
STAGE_DATE = {"full_laptop": "19-26 Oct 2026", "base_gemma": "Oct 2026", "hero2_gemma": "4-6 Nov 2026", "hero2_gemma_key": "4-6 Nov 2026",
              "hero2_gemma_all": "5-6 Nov 2026", "hero2_qwen36": "7-8 Nov 2026", "hero2_qwen36_key": "7-8 Nov 2026", "hero2_qwen36_all": "7-8 Nov 2026",
              "hero2_qwen36_topup": "8 Nov 2026", "hero2_qwen36_all_topup": "8 Nov 2026", "hero8_gemma": "9 Nov 2026", "hero8_gemma_key": "9 Nov 2026",
              "hero8_gemma_all": "9 Nov 2026", "hero8_qwen36": "9 Nov 2026", "hero8_qwen36_key": "9 Nov 2026", "hero8_qwen36_all": "9 Nov 2026"}


def keys_to_redact():
    env = pathlib.Path(os.environ.get("FORENSICS_ENV_FILE", H.parents[1] / ".env")).read_text()
    vals = []
    for name in ("OPENROUTER_API_KEY", "AGENT_OPENROUTER_API_KEY"):
        m = re.search(rf"^{name}=(.+)$", env, re.M)
        if m and len(m.group(1).strip()) >= 16:
            vals.append(m.group(1).strip().strip('"').strip("'"))
    tok = re.search(r"^HF_TOKEN=(.+)$", env, re.M)
    return vals, (tok.group(1).strip().strip('"').strip("'") if tok else None)


SHAPE = re.compile(r"sk-or-v1-[0-9a-f]{64}")
TAG = "[REDACTED-OPENROUTER-KEY]"


def redact_text(text, vals):
    for v in vals:
        text = text.replace(v, TAG)
    return SHAPE.sub(TAG, text)


def per_run_rows():
    """run path (relative, results/raw/...) -> (stage, environment, model, status, outcome, bad) from every per-run csv."""
    out, envmap = {}, {}
    for p in sorted((H / "results").glob("*per_run*.csv")):
        try:
            rows = list(csv.DictReader(open(p)))
        except Exception:
            continue
        for r in rows:
            run = r.get("run")
            if not run or "environment" not in r:
                continue
            out[run.rstrip("/")] = (r.get("stage", ""), r["environment"], r.get("model", ""), r.get("status", ""), r.get("outcome", ""), r.get("bad", ""))
            parts = pathlib.Path(run).parts   # results/raw/<stage>/<environment>/<env dir>/<model dir>/<fleet>/run-N
            if len(parts) >= 6 and parts[0] == "results":
                envmap[parts[4]] = parts[3]
    return out, envmap


def find_runs(stage, model_dir):
    root = R / stage
    if not root.is_dir():
        return []
    runs = []
    for p in root.rglob("run-*"):
        if p.is_dir() and re.fullmatch(r"run-\d+", p.name) and (model_dir is None or model_dir in p.parts) and \
                (any(p.glob("step-*/messages.json")) or (p / "rollout.log").exists()):
            runs.append(p)
    return sorted(runs)


def model_of(run):
    for tag, d in MODEL_DIRS.items():
        if d in run.parts:
            return tag
    return "unknown"


def stage_run(run, dest, vals):
    """Copy the allowed files of one run into dest, redacted. Returns (uploaded names, skipped notes)."""
    up, skipped = [], []

    def put(src, name):
        if src.stat().st_size > MAX_FILE:
            skipped.append(f"{name} ({src.stat().st_size // 1024 // 1024} MB)"); return
        d = dest / name; d.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = src.read_text()
            d.write_text(redact_text(text, vals))
        except UnicodeDecodeError:
            shutil.copy2(src, d)
        up.append(name)

    for name in ("rollout.log", "config.yaml", "run_meta.json", "error.json"):
        if (run / name).is_file():
            put(run / name, name)
    if not (run / "config.yaml").is_file() and (run.parent / "config.yaml").is_file():
        put(run.parent / "config.yaml", "config.yaml")   # the fleet's config is the run's config (same for every run of the fleet)
    steps = sorted(run.glob("step-*"), key=lambda q: int(q.name.split("-")[1]))
    if steps:
        last = steps[-1]
        for name in ("messages.json", "state.json"):
            if (last / name).is_file():
                put(last / name, name)
        (dest / "last_step.txt").write_text(f"{last.name}\n{len(steps)} step checkpoints on disk; only the last is here\n")
    fin = run / "final"
    if fin.is_dir():
        for f in sorted(fin.rglob("*")):
            if f.is_file() and f.suffix in (".json", ".txt", ".md", ".log", ".csv", ".yaml", ".py"):
                put(f, str(pathlib.Path("final") / f.relative_to(fin)))
    for name in ("blobs", "fs"):
        if (run / name).exists():
            skipped.append(f"{name}/ (data, not uploaded)")
    return up, skipped


def verify_clean(folder, vals):
    """Second, independent pass: grep the staging folder for any key value or key shape."""
    hits = 0
    for p in pathlib.Path(folder).rglob("*"):
        if p.is_file():
            try:
                t = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if SHAPE.search(t) or any(v in t for v in vals):
                hits += 1; print(f"KEY LEFT IN {p}", file=sys.stderr)
    return hits == 0


def readme(manifest_rows):
    by_stage = {}
    for r in manifest_rows:
        s = by_stage.setdefault(r["stage"], {"runs": 0, "models": set(), "envs": set()})
        s["runs"] += 1; s["models"].add(r["model"]); s["envs"].add(r["environment"])
    lines = ["---", "license: other", "pretty_name: gigi forensics transcripts", "---", "",
             "# Model forensics: agent transcripts of gemma-3 27B, Qwen3.6 27B (and Llama 3.3 70B) on the 29 task-gaming environments", "",
             "These are the full conversations from the SPAR model-forensics project (GG, Nov 2026): each model running the agent environments of "
             "\"Why do models task game?\", one folder per run. They are here so the team can work on the text (bag-of-words and token "
             "regressions as a stand-in for NLA and J-space) before the activations are captured on a GPU box by replaying these same transcripts.", "",
             "## What is in a run folder", "",
             "- `messages.json`: the whole conversation in OpenAI chat format (system, user, assistant with `tool_calls`, tool results), taken from the run's last step checkpoint. For gemma (text-mode tools) the assistant text also carries the ```tool_code``` block it wrote; the provider parsed it into `tool_calls`.",
             "- `rollout.log`: the same conversation as the harness printed it, with the tool results and the task's final result block.",
             "- `state.json`: the environment's state after the last step; `final/` the final verdicts where the task has a grader (`grading.json`, Claude Opus) or a final state file.",
             "- `config.yaml`, `run_meta.json`, `error.json` when present. `last_step.txt` says how many step checkpoints exist on the laptop.",
             "- Not here: the environment's data blobs and filesystem snapshots (`blobs/`, `fs/`) and any single file over 5 MB. `manifest.csv` lists what was skipped per run.", "",
             "## Settings (how a run could end)", ""]
    for k, v in SETTINGS.items():
        lines.append(f"- **{k}**: {v}.")
    lines += ["", "## Models and precision", ""]
    for tag, p in PRECISION.items():
        lines.append(f"- `{tag}` ({MODEL_DIRS[tag]}): {p}.")
    lines += ["", "Every call went through OpenRouter with the provider pinned and fallbacks off; sampling with each model's model-card settings "
              "(gemma temperature 1.0, top_p 0.95, top_k 64; Qwen3.6 non-thinking temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5).", "",
              "## Stages", "", "| stage folder | setting | when | models | runs | environments |", "|---|---|---|---|---|---|"]
    for s in sorted(by_stage):
        d = by_stage[s]
        lines.append(f"| `{s}` | {STAGE_SETTING.get(s, '?')} | {STAGE_DATE.get(s, '')} | {', '.join(sorted(d['models']))} | {d['runs']} | {len(d['envs'])} |")
    lines += ["", "`manifest.csv` has one row per run: stage, environment, environment folder, model, run path, status and the bad-act verdict "
              "(`bad` = 1 did the task's bad act, 0 did not, empty = no measurable value or not in a scored table), uploaded files, skipped files.", "",
              "Secrets were redacted before upload (`[REDACTED-OPENROUTER-KEY]`). Source and results pages: "
              "https://github.com/mandliya/spar-model-hermeneutics/tree/main/gigi/3-forensics (HERO_RESULTS.md)."]
    return "\n".join(lines) + "\n"


def free_gb():
    return shutil.disk_usage(str(H)).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", required=True)
    ap.add_argument("--model", default=None, help="tag filter: gemma-3-27b, qwen3.6-27b, llama-3.3-70b, qwen3-next-80b")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--public", action="store_true", help="create the repo public; default private (10 Nov 2026: GG approved the upload, the parent session chose private so the transcripts stay inside the org until the team decides)")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    vals, token = keys_to_redact()
    if not token:
        sys.exit("no HF_TOKEN in the env file")
    api = HfApi(token=token)
    model_dir = MODEL_DIRS[a.model] if a.model else None
    perrun, envmap = per_run_rows()
    if not a.dry_run:
        api.create_repo(a.repo, repo_type="dataset", private=not a.public, exist_ok=True)
    existing = set(api.list_repo_files(a.repo, repo_type="dataset")) if not a.dry_run else set()
    # the manifest is cumulative: read the one in the repo, add rows for what we upload now
    manifest = {}
    if "manifest.csv" in existing:
        p = api.hf_hub_download(a.repo, "manifest.csv", repo_type="dataset", cache_dir=tempfile.mkdtemp(prefix="hfman_"))
        for r in csv.DictReader(open(p)):
            manifest[r["run_path"]] = r
    log = H / "results" / "logs" / "hf_upload.txt"; log.parent.mkdir(parents=True, exist_ok=True)
    say = lambda *x: (print(*x, flush=True), open(log, "a").write(time.strftime("%F %T ") + " ".join(map(str, x)) + "\n"))
    for stage in a.stages:
        runs = find_runs(stage, model_dir)
        say(f"{stage}: {len(runs)} runs on disk" + (f" for {a.model}" if a.model else ""))
        if not runs:
            continue
        # group by environment folder: one upload (one commit) per environment
        by_env = {}
        for run in runs:
            rel = run.relative_to(R / stage).parts   # <environment>/<env dir>/<model dir>/<fleet>/run-N  (or without <environment>)
            by_env.setdefault(rel[0], []).append(run)
        for env_key, eruns in sorted(by_env.items()):
            if free_gb() < a.min_free_gb:
                say(f"STOP: only {free_gb():.1f} GB free on the Mac, under {a.min_free_gb}; stopped before {stage}/{env_key}"); return
            stg = pathlib.Path(tempfile.mkdtemp(prefix="hfstage_"))
            n_new = 0
            for run in eruns:
                rel = run.relative_to(R).as_posix()           # <stage>/<environment>/.../run-N
                run_path = f"results/raw/{rel}"
                if not a.force and f"{rel}/rollout.log" in existing:
                    continue
                dest = stg / rel
                up, skipped = stage_run(run, dest, vals)
                if not up:
                    shutil.rmtree(dest, ignore_errors=True); continue
                info = perrun.get(run_path)
                parts = run.relative_to(R / stage).parts
                env_dir = parts[1] if len(parts) >= 5 else parts[0]
                environment = info[1] if info else (parts[0] if len(parts) >= 5 else envmap.get(parts[0], parts[0]))
                manifest[run_path] = {"stage": stage, "environment": environment, "env_dir": env_dir, "model": info[2] if info and info[2] else model_of(run),
                                      "setting": STAGE_SETTING.get(stage, ""), "run_path": run_path, "repo_path": rel,
                                      "status": info[3] if info else "", "outcome": info[4] if info else "", "bad": info[5] if info else "",
                                      "files": " ".join(up), "skipped": "; ".join(skipped)}
                n_new += 1
            if n_new == 0:
                shutil.rmtree(stg, ignore_errors=True); say(f"  {stage}/{env_key}: already in the repo"); continue
            if not verify_clean(stg, vals):
                say(f"ABORT: a key survived redaction in the staging folder for {stage}/{env_key} ({stg}); nothing uploaded"); sys.exit(2)
            size_mb = sum(f.stat().st_size for f in stg.rglob("*") if f.is_file()) / 1e6
            if a.dry_run:
                say(f"  dry run: {stage}/{env_key}: {n_new} runs, {size_mb:.0f} MB would go up"); shutil.rmtree(stg, ignore_errors=True); continue
            for attempt in range(4):
                try:
                    api.upload_folder(folder_path=str(stg), path_in_repo="", repo_id=a.repo, repo_type="dataset",
                                      commit_message=f"{stage}/{env_key}: {n_new} runs")
                    break
                except Exception as e:
                    say(f"  upload attempt {attempt + 1} failed for {stage}/{env_key}: {e!r}"[:300]); time.sleep(30 * (attempt + 1))
            else:
                say(f"UPLOAD FAILED {stage}/{env_key}: staging kept at {stg}"); continue
            say(f"  uploaded {stage}/{env_key}: {n_new} runs, {size_mb:.0f} MB")
            shutil.rmtree(stg, ignore_errors=True)
        # manifest + README after each stage, so a stop mid-way leaves the repo consistent
        rows = sorted(manifest.values(), key=lambda r: (r["stage"], r["environment"], r["run_path"]))
        if rows and not a.dry_run:
            tmp = pathlib.Path(tempfile.mkdtemp(prefix="hfmeta_"))
            with open(tmp / "manifest.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
            (tmp / "README.md").write_text(readme(rows))
            api.upload_folder(folder_path=str(tmp), path_in_repo="", repo_id=a.repo, repo_type="dataset", commit_message=f"manifest + README after {stage}")
            shutil.rmtree(tmp, ignore_errors=True)
            say(f"  manifest: {len(rows)} runs in the repo")
    say("done")


if __name__ == "__main__":
    main()
