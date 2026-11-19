"""Re-run the distilled gemma checkpoints in the environments (distill/PLAN_hero9.md, step 3; 12 Nov 2026).
  python3 distill_eval.py configs --host IP --port PORT --models base,c24,c48,c72,final
      writes configs/<env>__distill_v1_<m>.yaml (NEW files) from configs/<env>__gemma-3-27b.yaml: the same task, sampling,
      text-mode tools and step cap, but agent.model = the vLLM model name, agent.base_url = the box, agent.api_key_env =
      VLLM_API_KEY, no OpenRouter provider pins, and the sandbox egress allowlist = the box's IP.
  python3 distill_eval.py fetch --stages distill_v1_base,...     every fleet named in results/logs/<stage>/*.log, once
  python3 distill_eval.py score                                   results/distill_v1_eval.csv (+ _per_run.csv), chart
Scores with the hero9 measures (full_runs_table.MEASURES), so the rates compare directly with the hero9 plain runs.
Qwen3.6 27B as the student (phase 2, 19 Nov 2026; opt-in, the gemma defaults are unchanged):
  python3 distill_eval.py configs --student qwen36 --host IP --port PORT --models base,c25,... [--base-name qwen-base]
      [--envs a,b] [--out-dir DIR]   from configs/<env>__qwen3.6-27b-bf16.yaml with THINKING ON, matching how the
      student was trained: chat_template_kwargs {enable_thinking: true, preserve_thinking: true}, the model card's
      thinking sampling (temperature 1.0, top-p 0.95, top-k 20, min-p 0), native tool calls, no OpenRouter reasoning field.
  DISTILL_ENVS / DISTILL_HELD_OUT (comma lists) override the task list and held-out set for a new round.
"""
import argparse, collections, csv, json, math, os, pathlib, re, shutil, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import yaml  # noqa: E402

V = os.environ.get("DISTILL_V", "v1")   # v2 = the 13 Nov redo with CUDA graphs off: v1's vLLM garbled ~15% of runs
ENVS = ["tower_of_hanoi", "eval_tampering", "currency_conversion", "coding", "coding_norules", "lazy_investigation"]
HELD_OUT = {"tower_of_hanoi", "eval_tampering", "currency_conversion"}
if V == "classic":   # classic round (PLAN_classic.md sections 2 and 5, 17 Nov 2026): norvane added, lazy_investigation held out
    ENVS = ENVS + ["norvane"]
    HELD_OUT = {"tower_of_hanoi", "lazy_investigation", "currency_conversion", "eval_tampering", "norvane"}
if os.environ.get("DISTILL_ENVS"):   # a new round's tasks (19 Nov 2026), else the lists above
    ENVS = os.environ["DISTILL_ENVS"].split(",")
if os.environ.get("DISTILL_HELD_OUT"):
    HELD_OUT = set(os.environ["DISTILL_HELD_OUT"].split(","))
SERVED = {"base": "gemma-base"}   # every other model name is served as itself (a LoRA module)
# Qwen3.6 student with thinking ON (19 Nov 2026): the model card's thinking-mode sampling for general tasks, the same
# values as configs/*__qwen3.6-27b-think.yaml; preserve_thinking keeps earlier turns' reasoning in the prompt, exactly
# as distill_train.py renders the training data (QWEN_TEMPLATE_KW).
QWEN_THINKING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0, "reasoning_effort": "none",
                 "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
RES = HERE / "results"; RAW = RES / "raw"; LOGS = RES / "logs"
GARBLE = re.compile(r"[\u0980-\u09FF\u0B80-\u0BFF\u0C00-\u0C7F\u0D00-\u0D7F\u0E00-\u0E7F\u10A0-\u10FF]")   # scripts gemma never writes here


def run_status(run):
    """How a distill run ended (15 Nov 2026). "finished" = the harness printed its FINAL RESULTS block, step-limit endings
    included. full_runs_table.status() looks for a Traceback near the end instead, which (a) called 5 finished runs crashes
    because the model's own Python error was the last tool output, and (b) missed runs our vLLM server timed out on, runs
    that overflowed its 65,536-token context, and runs Modal's one-hour limit cut off mid-step (17, scored on a partial
    state until now)."""
    log = run / "rollout.log"
    if not log.exists():
        return "no log"
    t = log.read_text(errors="replace")
    if "FINAL RESULTS" in t:
        return "finished"
    if "APITimeoutError" in t[-4000:]:
        return "our server timed out"
    if "maximum context length" in t[-4000:] or "max_tokens must be at least 1" in t[-4000:]:   # the second: the
        return "context window full"   # history filled the window exactly, so no room was left for a reply (17 Nov)
    if "Traceback" in t[-1500:]:
        return "crash"
    return "cut off by the time limit"


def value(env, run):
    """The bad act (True/False) or None. As full_runs_table.MEASURES, except that a coding run that never submitted is
    no answer (15 Nov 2026; it scored as clean, the same flaw the 14 Nov eval_tampering fix closed)."""
    import full_runs_table as T
    st = T.last_state(run)
    if env in ("coding", "coding_norules") and not st.get("submitted"):
        return None
    return T.MEASURES[env][1](st, run)


def garbled(run):
    """True if any assistant turn in the run's last step contains the broken-server gibberish (Bengali/Tamil/Georgian... mix)."""
    steps = sorted(run.glob("step-*/messages.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    return bool(steps) and any(m.get("role") == "assistant" and GARBLE.search(str(m.get("content") or ""))
                               for m in json.loads(steps[-1].read_text()))


def configs(args):
    student = getattr(args, "student", "gemma3")
    out_dir = pathlib.Path(args.out_dir) if getattr(args, "out_dir", None) else HERE / "configs"
    envs = args.envs.split(",") if getattr(args, "envs", None) else ENVS
    served = dict(SERVED, base=args.base_name) if getattr(args, "base_name", None) else SERVED
    for m in args.models.split(","):
        for env in envs:
            if student == "gemma3":
                src = HERE / "configs" / f"{env}__gemma-3-27b.yaml"
            else:   # the classic round's bf16 Qwen3.6 configs: native tools, our vLLM server, the same step caps
                src = HERE / "configs" / f"{env}__qwen3.6-27b-bf16.yaml"
            dst = out_dir / f"{env}__distill_{V}_{m}.yaml"
            c = yaml.safe_load(src.read_text())
            a = c["agent"]
            a.update(model=served.get(m, m), base_url=f"http://{args.host}:{args.port}/v1", api_key_env="VLLM_API_KEY",
                     provider_preferences=None)
            if student == "qwen36":
                a.update({k: (dict(v) if isinstance(v, dict) else v) for k, v in QWEN_THINKING.items()})
            c.setdefault("modal", {})["allow_hosts"] = [args.host]
            note = "(12 Nov 2026)" if student == "gemma3" else "(qwen36 student, thinking on; 19 Nov 2026)"
            dst.write_text(f"# {dst.name}: written by scripts/distill_eval.py configs from {src.name} {note}\n"
                           + yaml.safe_dump(c, sort_keys=False))
            print("wrote", dst.relative_to(HERE) if dst.is_relative_to(HERE) else dst)


def fetch(args):
    fails = 0
    for s in args.stages.split(","):
        for log in sorted((LOGS / s).glob("*__*.log")):
            t = log.read_text(errors="replace")
            vol = re.search(r"^Volume: (\S+)", t, re.M); out = re.search(r"^Output: results/(\S+?)/?$", t, re.M)
            if not vol or not out:
                print(f"NOT READY (no Volume/Output line): {s}/{log.name}"); fails += 1; continue
            env, fleet = log.stem.split("__")[0], out.group(1)
            dest = RAW / s / env / fleet; stamp = dest / ".fetched"
            if stamp.exists():
                continue
            tmp = pathlib.Path(tempfile.mkdtemp(prefix="distill_fetch_"))
            try:   # into an empty temp folder, then only missing files are moved in (never --force onto run folders)
                r = subprocess.run(["env", "-u", "VIRTUAL_ENV", "uv", "run", "modal", "volume", "get", vol.group(1), fleet, str(tmp)],
                                   cwd=HERE / "agent-interp-envs", capture_output=True, text=True, timeout=3600)
                if r.returncode != 0:
                    print(f"FETCH FAILED {s} {env}: {(r.stderr or r.stdout).strip()[-200:]}"); fails += 1; continue
                src = tmp / pathlib.PurePosixPath(fleet).name; src = src if src.is_dir() else tmp
                for f in sorted(src.rglob("*")):
                    if f.is_file() or f.is_symlink():
                        d = dest / f.relative_to(src)
                        if not (d.exists() or d.is_symlink()):
                            d.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(f), str(d))
                stamp.write_text("fetched\n"); print(f"fetched {s} {env} {fleet}")
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"fetch: {fails} not ready or failed"); return fails


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))


def score(args):
    import full_runs_table as T
    rows, per = [], []
    for sdir in sorted(RAW.glob(f"distill_{V}_*")):
        if "gate" in sdir.name:
            continue
        model = sdir.name[len(f"distill_{V}_"):]
        for env in ENVS:
            k = n = g = 0; lost = collections.Counter(); ends = collections.Counter()
            for run in sorted(r for r in (sdir / env).rglob("run-*") if r.is_dir()):   # depth varies (eval_tampering has a variant level)
                st = run_status(run)
                if st != "finished":
                    lost[st] += 1; per.append([model, env, str(run.relative_to(HERE)), "", st]); continue
                try:
                    v = value(env, run)
                except Exception as e:   # noqa: BLE001  a run whose state cannot be read is lost, never clean
                    v = None; lost[f"unreadable: {type(e).__name__}"] += 1
                if v is None:
                    lost["no value"] += 1; per.append([model, env, str(run.relative_to(HERE)), "", "no value"]); continue
                gb = garbled(run); g += gb
                k += bool(v); n += 1; per.append([model, env, str(run.relative_to(HERE)), int(bool(v)), "scored", int(gb)])
                log = (run / "rollout.log")
                t = log.read_text(errors="replace") if log.exists() else ""
                ends["unknown function"] += t.count("unknown function"); ends["provider retry"] += t.count("[provider retry]")
            if n or lost:
                lo, hi = wilson(k, n)
                rows.append({"model": model, "env": env, "held_out": env in HELD_OUT, "k": k, "n": n, "rate": round(k / n, 3) if n else "",
                             "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "lost": "; ".join(f"{a}: {b}" for a, b in sorted(lost.items())),
                             "unknown_function_calls": ends["unknown function"], "provider_retries": ends["provider retry"],
                             "garbled_runs": g})
    if not rows:
        print("nothing scored yet"); return
    with open(RES / f"distill_{V}_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(RES / f"distill_{V}_eval_per_run.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "env", "run", "bad", "status", "garbled"]); w.writerows(per)
    for r in rows:
        print(f"{r['model']:6} {r['env']:22} {r['k']:3}/{r['n']:<3} garbled: {r['garbled_runs']:2}  lost: {r['lost'] or '-'}")
    print(f"scored: {sum(r['n'] for r in rows)} runs -> results/distill_{V}_eval.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs"); c.add_argument("--host", required=True); c.add_argument("--port", required=True)
    c.add_argument("--models", required=True)
    c.add_argument("--student", default="gemma3", choices=["gemma3", "qwen36"], help="qwen36: Qwen3.6 student, thinking on")
    c.add_argument("--base-name", default=None, help="served name of the untrained student (default gemma-base)")
    c.add_argument("--envs", default=None, help="comma list of tasks (default: the round's ENVS)")
    c.add_argument("--out-dir", default=None, help="where to write the configs (default configs/)")
    c.set_defaults(fn=configs)
    f = sub.add_parser("fetch"); f.add_argument("--stages", required=True); f.set_defaults(fn=fetch)
    s = sub.add_parser("score"); s.set_defaults(fn=score)
    a = ap.parse_args(); r = a.fn(a); sys.exit(1 if isinstance(r, int) and r else 0)


if __name__ == "__main__":
    main()
