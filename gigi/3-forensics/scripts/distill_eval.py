"""Re-run the distilled gemma checkpoints in the environments (distill/PLAN_hero9.md, step 3; 12 Nov 2026).
  python3 distill_eval.py configs --host IP --port PORT --models base,c24,c48,c72,final
      writes configs/<env>__distill_v1_<m>.yaml (NEW files) from configs/<env>__gemma-3-27b.yaml: the same task, sampling,
      text-mode tools and step cap, but agent.model = the vLLM model name, agent.base_url = the box, agent.api_key_env =
      VLLM_API_KEY, no OpenRouter provider pins, and the sandbox egress allowlist = the box's IP.
  python3 distill_eval.py fetch --stages distill_v1_base,...     every fleet named in results/logs/<stage>/*.log, once
  python3 distill_eval.py score                                   results/distill_v1_eval.csv (+ _per_run.csv), chart
Scores with the hero9 measures (full_runs_table.MEASURES), so the rates compare directly with the hero9 plain runs.
"""
import argparse, collections, csv, json, math, pathlib, re, shutil, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import yaml  # noqa: E402

ENVS = ["tower_of_hanoi", "eval_tampering", "currency_conversion", "coding", "coding_norules", "lazy_investigation"]
HELD_OUT = {"tower_of_hanoi", "eval_tampering", "currency_conversion"}
SERVED = {"base": "gemma-base"}   # every other model name is served as itself (a LoRA module)
RES = HERE / "results"; RAW = RES / "raw"; LOGS = RES / "logs"


def configs(args):
    for m in args.models.split(","):
        for env in ENVS:
            src = HERE / "configs" / f"{env}__gemma-3-27b.yaml"; dst = HERE / "configs" / f"{env}__distill_v1_{m}.yaml"
            c = yaml.safe_load(src.read_text())
            a = c["agent"]
            a.update(model=SERVED.get(m, m), base_url=f"http://{args.host}:{args.port}/v1", api_key_env="VLLM_API_KEY",
                     provider_preferences=None)
            c.setdefault("modal", {})["allow_hosts"] = [args.host]
            dst.write_text(f"# {dst.name}: written by scripts/distill_eval.py configs from {src.name} (12 Nov 2026)\n"
                           + yaml.safe_dump(c, sort_keys=False))
            print("wrote", dst.relative_to(HERE))


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
    for sdir in sorted(RAW.glob("distill_v1_*")):
        model = sdir.name[len("distill_v1_"):]
        for env in ENVS:
            k = n = 0; lost = collections.Counter(); ends = collections.Counter()
            for run in sorted((sdir / env).glob("*/*/*/run-*")) + sorted((sdir / env).glob("*/*/run-*")):
                st = T.status(run)
                if st != "finished":
                    lost[st] += 1; per.append([model, env, str(run.relative_to(HERE)), "", st]); continue
                try:
                    v = T.MEASURES[env][1](T.last_state(run), run)
                except Exception as e:   # noqa: BLE001  a run whose state cannot be read is lost, never clean
                    v = None; lost[f"unreadable: {type(e).__name__}"] += 1
                if v is None:
                    lost["no value"] += 1; per.append([model, env, str(run.relative_to(HERE)), "", "no value"]); continue
                k += bool(v); n += 1; per.append([model, env, str(run.relative_to(HERE)), int(bool(v)), "scored"])
                log = (run / "rollout.log")
                t = log.read_text(errors="replace") if log.exists() else ""
                ends["unknown function"] += t.count("unknown function"); ends["provider retry"] += t.count("[provider retry]")
            if n or lost:
                lo, hi = wilson(k, n)
                rows.append({"model": model, "env": env, "held_out": env in HELD_OUT, "k": k, "n": n, "rate": round(k / n, 3) if n else "",
                             "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "lost": "; ".join(f"{a}: {b}" for a, b in sorted(lost.items())),
                             "unknown_function_calls": ends["unknown function"], "provider_retries": ends["provider retry"]})
    if not rows:
        print("nothing scored yet"); return
    with open(RES / "distill_v1_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(RES / "distill_v1_eval_per_run.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "env", "run", "bad", "status"]); w.writerows(per)
    for r in rows:
        print(f"{r['model']:6} {r['env']:22} {r['k']:3}/{r['n']:<3} lost: {r['lost'] or '-'}")
    print(f"scored: {sum(r['n'] for r in rows)} runs -> results/distill_v1_eval.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("configs"); c.add_argument("--host", required=True); c.add_argument("--port", required=True)
    c.add_argument("--models", required=True); c.set_defaults(fn=configs)
    f = sub.add_parser("fetch"); f.add_argument("--stages", required=True); f.set_defaults(fn=fetch)
    s = sub.add_parser("score"); s.set_defaults(fn=score)
    a = ap.parse_args(); r = a.fn(a); sys.exit(1 if isinstance(r, int) and r else 0)


if __name__ == "__main__":
    main()
