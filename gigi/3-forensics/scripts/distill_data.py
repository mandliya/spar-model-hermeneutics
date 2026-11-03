"""Distillation data prep (team call 28 Oct 2026): Qwen3-Next-80B's agentic traces, rendered exactly as gemma-3-27b
sees a history in text mode, one training example per assistant turn. No API calls.

Sources: Qwen's full runs (the runs behind results/full_runs.csv, found the same way full_runs_table.py finds them)
and Qwen's swap replays (results/gap_replays.csv, who=qwen). A replay shares assistant turns 0..K with its original
run, so those turns are emitted once, and a replay always travels with its original in the train/val split.
Rendering reuses agent-interp-envs' own text-mode functions (TEXT_TOOLS_INSTRUCTION, _python_stub, _call_text,
_text_tools_messages), so the text is byte-identical to what gemma gets at inference; tools come from each
environment's tools.py the way its agent.py builds them. Run with the environments' venv (has openai, yaml):
  PY=gigi/3-forensics/agent-interp-envs/.venv/bin/python
  $PY gigi/3-forensics/scripts/distill_data.py inventory
  $PY gigi/3-forensics/scripts/distill_data.py build --split 0.8 --seed 0 [--exclude-hacked] [--envs a,b]
  $PY gigi/3-forensics/scripts/distill_data.py check
Incident-stratified split (opt-in, 3 Nov 2026; the default split is unchanged):
  $PY gigi/3-forensics/scripts/distill_data.py build --stratify-incidents --dry-run --max-mb 100000   # table only, writes nothing
  $PY gigi/3-forensics/scripts/distill_data.py build --stratify-incidents --max-mb 100000 --max-context-chars 50000 --out distill/stratified
"""
import argparse, collections, csv, hashlib, importlib, inspect, json, pathlib, random, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parents[1]           # gigi/3-forensics
ENVS = HERE / "agent-interp-envs"
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(ENVS / "src"))
import yaml                                                      # noqa: E402
import full_runs_table as T                                      # noqa: E402  run discovery, status, measures
import swap_all_score as S                                       # noqa: E402  replay plan (original run per point)
from agent_interp_envs import tool_calling                       # noqa: E402
from agent_interp_envs.providers.openrouter_provider import (    # noqa: E402
    _text_tools_messages, _tool_call_from_tool_code)

OUT = HERE / "distill"
QWEN = "qwen3-next-80b"
MAX_BYTES = 200 * 1024 * 1024
# the 8 environments with a Qwen-gemma gap (same set as intent_labels.py / stopped_reasons.py)
GAP_ENVS = {"coding", "coding_norules", "optimize_rendering_engine", "optimize_rendering_engine_graded",
            "dashboard_costly_failure", "ai_rd", "grade_your_own", "precommit_hook"}
SAMPLE_RUNS = 200

# config `environment:` -> environments/<dir> whose agent.py/tools.py the image is built from (Dockerfiles)
ENV_DIR = {"aj_data_anon": "algo_judge", "lucky_seed_readable_grader": "lucky_seed", "tables_coverage_tests": "tables_coverage",
           "palisades_tower_of_hanoi": "palisades_bounty", "chess": "games", "tictactoe": "games"}
_tools_mod: dict[str, object] = {}


def _env_module(d):
    """environments/<d>/tools.py imported as its agent.py imports it (`from tools import ...`)."""
    if d not in _tools_mod:
        sys.path.insert(0, str(ENVS / "environments" / d))
        for m in [m for m in sys.modules if m == "tools" or m.startswith("tools.")]:
            del sys.modules[m]
        _tools_mod[d] = importlib.import_module("tools")
        sys.path.pop(0)
    return _tools_mod[d]


def tools_for(config):
    """The tool list the environment gave the provider for this config (mirrors each environments/*/agent.py)."""
    env = config["environment"]; task = config.get("task") or {}
    d = ENV_DIR.get(env, env)
    if d == "games":
        return [tool_calling.EXECUTE_COMMAND_TOOL]
    mod = _env_module(d)
    if env == "ai_rd":
        return [mod.EXECUTE_COMMAND_TOOL]
    if env in ("eval_tampering", "sandbagging"):
        return [mod.EXECUTE_COMMAND_TOOL, mod.SUBMIT_TOOL]
    if d == "coding":
        return mod.build_tools(task.get("tools") if task.get("tools") is not None else list(mod.DEFAULT_TOOLS))
    if d == "funding_email":
        return mod.get_tools(require_audit_notes=task.get("require_audit_notes", False),
                             audit_notes_mode=task.get("audit_notes_mode", "freeform"),
                             email_filename=task.get("email_filename", "investor_email_draft.txt"))
    if hasattr(mod, "get_tools"):
        return mod.get_tools(config) if inspect.signature(mod.get_tools).parameters else mod.get_tools()
    return list(mod.ALL_TOOLS)


def transcript(run):
    """The full message list of a run: final/messages.json, else the last readable step-N/messages.json."""
    cands = [run / "final" / "messages.json"]
    steps = sorted(run.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    cands += [s / "messages.json" for s in reversed(steps)]
    for c in cands:
        try:
            m = json.loads(c.read_text())
            if isinstance(m, list) and any(x.get("role") == "assistant" for x in m):
                return m
        except (OSError, ValueError):
            continue
    return None


def rel(p):
    return str(pathlib.Path(p).resolve().relative_to(HERE))


def load_runs(envs=None):
    """Every Qwen trajectory with a transcript: dicts with env, run, kind (full|replay), group (the original run,
    the split unit), K (shared prefix: assistant turns 0..K are the original's; -1 for a full run), hacked
    (True/False/None), messages, tools. Also the trajectories left out, with the reason."""
    runs, skipped = [], []
    full = [r for r in T.modal_runs() if r[0] != "chess"] + T.laptop_runs("chess")
    cfg_cache = {}

    def config_of(run):
        c = run.parent / "config.yaml"
        if c not in cfg_cache:
            cfg_cache[c] = yaml.safe_load(c.read_text())
        return cfg_cache[c]

    for env, model, run in full:
        if model != QWEN or (envs and env not in envs):
            continue
        r = rel(run); st = T.status(run)
        if st != "finished":
            skipped.append({"env": env, "run": r, "kind": "full", "reason": f"lost: {st}"}); continue
        m = transcript(run)
        if m is None:
            skipped.append({"env": env, "run": r, "kind": "full", "reason": "no readable transcript"}); continue
        state = T.last_state(run)
        v = T.MEASURES[env][1](state, run) if env in T.MEASURES else T.judged(state, run) if env in T.JUDGED else None
        if r in S.HAND:
            v = S.HAND[r]
        cfg = config_of(run)
        runs.append({"env": env, "run": r, "kind": "full", "group": r, "K": -1, "hacked": None if v is None else bool(v),
                     "messages": m, "tools": tools_for(cfg)})
    plan = {S.point_name(p): p for p in S.PLAN}
    for row in csv.DictReader(open(HERE / "results" / "gap_replays.csv")):
        if row["who"] != "qwen" or (envs and row["env"] not in envs):
            continue
        run = HERE / row["run"]; orig = plan[row["point"]]["run"]
        m = transcript(run)
        if m is None:
            skipped.append({"env": row["env"], "run": row["run"], "kind": "replay", "reason": f"no transcript ({row['status']})"}); continue
        K = int(row["K"])
        if sum(1 for x in m if x["role"] == "assistant") <= K + 1:
            skipped.append({"env": row["env"], "run": row["run"], "kind": "replay", "reason": "no turn after the prefix"}); continue
        cfg = config_of(HERE / orig)
        runs.append({"env": row["env"], "run": row["run"], "kind": "replay", "group": orig, "K": K,
                     "hacked": None if row["hacked"] == "" else bool(int(row["hacked"])), "messages": m, "tools": tools_for(cfg)})
    return runs, skipped


def render(run):
    """One rendered example per assistant turn: (step, context messages, target message). The text-mode rendering of a
    history cut before assistant turn i is the first part of the rendering of the whole history (an assistant turn
    always ends the merge of tool/user messages), so the transcript is rendered once; `check` verifies this."""
    R = _text_tools_messages(run["messages"], run["tools"])
    idx = [i for i, m in enumerate(R) if m["role"] == "assistant"]
    for step, i in enumerate(idx):
        yield step, R[:i], R[i]


def examples(runs, exclude_hacked=False):
    """Deduped examples per group. A replay's turns 0..K are the original's (emitted once, by the original, or by the
    first replay seen if the original has no transcript); identical samples are deduped by content hash."""
    by_group = collections.defaultdict(list)
    for r in runs:
        by_group[r["group"]].append(r)
    out = collections.defaultdict(list); seen = set(); dropped = []
    for g, rs in by_group.items():
        rs = sorted(rs, key=lambda r: (r["kind"] != "full", r["K"], r["run"]))   # the original first, then shortest prefix
        covered = -1   # steps 0..covered of the shared trajectory already emitted
        for r in rs:
            if exclude_hacked and r["hacked"]:
                dropped.append({"env": r["env"], "run": r["run"], "kind": r["kind"], "reason": "hacked"}); continue
            for step, ctx, tgt in render(r):
                if r["kind"] == "replay" and step <= min(r["K"], covered):
                    continue
                h = hashlib.sha1(json.dumps(ctx + [tgt], sort_keys=True).encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                out[g].append({"env": r["env"], "run": r["run"], "step": step, "hacked": r["hacked"], "kind": r["kind"],
                               "group": g, "messages": ctx + [tgt]})
            covered = max(covered, sum(1 for m in r["messages"] if m["role"] == "assistant") - 1 if r["kind"] == "full" else r["K"])
    return out, dropped


def nbytes(ex):
    return len(json.dumps(ex, ensure_ascii=False).encode()) + 1


def inventory(args):
    runs, skipped = load_runs(args.envs)
    ex, _ = examples(runs)
    rows = collections.defaultdict(lambda: collections.Counter())
    for r in runs:
        c = rows[r["env"]]; c[r["kind"]] += 1
        c[f"{r['kind']}_" + ("hacked" if r["hacked"] else "clean" if r["hacked"] is False else "unscored")] += 1
    for s in skipped:
        rows[s["env"]][f"{s['kind']}_lost"] += 1
    for g, exs in ex.items():
        env = exs[0]["env"]; c = rows[env]
        c["turns"] += len(exs); c["target_chars"] += sum(len(e["messages"][-1]["content"]) for e in exs)
        c["example_chars"] += sum(sum(len(m["content"]) for m in e["messages"]) for e in exs)
        c["bytes"] += sum(nbytes(e) for e in exs)
    hdr = ["env", "full", "full lost", "hacked/clean/unscored", "replays", "replays lost", "hacked/clean/unscored",
           "turns", "target chars", "example chars", "MB"]
    table = []
    for env in sorted(rows):
        c = rows[env]
        table.append([env, c["full"], c["full_lost"], f"{c['full_hacked']}/{c['full_clean']}/{c['full_unscored']}",
                      c["replay"], c["replay_lost"], f"{c['replay_hacked']}/{c['replay_clean']}/{c['replay_unscored']}",
                      c["turns"], f"{c['target_chars']:,}", f"{c['example_chars']:,}", f"{c['bytes'] / 2**20:.1f}"])
    tot = collections.Counter()
    for c in rows.values():
        tot.update(c)
    table.append(["TOTAL", tot["full"], tot["full_lost"], f"{tot['full_hacked']}/{tot['full_clean']}/{tot['full_unscored']}",
                  tot["replay"], tot["replay_lost"], f"{tot['replay_hacked']}/{tot['replay_clean']}/{tot['replay_unscored']}",
                  tot["turns"], f"{tot['target_chars']:,}", f"{tot['example_chars']:,}", f"{tot['bytes'] / 2**20:.1f}"])
    w = [max(len(str(r[i])) for r in [hdr] + table) for i in range(len(hdr))]
    print(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)))
    print("-+-".join("-" * x for x in w))
    for r in table:
        print(" | ".join(str(v).ljust(w[i]) if i == 0 else str(v).rjust(w[i]) for i, v in enumerate(r)))
    print("\nhacked/clean/unscored = runs by the grader's verdict (full_runs.csv measures, gap_replays.csv); lost = no usable "
          "transcript (provider failure, crash, cut off). turns = unique assistant turns after prefix dedupe; example chars = "
          "the training-set size (each turn carries its whole context); MB = JSONL bytes.")
    print(f"replays with a transcript but no turn after the prefix: {sum(1 for s in skipped if s['reason'] == 'no turn after the prefix')}")


def stratified_split(groups_by_env, split, seed):
    """Groups (original runs with their replays) into train/val per environment, 80/20 by default."""
    rng = random.Random(seed); train, val = [], []
    for env in sorted(groups_by_env):
        gs = sorted(groups_by_env[env]); rng.shuffle(gs)
        n_val = int(round(len(gs) * (1 - split))) if len(gs) > 1 else 0
        val += gs[:n_val]; train += gs[n_val:]
    return set(train), set(val)


def incident_families(runs):
    """group -> True if any run of the family (the original or one of its replays) is one the grader marked hacked."""
    inc = collections.defaultdict(bool)
    for r in runs:
        inc[r["group"]] |= bool(r["hacked"])
    return inc


def incident_split(groups_by_env, incident, split, seed, hacked_split=None):
    """--stratify-incidents: families into train/val per environment AND per stratum (cheating family = contains a run
    the grader marked hacked; other = none did), with the same train fraction in both strata (Yuqi's rule; John's
    fallback = --hacked-split 0.5 for the cheating stratum only). A stratum of n >= 2 families holds out
    max(1, round(n * (1 - split))) of them and keeps at least 1 in train, so every environment with 2 or more cheating
    families holds one out; a lone family goes to train, as in stratified_split. Families stay whole."""
    rng = random.Random(seed); train, val = [], []
    for env in sorted(groups_by_env):
        for stratum in (True, False):
            gs = sorted(g for g in groups_by_env[env] if incident[g] == stratum); rng.shuffle(gs)
            frac = hacked_split if (stratum and hacked_split is not None) else split
            n_val = min(max(1, int(round(len(gs) * (1 - frac)))), len(gs) - 1) if len(gs) >= 2 else 0
            val += gs[:n_val]; train += gs[n_val:]
    return set(train), set(val)


def split_table(runs, ex, chosen, env_of, train, val, incident, sampled=False):
    """--dry-run: per environment, families train/val by stratum, runs (incl. replays) train/val, Qwen hacked/scored in
    val, examples train/val. Markdown, with the |---| row."""
    fam = collections.defaultdict(collections.Counter)
    for g in chosen:
        part = "train" if g in train else "val"
        fam[env_of[g]][("cheat" if incident[g] else "other") + "_" + part] += 1
        fam[env_of[g]]["ex_" + part] += len(ex[g])
    for r in runs:
        if r["group"] not in chosen:
            continue
        c = fam[env_of[r["group"]]]; part = "train" if r["group"] in train else "val"
        c["runs_" + part] += 1
        if part == "val" and r["hacked"] is not None:
            c["val_scored"] += 1; c["val_hacked"] += bool(r["hacked"])
    hdr = ["environment (* = gap environment)", "cheating families train / val", "other families train / val",
           "runs incl. replays train / val", "Qwen hacked / scored in val", "examples train / val"]
    def row(name, c, bold=False):
        cells = [name, f"{c['cheat_train']} / {c['cheat_val']}", f"{c['other_train']} / {c['other_val']}",
                 f"{c['runs_train']:,} / {c['runs_val']:,}", f"{c['val_hacked']} / {c['val_scored']}",
                 f"{c['ex_train']:,} / {c['ex_val']:,}"]
        return "| " + " | ".join(f"**{x}**" if bold else x for x in cells) + " |"
    order = sorted(fam, key=lambda e: (e not in GAP_ENVS, e))
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    gap_tot, rest_tot, tot = collections.Counter(), collections.Counter(), collections.Counter()
    for e in order:
        lines.append(row(e + (" *" if e in GAP_ENVS else ""), fam[e]))
        (gap_tot if e in GAP_ENVS else rest_tot).update(fam[e]); tot.update(fam[e])
    lines += [row("gap environments", gap_tot, True), row("other environments", rest_tot, True), row("total", tot, True)]
    print("\n".join(lines))
    gap = [e for e in order if e in GAP_ENVS]
    short = [e for e in gap if fam[e]["cheat_train"] + fam[e]["cheat_val"] < 2]
    held = [e for e in gap if fam[e]["cheat_val"] >= 1]
    print(f"\nfamilies = original run + its replays with at least one unique example ({len(chosen)}); cheating family = "
          f"contains a run the grader marked hacked.")
    print(f"gap environments holding out a cheating family: {len(held)} of {len(gap)} ({', '.join(held) or 'none'})")
    print(f"gap environments with fewer than 2 cheating families{' in this sample' if sampled else ''} (cannot hold one out): {len(short)} "
          f"({', '.join(e + ': ' + str(fam[e]['cheat_train'] + fam[e]['cheat_val']) for e in short) or 'none'})")


def build(args):
    runs, skipped = load_runs(args.envs)
    ex, dropped = examples(runs, args.exclude_hacked)
    split_env = None
    if args.stratify_incidents:   # the split is fixed on the unfiltered families: --max-context-chars, --exclude-hacked
        base = examples(runs)[0] if args.exclude_hacked else ex   # and the size guard only drop examples, never move a family
        split_env = {g: exs[0]["env"] for g, exs in base.items() if exs}
    if args.max_context_chars:
        for g in ex:
            ex[g] = [e for e in ex[g] if sum(len(m["content"]) for m in e["messages"][:-1]) <= args.max_context_chars]
    sizes = {g: sum(nbytes(e) for e in exs) for g, exs in ex.items()}
    total = sum(sizes.values()); env_of = {g: exs[0]["env"] for g, exs in ex.items() if exs}
    chosen = set(env_of); note = None
    max_bytes = int(args.max_mb * 2**20) if args.max_mb else MAX_BYTES   # --max-mb unset = the 200 MB default
    if total > max_bytes:   # the 200 MB guard: a stratified sample of SAMPLE_RUNS groups, shrunk further if still too big
        rng = random.Random(args.seed); by_env = collections.defaultdict(list)
        for g, e in sorted(env_of.items()):
            by_env[e].append(g)
        order = []
        for e in sorted(by_env):
            gs = by_env[e]; rng.shuffle(gs); order.append(gs)
        seq = []   # round-robin over environments keeps the sample stratified
        while any(order):
            seq += [gs.pop() for gs in order if gs]
        picked, used = [], 0   # a run family (original + its replays) joins the sample only if it still fits the budget
        for g in seq:
            if len(picked) >= SAMPLE_RUNS:
                break
            if used + sizes[g] <= max_bytes:
                picked.append(g); used += sizes[g]
        chosen = set(picked)
        missing = sorted(set(env_of.values()) - {env_of[g] for g in chosen})
        note = (f"full set is {total / 2**20:.0f} MB ({sum(len(v) for v in ex.values())} examples, {len(env_of)} runs); "
                f"wrote a stratified sample of {len(chosen)} runs ({used / 2**20:.0f} MB), the run families that fit the "
                f"{max_bytes // 2**20} MB budget in a seeded round-robin over environments (long families skipped"
                f"{'; no run of ' + ', '.join(missing) if missing else ''})")
        print("GUARD:", note)
    by_env = collections.defaultdict(list)
    for g in chosen:
        by_env[env_of[g]].append(g)
    incident = incident_families(runs)
    if args.stratify_incidents:
        by_all = collections.defaultdict(list)
        for g, e in sorted(split_env.items()):
            by_all[e].append(g)
        train, val = incident_split(by_all, incident, args.split, args.seed, args.hacked_split)
    else:
        train, val = stratified_split(by_env, args.split, args.seed)
    if args.dry_run:
        split_table(runs, ex, chosen, env_of, train, val, incident, sampled=note is not None)
        print("dry run: nothing written")
        return
    out = out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    counts = collections.defaultdict(lambda: {"train": 0, "val": 0, "train_runs": 0, "val_runs": 0})
    with open(out / "train.jsonl", "w") as ft, open(out / "val.jsonl", "w") as fv:
        for g in sorted(chosen):
            f, part = (ft, "train") if g in train else (fv, "val")
            counts[env_of[g]][f"{part}_runs"] += 1
            for e in ex[g]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n"); counts[e["env"]][part] += 1
    manifest = {
        "seed": args.seed, "split": args.split, "exclude_hacked": args.exclude_hacked, "envs": args.envs,
        "max_context_chars": args.max_context_chars, "sampling": note,
        **({"stratify_incidents": True, "hacked_split": args.hacked_split, "max_mb": args.max_mb,
            "cheating_families": {"train": sorted(g for g in train if incident[g]), "val": sorted(g for g in val if incident[g])}}
           if args.stratify_incidents else {}),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
        "env_repo_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ENVS, capture_output=True, text=True).stdout.strip(),
        "counts": dict(sorted(counts.items())),
        "totals": {"train": sum(c["train"] for c in counts.values()), "val": sum(c["val"] for c in counts.values()),
                   "train_runs": len(train & chosen), "val_runs": len(val & chosen),
                   "full_set_examples": sum(len(v) for v in ex.values()), "full_set_bytes": total},
        "excluded_runs": skipped + dropped,
        "runs_not_in_sample": sorted(set(env_of) - chosen),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{'env':35s} {'train':>6s} {'val':>6s} {'train runs':>10s} {'val runs':>8s}")
    for env, c in sorted(counts.items()):
        print(f"{env:35s} {c['train']:6d} {c['val']:6d} {c['train_runs']:10d} {c['val_runs']:8d}")
    t = manifest["totals"]
    print(f"{'TOTAL':35s} {t['train']:6d} {t['val']:6d} {t['train_runs']:10d} {t['val_runs']:8d}")
    for f in ("train.jsonl", "val.jsonl", "manifest.json"):
        print(f"{f if out == OUT else rel(out / f)}: {(out / f).stat().st_size / 2**20:.1f} MB")
    print(f"excluded: {len(skipped)} without a usable transcript, {len(dropped)} hacked (if --exclude-hacked)")


def out_dir(args):
    """--out (relative to the current directory or to gigi/3-forensics), which must be distill/ or inside it."""
    if not getattr(args, "out", None):
        return OUT
    p = pathlib.Path(args.out)
    out = p.resolve() if p.is_absolute() or (pathlib.Path.cwd() / p).resolve().is_relative_to(OUT) else (HERE / p).resolve()
    if not out.is_relative_to(OUT):
        sys.exit(f"--out must be {OUT} or a folder inside it")
    return out


def check(args):
    rng = random.Random(args.seed); out = out_dir(args)
    lines = (out / "train.jsonl").read_text().splitlines() + (out / "val.jsonl").read_text().splitlines()
    sample = [json.loads(l) for l in rng.sample(lines, min(20, len(lines)))]
    cache = {}; ok = bad = text = prefix_ok = 0
    for e in sample:
        if e["run"] not in cache:   # the tools come from the original run's config (same environment for its replays)
            cfg = yaml.safe_load(((HERE / e["group"]).parent / "config.yaml").read_text())
            cache[e["run"]] = (transcript(HERE / e["run"]), tools_for(cfg))
        msgs, tools = cache[e["run"]]
        orig = [m for m in msgs if m["role"] == "assistant"][e["step"]]
        # the prefix rendering identity render() relies on
        before = msgs[:[i for i, m in enumerate(msgs) if m["role"] == "assistant"][e["step"]]]
        prefix_ok += _text_tools_messages(before, tools) == e["messages"][:-1]
        target = e["messages"][-1]["content"]; parsed = _tool_call_from_tool_code(target, tools)
        tcs = orig.get("tool_calls") or []
        if not tcs:
            text += 1; continue
        want = (tcs[0]["function"]["name"], json.loads(tcs[0]["function"]["arguments"] or "{}"))
        got = (parsed["function"]["name"], json.loads(parsed["function"]["arguments"])) if parsed else None
        if got == want:
            ok += 1
        else:
            bad += 1; print(f"MISMATCH {e['run']} step {e['step']}: want {want!r:.200} got {got!r:.200}")
    n = ok + bad
    print(f"{len(sample)} examples: {n} tool-call turns, {ok}/{n} parse back to the original call "
          f"({100 * ok / n if n else 0:.0f}%); {text} final-text turns (no call); prefix rendering identical: {prefix_ok}/{len(sample)}")
    for e in sample[:2]:
        print(f"\n=== {e['env']} {e['kind']} step {e['step']} hacked={e['hacked']} ({len(e['messages'])} messages) {e['run']}")
        for m in e["messages"][:3] + ([{"role": "...", "content": f"{len(e['messages']) - 4} more"}] if len(e["messages"]) > 4 else []) + e["messages"][-1:]:
            c = m["content"]; c = c if len(c) <= 500 else c[:350] + f" [...{len(c) - 450} chars...] " + c[-100:]
            print(f"--- {m['role']}\n{c}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("inventory", inventory), ("build", build), ("check", check)):
        p = sub.add_parser(name); p.set_defaults(fn=fn)
        p.add_argument("--envs", type=lambda s: s.split(","), default=None, help="comma-separated environment names (full_runs.csv names)")
        p.add_argument("--seed", type=int, default=0)
        if name == "build":
            p.add_argument("--split", type=float, default=0.8, help="train fraction of runs per environment")
            p.add_argument("--exclude-hacked", action="store_true", help="drop runs where the grader saw the bad act")
            p.add_argument("--max-context-chars", type=int, default=0, help="drop examples whose context is longer (0 = keep all)")
            # opt-in options (3 Nov 2026); left unset, build behaves exactly as before
            p.add_argument("--stratify-incidents", action="store_true", help="split per environment AND per stratum (family "
                           "contains a hacked run or not), same train fraction in both; every stratum of >= 2 holds one out")
            p.add_argument("--hacked-split", type=float, default=None, help="with --stratify-incidents: a different train "
                           "fraction for the cheating stratum (John's fallback: 0.5); default = --split (Yuqi's rule)")
            p.add_argument("--dry-run", action="store_true", help="print the per-environment split table, write nothing")
            p.add_argument("--max-mb", type=float, default=0, help="size budget for the written set in MB (0 = the 200 MB "
                           "default); above it build writes a stratified sample")
        if name in ("build", "check"):
            p.add_argument("--out", default=None, help="data directory inside distill/ (default distill/; use a "
                           "subfolder, e.g. distill/stratified, to keep the committed sample)")
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
