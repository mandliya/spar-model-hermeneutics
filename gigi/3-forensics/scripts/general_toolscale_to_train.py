"""general_toolscale_to_train.py (24 Nov 2026): the MIDDLE general training set for Qwen3.6 27B (phase2/PLAN_next_24nov.md,
direction 2; phase2/general_set/PIPELINE.md). It turns the public dataset
zake7749/deepseek-v4-pro-agent-tool-calling-trajectory (DeepSeek V4 Pro doing ToolScale customer-service conversations
with tools: bank, movie, ecommerce, medicine, basketball) into exactly the training format of the DeepSeek round (p2ds,
distill/p2_deepseekv4_v1): one JSON line per conversation with "format": "qwen36", the tool schemas, native Qwen3.6
messages (the teacher's reasoning in reasoning_content, tool calls with their arguments as a dict, tool results as role
"tool") and train_turns = the assistant turns that carry the loss.

Nothing here is re-implemented. The functions that made p2ds's data are loaded from scripts/distill_data.py as they are
(qwen_messages, trajectories, stratified_split, msg_chars, ...), so the cut (--max-context-chars 105000, then
distill_train.py's 32,768-token cut), the 80/20 split per domain with seed 0 and the file layout are p2ds's own code.
They are read out of the file by name instead of imported, because importing distill_data.py needs the harness's
environment and our results tables, which this job does not use.

The parquet has 2,510 rows that are 284 conversations: each row is the same conversation cut at one more assistant
turn. Only the longest row per (domain, id) is kept; the others are checked to be its first part.

LICENCE: the dataset states no licence. The converted files must not be committed to git or published anywhere (not to
Hugging Face either). --out must be outside the repo, and the script refuses a folder inside it.

The parquet is downloaded data, so run with -I from another folder than the data:
  uv run --no-project --with pyarrow --with transformers==5.23.0 --with jinja2 python -I \
     gigi/3-forensics/scripts/general_toolscale_to_train.py build --parquet <dir>/train.parquet --out <folder outside the repo> \
     --tokenizer <folder with Qwen3.6's tokenizer.json, tokenizer_config.json, chat_template.jinja>
  ... general_toolscale_to_train.py stats --data <folder with train.jsonl> --tokenizer <folder>     # sizes of any set in
      this format (p2ds's own train.jsonl too), counted with distill_train.py's own encoder
Without --tokenizer, build still writes the files and reports characters only.
"""
import argparse, ast, collections, hashlib, json, pathlib, random, statistics, subprocess, sys

SCRIPTS = pathlib.Path(__file__).resolve().parent
SOURCE = "zake7749/deepseek-v4-pro-agent-tool-calling-trajectory"
MAX_CONTEXT_CHARS = 105_000   # p2ds's build flag (distill/p2_deepseekv4_v1/manifest.json)
MAX_LEN = 32_768              # p2ds's training cut (p2q_train.sh --max-len)
NEEDED = ("qwen_messages", "student_messages", "msg_chars", "_no_reasoning", "_student_fields", "trajectories",
          "nbytes", "stratified_split")


def p2ds_functions():
    """The data-build functions of scripts/distill_data.py, unchanged, with the student set to qwen36 as in p2ds's build
    (`--student qwen36 --trajectories`). Only the named function definitions are executed, none of the file's imports."""
    src = (SCRIPTS / "distill_data.py").read_text()
    tree = ast.parse(src)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in NEEDED]
    missing = set(NEEDED) - {n.name for n in defs}
    if missing:
        sys.exit(f"distill_data.py no longer defines {sorted(missing)}: this converter must be re-checked against it")
    ns = {"json": json, "collections": collections, "hashlib": hashlib, "random": random,
          "STUDENT": "qwen36", "_QWEN_SKIP": {}, "_text_tools_messages": None}
    exec(compile(ast.Module(body=defs, type_ignores=[]), str(SCRIPTS / "distill_data.py"), "exec"), ns)   # noqa: S102
    return ns


def load_conversations(parquet, min_score=0.0):
    """The longest row per (domain, id) as run dicts in distill_data.py's shape, plus a report of what was checked."""
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["domain"], r["id"])].append(r)
    rep = collections.Counter(rows=len(rows), conversations=len(by))
    runs = []
    for (domain, cid), rs in sorted(by.items()):
        full = max(rs, key=lambda r: len(r["messages"]))
        msgs = full["messages"]
        for r in rs:   # every shorter row is the longest one cut after an earlier assistant turn
            n = len(r["messages"])
            rep["rows_that_are_a_prefix"] += [dict(m) for m in r["messages"]] == [dict(m) for m in msgs[:n]]
        ids = set(); ok = msgs[-1]["role"] == "assistant"
        for m in msgs:   # every tool result answers a call made earlier in the conversation
            if m["role"] == "assistant":
                ids |= {tc["id"] for tc in m.get("tool_calls") or []}
            elif m["role"] == "tool" and m.get("tool_call_id") not in ids:
                ok = False
        rep["well_formed"] += ok
        if full["score"] < min_score:
            rep["below_min_score"] += 1; continue
        run = f"{SOURCE}/{domain}/{cid}"
        runs.append({"env": f"toolscale_{domain}", "run": run, "kind": "full", "group": run, "K": -1, "hacked": None,
                     "messages": [{k: v for k, v in m.items() if v is not None} for m in msgs],
                     "tools": json.loads(full["tools"]), "score": full["score"]})
    return runs, rep


def build(a):
    out = pathlib.Path(a.out).resolve()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=SCRIPTS, capture_output=True, text=True).stdout.strip()
    if top and out.is_relative_to(pathlib.Path(top).resolve()):
        sys.exit(f"--out {out} is inside the repo. The source dataset has no licence, so the converted set must stay "
                 f"out of git: choose a folder outside {top}.")
    D = p2ds_functions()
    runs, rep = load_conversations(a.parquet, a.min_score)
    score_of = {r["run"]: r["score"] for r in runs}
    ex, dropped = D["trajectories"](runs, False, a.max_context_chars)   # p2ds: build --trajectories --max-context-chars 105000
    env_of = {g: exs[0]["env"] for g, exs in ex.items() if exs}
    chosen = set(env_of)
    by_env = collections.defaultdict(list)
    for g in chosen:
        by_env[env_of[g]].append(g)
    train, val = D["stratified_split"](by_env, a.split, a.seed)       # p2ds: --split 0.8 --seed 0, per task; here per domain
    out.mkdir(parents=True, exist_ok=True)
    counts = collections.defaultdict(lambda: {"train": 0, "val": 0, "train_runs": 0, "val_runs": 0})
    cut = 0
    with open(out / "train.jsonl", "w") as ft, open(out / "val.jsonl", "w") as fv:
        for g in sorted(chosen):                                        # the same order as distill_data.build writes
            f, part = (ft, "train") if g in train else (fv, "val")
            counts[env_of[g]][f"{part}_runs"] += 1
            for e in ex[g]:
                n_turns = sum(1 for m in next(r for r in runs if r["run"] == e["run"])["messages"] if m["role"] == "assistant")
                cut += len(e["train_turns"]) < n_turns
                f.write(json.dumps({**e, "score": score_of[e["run"]]}, ensure_ascii=False) + "\n"); counts[e["env"]][part] += 1
    manifest = {
        "source": SOURCE, "source_file": str(a.parquet), "licence": "none stated: do not commit, do not republish",
        "seed": a.seed, "split": a.split, "trajectories": True, "student": "qwen36", "max_context_chars": a.max_context_chars,
        "min_score": a.min_score, "checks": dict(rep), "runs_cut_by_max_context_chars": cut,
        "trained_turns_without_reasoning": D["_no_reasoning"](ex, chosen),
        "counts": dict(sorted(counts.items())),
        "totals": {"train": sum(c["train"] for c in counts.values()), "val": sum(c["val"] for c in counts.values()),
                   "train_runs": len(train & chosen), "val_runs": len(val & chosen),
                   "full_set_examples": sum(len(v) for v in ex.values()),
                   "full_set_bytes": sum(D["nbytes"](e) for v in ex.values() for e in v)},
        "excluded_runs": dropped,
        "action_match_score": dict(sorted(collections.Counter(round(s, 2) for s in score_of.values()).items())),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"rows {rep['rows']}, conversations {rep['conversations']}; rows that are the first part of their "
          f"conversation's longest row: {rep['rows_that_are_a_prefix']} of {rep['rows']}; conversations that end on an "
          f"assistant turn with every tool result answering an earlier call: {rep['well_formed']} of {rep['conversations']}")
    print(f"{'domain':24s} {'train':>6s} {'val':>6s}")
    for env, c in sorted(counts.items()):
        print(f"{env:24s} {c['train']:6d} {c['val']:6d}")
    t = manifest["totals"]
    print(f"{'TOTAL':24s} {t['train']:6d} {t['val']:6d}")
    print(f"left out: {len(dropped)} (tool-call arguments that are not a JSON object); conversations cut by the "
          f"{a.max_context_chars:,}-character rule: {cut}; trained turns without reasoning: "
          f"{manifest['trained_turns_without_reasoning']['without']} of {manifest['trained_turns_without_reasoning']['of']}")
    for f in ("train.jsonl", "val.jsonl", "manifest.json"):
        print(f"{out / f}: {(out / f).stat().st_size / 2**20:.2f} MB")
    if a.tokenizer:
        for part in ("train", "val"):
            s = sizes(out / f"{part}.jsonl", a.tokenizer, a.max_len)
            (out / f"sizes_{part}.json").write_text(json.dumps(s, indent=1)); show(part, s)


def sizes(path, tok_dir, max_len):
    """Sizes of a qwen36 trajectory file as training sees it: every example goes through distill_train.encode_turns with
    the template arguments of the training script, so "tokens" is what one pass reads and "trained tokens" is what the
    loss is taken on (reasoning, </think>, reply, tool calls, <|im_end|> of each kept turn)."""
    sys.path.insert(0, str(SCRIPTS))
    import distill_train as DT
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    c = collections.Counter(); lens = []; reason_tok = []; reason_chars = []; turn_tok = []; per_env = collections.defaultdict(collections.Counter)
    for line in open(path):
        e = json.loads(line); c["examples"] += 1
        if e.get("format") != "qwen36" or "train_turns" not in e:
            sys.exit(f"{path}: not a qwen36 trajectory example")
        enc, why = DT.encode_turns(tok, e["messages"], e["train_turns"], max_len, e["tools"], DT.QWEN_TEMPLATE_KW)
        if enc is None:
            c["dropped: " + why] += 1; continue
        ids, labels = enc
        n_tgt = sum(1 for x in labels if x != -100); kept = sum(1 for x in labels if x == end)
        c["kept"] += 1; c["tokens"] += len(ids); c["trained_tokens"] += n_tgt; c["trained_turns"] += kept
        c["turns_in_file"] += len(e["train_turns"]); c["cut_by_max_len"] += kept < len(e["train_turns"])
        lens.append(len(ids)); pe = per_env[e["env"]]; pe["kept"] += 1; pe["tokens"] += len(ids); pe["trained_tokens"] += n_tgt
        a = [m for m in e["messages"] if m["role"] == "assistant"]
        for t in sorted(e["train_turns"])[:kept]:   # the turns the token cut kept (it only ever drops the last ones)
            rc = a[t].get("reasoning_content") or ""
            reason_chars.append(len(rc)); reason_tok.append(len(tok(rc, add_special_tokens=False)["input_ids"]) if rc else 0)
            c["tool_call_turns"] += bool(a[t].get("tool_calls")); c["tool_calls"] += len(a[t].get("tool_calls") or [])
        turn_tok.append(n_tgt / max(kept, 1))
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else 0
    return {"file": str(path), "max_len": max_len, **dict(c),
            "tokens_per_example": {"mean": round(statistics.mean(lens)) if lens else 0, "median": q(lens, .5), "p90": q(lens, .9), "max": max(lens, default=0)},
            "reasoning_tokens_per_trained_turn": {"mean": round(statistics.mean(reason_tok), 1) if reason_tok else 0,
                                                  "median": q(reason_tok, .5), "p90": q(reason_tok, .9), "max": max(reason_tok, default=0)},
            "reasoning_chars_per_trained_turn": {"mean": round(statistics.mean(reason_chars), 1) if reason_chars else 0, "median": q(reason_chars, .5)},
            "reasoning_tokens": sum(reason_tok), "turns_without_reasoning": sum(1 for x in reason_tok if not x),
            "trained_tokens_per_turn_mean": round(c["trained_tokens"] / max(c["trained_turns"], 1), 1),
            "per_env": {k: dict(v) for k, v in sorted(per_env.items())}}


def show(name, s):
    print(f"\n{name}: {s.get('kept', 0)} of {s['examples']} examples kept at {s['max_len']:,} tokens "
          f"({s.get('cut_by_max_len', 0)} lost their last turns to the token cut)")
    print(f"  tokens in one pass: {s.get('tokens', 0):,}; of them trained on (the model's own turns): {s.get('trained_tokens', 0):,}")
    t = s["tokens_per_example"]; print(f"  tokens per example: mean {t['mean']:,}, median {t['median']:,}, 90% under {t['p90']:,}, longest {t['max']:,}")
    r = s["reasoning_tokens_per_trained_turn"]
    print(f"  trained turns: {s.get('trained_turns', 0):,} ({s.get('tool_call_turns', 0):,} with a tool call, {s.get('tool_calls', 0):,} calls); "
          f"{s['trained_tokens_per_turn_mean']} trained tokens per turn")
    print(f"  reasoning per trained turn: mean {r['mean']} tokens, median {r['median']}, 90% under {r['p90']}, longest {r['max']:,}; "
          f"{s['turns_without_reasoning']} turns have none; reasoning is {s['reasoning_tokens']:,} tokens in all")


def stats(a):
    d = pathlib.Path(a.data)
    for part in a.splits.split(","):
        if (d / f"{part}.jsonl").exists():
            s = sizes(d / f"{part}.jsonl", a.tokenizer, a.max_len); show(part, s)
            if a.save:
                pathlib.Path(a.save).mkdir(parents=True, exist_ok=True)
                (pathlib.Path(a.save) / f"sizes_{part}.json").write_text(json.dumps(s, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.set_defaults(fn=build)
    b.add_argument("--parquet", required=True, help="the dataset's train.parquet")
    b.add_argument("--out", required=True, help="output folder, OUTSIDE the repo")
    b.add_argument("--split", type=float, default=0.8); b.add_argument("--seed", type=int, default=0)
    b.add_argument("--max-context-chars", type=int, default=MAX_CONTEXT_CHARS)
    b.add_argument("--min-score", type=float, default=0.0, help="keep conversations with at least this action-match "
                   "score (default 0 = all 284; the dataset's own filter already ran)")
    s = sub.add_parser("stats"); s.set_defaults(fn=stats)
    s.add_argument("--data", required=True, help="folder with train.jsonl / val.jsonl in the qwen36 trajectory format")
    s.add_argument("--splits", default="train,val"); s.add_argument("--save", default=None, help="folder for sizes_<split>.json")
    for p in (b, s):
        p.add_argument("--tokenizer", default=None if p is b else argparse.SUPPRESS, required=p is s,
                       help="folder with Qwen3.6's tokenizer files (public, free to download)")
        p.add_argument("--max-len", type=int, default=MAX_LEN, help="training's token cut")
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
