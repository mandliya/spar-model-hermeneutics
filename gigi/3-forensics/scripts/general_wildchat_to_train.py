"""FAR rung of the general training set (24 Nov 2026): DeepSeek V4 Pro's answers to the WildChat questions
(scripts/general_wildchat_generate.py -> results/general_wildchat/raw.jsonl) as training data for Qwen3.6 27B, in the
format the DeepSeek round trained on (distill/p2_deepseekv4_v1, built by distill_data.py build --hero9 --trajectories
--student qwen36 --split 0.8 --seed 0; checked against its manifest and files on the launch box). No API calls.

  PY=gigi/3-forensics/agent-interp-envs/.venv/bin/python     (distill_data.py needs the environments' venv)
  $PY gigi/3-forensics/scripts/general_wildchat_to_train.py --dry-run       # the split table, writes nothing
  $PY gigi/3-forensics/scripts/general_wildchat_to_train.py                 # -> distill/general_wildchat_v1/

One example per question, the same keys as a p2ds example:
  {"env": "wildchat_<kind>", "run": "general_wildchat/<id>", "step": 0, "hacked": false, "kind": "full",
   "group": "general_wildchat/<id>", "train_turns": [0], "format": "qwen36", "tools": [],
   "messages": [{"role": "user", "content": <question>},
                {"role": "assistant", "content": <answer>, "reasoning_content": <DeepSeek's reasoning>}]}
  - The assistant message is made by distill_data.qwen_messages, the function that made every p2ds message: the
    reasoning goes in reasoning_content (OpenRouter's `reasoning` field, mirrored as the harness mirrors it), and
    Qwen3.6's chat template wraps it in <think>...</think> before the answer. distill_train.py renders the example with
    apply_chat_template(messages, tools=[], preserve_thinking=True, enable_thinking=True) and trains on the reasoning,
    "</think>", the answer and <|im_end|> (encode_turns), as in p2ds.
  - "tools" is an empty list (a plain question has none), so the template writes no tool block; there is no system
    prompt, because none was sent to DeepSeek.
  - "env" carries the kind of question, so the trainer's per-"env" counts show the mix.
Split: distill_data.stratified_split with split 0.8 and seed 0, the p2ds call, with the kinds of question in the place
of the tasks: 80% train, 20% val within each kind. No test.jsonl is written (see --help of p2q_train.sh's loss step:
it reads val and test; point it at val only, or copy p2ds's test.jsonl to compare on the same held-out runs).

A question is left out, and listed in the manifest with the reason, when it has no usable reply (see
general_wildchat_generate.verdict: not HTTP 200, not from the pinned provider, cut before the end, or an empty text).
A reply whose text is null stays out; nothing is filled in. An answer without reasoning is kept and counted
(trained_turns_without_reasoning), as distill_data.py does.
Refuses to overwrite an existing data folder unless --force.
"""
import argparse, collections, hashlib, json, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import general_wildchat_generate as G   # noqa: E402  parse / verdict / cost_of: one definition of "usable reply"


def harness_message(msg: dict) -> dict:
    """OpenRouter's reply message as the harness stores an assistant turn (openrouter_provider.invoke): `reasoning`
    mirrored into `reasoning_content`, and the other way round."""
    m = dict(msg)
    if m.get("reasoning") and not m.get("reasoning_content"):
        m["reasoning_content"] = m["reasoning"]
    if m.get("reasoning_content") and not m.get("reasoning"):
        m["reasoning"] = m["reasoning_content"]
    return m


def load(raw_path, questions_path):
    """(question rows, {id: (reply body, attempt)} for the first usable reply, {id: last reason} for the rest, totals)."""
    qs = [json.loads(l) for l in open(questions_path) if l.strip()]
    good, why, tot = {}, {}, collections.Counter()
    with open(raw_path) as f:
        for line in f:
            if not line.strip():
                continue
            tot["attempts"] += 1
            try:
                r = json.loads(line)
            except ValueError:
                tot["unreadable lines"] += 1; continue
            body = G.parse(r.get("raw")); ok, reason = G.verdict(r.get("status"), body)
            tot["cost"] += G.cost_of(body) or 0.0
            if ok and r["id"] not in good:
                good[r["id"]] = (body, r.get("attempt"))
            elif not ok:
                why[r["id"]] = reason if r.get("error") is None else r["error"]
    return qs, good, why, tot


def build(a):
    import distill_data as DD   # the p2ds builder: its message renderer and its split
    DD.STUDENT = "qwen36"
    if not pathlib.Path(a.raw).exists():
        sys.exit(f"{a.raw} does not exist yet: run general_wildchat_generate.py first (it costs money: ask GG)")
    qs, good, why, tot = load(a.raw, a.questions)
    ex, dropped, usage = {}, [], collections.Counter()
    for q in qs:
        run = f"general_wildchat/{q['id']}"
        if q["id"] not in good:
            dropped.append({"id": q["id"], "reason": why.get(q["id"], "no reply in raw.jsonl")}); continue
        body, attempt = good[q["id"]]
        sent = [{"role": "user", "content": q["question"]}, harness_message(body["choices"][0]["message"])]
        try:
            msgs = DD.qwen_messages(sent)
        except (ValueError, KeyError) as e:
            dropped.append({"id": q["id"], "reason": f"{type(e).__name__}: {e}"}); continue
        if msgs[-1].get("tool_calls"):
            dropped.append({"id": q["id"], "reason": "the reply has a tool call although none was offered"}); continue
        if a.max_chars and sum(DD.msg_chars(m) for m in msgs) > a.max_chars:
            dropped.append({"id": q["id"], "reason": f"longer than {a.max_chars} characters"}); continue
        u = body.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens"):
            usage[k] += u.get(k) or 0
        usage["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        ex[run] = {"env": f"wildchat_{q['kind']}", "run": run, "step": 0, "hacked": False, "kind": "full", "group": run,
                   "train_turns": [0], "messages": msgs, "format": "qwen36", "tools": []}
    by_env = collections.defaultdict(list)
    for g, e in ex.items():
        by_env[e["env"]].append(g)
    train, val = DD.stratified_split(by_env, a.split, a.seed)
    counts = {env: {"train": sum(g in train for g in gs), "val": sum(g in val for g in gs)} for env, gs in sorted(by_env.items())}
    no_reason = sum(not (e["messages"][-1].get("reasoning_content") or "").strip() for e in ex.values())
    print(f"{len(qs)} questions; {len(ex)} with a usable answer; {len(dropped)} left out; "
          f"{tot['attempts']} calls in raw.jsonl, ${tot['cost']:.2f} billed")
    for reason, n in collections.Counter(d["reason"] for d in dropped).most_common():
        print(f"  left out, {reason}: {n}")
    print(f"{'kind of question':32s} {'train':>6s} {'val':>6s}")
    for env, c in counts.items():
        print(f"{env:32s} {c['train']:6d} {c['val']:6d}")
    print(f"{'TOTAL':32s} {len(train):6d} {len(val):6d}")
    print(f"answers without reasoning: {no_reason} of {len(ex)}")
    if ex:
        n = len(ex)
        print(f"per answer, as billed: {usage['completion_tokens'] / n:,.0f} output tokens, of which "
              f"{usage['reasoning_tokens'] / n:,.0f} reasoning; {usage['prompt_tokens'] / n:,.0f} input tokens")
    if a.dry_run:
        print("dry run: nothing written"); return
    out = pathlib.Path(a.out)
    if not out.resolve().is_relative_to((HERE / "distill").resolve()):
        sys.exit(f"--out must be inside {HERE / 'distill'}")
    if out.exists() and any(out.iterdir()) and not a.force:
        sys.exit(f"{out} exists and is not empty; pass --force to write it again")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "train.jsonl", "w") as ft, open(out / "val.jsonl", "w") as fv:
        for g in sorted(ex):
            (ft if g in train else fv).write(json.dumps(ex[g], ensure_ascii=False) + "\n")
    t = G.settings()
    manifest = {
        "what": "DeepSeek V4 Pro's single-turn answers to WildChat-1M questions (general training set, far rung)",
        "seed": a.seed, "split": a.split, "student": "qwen36", "trajectories": True, "max_context_chars": a.max_chars,
        "teacher": {"model": t["model"], "provider": t["pin"], "quantization": t["pins"][t["pin"]],
                    "allow_fallbacks": False, "reasoning_effort": t["reasoning_effort"],
                    "temperature": t["temperature"], "top_p": t["top_p"], "max_tokens": None,
                    "system_prompt": None, "tools": None},
        "questions": {"file": str(pathlib.Path(a.questions).resolve().relative_to(HERE.resolve()))
                      if pathlib.Path(a.questions).resolve().is_relative_to(HERE.resolve()) else a.questions,
                      "sha256": hashlib.sha256(pathlib.Path(a.questions).read_bytes()).hexdigest(), "n": len(qs),
                      "source": "allenai/WildChat-1M", "licence": "ODC-BY 1.0"},
        "raw": {"calls": tot["attempts"], "unreadable_lines": tot["unreadable lines"], "cost_usd": round(tot["cost"], 4)},
        "usage_of_kept_answers": dict(usage),
        "trained_turns_without_reasoning": {"without": no_reason, "of": len(ex)},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip(),
        "counts": counts, "totals": {"train": len(train), "val": len(val), "test": 0, "train_runs": len(train),
                                     "val_runs": len(val), "test_runs": 0},
        "excluded_runs": dropped,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for f in ("train.jsonl", "val.jsonl", "manifest.json"):
        print(f"{out / f}: {(out / f).stat().st_size / 2**20:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(HERE / "results/general_wildchat/raw.jsonl"))
    ap.add_argument("--questions", default=str(HERE / "general/wildchat/questions.jsonl"))
    ap.add_argument("--out", default=str(HERE / "distill/general_wildchat_v1"))
    ap.add_argument("--split", type=float, default=0.8, help="train share within each kind of question (p2ds: 0.8)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars", type=int, default=0, help="leave out an example longer than this many characters, "
                    "question + reasoning + answer (0 = keep all; the trainer drops what does not fit --max-len anyway)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    build(ap.parse_args())


if __name__ == "__main__":
    main()
