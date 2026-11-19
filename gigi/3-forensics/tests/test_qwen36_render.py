"""Renderer check for Qwen3.6 as the distillation student (phase 2, 19 Nov 2026). No GPU, no weights: the tokenizer only.

  1. build (the environments' venv, which has agent_interp_envs):
       agent-interp-envs/.venv/bin/python tests/test_qwen36_render.py build --out /tmp/q36.jsonl
     Real Qwen3.6 transcripts through distill_data.py's native renderer (--student qwen36): 6 hero9 runs (thinking off,
     tool calls, several tasks) and 3 hero2_qwen36_think runs (thinking on: reasoning in every turn). Writes both the
     trajectory examples and per-turn examples of one run.
  2. verify (an env with transformers 5.x and the Qwen3.6 tokenizer files):
       python tests/test_qwen36_render.py verify --examples /tmp/q36.jsonl --tokenizer <dir or Qwen/Qwen3.6-27B>
     For every example, with distill_train.encode_turns / encode and the chat template the re-run server uses
     (apply_chat_template(messages, tools=..., preserve_thinking=True, enable_thinking=True)):
       a. no example dropped (the prefix property holds turn by turn);
       b. the training text (decoded ids) equals the template's rendering of the whole history;
       c. the training token ids equal the tokenization of that rendering as one string (as the server tokenizes a prompt);
       d. the loss mask covers exactly each turn's completion (reasoning, </think>, reply, tool calls, <|im_end|>) and
          sits inside the template's own assistant span ({% generation %}; return_assistant_tokens_mask), the only
          assistant tokens left out being each turn's opening "<think>\\n", which the generation prompt supplies;
       e. teacher reasoning sits inside <think>...</think> of its turn, in every turn (preserve_thinking);
       f. every tool call renders as Qwen XML (<function=NAME>).
"""
import argparse, json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))


def build(a):
    import distill_data as DD
    DD.STUDENT = "qwen36"
    plain, _ = DD.load_runs_hero9(envs=["coding", "currency_conversion", "eval_tampering", "tower_of_hanoi",
                                        "lazy_investigation", "funding_email"],
                                  full_csvs=["hero9_full_runs__qwen3.6-27b_per_run.csv",
                                             "hero9_full_runs__qwen3.6-27b__other21_per_run.csv"], swap=False)
    think, _ = DD.load_runs_hero9(full_csvs=["hero2_full_runs__qwen3.6-27b-think_per_run.csv"], swap=False)
    pick, seen = [], set()
    for r in plain:   # one run per task, with tool calls, short enough to keep the test quick
        if r["env"] in seen or not any(m.get("tool_calls") for m in r["messages"]):
            continue
        if len(json.dumps(r["messages"])) > 120_000:
            continue
        pick.append(r); seen.add(r["env"])
    th = [r for r in think if any(isinstance(m.get("reasoning_content") or m.get("reasoning"), str) and
                                  (m.get("reasoning_content") or m.get("reasoning")).strip() for m in r["messages"])]
    th = sorted(th, key=lambda r: len(json.dumps(r["messages"])))[:3]
    # private keys never reach the training text: the harness now stores a per-call cost record as "_usage" in each
    # assistant message (19 Nov 2026); any "_"-prefixed key must be dropped by the renderer
    import copy
    r0 = copy.deepcopy(th[0])
    for m in r0["messages"]:
        if m["role"] == "assistant":
            m["_usage"] = {"cost": 0.0012, "prompt_tokens": 1234}; m["_anything_private"] = "x"
    tainted, clean = DD.qwen_messages(r0["messages"]), DD.qwen_messages(th[0]["messages"])
    leaked = [k for m in tainted for k in m if k.startswith("_")]
    assert not leaked and tainted == clean and "_usage" not in json.dumps(tainted), f"private keys leaked: {leaked}"
    print("private keys: '_usage' and '_anything_private' added to every assistant message are dropped by the renderer "
          f"(rendering identical to the clean run: {tainted == clean})")
    traj, dropped = DD.trajectories(pick + th, max_context_chars=105_000)
    per, _ = DD.examples(pick[:1])
    with open(a.out, "w") as f:
        for g, exs in traj.items():
            for e in exs:
                f.write(json.dumps({**e, "mode": "trajectory"}) + "\n")
        for g, exs in per.items():
            for e in exs[:4]:
                f.write(json.dumps({**e, "mode": "per-turn"}) + "\n")
    print(f"picked {len(pick)} thinking-off runs ({sorted(seen)}) and {len(th)} thinking-on runs; "
          f"{sum(len(v) for v in traj.values())} trajectory examples, dropped {dropped}; wrote {a.out}")


def verify(a):
    import transformers
    from transformers import AutoTokenizer
    import distill_train as DT
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    KW = DT.QWEN_TEMPLATE_KW
    exs = [json.loads(l) for l in open(a.examples)]
    print(f"transformers {transformers.__version__}; {len(exs)} examples")
    totals = {"examples": 0, "dropped": 0, "text_equal": 0, "ids_equal": 0, "mask_exact": 0, "mask_in_assistant": 0,
              "reasoning_turns": 0, "reasoning_in_think": 0, "tool_calls": 0, "tool_calls_xml": 0,
              "think_open_tokens_outside_mask": 0, "trained_tokens": 0}
    for e in exs:
        totals["examples"] += 1
        msgs, tools = e["messages"], e["tools"]
        turns = e.get("train_turns") or [sum(1 for m in msgs if m["role"] == "assistant") - 1]
        if "train_turns" in e:
            enc, why = DT.encode_turns(tok, msgs, e["train_turns"], 10**9, tools, KW)
        else:
            enc, why = DT.encode(tok, msgs, 10**9, tools, KW)
        if enc is None:
            totals["dropped"] += 1; print("DROPPED", e["run"], why); continue
        ids, labels = enc
        full = tok.apply_chat_template(msgs, tools=tools, tokenize=False, **KW)
        text = tok.decode(ids, skip_special_tokens=False)
        want_text = full.rstrip("\n") if "train_turns" in e else (
            tok.apply_chat_template(msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True, **KW)
            + full[len(tok.apply_chat_template(msgs[:-1], tools=tools, tokenize=False, add_generation_prompt=True, **KW)):].rstrip("\n"))
        totals["text_equal"] += text == want_text
        whole = tok(want_text, add_special_tokens=False)["input_ids"]
        same = ids == whole
        totals["ids_equal"] += same
        if not same:
            k = next((i for i, (x, y) in enumerate(zip(ids, whole)) if x != y), min(len(ids), len(whole)))
            print(f"IDS DIFFER {e['run']}: at token {k}: ours {tok.decode(ids[k:k + 6])!r} vs one-string "
                  f"{tok.decode(whole[k:k + 6])!r} (lengths {len(ids)} vs {len(whole)})")
        # d. loss mask = the completions of the trained turns, each inside the template's assistant span
        idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        want_comp = []
        for t, i in enumerate(idx):
            if t not in set(turns):
                continue
            p = tok.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True, **KW)
            f_ = tok.apply_chat_template(msgs[:i + 1], tools=tools, tokenize=False, **KW)
            want_comp.append(f_[len(p):].rstrip("\n"))
        spans, cur = [], []
        for x, y in zip(ids, labels):
            if y != -100:
                cur.append(x)
            elif cur:
                spans.append(cur); cur = []
        if cur:
            spans.append(cur)
        got_comp = [tok.decode(s, skip_special_tokens=False) for s in spans]
        exact = got_comp == want_comp and all(c.endswith("<|im_end|>") and "</think>" in c for c in got_comp)
        totals["mask_exact"] += exact
        totals["trained_tokens"] += sum(1 for y in labels if y != -100)
        if not exact:
            print(f"MASK MISMATCH {e['run']}: {len(got_comp)} spans vs {len(want_comp)} turns; first got "
                  f"{got_comp[0][:120] if got_comp else None!r} want {want_comp[0][:120] if want_comp else None!r}")
        if same:
            hf = tok.apply_chat_template(msgs, tools=tools, tokenize=True, return_dict=True,
                                         return_assistant_tokens_mask=True, **KW)
            am = hf["assistant_masks"][:len(ids)]
            inside = all(am[i] == 1 for i, y in enumerate(labels) if y != -100)
            totals["mask_in_assistant"] += inside
            extra = [i for i, v in enumerate(am) if v == 1 and labels[i] == -100]
            # assistant tokens we do not train: each untrained turn whole, and each trained turn's "<think>\n" opener
            opener = sum(1 for i in extra if tok.decode([ids[i]]) in ("<think>", "\n") and i + 1 < len(labels)
                         and (labels[i + 1] != -100 or (i + 2 < len(labels) and labels[i + 2] != -100)))
            totals["think_open_tokens_outside_mask"] += opener
        for m in msgs:
            if m["role"] != "assistant":
                continue
            rc = m.get("reasoning_content")
            if rc:
                totals["reasoning_turns"] += 1
                totals["reasoning_in_think"] += ("<think>\n" + rc.strip() + "\n</think>") in full
            for tc in m.get("tool_calls") or []:
                totals["tool_calls"] += 1
                totals["tool_calls_xml"] += f"<function={tc['function']['name']}>" in full
    print(json.dumps(totals, indent=1))
    ok = (totals["dropped"] == 0 and totals["text_equal"] == totals["examples"] and totals["mask_exact"] == totals["examples"]
          and totals["reasoning_in_think"] == totals["reasoning_turns"] and totals["tool_calls_xml"] == totals["tool_calls"])
    print("RENDER CHECK", "PASSED" if ok else "FAILED",
          f"(token ids equal to one-string tokenization in {totals['ids_equal']}/{totals['examples']})")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--out", required=True); b.set_defaults(fn=build)
    v = sub.add_parser("verify"); v.add_argument("--examples", required=True); v.add_argument("--tokenizer", required=True)
    v.set_defaults(fn=verify)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
