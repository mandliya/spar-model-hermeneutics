"""general_tbench_selftest.py (24 Nov 2026): the agent's loop against a scripted fake model and a fake container.
No network, no key, no Harbor, no money. Run before every launch (general_tbench_launch.sh does):
  python3 scripts/general_tbench_selftest.py
Checks: the raw reply is on disk before parsing; reasoning is stored on every turn and sent back; a 429 and a 503 are
retried; two calls in one turn are resampled 3 times and then answered with the harness's text; a plain reply ends the
run with the FINAL RESULTS block; the step cap ends a run; the spend cap stops a run before the next request; the
converter reads the result and writes the p2ds shape.
"""
import asyncio
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import general_tbench_agent as A   # noqa: E402


def reply(content=None, calls=None, reasoning="thinking...", cost=0.01):
    msg = {"role": "assistant", "content": content, "reasoning": reasoning,
           "reasoning_details": [{"type": "reasoning.text", "text": reasoning}]}
    if calls:
        msg["tool_calls"] = [{"id": f"call_{i}", "type": "function", "index": i,
                              "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    return 200, json.dumps({"id": "gen-x", "provider": "SiliconFlow", "model": "deepseek/deepseek-v4-pro",
                            "choices": [{"message": msg, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": cost,
                                      "prompt_tokens_details": {"cached_tokens": 50},
                                      "completion_tokens_details": {"reasoning_tokens": 10}}})


def run_case(script, max_steps=100, cap=100.0, prior=0.0):
    d = pathlib.Path(tempfile.mkdtemp(prefix="gt_selftest_")) / "job" / "fix-git__abc" / "agent"
    s = A.settings(); s["max_steps"] = max_steps
    sent = []; script = list(script)
    led = A.Ledger(d.parents[2] / "ledger.jsonl")
    if prior:
        led.add({"cost": prior})

    async def post(body):
        sent.append(json.loads(json.dumps(body)))
        raw = d / "raw" / f"call-{len(sent):04d}.json"
        assert not raw.exists(), "raw file must not exist before the reply"
        return script.pop(0)

    async def execute(command):
        return 0, f"ran: {command}"

    async def no_sleep(_):
        return None

    run = A.Run(s, d, "Do the task.", execute, post, led, cap, {"task": "fix-git"}, sleep=no_sleep)
    err = None
    try:
        asyncio.run(run.loop())
    except Exception as e:   # noqa: BLE001
        err = e
    return d, run, sent, err


def main():
    cmd = lambda c: [("execute_command", {"command": c})]   # noqa: E731
    # 1. the normal path, with a 429, a 503 and a resampled double call on the way
    two = [("execute_command", {"command": "a"}), ("execute_command", {"command": "b"})]
    d, run, sent, err = run_case([
        reply(calls=cmd("ls -la")),
        (429, json.dumps({"error": {"code": 429, "message": "rate limited"}})),
        (503, "upstream unavailable"),
        reply(calls=two), reply(calls=two), reply(calls=two),
        reply(calls=cmd("git status")),
        reply(content="All done."),
    ])
    assert err is None and run.ended == "plain reply", (err, run.ended)
    m = json.loads((d / "final/messages.json").read_text())
    roles = [x["role"] for x in m]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "tool", "assistant", "tool", "assistant"], roles
    assert m[3]["content"] == "Exit code: 0\nOutput:\nran: ls -la"
    assert m[5]["content"].startswith("Exit code: 1\nOutput:\nYou called 2 functions, but you must call ONLY ONE function per turn.")
    assert all(x.get("reasoning") and x.get("reasoning_content") and x.get("_usage") for x in m if x["role"] == "assistant")
    assert len(list((d / "raw").glob("call-*.json"))) == 8 and (d / "raw/call-0003.json").read_text() == "upstream unavailable"
    last = sent[-1]["messages"]
    assert all("_usage" not in x for x in last) and last[2]["reasoning"] == "thinking..." and last[2]["reasoning_details"]
    assert sent[0]["provider"] == {"only": ["siliconflow"], "quantizations": ["fp8"], "allow_fallbacks": False}
    assert sent[0]["reasoning"] == {"effort": "xhigh"} and sent[0]["temperature"] == 1.0 and sent[0]["top_p"] == 1.0
    assert "FINAL RESULTS" in (d / "rollout.log").read_text()
    end = json.loads((d / "final/episode_end.json").read_text())
    assert end["ended_by_itself"] and end["steps"] == 4 and abs(end["cost"] - 0.06) < 1e-9, end   # 6 billed replies
    assert abs(A.Ledger(d.parents[2] / "ledger.jsonl").refresh() - 0.06) < 1e-9
    calls = [json.loads(x) for x in (d / "calls.jsonl").read_text().splitlines()]
    assert [c["kept"] for c in calls] == [True, False, False, True, True, True, True, True]
    jobs = d.parents[2]
    # 2. the step cap
    d2, run2, _, err2 = run_case([reply(calls=cmd("echo 1")), reply(calls=cmd("echo 2")), reply(calls=cmd("echo 3"))], max_steps=2)
    assert err2 is None and run2.ended == "step limit" and run2.step == 2 and "FINAL RESULTS" in (d2 / "rollout.log").read_text()
    # 3. the spend cap: the ledger already holds $5 of a $5 cap, so no request is sent
    d3, run3, sent3, err3 = run_case([reply(calls=cmd("ls"))], cap=5.0, prior=5.0)
    assert isinstance(err3, A.SpendCapReached) and not sent3 and run3.ended == "spend cap"
    assert "FINAL RESULTS" not in (d3 / "rollout.log").read_text()
    # 4. a reply from another host than the pinned one stops the run (only checked against the real OpenRouter URL)
    other = json.loads(reply(calls=cmd("ls"))[1]); other["provider"] = "DeepInfra"
    d4, run4, _, err4 = run_case([(200, json.dumps(other))])
    assert isinstance(err4, A.PinViolated), err4
    # 5. the converter on run 1
    out = pathlib.Path(tempfile.mkdtemp(prefix="gt_selftest_out_")) / "set"
    conv = [sys.executable, str(HERE / "general_tbench_convert.py")]
    subprocess.run(conv + ["table", str(jobs)], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(conv + ["build", str(jobs), "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    e = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert e["format"] == "qwen36" and e["env"] == "tbench2/fix-git" and e["train_turns"] == [0, 1, 2, 3], e["train_turns"]
    a0 = e["messages"][2]
    assert a0["reasoning_content"] == "thinking..." and a0["tool_calls"][0]["function"]["arguments"] == {"command": "ls -la"}
    assert e["tools"] == [A.EXECUTE_COMMAND_TOOL] and set(a0) == {"role", "content", "reasoning_content", "tool_calls"}
    print("SELFTEST OK: loop, retries, resampling, step cap, spend cap, provider pin, raw replies, converter")


if __name__ == "__main__":
    main()
