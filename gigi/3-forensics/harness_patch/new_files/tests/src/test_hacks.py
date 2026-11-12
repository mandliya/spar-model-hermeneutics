"""The five hard hacks (agent.hack, hacks.py) on top of the hard end rule."""
import json

from agent_interp_envs.end_task_required import END_TASK
from agent_interp_envs.hacks import HACKS, prompt_addition
from agent_interp_envs.providers import create_provider
from agent_interp_envs.tool_calling import EXECUTE_COMMAND_TOOL, validate_and_get_command
from agent_interp_envs.types import LLMResponse, ToolCall, ToolResult


def _provider(hack, replies, **kw):
    p = create_provider(provider="mock", model="mock",
                        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
                        tools=[EXECUTE_COMMAND_TOOL], end_task_required=True, end_task_reprompts=3, hack=hack, **kw)
    queue = list(replies)
    # the hack wrapper and the rule wrapper are installed over the mock's invoke; swap the innermost for a script
    inner = p.invoke
    def raw():
        r = queue.pop(0)
        if isinstance(r, str):
            p.messages.append({"role": "assistant", "content": r})
            return LLMResponse(reasoning=None, response=r, tool_calls=None)
        name, args = r
        tc = ToolCall(id=f"id{len(p.messages)}", name=name, arguments=json.dumps(args))
        p.messages.append({"role": "assistant", "content": None,
                           "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": name, "arguments": tc.arguments}}]})
        return LLMResponse(reasoning=None, response=None, tool_calls=[tc])
    # find the mock's own invoke at the bottom of the closure chain and replace it
    f = inner
    while getattr(f, "__closure__", None):
        cells = {k: c.cell_contents for k, c in zip(f.__code__.co_freevars, f.__closure__)}
        nxt = cells.get("orig") or cells.get("orig_invoke")
        if nxt is None or not getattr(nxt, "__closure__", None):
            # f is the lowest wrapper: point its orig at the script
            for k, c in zip(f.__code__.co_freevars, f.__closure__):
                if k in ("orig", "orig_invoke"):
                    c.cell_contents = raw
            break
        f = nxt
    return p


def _step(p, n=3):
    r = validate_and_get_command(p, max_retries=n)
    if r.success and r.tool_call.name == "execute_command":
        p.add_tool_result(ToolResult(id=r.tool_call.id, name="execute_command", content="ok"))
    return r


def test_prompt_additions_exist_for_the_prompt_hacks_and_land_in_the_system_prompt():
    for h in HACKS:
        assert isinstance(prompt_addition(h, END_TASK), str)
    p = _provider("do_the_work", [])
    assert "complete the task yourself" in p.messages[0]["content"] and f"`{END_TASK}`" in p.messages[0]["content"]
    p = _provider("worked_example", [])
    assert "pytest -q" in p.messages[0]["content"]
    p = _provider("worked_example_control", [])
    assert "1 failed" in p.messages[0]["content"] and "all tests pass" not in p.messages[0]["content"]
    p = _provider("worked_example_neutral", [])
    assert "solution.py" not in p.messages[0]["content"] and "read the files that matter" in p.messages[0]["content"]


def test_no_early_exit_refuses_then_accepts():
    ls = ("execute_command", {"command": "ls"}); end = (END_TASK, {"report": "done"})
    p = _provider("no_early_exit", [ls, end, ls, end], hack_min_steps=2)
    assert _step(p).command == "ls"
    r = _step(p)                              # end_task after 1 step: refused, the model's next reply (ls) comes back
    assert r.success and r.command == "ls"
    assert any(str(m.get("content", "")).startswith("Too early to finish") for m in p.messages if m.get("role") == "tool")
    r = _step(p)                              # 2 steps done: end_task accepted, the rule ends the run
    assert r.error_type == "no_tool_call" and p.end_task_called


def test_proof_of_work_refuses_until_a_file_changes(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    ls = ("execute_command", {"command": "ls"}); end = (END_TASK, {"report": "done"})
    p = _provider("proof_of_work", [ls, end, ls, end], hack_workspace=str(tmp_path))
    _step(p)
    r = _step(p)                              # nothing changed: refused
    assert r.command == "ls"
    assert any(str(m.get("content", "")).startswith("Not finished: no file") for m in p.messages if m.get("role") == "tool")
    (tmp_path / "a.py").write_text("x = 2\n\n\n")   # the size changes, so no mtime race
    r = _step(p)
    assert r.error_type == "no_tool_call" and p.end_task_called


def test_refusals_are_bounded():
    end = (END_TASK, {"report": "done"})
    p = _provider("no_early_exit", [end] * 4, hack_min_steps=50, hack_max_refusals=2)
    r = validate_and_get_command(p, max_retries=3)
    assert r.error_type == "no_tool_call" and p.end_task_called
    assert sum(1 for m in p.messages if str(m.get("content", "")).startswith("Too early")) == 2


def test_plan_first_asks_for_a_plan_then_a_tool_and_adds_progress_lines():
    ls = ("execute_command", {"command": "ls"})
    p = _provider("plan_first", [ls, "1. look 2. fix 3. test", ls, ls])
    r = _step(p)
    assert r.success and r.command == "ls"
    users = [m["content"] for m in p.messages if m["role"] == "user"]
    assert any("numbered plan" in u for u in users) and any("step 1 of your plan" in u for u in users)
    assert "1. look 2. fix 3. test" in [m.get("content") for m in p.messages]
    assert p.messages[-1]["role"] == "tool" and "[1 tool step done." in p.messages[-1]["content"]
    _step(p)
    assert "[2 tool steps done." in p.messages[-1]["content"]


def test_a_premature_submit_is_refused_like_a_premature_end_task():
    sub = ("execute_command", {"command": "cd /agent && python submit_solution.py"}); ls = ("execute_command", {"command": "ls"})
    p = _provider("no_early_exit", [sub, ls, sub], hack_min_steps=1)
    r = _step(p)                              # submit at 0 steps: refused, the next reply (ls) comes through
    assert r.success and r.command == "ls"
    assert any(str(m.get("content", "")).startswith("Too early to submit") for m in p.messages if m.get("role") == "tool")
    r = _step(p)                              # 1 step done: the submit goes through to the environment
    assert r.success and "submit_solution.py" in r.command
    p = _provider("proof_of_work", [("execute_command", {"command": "cat submit_solution.py"})], hack_workspace="/nonexistent")
    assert _step(p).success                   # reading the script is not submitting
