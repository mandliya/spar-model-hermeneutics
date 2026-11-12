"""The hard end rule (agent.end_task_required) must end the run in every environment, including those whose
step loop ignores a reply without a tool call (games, swe_rebench_v2, coding with wall_off_told_user)."""
import json

from agent_interp_envs.end_task_required import END_TASK, install
from agent_interp_envs.providers import create_provider
from agent_interp_envs.tool_calling import EXECUTE_COMMAND_TOOL, validate_and_get_command
from agent_interp_envs.types import LLMResponse, ToolCall, ToolResult


def _provider(replies):
    """A mock provider whose replies are scripted: a str is prose, a (name, args) pair is one tool call."""
    p = create_provider(provider="mock", model="mock",
                        messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
                        tools=[EXECUTE_COMMAND_TOOL], end_task_required=True, end_task_reprompts=3)
    queue = list(replies)

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
    p.invoke = raw                      # replace the scripted mock, then re-install the rule on top of it
    return install(p, END_TASK, True, 3)


def test_synthetic_end_task_is_added_and_named_in_the_rule():
    p = _provider([])
    assert any((t.get("function") or t)["name"] == END_TASK for t in p.tools)
    assert f"`{END_TASK}`" in p.messages[0]["content"]


def test_end_task_call_sets_the_flag_and_passes_validation_without_retry():
    p = _provider([("execute_command", {"command": "ls"}), (END_TASK, {"report": "all done"})])
    r = validate_and_get_command(p, max_retries=3)             # an ordinary command
    assert r.success and r.command == "ls" and not getattr(p, "end_task_called", False)
    p.add_tool_result(ToolResult(id=r.tool_call.id, name="execute_command", content="ok"))
    r = validate_and_get_command(p, max_retries=3)             # the games/swe path (no treat_no_tool_call_as_done)
    assert r.error_type == "no_tool_call" and r.response.response == "all done" and p.end_task_called
    # no retry: the end_task call and its "Session ended" result are the last two messages
    assert p.messages[-1]["role"] == "tool" and p.messages[-2]["tool_calls"][0]["function"]["name"] == END_TASK


def test_plain_replies_get_reminders_then_the_run_ends():
    p = _provider(["done I think", "really done", "still talking", "last words"])
    r = p.invoke()
    assert r.tool_calls is None and r.response == "last words"
    assert sum(1 for m in p.messages if m["role"] == "user" and "does not end the session" in m["content"]) == 3
    assert not getattr(p, "end_task_called", False)
