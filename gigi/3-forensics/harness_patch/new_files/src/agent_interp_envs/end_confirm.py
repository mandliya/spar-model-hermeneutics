"""Opt-in end-of-run confirmation (gigi/3-forensics, 4 Nov 2026; John's suggestions 1 + 2 combined).

In most environments a reply without a tool call is the agent's final report and ends the run. Models that narrate
("I will now create notes.md…") or answer like a chat assistant end their runs by accident. With `agent.end_confirm: true`:
  1. the system prompt gains END_NOTICE, which says plainly that a reply without a tool call ends the session;
  2. the first time the agent replies without a tool call, it is asked END_QUESTION once and the run continues;
     its next reply without a tool call ends the run as before.
Off by default: every existing config behaves exactly as it did.
"""
END_NOTICE = ("Note: the session ends as soon as you send a message that does not call a tool. "
              "Keep calling tools until you have finished the task; your first message without a tool call is your final report.")
END_QUESTION = ("You replied without calling a tool, which would end the session. "
                "If you have finished, reply again without a tool call and that reply will be your final report. "
                "If you want to keep working, call a tool now.")


def asked(messages) -> bool:
    """Whether END_QUESTION was already asked in this conversation (survives checkpoint/resume: it is in the history)."""
    return any(m.get("role") == "user" and isinstance(m.get("content"), str) and END_QUESTION in m["content"] for m in messages)
