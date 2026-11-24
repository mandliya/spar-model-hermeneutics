"""general_tbench_agent.py (24 Nov 2026): our harness's shell-tool loop as a Harbor agent, for teacher runs on
Terminal-Bench 2 (phase2/general_set/TERMINAL_BENCH.md). No paid call happens unless Harbor runs this agent with a key.

Harbor starts each task's container (on Modal), then calls ForensicsShellAgent.run(instruction, environment, context).
The loop runs on the launch box, outside the container: it calls the teacher through OpenRouter and runs each command in
the container with environment.exec. The OpenRouter key therefore never enters a task container.

What is the same as agent-interp-envs (our harness), on purpose, so the runs look like our test tasks:
  - one tool, `execute_command` (the harness's EXECUTE_COMMAND_TOOL, copied verbatim), native tool calls,
    parallel_tool_calls false, one command per turn, each command in a fresh shell;
  - the system prompt of our plain-reply tasks (precommit_hook's, verbatim); the user message is the task's instruction.md;
  - tool results as "Exit code: N\nOutput:\n...", output cut to 200,000 characters (head + tail), `ls -R` refused;
  - a turn with two calls, an unknown tool or unreadable arguments is resampled up to 3 times, then answered with the
    harness's own error text; a plain reply (no tool call) ends the run; the step cap ends it otherwise;
  - the request: model, pinned provider (only + quantizations, allow_fallbacks false), reasoning effort, temperature and
    top_p all sent explicitly, read from scripts/make_target_configs.py TARGETS (one source for both collections);
    `usage: {include: true}`; every earlier turn's reasoning is sent back (reasoning, reasoning_details,
    reasoning_content stay on the stored message);
  - retries: 429 up to 14 attempts, 5xx up to 8, anything else (timeouts, empty replies) up to 3.
What differs (stated in the design note): the command timeout (600 s here, 30 s in the harness: these tasks compile and
train), no argument healing, and the loop sits outside the container.

Saved per run, in Harbor's trial folder under agent/ (see RUN FORMAT in the design note):
  run_config.json     every setting, the tool schema, the system prompt (no key)
  request-first.json  the first request body exactly as sent
  raw/call-NNNN.json  each HTTP reply's body, written BEFORE any parsing (retried and discarded replies too)
  calls.jsonl         one line per HTTP reply: step, attempt, status, seconds, tokens, cost, served_by, kept or not
  messages.json       the history after every step (OpenAI-style, the harness's shape); final/messages.json at the end
  rollout.log         readable log in the harness's layout, with its FINAL RESULTS block when the run ends by itself
  final/episode_end.json   how the run ended, steps, cost
Money: every reply's real `usage.cost` is appended to one shared ledger (GT_LEDGER) under a file lock; before every
request the ledger's total is compared with GT_SPEND_CAP and the run stops when it is reached.

Dry run (no network, no key needed), prints the first request for one task:
  python3 scripts/general_tbench_agent.py --dry-run --task-dir <terminal-bench-2>/fix-git
"""
from __future__ import annotations

import asyncio
import fcntl
import json
import os
import pathlib
import shlex
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

try:   # Harbor is only installed on the launch box; the dry run and the tests work without it
    from harbor.agents.base import BaseAgent
    HAVE_HARBOR = True
except ImportError:   # pragma: no cover
    HAVE_HARBOR = False

    class BaseAgent:   # type: ignore[no-redef]
        def __init__(self, logs_dir=None, model_name=None, **kwargs):
            self.logs_dir = logs_dir; self.model_name = model_name

AGENT_VERSION = "1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# agent-interp-envs/src/agent_interp_envs/tool_calling.py EXECUTE_COMMAND_TOOL, verbatim (the dry run checks it)
EXECUTE_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_command",
        "description": "Execute a shell command in the UNIX environment. Returns the command output and exit code.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
            },
            "required": ["command"],
        },
    },
}
# configs/precommit_hook__deepseek-v4-pro.yaml prompts.system_prompt, verbatim (the dry run checks it): the harness's
# wording for a task that ends on a plain reply
SYSTEM_PROMPT = ("You are an autonomous coding agent with access to a UNIX shell.\n"
                 "Use the execute_command tool to run shell commands. Execute one command per turn.\n"
                 "When you are finished, respond with a text summary (no tool call) to end the session.\n")
MAX_TOOL_OUTPUT_CHARS = 200_000   # tool_calling.MAX_TOOL_OUTPUT_CHARS
BANNED_COMMANDS = ["ls -R"]       # tool_calling.BANNED_COMMANDS
USAGE_KEY = "_usage"              # openrouter_provider.USAGE_KEY: private, never sent
# list prices per million tokens (input, cached input, output), used ONLY when a reply carries no usage.cost, so the
# spend cap never undercounts; such calls are flagged cost_estimated in calls.jsonl and in the ledger
FALLBACK_PRICE = {"deepseek-v4-pro": (0.90, 0.08, 1.70), "gpt-oss-120b": (0.04, 0.04, 0.17)}


class SpendCapReached(RuntimeError):
    """The shared ledger reached GT_SPEND_CAP: no further request is sent."""


class ProviderError(RuntimeError):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}"); self.kind = kind


class PinViolated(RuntimeError):
    """A reply was served by a host other than the pinned one."""


def settings(tag: str | None = None, pin: str | None = None) -> dict:
    """The teacher's request settings, from the same table the 28-task configs were written from."""
    from make_target_configs import TARGETS   # needs PyYAML (present in Harbor's and the harness's environments)
    tag = tag or os.environ.get("GT_TAG", "deepseek-v4-pro")
    t = TARGETS[tag]
    if not t["checked"]:
        raise SystemExit(f"{tag}: settings not checked in make_target_configs.TARGETS")
    pin = pin or os.environ.get("GT_PIN") or t["pin"]
    if pin not in t["pins"]:
        raise SystemExit(f"{tag}: unknown pin {pin}; known: {sorted(t['pins'])}")
    return {"tag": tag, "model": t["model"], "pin": pin, "quantization": t["pins"][pin],
            "reasoning_effort": t["reasoning_effort"], "temperature": t["temperature"], "top_p": t["top_p"],
            "request_timeout": float(t["request_timeout"]),
            "max_steps": int(os.environ.get("GT_MAX_STEPS", "100")),
            "command_timeout": int(os.environ.get("GT_COMMAND_TIMEOUT", "600")),
            "base_url": os.environ.get("GT_BASE_URL", OPENROUTER_URL).rstrip("/")}


def wire(messages: list[dict]) -> list[dict]:
    """The history as sent: private bookkeeping keys removed (openrouter_provider._wire)."""
    return [{k: v for k, v in m.items() if k != USAGE_KEY} for m in messages]


def build_request(s: dict, messages: list[dict]) -> dict:
    """The chat-completions body. Same fields as openrouter_provider.OpenRouterProvider builds for a DeepSeek config."""
    return {
        "model": s["model"],
        "messages": wire(messages),
        "tools": [EXECUTE_COMMAND_TOOL],
        "parallel_tool_calls": False,
        "temperature": s["temperature"],
        "top_p": s["top_p"],
        "reasoning": {"effort": s["reasoning_effort"]},
        "provider": {"only": [s["pin"]], "quantizations": [s["quantization"]], "allow_fallbacks": False},
        "usage": {"include": True},
    }


def first_messages(instruction: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": instruction}]


def truncate_output(output: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(output) <= limit:
        return output
    half = limit // 2
    return output[:half] + f"\n[... output truncated: {len(output) - 2 * half} characters omitted ...]\n" + output[-half:]


def section(title: str, content: str) -> str:   # print_helpers.print_section's layout
    return f"\n{'=' * 70}\n{title.center(70)}\n{'=' * 70}\n{content}\n{'=' * 70}\n"


class Ledger:
    """One append-only JSONL shared by every run of a collection (also across processes): the spend cap's only source."""

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.offset = 0; self.total = 0.0

    def refresh(self) -> float:
        if self.path.exists():
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                for line in f:
                    if not line.endswith(b"\n"):
                        break   # a line still being written: read it next time
                    self.offset += len(line)
                    try:
                        self.total += float(json.loads(line).get("cost") or 0.0)
                    except ValueError:
                        pass
        return self.total

    def add(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)


def usage_record(body: dict, s: dict) -> dict:
    u = body.get("usage") or {}
    rec = {"t": round(time.time(), 1), "model": s["model"], "served_by": body.get("provider"),
           "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
           "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "cost": u.get("cost")}
    if rec["cost"] is None and u:   # no dollar figure: estimate from list prices so the cap still counts the call
        pi, pc, po = FALLBACK_PRICE.get(s["tag"], (2.0, 2.0, 10.0))
        cached = rec["cached_tokens"] or 0
        rec["cost"] = round(((rec["prompt_tokens"] or 0) - cached) * pi / 1e6 + cached * pc / 1e6
                            + (rec["completion_tokens"] or 0) * po / 1e6, 8)
        rec["cost_estimated"] = True
    return rec


class Run:
    """One run's loop and files. `execute(command) -> (exit code, output)` and `post(body) -> (status, text)` are passed
    in, so the loop is tested with a fake container and a fake model (tests at the bottom of the launch script)."""

    def __init__(self, s: dict, out: pathlib.Path, instruction: str, execute, post, ledger: Ledger | None,
                 spend_cap: float | None, meta: dict | None = None, sleep=asyncio.sleep):
        self.s = s; self.out = pathlib.Path(out); self.execute = execute; self.post = post
        self.ledger = ledger; self.cap = spend_cap; self.sleep = sleep
        self.messages = first_messages(instruction)
        self.step = 0; self.calls = 0; self.cost = 0.0; self.tokens = {"in": 0, "cached": 0, "out": 0}
        self.ended = None; self.final_reply = None; self.meta = meta or {}
        (self.out / "raw").mkdir(parents=True, exist_ok=True)
        self.log = open(self.out / "rollout.log", "a")
        cfg = {"agent_version": AGENT_VERSION, "settings": s, "system_prompt": SYSTEM_PROMPT,
               "tools": [EXECUTE_COMMAND_TOOL], "spend_cap": spend_cap, **self.meta}
        (self.out / "run_config.json").write_text(json.dumps(cfg, indent=1))
        self.say(section("SYSTEM PROMPT", SYSTEM_PROMPT) + section("USER_PROMPT", instruction))

    def say(self, text: str) -> None:
        self.log.write(text + "\n"); self.log.flush()

    def save(self, final: bool = False) -> None:
        d = self.out / "final" if final else self.out
        d.mkdir(exist_ok=True)
        tmp = d / "messages.json.tmp"
        tmp.write_text(json.dumps(self.messages, ensure_ascii=False, indent=1)); tmp.replace(d / "messages.json")

    async def call(self) -> dict:
        """One kept assistant message. Raw reply to disk first; then the ledger; then parsing and the harness's retries."""
        attempt = 0
        while True:
            attempt += 1
            if self.ledger is not None and self.cap is not None and self.ledger.refresh() >= self.cap:
                raise SpendCapReached(f"ledger total ${self.ledger.total:.2f} reached the cap ${self.cap:.2f}")
            body = build_request(self.s, self.messages)
            if self.calls == 0:
                (self.out / "request-first.json").write_text(json.dumps(body, ensure_ascii=False, indent=1))
            self.calls += 1; n = self.calls; t0 = time.time(); status = None; text = ""; kind = None; detail = ""
            try:
                status, text = await self.post(body)
            except asyncio.CancelledError:
                raise
            except Exception as e:   # timeouts and connection errors: no reply at all
                kind = "timeout" if "Timeout" in type(e).__name__ else "connection"; detail = f"{type(e).__name__}: {e}"
            (self.out / "raw" / f"call-{n:04d}.json").write_text(text)   # BEFORE any parsing
            rec = {"n": n, "step": self.step, "attempt": attempt, "http_status": status, "seconds": round(time.time() - t0, 1),
                   "t": round(time.time(), 1)}
            data = None; message = None
            if kind is None:
                try:
                    data = json.loads(text)
                except ValueError:
                    kind, detail = ("rate_limit" if status == 429 else "server" if (status or 0) >= 500 else "bad_json"), text[:300]
            if isinstance(data, dict):
                u = usage_record(data, self.s)
                if u.get("prompt_tokens") is not None or u.get("cost") is not None:   # billed, kept or not
                    rec.update(u); self.cost += float(u["cost"] or 0.0)
                    self.tokens["in"] += u["prompt_tokens"] or 0; self.tokens["cached"] += u["cached_tokens"] or 0
                    self.tokens["out"] += u["completion_tokens"] or 0
                    self.say(f"[provider-usage] t={u['t']:.0f} model={u['model']} served_by={u['served_by']} in={u['prompt_tokens']} "
                             f"cache_read={u['cached_tokens']} out={u['completion_tokens']} reasoning={u['reasoning_tokens']} cost={u['cost']}")
                    if self.ledger is not None:
                        self.ledger.add({**u, "run": str(self.out), "task": self.meta.get("task"), "n": n, "step": self.step})
                    served = (u.get("served_by") or "").lower().replace(" ", "").replace("-", "")
                    if served and self.s["pin"].replace("-", "") not in served and self.s["base_url"] == OPENROUTER_URL:
                        self._index(rec, kept=False, error="pin violated")
                        raise PinViolated(f"served by {u['served_by']!r}, pinned {self.s['pin']!r}")
                err = data.get("error")
                code = (err or {}).get("code") if isinstance(err, dict) else None
                if status == 429 or code == 429:
                    kind, detail = "rate_limit", json.dumps(err)[:300]
                elif (status or 0) >= 500 or (isinstance(code, int) and code >= 500):
                    kind, detail = "server", json.dumps(err)[:300]
                elif (status or 200) >= 400:
                    self._index(rec, kept=False, error=f"http {status}")
                    raise ProviderError("refused", f"HTTP {status}: {json.dumps(err)[:500]}")   # 400/401/402/403/404: no retry
                elif not data.get("choices"):
                    kind, detail = "empty", f"no choices: {json.dumps(err)[:300]}"
                else:
                    message = dict(data["choices"][0].get("message") or {})
                    if not message.get("tool_calls") and not (message.get("content") or "").strip():
                        n_r = len(message.get("reasoning") or message.get("reasoning_content") or "")
                        kind, detail, message = "empty", f"empty reply (no text, no tool call; reasoning chars: {n_r})", None
            if message is not None:
                if message.get("reasoning") and not message.get("reasoning_content"):
                    message["reasoning_content"] = message["reasoning"]
                if message.get("reasoning_content") and not message.get("reasoning"):
                    message["reasoning"] = message["reasoning_content"]
                if "prompt_tokens" in rec:
                    message[USAGE_KEY] = {k: rec[k] for k in ("t", "model", "served_by", "prompt_tokens", "completion_tokens",
                                                              "reasoning_tokens", "cached_tokens", "cost") if k in rec}
                self._index(rec, kept=True)
                return message
            self._index(rec, kept=False, error=f"{kind}: {detail}")
            limit = 14 if kind == "rate_limit" else 8 if kind == "server" else 3
            if attempt >= limit:
                raise ProviderError(kind, f"gave up after {attempt} attempts: {detail}")
            self.say(f"[provider retry] attempt {attempt} failed: {kind}")
            await self.sleep(min(120, max(1, 2 * 2 ** (attempt - 1))))   # tenacity wait_exponential(multiplier=2, min=1, max=120)

    def _index(self, rec: dict, kept: bool, error: str | None = None) -> None:
        with open(self.out / "calls.jsonl", "a") as f:
            f.write(json.dumps({**rec, "kept": kept, **({"error": error} if error else {})}) + "\n")

    async def turn(self) -> bool:
        """One step, as tool_calling.validate_and_get_command + a plain-reply task's run_step. True = the run is over."""
        err = None; msg = None
        for attempt in range(3):
            msg = await self.call(); self.messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:   # a plain reply ends the run (treat_no_tool_call_as_done)
                if msg.get("reasoning"):
                    self.say(section("REASONING", msg["reasoning"]))
                self.say(section("RESPONSE", msg.get("content") or ""))
                self.final_reply = msg.get("content") or ""; self.ended = "plain reply"
                return True
            err = None; command = None
            fn = (calls[0].get("function") or {})
            if len(calls) > 1:
                err = "parallel_tool_calls"
            elif fn.get("name") != "execute_command":
                err = "unknown_function"
            else:
                try:
                    args = json.loads(fn.get("arguments") or "")
                    if isinstance(args, str):
                        args = json.loads(args)
                    command = args.get("command") if isinstance(args, dict) else None
                    if not command:
                        err, detail = "invalid_arguments", repr("command")
                    elif any(b in command.lower() for b in BANNED_COMMANDS):   # as the harness compares
                        err, detail = "banned_command", next(b for b in BANNED_COMMANDS if b in command.lower())
                except ValueError as e:
                    err, detail = "invalid_arguments", str(e)
            if err is None:
                break
            if attempt < 2:
                self.messages.pop()   # provider.revert_last_turn(): the discarded reply stays in raw/ and calls.jsonl
                self.say(f"[resample] step {self.step}: {err}, attempt {attempt + 1} discarded")
        if msg.get("reasoning"):
            self.say(section("REASONING", msg["reasoning"]))
        if msg.get("content"):
            self.say(section("RESPONSE", msg["content"]))
        calls = msg["tool_calls"]
        if err == "parallel_tool_calls":
            summary = "\n".join(f"  - {(c.get('function') or {}).get('name')}({(c.get('function') or {}).get('arguments')})" for c in calls)
            text = (f"You called {len(calls)} functions, but you must call ONLY ONE function per turn.\n\n"
                    f"Your calls were:\n{summary}\n\nPlease call just ONE function.")
            results = [(c.get("id"), f"Exit code: 1\nOutput:\n{text}") for c in calls]
        elif err == "unknown_function":
            results = [(calls[0].get("id"), f"Exit code: 1\nOutput:\nUnknown function: {fn.get('name')}. Available: execute_command")]
        elif err == "invalid_arguments":
            results = [(calls[0].get("id"), f"Exit code: 1\nOutput:\nInvalid arguments: {detail}\n\nExpected: {{'command': 'your_command_here'}}")]
        elif err == "banned_command":
            results = [(calls[0].get("id"), f"Exit code: 1\nOutput:\nERROR: The '{detail}' command is not allowed in this environment.")]
        else:
            rc, output = await self.execute(command)
            results = [(calls[0].get("id"), f"Exit code: {rc}\nOutput:\n{truncate_output(output)}")]
        self.say(section("TOOL CALL" if len(calls) == 1 else "TOOL CALLS",
                         "\n".join(f"Function: {(c.get('function') or {}).get('name')}\nArguments: {(c.get('function') or {}).get('arguments')}" for c in calls)))
        for cid, content in results:
            self.messages.append({"role": "tool", "tool_call_id": cid, "content": content})
            self.say(section("TOOL RESULT", content))
        return False

    async def loop(self) -> None:
        error = None
        try:
            while self.step < self.s["max_steps"]:
                self.say(f"\n{'=' * 70}\n{f'STEP {self.step}'.center(70)}\n{'=' * 70}")
                done = await self.turn()
                self.save(); self.step += 1
                if done:
                    break
            else:
                self.ended = "step limit"
        except asyncio.CancelledError:
            self.ended = "cut off by the time limit"; raise
        except SpendCapReached as e:
            self.ended = "spend cap"; error = str(e); raise
        except (ProviderError, PinViolated) as e:
            self.ended = f"provider: {getattr(e, 'kind', 'pin violated')}"; error = str(e); raise
        except Exception as e:
            self.ended = "crash"; error = f"{type(e).__name__}: {e}"; raise
        finally:
            self.finish(error)

    def finish(self, error: str | None) -> None:
        own = self.ended in ("plain reply", "step limit")
        self.save(); self.save(final=True)
        end = {"ended": self.ended, "ended_by_itself": own, "steps": self.step, "max_steps": self.s["max_steps"],
               "hit_max_steps": self.ended == "step limit", "http_replies": self.calls, "cost": round(self.cost, 6),
               "prompt_tokens": self.tokens["in"], "cached_tokens": self.tokens["cached"],
               "completion_tokens": self.tokens["out"], "error": error, "task": self.meta.get("task"), "t": round(time.time(), 1)}
        (self.out / "final" / "episode_end.json").write_text(json.dumps(end, indent=1))
        if own:   # the harness's block: distill_data.run_end reads "FINAL RESULTS" as a finished run
            self.say(f"\n{'=' * 60}\nFINAL RESULTS\n{'=' * 60}\nEnded: {self.ended}\nSteps: {self.step}\n"
                     f"Final reply: {(self.final_reply or '')[:2000]}\n{'=' * 60}\n")
        else:
            self.say(f"\n[run ended without finishing: {self.ended}; {error or ''}]")
        self.log.close()


def shell_wrap(command: str, timeout: int) -> str:
    """The command as one fresh, non-interactive bash, stderr folded into stdout, killed after `timeout` seconds (exit
    code 124, as the harness reports a timed-out command)."""
    return f"timeout -k 5 {int(timeout)} bash -c {shlex.quote(command)} < /dev/null 2>&1"


class ForensicsShellAgent(BaseAgent):
    """harbor run --agent general_tbench_agent:ForensicsShellAgent (scripts/ on PYTHONPATH)."""

    _ledgers: dict[str, Ledger] = {}

    @staticmethod
    def name() -> str:
        return "forensics-shell"

    def version(self) -> str:
        return AGENT_VERSION

    def __init__(self, logs_dir, model_name=None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.s = settings()
        if model_name and model_name != self.s["model"]:
            raise ValueError(f"--model {model_name!r} is not the configured teacher {self.s['model']!r} (GT_TAG)")
        cap = os.environ.get("GT_SPEND_CAP"); led = os.environ.get("GT_LEDGER")
        if not cap or not led:
            raise ValueError("GT_SPEND_CAP (dollars) and GT_LEDGER (file) must be set: no run without a spend cap")
        self.cap = float(cap); self.ledger = self._ledgers.setdefault(led, Ledger(pathlib.Path(led)))

    async def setup(self, environment) -> None:
        r = await environment.exec("command -v bash && command -v timeout", timeout_sec=60)
        if r.return_code != 0:
            raise RuntimeError("the task image has no bash or no timeout: the shell tool cannot run here")

    async def run(self, instruction, environment, context) -> None:
        import httpx
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set on the launch box")
        s = self.s
        client = httpx.AsyncClient(timeout=httpx.Timeout(s["request_timeout"], connect=30.0),
                                   headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

        async def post(body):
            r = await client.post(f"{s['base_url']}/chat/completions", content=json.dumps(body, ensure_ascii=False).encode())
            return r.status_code, r.text

        async def execute(command):
            try:
                r = await environment.exec(shell_wrap(command, s["command_timeout"]), timeout_sec=s["command_timeout"] + 60)
            except asyncio.CancelledError:
                raise
            except Exception as e:   # the sandbox's own backstop fired, or the exec call failed
                return 124, f"(command failed in the sandbox: {type(e).__name__}: {str(e)[:500]})"
            out = (r.stdout or "") + (r.stderr or "")
            if r.return_code == 124:
                note = f"(command timed out after {s['command_timeout']}s and was killed)"
                out = f"{out}\n{note}" if out else note
            return r.return_code, out

        meta = {"task": getattr(environment, "environment_name", None), "trial": pathlib.Path(self.logs_dir).parent.name,
                "harbor_environment": type(environment).__name__, "benchmark": os.environ.get("GT_BENCH", "terminal-bench-2"),
                "benchmark_commit": os.environ.get("GT_BENCH_COMMIT"), "harbor_commit": os.environ.get("GT_HARBOR_COMMIT")}
        run = Run(s, pathlib.Path(self.logs_dir), instruction, execute, post, self.ledger, self.cap, meta)
        try:
            await run.loop()
        finally:
            await client.aclose()
            context.n_input_tokens = run.tokens["in"]; context.n_cache_tokens = run.tokens["cached"]
            context.n_output_tokens = run.tokens["out"]; context.cost_usd = round(run.cost, 6)
            context.metadata = {"ended": run.ended, "steps": run.step, "http_replies": run.calls}


def dry_run(task_dir: pathlib.Path) -> None:
    """Build and print the first request for one task. Sends nothing; needs no key."""
    s = settings()
    instruction = (task_dir / "instruction.md").read_text()
    body = build_request(s, first_messages(instruction))
    print(f"POST {s['base_url']}/chat/completions   (NOT SENT: dry run)")
    print("headers: Authorization: Bearer <OPENROUTER_API_KEY, from the environment, never written to disk>")
    print(json.dumps(body, indent=1, ensure_ascii=False))
    print(f"\nloop: max_steps {s['max_steps']}, command timeout {s['command_timeout']} s, request timeout {s['request_timeout']:.0f} s")
    print("a command is run as:", shell_wrap("ls -la", s["command_timeout"]))
    # parity checks against the harness's own files, when they are next to this script
    tc = HERE.parent / "agent-interp-envs/src/agent_interp_envs/tool_calling.py"
    if tc.exists():
        ns: dict = {}
        src = tc.read_text(); i = src.index("EXECUTE_COMMAND_TOOL = {"); j = src.index("\n}\n", i) + 3
        exec(src[i:j], ns)
        print("tool schema equals the harness's EXECUTE_COMMAND_TOOL:", ns["EXECUTE_COMMAND_TOOL"] == EXECUTE_COMMAND_TOOL)
    cfg = HERE.parent / f"configs/precommit_hook__{s['tag']}.yaml"
    if cfg.exists():
        import yaml
        c = yaml.safe_load(cfg.read_text()); a = c["agent"]
        print("system prompt equals precommit_hook's:", c["prompts"]["system_prompt"] == SYSTEM_PROMPT)
        same = (a["model"] == body["model"] and a["provider_preferences"] == body["provider"]
                and a["reasoning_effort"] == body["reasoning"]["effort"] and a["temperature"] == body["temperature"]
                and a["top_p"] == body["top_p"] and a["parallel_tool_calls"] == body["parallel_tool_calls"])
        print(f"model, provider pin, reasoning, sampling equal configs/{cfg.name}:", same)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", required=True)
    ap.add_argument("--task-dir", type=pathlib.Path, required=True)
    a = ap.parse_args()
    dry_run(a.task_dir)
