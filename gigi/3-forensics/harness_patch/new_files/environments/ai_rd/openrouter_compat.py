"""gigi/3-forensics (26 Oct 2026): keep ai_rd's scorer working after OpenRouter retired `openai/gpt-3.5-turbo`.

Installed in the image's site-packages (outside the agent's /agent workspace, so score.py and api_utils.py stay
byte-identical to the 21 Oct runs and their checkpoints) and loaded at interpreter start by a .pth file. Any
chat-completions call that asks for `openai/gpt-3.5-turbo` is sent to the dated `openai/gpt-3.5-turbo-16k` on Azure
(OpenAI's own endpoint rejects it as deprecated). A 429 for it, or for score.py's answer extractor
`google/gemini-2.5-flash-lite` (rate-limited upstream 102 times in the 21 Oct ai_rd replays, so agents saw the
scorer crash), is waited out (up to ~5 min) instead of raised: a busy upstream slows a score.py run down rather
than crashing it, the same for both models. Calls for any other model are untouched.
"""
import importlib.abc
import importlib.util
import sys
import time

ALIAS = {"openai/gpt-3.5-turbo": "openai/gpt-3.5-turbo-16k"}
WAIT_OUT = {"openai/gpt-3.5-turbo", "google/gemini-2.5-flash-lite"}
PROVIDER = {"only": ["azure"], "allow_fallbacks": False}
WAITS = [2, 4, 8, 15, 30, 30, 60, 60, 60, 60]


def _rewrite(kwargs):
    """True if the call is one whose 429s are waited out; aliases the retired model name."""
    model = kwargs.get("model")
    if model not in WAIT_OUT:
        return False
    if model not in ALIAS:
        return True
    kwargs["model"] = ALIAS[model]
    extra = dict(kwargs.get("extra_body") or {})
    extra["provider"] = dict(PROVIDER)
    kwargs["extra_body"] = extra
    return True


def _patch():
    from openai import RateLimitError
    from openai.resources.chat import completions as cc

    sync_create, async_create = cc.Completions.create, cc.AsyncCompletions.create

    def create(self, *args, **kwargs):
        if not _rewrite(kwargs):
            return sync_create(self, *args, **kwargs)
        for w in WAITS + [None]:
            try:
                return sync_create(self, *args, **kwargs)
            except RateLimitError:
                if w is None:
                    raise
                time.sleep(w)

    async def acreate(self, *args, **kwargs):
        import asyncio
        if not _rewrite(kwargs):
            return await async_create(self, *args, **kwargs)
        for w in WAITS + [None]:
            try:
                return await async_create(self, *args, **kwargs)
            except RateLimitError:
                if w is None:
                    raise
                await asyncio.sleep(w)

    cc.Completions.create, cc.AsyncCompletions.create = create, acreate


class _Finder(importlib.abc.MetaPathFinder):
    """Patch once the openai package has been imported (never imports it on its own)."""

    def find_spec(self, name, path, target=None):
        if name != "openai":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            try:
                _patch()
            except Exception as e:   # never break an import
                print(f"[openrouter_compat] not patched: {e}", file=sys.stderr)

        loader.exec_module = exec_module
        return spec


if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
