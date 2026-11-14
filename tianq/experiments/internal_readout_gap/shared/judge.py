"""A judge model that answers a prompt with one whole number from 0 to 10: the one place that calls it.

JUDGE through OpenRouter, provider pinned, no fallbacks. Every answer is cached one per line in the caller's folder
(one file per group of questions), so a rerun only asks what is missing, and the judge stops before the folder's total
cost passes the caller's budget. Both experiments use this: distillation_similarity to rate how alike two readouts
are, lie_detection_gap to rate how much a readout looks like a lie.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import re
import threading
import time

from openai import OpenAI

JUDGE = "openai/gpt-5-mini"
PROVIDER = {"order": ["openai"], "allow_fallbacks": False}
THREADS = 32
ATTEMPTS = 5
READOUT_KINDS = {
    "NLA sentences": "Each readout is a few short descriptions written by a tool that turns an internal state into sentences.",
    "J-lens words": "Each readout is a list of words the model was leaning towards, most likely first, possibly in several languages."}


class Judge:
    def __init__(self, folder: pathlib.Path, budget: float) -> None:
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        self.folder, self.budget = folder, budget
        folder.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        for path in folder.glob("*.jsonl"):
            for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
                self.cache[row["key"]] = row
        self.spent = sum(r["cost"] for r in self.cache.values())

    def rate(self, key: str, prompt: str | None) -> float:
        """The judge's 0-10 answer to `prompt`; NaN if the answer is not a number, or if `prompt` is None (nothing
        to show the judge). `key` is "<group> || <item>": the cache key, and <group> names the cache file."""
        if key in self.cache:
            return self.cache[key]["rating"]
        if prompt is None:
            row = {"key": key, "rating": float("nan"), "answer": "", "cost": 0.0}
        else:
            with self.lock:
                if self.spent >= self.budget:
                    raise SystemExit(f"stopped at ${self.spent:.2f}: budget ${self.budget:.0f}")
            for attempt in range(ATTEMPTS):   # OpenRouter sometimes returns a provider error as a body with no choices
                r = self.client.chat.completions.create(
                    model=JUDGE, messages=[{"role": "user", "content": prompt}],
                    extra_body={"provider": PROVIDER, "usage": {"include": True}, "reasoning": {"effort": "low"}})
                if r.choices:
                    break
                time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(f"the judge returned no answer {ATTEMPTS} times for {key}")
            text = (r.choices[0].message.content or "").strip()
            m = re.fullmatch(r"(10|\d)", text)
            cost = float(getattr(r.usage, "cost", 0) or (r.usage.model_extra or {}).get("cost", 0) or 0)
            row = {"key": key, "rating": float(m.group(1)) if m else float("nan"), "answer": text, "cost": cost}
        with self.lock:
            self.cache[key] = row
            self.spent += row["cost"]
            with open(self.folder / f"{key.split(' || ')[0].replace(' | ', '_').replace(' ', '_')}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        return row["rating"]

    def rate_all(self, jobs: list[tuple[str, str | None]]) -> list[float]:
        """rate() for every (key, prompt), THREADS at a time, in order."""
        with concurrent.futures.ThreadPoolExecutor(THREADS) as pool:
            return list(pool.map(lambda j: self.rate(*j), jobs))
