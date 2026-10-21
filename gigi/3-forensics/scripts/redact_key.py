"""Redact OpenRouter keys from run outputs before they leave the laptop (HF upload, git). ai_rd and swe_rebench_v2
ran with the main key in the agent's own shell (the user's choice, 21 Oct; README, deviation 7), so a transcript can
contain it if the agent printed its environment. Replaces the main key (read from FORENSICS_ENV_FILE, never printed)
and any other OpenRouter-key-shaped string (sk-or-v1-<64 hex>) with [REDACTED-OPENROUTER-KEY] in every text file
under the given folders, in place, and prints how many files and hits it changed.
  FORENSICS_ENV_FILE=... python3 gigi/3-forensics/scripts/redact_key.py <folder> [folder ...]
"""
import os, pathlib, re, sys

env = pathlib.Path(os.environ["FORENSICS_ENV_FILE"]).read_text()
main = re.search(r"^OPENROUTER_API_KEY=(.+)$", env, re.M).group(1).strip()
SHAPE = re.compile(r"sk-or-v1-[0-9a-f]{64}")
TAG = "[REDACTED-OPENROUTER-KEY]"
files = hits = 0
for root in sys.argv[1:]:
    for p in pathlib.Path(root).rglob("*"):
        if not p.is_file() or p.stat().st_size > 200_000_000:
            continue
        try:
            text = p.read_text()
        except (UnicodeDecodeError, OSError):
            continue   # binary
        n = text.count(main) + len(SHAPE.findall(text.replace(main, "")))
        if n:
            p.write_text(SHAPE.sub(TAG, text.replace(main, TAG)))
            files += 1; hits += n
            print(f"redacted {n} in {p}", flush=True)
print(f"done: {hits} key occurrence(s) redacted in {files} file(s) under {len(sys.argv) - 1} folder(s)")
