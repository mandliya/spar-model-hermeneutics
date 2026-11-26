"""Blank this box's keys wherever a log printed them, so the log can go to HF.

    python3 blank_keys.py /workspace/logs/classic_nla_install_training_stack.log [more files or folders]

Why a log holds keys at all: Miles' build_conda.sh runs with `set -x` and sources ~/.bashrc, which on a Vast box reads
/workspace/.env and the box's own start-up variables, so the trace prints each one. HF refuses a commit that holds a
live token. Here each key's value is replaced in place by <removed NAME>; nothing else in the file changes. The keys
are the variables in /workspace/.env and in the box's start-up environment whose names say key, token, secret or
password.
Prints how many replacements each file had, never the values.
"""

from __future__ import annotations

import pathlib
import re
import sys

ENV_FILE = pathlib.Path("/workspace/.env")
BOX_VARIABLES = pathlib.Path("/proc/1/environ")
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)
SHORTEST = 12   # a shorter value is a setting, not a key, and could match ordinary text


def keys() -> dict[bytes, bytes]:
    """value -> name, longest values first so a key that contains another is replaced whole."""
    lines = ENV_FILE.read_bytes().splitlines() + BOX_VARIABLES.read_bytes().split(b"\0")
    pairs = [line.split(b"=", 1) for line in lines if b"=" in line and not line.startswith(b"#")]
    found = {value.strip().strip(b"'\""): name.strip().removeprefix(b"export ") for name, value in pairs
             if SECRET_NAME.search(name.decode())}
    return dict(sorted(((v, n) for v, n in found.items() if len(v) >= SHORTEST), key=lambda item: -len(item[0])))


def main() -> None:
    secrets = keys()
    files = [f for arg in sys.argv[1:] for f in ([pathlib.Path(arg)] if pathlib.Path(arg).is_file() else sorted(pathlib.Path(arg).rglob("*"))) if f.is_file()]
    for f in files:
        text = f.read_bytes()
        count = sum(text.count(value) for value in secrets)
        for value, name in secrets.items():
            text = text.replace(value, b"<removed " + name + b">")
        if count:
            f.write_bytes(text)
        print(f"{f}: {count} key values blanked")


if __name__ == "__main__":
    main()
