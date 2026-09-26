"""Push raw execution logs to the Hub whole, with only secrets masked.

Every log under a log directory is uploaded in full: the exact commands, their
complete output (tqdm carriage returns and all), and exit codes. Nothing is
truncated. The single edit is masking credentials at upload time, because
these logs capture raw terminal output and a failed request once printed an HF
token straight into it. The copy on the box stays byte-for-byte raw.

Deliberately imports nothing but huggingface_hub, so the same push runs from a
laptop with no ML stack installed:

    python -m dyl_repro.core.logs <log_dir>
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import HfHubHTTPError

LOGS_PREFIX = "logs"
MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 5.0

SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    # vastai 2FA: `tfa login ... --secret <S> -c <CODE>`
    re.compile(rb"(?<=--secret )\S+"),
    # vastai 2FA: `tfa send-email` prints "Secret token: <S>"
    re.compile(rb"(?i)(?<=secret token: )\S+"),
    # vastai `create instance` prints {'instance_api_key': '<K>'}
    re.compile(rb"(?<=api_key': ')[^']+"),
    re.compile(rb"(?<=tfa login )(.*?-c )\S+"),
)
MASK = b"***REDACTED***"


def redacted(data: bytes) -> bytes:
    token = os.environ.get("HF_TOKEN", "").encode()
    if token:
        data = data.replace(token, MASK)
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            data = pattern.sub(lambda m: m.group(1) + MASK, data)
        else:
            data = pattern.sub(MASK, data)
    return data


def log_operations(log_dir: Path) -> list[CommitOperationAdd]:
    if not log_dir.is_dir():
        return []
    return [
        CommitOperationAdd(path_in_repo=f"{LOGS_PREFIX}/{path.name}", path_or_fileobj=redacted(path.read_bytes()))
        for path in sorted(log_dir.iterdir())
        if path.is_file()
    ]


def with_retries(action: Callable[[], object], what: str) -> None:
    """Retry transient Hub failures. A retry resends the same thing; if the Hub
    stays down the error propagates, since work that exists only on a rented
    box is not saved."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            action()
            return
        except (HfHubHTTPError, OSError) as exc:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(f"gave up on {what} after {MAX_ATTEMPTS} attempts") from exc
            delay = BACKOFF_SECONDS * 2 ** (attempt - 1)
            print(f"    {what} failed ({exc!r}), retry {attempt} in {delay:.0f}s", flush=True)
            time.sleep(delay)


def push(repo_id: str, token: str, log_dir: Path) -> None:
    operations = log_operations(log_dir)
    if not operations:
        print(f"no logs in {log_dir}", flush=True)
        return
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    with_retries(
        lambda: api.create_commit(
            repo_id,
            operations=operations,
            commit_message=f"logs: {len(operations)} files from {log_dir.name}",
            repo_type="dataset",
        ),
        what="log push",
    )
    print(f"pushed {len(operations)} log files to {repo_id}/{LOGS_PREFIX}", flush=True)


if __name__ == "__main__":
    push(os.environ["HF_DATASET_REPO"], os.environ["HF_TOKEN"], Path(sys.argv[1]))
