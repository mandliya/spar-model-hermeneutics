"""HuggingFace as the durable home for everything this run produces.

Durability and transport are deliberately separated.

Every finished row is written to local disk the moment it completes — one row,
one write, never batched — so a process crash loses nothing. Those files are
then synced to the Hub in periodic folder uploads.

The split is forced by the platform, not chosen for convenience. Streaming each
row straight to the Hub means two commits per row, and HuggingFace caps a
repository at 128 commits per hour; 310 rows needs 620. The first attempt at
per-row uploads died at row 56 with a 429. `upload_folder` is what the Hub's own
rate-limit message points at, and it turns the whole run into a handful of
commits while leaving the per-row durability guarantee intact.

Two files per row, because the shapes want different encodings:

    rows/<uid>.json          the full end-to-end record, nothing truncated
    activations/<uid>.npy    [n_layers+1, D] float32

float32 rather than float16: Gemma-3 carries a few "massive activation"
dimensions reaching ~50,000 in deep layers, and float16 tops out at 65,504.
Storing fp16 overflowed one dimension per layer to inf, which silently poisoned
every probe logit. The model runs in bfloat16, whose exponent range matches
float32, so this cast is lossless as well as safe.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

ROWS_PREFIX = "rows"
ACTIVATIONS_PREFIX = "activations"
SUMMARY_PREFIX = "summary"

MAX_ATTEMPTS = 4
BACKOFF_SECONDS = 5.0

# Rows between Hub syncs. At ~3.4s/row this is about three minutes of work at
# risk if the box itself disappears, against roughly seven commits for a full
# 310-row run — comfortably inside the hourly ceiling.
SYNC_EVERY = 50


@dataclass
class HfStore:
    repo_id: str
    token: str
    scratch: Path
    # Which experiment these rows belong to: "city" for the in-domain
    # reproduction, "ascii" for the transfer. Both write to one repo, because
    # they are two halves of one result, and without a namespace they would
    # share a single flat rows/ prefix — 300 city rollouts interleaved with
    # 310 ASCII rows, distinguishable only by reading the uids.
    namespace: str = ""
    _api: HfApi | None = None
    _pending: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._api = HfApi(token=self.token)
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        self.activations_dir.mkdir(parents=True, exist_ok=True)

    # The local layout under scratch/ mirrors the repo layout exactly, so the
    # folder upload is a plain mirror and needs no path translation.

    @property
    def base(self) -> Path:
        return self.scratch / self.namespace if self.namespace else self.scratch

    @property
    def rows_dir(self) -> Path:
        return self.base / ROWS_PREFIX

    @property
    def activations_dir(self) -> Path:
        return self.base / ACTIVATIONS_PREFIX

    def repo_path(self, *parts: str) -> str:
        pieces = [self.namespace, *parts] if self.namespace else list(parts)
        return "/".join(pieces)

    @property
    def api(self) -> HfApi:
        assert self._api is not None
        return self._api

    def ensure_repo(self) -> None:
        """Create the dataset repo if absent, always private.

        Private is the default for research output here, and `exist_ok=True`
        does not downgrade an existing repo's visibility, so this cannot
        silently re-expose something that was deliberately locked down.

        Public repos are throttled less aggressively on the free tier, which is
        why this started out public — but that tradeoff disappeared once the
        account had PRO, and it was never a good reason to publish model
        outputs by default.
        """
        self.api.create_repo(
            repo_id=self.repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )

    def completed_uids(self) -> set[str]:
        """Rows already done, locally or on the Hub.

        The union matters for resume. Local files are ahead of the Hub between
        syncs, and the Hub is what survives the box being destroyed, so neither
        source alone is sufficient.
        """
        local = {p.stem for p in self.rows_dir.glob("*.json")}
        wanted = self.repo_path(ROWS_PREFIX) + "/"
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        except HfHubHTTPError:
            return local
        remote = {
            Path(f).stem
            for f in files
            if f.startswith(wanted) and f.endswith(".json")
        }
        return local | remote

    def push_row(self, uid: str, record: dict, activations: np.ndarray | None) -> None:
        """Persist one unit of work. Local write is immediate and unconditional."""
        if activations is not None:
            np.save(self.activations_dir / f"{uid}.npy", activations)

        path = self.rows_dir / f"{uid}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        self._pending += 1
        if self._pending >= SYNC_EVERY:
            self.sync()

    def sync(self) -> None:
        """Push everything written since the last sync, as one commit."""
        if self._pending == 0:
            return
        print(f"    syncing {self._pending} rows to {self.repo_id} ...", flush=True)
        self._with_retries(
            lambda: self.api.upload_folder(
                folder_path=str(self.scratch),
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"add {self._pending} rows",
            ),
            what="folder sync",
        )
        self._pending = 0

    def push_summary(self, name: str, payload: dict) -> None:
        local = self.scratch / SUMMARY_PREFIX
        local.mkdir(exist_ok=True)
        (local / f"{name}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        self._pending += 1
        self.sync()

    def push_file(self, local_path: Path, path_in_repo: str) -> None:
        """Upload one file as-is, outside the per-row batching."""
        self._with_retries(
            lambda: self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
            ),
            what=f"upload of {path_in_repo}",
        )

    def load_record(self, uid: str) -> dict:
        local = self.rows_dir / f"{uid}.json"
        if local.exists():
            return json.loads(local.read_text(encoding="utf-8"))
        fetched = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=self.repo_path(ROWS_PREFIX, f"{uid}.json"),
            token=self.token,
        )
        return json.loads(Path(fetched).read_text(encoding="utf-8"))

    def _with_retries(self, action, what: str) -> None:
        """Retry transient Hub failures with backoff.

        A retry here is never a fallback to a different value — if the Hub stays
        unreachable the error propagates, because a row that exists only on an
        ephemeral rented box is not actually saved.
        """
        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                action()
                return
            except (HfHubHTTPError, OSError) as exc:
                last = exc
                if attempt == MAX_ATTEMPTS:
                    break
                delay = BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(
                    f"    {what} failed ({exc.__class__.__name__}), "
                    f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError(f"gave up on {what} after {MAX_ATTEMPTS} attempts") from last
