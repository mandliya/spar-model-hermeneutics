"""Per-row storage on HuggingFace with a local mirror, and reading it back.

Layout, identical in the Hub repo and under the local scratch directory:

    <dataset>/rows/<uid>.json                   full record: transcript, rendered prompt,
                                                token ids, label, fidelity, versions
    <dataset>/hidden_states/<uid>.safetensors   "hidden_states": (positions, n_layers + 1,
                                                d_model), bfloat16; the record names the positions
    <dataset>/results/<name>.json               evaluation outputs, per-row scores included
    <dataset>/charts/<name>.png                 charts drawn from a result
    <dataset>/probes/<name>.pt                  probes trained on this dataset (core/training.py)
    logs/<name>.log                             raw execution logs (see core/logs.py)

bfloat16 on disk because the model computes in bfloat16: saving the tensor as
produced is lossless at half the size of float32. float16 would NOT be:
Gemma's massive-activation dimensions exceed its range, which silently
corrupted week 1's first run.

Durability has two halves. Each row is written locally the moment it finishes,
and everything written since the last sync (plus the current logs) is pushed as
one commit every SYNC_EVERY rows, because the Hub caps a repo at 128 commits
per hour. On startup, rows that reached local disk but never the Hub are pushed
first, then the run resumes from the gap.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download
from huggingface_hub import constants as hf_constants
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from lie_detectors import load_probe
from safetensors import safe_open
from safetensors.torch import save_file

from dyl_repro.core.logs import log_operations, with_retries
from dyl_repro.core.probe import PROBE_POSITION, DylProbe

SCRATCH = Path(__file__).resolve().parents[2] / "scratch"
LOG_DIR = SCRATCH / "logs"
SYNC_EVERY = 250
TENSOR_KEY = "hidden_states"
# The Hub's rate-limit window: a 429 clears once it has passed.
RATE_LIMIT_WAIT_SECONDS = 300
CONNECTION_WAIT_SECONDS = 60
MAX_IDLE_ROUNDS = 5


@contextmanager
def _without_xet() -> Iterator[None]:
    """Download over the plain /resolve/ route instead of Xet (see RowStore.pull).

    HF_HUB_DISABLE_XET is huggingface_hub's own switch, read at call time
    (file_download.py L1722 checks is_xet_available(), which returns False when
    it is set). Flipped only for the duration, so uploads are unaffected."""
    previous = hf_constants.HF_HUB_DISABLE_XET
    hf_constants.HF_HUB_DISABLE_XET = True
    try:
        yield
    finally:
        hf_constants.HF_HUB_DISABLE_XET = previous


@dataclass
class RowStore:
    repo_id: str
    token: str
    scratch: Path
    dataset: str
    _pending: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.api = HfApi(token=self.token)
        (self.scratch / self.dataset / "rows").mkdir(parents=True, exist_ok=True)
        (self.scratch / self.dataset / "hidden_states").mkdir(parents=True, exist_ok=True)

    def _row_paths(self, uid: str) -> tuple[str, str]:
        return (
            f"{self.dataset}/hidden_states/{uid}.safetensors",
            f"{self.dataset}/rows/{uid}.json",
        )

    def ensure_repo(self) -> None:
        # Private, and exist_ok never changes an existing repo's visibility.
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)

    def done_uids(self) -> set[str]:
        """Rows finished locally or on the Hub. Pushes any finished only locally.

        A crash between syncs leaves rows on local disk that never reached the
        Hub. Counting them as done without pushing them would lose them for good
        when the box goes away, so they are pushed here before anything else.
        """
        local = {p.stem for p in (self.scratch / self.dataset / "rows").glob("*.json")}
        prefix = f"{self.dataset}/rows/"
        remote = {
            Path(f).stem
            for f in self.api.list_repo_files(self.repo_id, repo_type="dataset")
            if f.startswith(prefix) and f.endswith(".json")
        }
        unsynced = sorted(local - remote)
        if unsynced:
            print(f"    {len(unsynced)} rows were stored locally but never pushed; pushing now", flush=True)
            for uid in unsynced:
                self._pending += self._row_paths(uid)
            self.sync()
        return local | remote

    def put(self, uid: str, record: dict, hidden_states: torch.Tensor, positions: tuple[str, ...]) -> None:
        tensor_path, record_path = self._row_paths(uid)
        save_file(
            {TENSOR_KEY: hidden_states.contiguous()},
            self.scratch / tensor_path,
            metadata={"positions": ",".join(positions)},
        )
        # The record is written last: its presence is what marks the row done.
        (self.scratch / record_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
        self._pending += [tensor_path, record_path]
        if len(self._pending) >= 2 * SYNC_EVERY:
            self.sync()

    def put_result(self, name: str, payload: dict) -> None:
        path = f"{self.dataset}/results/{name}.json"
        (self.scratch / path).parent.mkdir(parents=True, exist_ok=True)
        (self.scratch / path).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        self._pending.append(path)
        self.sync()

    def put_probe(self, probe: DylProbe) -> None:
        """A probe we trained, as {"state_dict", "hyperparams"}: the layout
        of AISI's released checkpoints, so lie_detectors.load_probe reads it back
        (loading.py L48-L55):
        https://github.com/UKGovernmentBEIS/lie_detectors/blob/88043087688308bc9cefe7cbdbf9b3806a679149/src/lie_detectors/loading.py#L48-L55
        """
        path = f"{self.dataset}/probes/{probe.checkpoint}.pt"
        (self.scratch / path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": probe.module.state_dict(), "hyperparams": probe.module.hyperparams.model_dump(mode="json")},
            self.scratch / path,
        )
        self._pending.append(path)

    def probe(self, name: str) -> DylProbe:
        """A probe stored by put_probe, loaded back through the library's own loader."""
        path = hf_hub_download(self.repo_id, f"{self.dataset}/probes/{name}.pt", repo_type="dataset",
                               local_dir=self.scratch, token=self.token)
        module = load_probe(path)
        # Probes saved before read positions were recorded all read PROBE_POSITION.
        position = (module.hyperparams.model_extra or {}).get("position", PROBE_POSITION)
        return DylProbe(name, module, trained_on=module.hyperparams.train_dataset, position=position)

    def stored(self) -> set[str]:
        """Every path in the Hub repo, from one listing call."""
        return set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))

    def put_chart(self, name: str, figure) -> None:
        """A matplotlib figure, pushed with the next sync."""
        path = f"{self.dataset}/charts/{name}.png"
        (self.scratch / path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(self.scratch / path, dpi=150)
        self._pending.append(path)

    def sync(self) -> None:
        """Push everything written since the last sync, plus the current logs, as one commit."""
        if not self._pending:
            return
        paths = list(self._pending)
        print(f"    syncing {len(paths)} files to {self.repo_id} ...", flush=True)
        operations = [
            CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(self.scratch / p)) for p in paths
        ] + log_operations(LOG_DIR)
        with_retries(
            lambda: self.api.create_commit(
                self.repo_id,
                operations=operations,
                commit_message=f"{self.dataset}: add {len(paths)} files",
                repo_type="dataset",
            ),
            what="commit",
        )
        self._pending.clear()

    def result(self, name: str) -> dict:
        """One stored result, fetched into scratch at its current version on the Hub
        (re-downloaded when the Hub copy changed, e.g. after a re-evaluation)."""
        path = hf_hub_download(
            self.repo_id, f"{self.dataset}/results/{name}.json", repo_type="dataset",
            local_dir=self.scratch, token=self.token,
        )
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def pull(self) -> None:
        """Download this dataset's rows and hidden states that are missing locally.

        One listing call, then only the missing files. Asking the Hub about every
        file instead (a plain snapshot_download over a scratch the run itself
        wrote) spends one request per file, and 10k files trips the Hub's
        12k-requests-per-5-minutes limit even when nothing needs downloading.

        Downloads skip Xet storage. A Xet download first asks the Hub API for a
        read token, one call per file, and the API bucket allows 2,500 calls per
        5 minutes on a PRO account; a 5000-row dataset is 10k files. The plain
        route (`/resolve/` URLs) counts against the Resolvers bucket instead,
        12,000 per 5 minutes on PRO (https://huggingface.co/docs/hub/rate-limits).
        Only downloads here: uploads (sync) keep the default path.

        If a pull still hits a 429, it waits out the window and resumes from what
        reached disk. A dropped connection (the CDN resets some long pulls; the
        library then fails the file lookup with LocalEntryNotFoundError) or a
        response that arrives corrupted (ContentDecodingError, seen once on the
        listing itself) waits a minute and resumes the same way. MAX_IDLE_ROUNDS rounds in a row that get
        nothing through raise, so an outage fails loudly instead of looping.
        """
        wanted = (f"{self.dataset}/rows/", f"{self.dataset}/hidden_states/")
        before, idle = None, 0
        while True:
            try:
                missing = [
                    f
                    for f in self.api.list_repo_files(self.repo_id, repo_type="dataset")
                    if f.startswith(wanted) and not (self.scratch / f).exists()
                ]
                print(f"    pull: {len(missing)} files missing locally", flush=True)
                idle = idle + 1 if before is not None and len(missing) >= before else 0
                if idle >= MAX_IDLE_ROUNDS:
                    raise RuntimeError(f"pull made no progress in {idle} rounds ({len(missing)} files still missing)")
                before = len(missing)
                if missing:
                    with _without_xet():
                        snapshot_download(
                            self.repo_id,
                            repo_type="dataset",
                            allow_patterns=missing,
                            local_dir=self.scratch,
                            token=self.token,
                        )
                return
            except HfHubHTTPError as exc:
                if exc.response is None or exc.response.status_code != 429:
                    raise
                print(f"    pull: rate-limited by the Hub; waiting {RATE_LIMIT_WAIT_SECONDS}s, then resuming",
                      flush=True)
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
            except (LocalEntryNotFoundError, requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.ContentDecodingError) as exc:
                print(f"    pull: connection dropped ({type(exc).__name__}); waiting {CONNECTION_WAIT_SECONDS}s,"
                      " then resuming", flush=True)
                time.sleep(CONNECTION_WAIT_SECONDS)


class ActivationCache:
    """Read one dataset's stored rows back, as numpy, for scoring."""

    def __init__(self, scratch: Path, dataset: str) -> None:
        self.root = scratch / dataset
        paths = sorted((self.root / "rows").glob("*.json"))
        self.records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]

    def _tensor_path(self, record: dict) -> Path:
        return self.root / "hidden_states" / f"{record['uid']}.safetensors"

    def iter_rows(self) -> Iterator[tuple[dict, np.ndarray]]:
        """(record, float32 array of shape (positions, n_layers + 1, d_model)) per row."""
        for record in self.records:
            with safe_open(self._tensor_path(record), framework="pt") as f:
                yield record, f.get_tensor(TENSOR_KEY).float().numpy()

    def matrix(self, position: str, hidden_states_index: int) -> tuple[list[str], list[int | None], np.ndarray]:
        """(uids, labels, activations (N, d_model)) over every stored row, labelled or not."""
        rows = []
        for record in self.records:
            # Each row names its stored positions: POSITIONS, or REPLY_POSITIONS for a no-follow-up run.
            p = record["hidden_states"]["positions"].index(position)
            with safe_open(self._tensor_path(record), framework="pt") as f:
                rows.append(f.get_slice(TENSOR_KEY)[p, hidden_states_index].float().numpy())
        return [r["uid"] for r in self.records], [r["label"] for r in self.records], np.stack(rows)
