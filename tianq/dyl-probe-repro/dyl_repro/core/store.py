"""Per-row storage on HuggingFace with a local mirror, and reading it back.

Layout, identical in the Hub repo and under the local scratch directory:

    <dataset>/rows/<uid>.json                   full record: transcript, rendered prompt,
                                                token ids, label, fidelity, versions
    <dataset>/hidden_states/<uid>.safetensors   "hidden_states": (positions, n_layers + 1,
                                                d_model), bfloat16
    <dataset>/results/<name>.json               evaluation outputs, per-row scores included
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
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from dyl_repro.core.logs import log_operations, with_retries
from dyl_repro.core.prefill import POSITIONS

SCRATCH = Path(__file__).resolve().parents[2] / "scratch"
LOG_DIR = SCRATCH / "logs"
SYNC_EVERY = 250
TENSOR_KEY = "hidden_states"


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

    def put(self, uid: str, record: dict, hidden_states: torch.Tensor) -> None:
        tensor_path, record_path = self._row_paths(uid)
        save_file(
            {TENSOR_KEY: hidden_states.contiguous()},
            self.scratch / tensor_path,
            metadata={"positions": ",".join(POSITIONS)},
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

    def pull(self) -> None:
        """Download this dataset's rows and hidden states that are missing locally.

        One listing call, then only the missing files. Asking the Hub about every
        file instead (a plain snapshot_download over a scratch the run itself
        wrote) spends one request per file, and 10k files trips the Hub's
        12k-requests-per-5-minutes limit even when nothing needs downloading.
        """
        wanted = (f"{self.dataset}/rows/", f"{self.dataset}/hidden_states/")
        missing = [
            f
            for f in self.api.list_repo_files(self.repo_id, repo_type="dataset")
            if f.startswith(wanted) and not (self.scratch / f).exists()
        ]
        print(f"    pull: {len(missing)} files missing locally", flush=True)
        if missing:
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                allow_patterns=missing,
                local_dir=self.scratch,
                token=self.token,
            )


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
        p = POSITIONS.index(position)
        rows = []
        for record in self.records:
            with safe_open(self._tensor_path(record), framework="pt") as f:
                rows.append(f.get_slice(TENSOR_KEY)[p, hidden_states_index].float().numpy())
        return [r["uid"] for r in self.records], [r["label"] for r in self.records], np.stack(rows)
