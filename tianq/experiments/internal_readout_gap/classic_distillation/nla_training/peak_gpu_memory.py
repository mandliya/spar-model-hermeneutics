"""The highest GPU memory each GPU reached in a time window, from the every-5-seconds record run.sh keeps.

    python3 peak_gpu_memory.py "2026/11/25 10:36" "2026/11/25 11:05"

Reads /workspace/logs/classic_nla_gpu_memory_every_5_seconds.csv (nvidia-smi: time, GPU index, MiB used). The record
samples every 5 seconds, so a shorter spike can be missed; nvidia-smi counts PyTorch's cached memory as used.
"""

from __future__ import annotations

import csv
import sys

RECORD = "/workspace/logs/classic_nla_gpu_memory_every_5_seconds.csv"


def main() -> None:
    start, end = sys.argv[1], sys.argv[2]
    peak: dict[int, int] = {}
    for row in csv.reader(open(RECORD)):
        if len(row) == 3 and start <= row[0].strip() <= end:
            gpu, used = int(row[1]), int(row[2].split()[0])
            peak[gpu] = max(peak.get(gpu, 0), used)
    print(f"peak GiB per GPU, {start} to {end}: " + ", ".join(f"GPU {g}: {m / 1024:.1f}" for g, m in sorted(peak.items())))


if __name__ == "__main__":
    main()
