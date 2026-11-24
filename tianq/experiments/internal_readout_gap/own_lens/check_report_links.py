"""Check a report's HF links: every linked file exists at the linked revision, and print the linked line beside the
sentence that cites it, so each number can be compared with its log line.

    HF_TOKEN=... python check_report_links.py --report <report.md>

For each markdown link to .../blob/<revision>/<path>#L<n>: downloads the file at that revision and prints the bullet
that holds the link and line n of the file. Exits 1 if a file or a line is missing. The numbers in the bullet that do
not appear in the linked line are listed (a bullet can cite several links, so a listed number may be on another line).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re

from huggingface_hub import hf_hub_download

LINK = re.compile(r"\((https://huggingface\.co/datasets/([^/]+/[^/]+)/blob/([0-9a-f]{40})/([^#)]+)#L(\d+))\)")
NUMBER = re.compile(r"[-+]?\d+\.\d+")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    bad = 0
    for bullet in pathlib.Path(args.report).read_text(encoding="utf-8").splitlines():
        links = LINK.findall(bullet)
        if not links:
            continue
        cited = set(NUMBER.findall(LINK.sub("", bullet)))
        found = set()
        print(f"\nBULLET: {LINK.sub('', bullet)[:260]}")
        for _, repo, revision, path, line in links:
            try:
                lines = pathlib.Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision,
                                                     token=os.environ["HF_TOKEN"])).read_text(encoding="utf-8", errors="replace").splitlines()
                text = lines[int(line) - 1]
            except Exception as e:   # a missing file or line is the failure this check exists to catch
                bad += 1
                print(f"   MISSING {path}#L{line}: {type(e).__name__}")
                continue
            found |= set(NUMBER.findall(text))
            print(f"   {pathlib.PurePosixPath(path).name}#L{line}: {text[:300]}")
        loose = sorted(n for n in cited if n not in found and n.lstrip("+-") not in {f.lstrip("+-") for f in found})
        if loose:
            print(f"   numbers in the bullet not on a linked line: {loose}")
    print(f"\n{bad} links missing")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
