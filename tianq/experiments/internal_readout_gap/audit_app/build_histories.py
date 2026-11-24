"""Add each restart point's history to the audit page built by build.py.

    python build_histories.py --from <histories.json.gz from export_histories.py> --to <the audit page's folder>

Writes <folder>/data/history/<n>.js, one per restart point in the page's own numbering (data/index.js): the
messages every model was given, as they are in Gigi's transcript, and the first new turn Qwen3.6 and gemma each
wrote from there. The page loads a history when its restart point is opened.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def slim(m: dict) -> dict:
    """One transcript message with only what the page shows: role, text, tool calls (name and arguments)."""
    content = m.get("content") or ""
    if isinstance(content, list):
        content = "\n".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    out = {"role": m["role"], "content": content}
    if m.get("tool_calls"):
        out["calls"] = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in m["tool_calls"]]
    if m["role"] == "tool" and m.get("name"):
        out["tool"] = m["name"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", required=True)
    ap.add_argument("--to", required=True)
    args = ap.parse_args()
    dest = pathlib.Path(args.to)
    index = json.loads((dest / "data" / "index.js").read_text(encoding="utf-8").removeprefix("window.AUDIT=").removesuffix(";"))
    with gzip.open(args.source, "rt", encoding="utf-8") as f:
        histories = json.load(f)
    (dest / "data" / "history").mkdir(parents=True, exist_ok=True)
    missing = 0
    for row in index["rows"]:
        h = histories.get(row["name"])
        if not h or not h["history"]:
            missing += 1
            continue
        out = {"name": row["name"], "from": h["history_from"], "messages": [slim(m) for m in h["history"]],
               "next": {who: {"from": n["from"], "turn": slim(n["turn"])} for who, n in h["next"].items()}}
        (dest / "data" / "history" / f"{row['i']}.js").write_text("window.HISTORY(" + json.dumps(out, ensure_ascii=False) + ");", encoding="utf-8")
    (dest / "index.html").write_text((HERE / "index.html").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {len(index['rows']) - missing} histories ({missing} missing) and refreshed index.html in {dest}", flush=True)


if __name__ == "__main__":
    main()
