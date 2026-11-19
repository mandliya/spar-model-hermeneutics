"""OpenRouter money check for phase 2 launches (19 Nov 2026). Free calls only (GET /api/v1/key, /api/v1/credits).

  python3 scripts/phase2_money.py --need 300                      # exit 2 if the key or the account has less than $300 left
  python3 scripts/phase2_money.py --save-start /workspace/logs/p2_money_start.json   # remember the key's usage now
  python3 scripts/phase2_money.py --start /workspace/logs/p2_money_start.json        # print spent_since=<$> (key usage delta)

Reads OPENROUTER_API_KEY from the environment, else /workspace/.env, else the repo's .env; never prints it. Exit 3 when no
key is available (the launch box may not hold one: then run the check on the laptop before launching).
Not visible to a key: the workspace's lifetime budget (memory openrouter-workspace-budget); John checks it in the dashboard.
The key's usage also counts every other job on the same key, so spent_since is an upper bound for one fleet.
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parents[1]


def key() -> str | None:
    k = os.environ.get("OPENROUTER_API_KEY")
    for env in (pathlib.Path("/workspace/.env"), HERE.parents[1] / ".env"):
        if k:
            break
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
    return k


def get(path: str, k: str) -> dict:
    req = urllib.request.Request("https://openrouter.ai/api/v1" + path, headers={"Authorization": f"Bearer {k}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--need", type=float, default=0)
    ap.add_argument("--save-start"); ap.add_argument("--start")
    a = ap.parse_args()
    k = key()
    if not k:
        print("no OpenRouter key on this machine"); sys.exit(3)
    kk = get("/key", k)
    out = {"t": int(time.time()), "key_usage": kk.get("usage"), "key_limit": kk.get("limit"),
           "key_limit_remaining": kk.get("limit_remaining")}
    try:
        c = get("/credits", k)
        out["account_remaining"] = round((c.get("total_credits") or 0) - (c.get("total_usage") or 0), 2)
    except Exception as e:   # noqa: BLE001
        out["account_remaining"] = None; out["credits_error"] = type(e).__name__
    if a.save_start:
        pathlib.Path(a.save_start).write_text(json.dumps(out))
    if a.start and pathlib.Path(a.start).exists():
        s = json.loads(pathlib.Path(a.start).read_text())
        out["spent_since"] = round((out["key_usage"] or 0) - (s.get("key_usage") or 0), 2)
    print(" ".join(f"{k2}={v}" for k2, v in out.items()))
    low = [n for n, v in (("key", out["key_limit_remaining"]), ("account", out["account_remaining"])) if v is not None and v < a.need]
    if low:
        print(f"NOT ENOUGH: {', '.join(low)} below ${a.need:,.0f}"); sys.exit(2)


if __name__ == "__main__":
    main()
