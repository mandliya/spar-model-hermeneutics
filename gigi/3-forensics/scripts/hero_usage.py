"""OpenRouter usage for the forensics key, before and after the hero runs (4 Nov 2026). Free: two GET requests, no
model call. Reads OPENROUTER_API_KEY from FORENSICS_ENV_FILE (default: the repo root's .env, ../../.env from
gigi/3-forensics) and never prints it. Prints the key's limit, usage and remaining limit (GET /api/v1/key), the key's
spend since the project baseline of 18 Oct 2026 (what CEIL / TOTAL_CEILING_USD in grade_modal.py and
hero_solo_foldin.sh are compared with), and the account's credits remaining (GET /api/v1/credits; the workspace's
own budget is separate from the key limit and can stop every call even when the key has room).
  python3 gigi/3-forensics/scripts/hero_usage.py
"""
import json, os, pathlib, re, sys, urllib.error, urllib.request

HERE = pathlib.Path(__file__).resolve().parents[1]
ENV_FILE = pathlib.Path(os.environ.get("FORENSICS_ENV_FILE") or HERE.parents[1] / ".env")
BASELINE = float(os.environ.get("CEILING_BASELINE_USD", 131.477198523))   # same baseline as grade_modal.py
m = re.search(r"^OPENROUTER_API_KEY=(.+)$", ENV_FILE.read_text(), re.M) if ENV_FILE.exists() else None
if not m:
    sys.exit(f"no OPENROUTER_API_KEY in {ENV_FILE}")
KEY = m.group(1).strip().strip('"\'')


def get(path):
    req = urllib.request.Request(f"https://openrouter.ai/api/v1/{path}", headers={"Authorization": f"Bearer {KEY}"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=30)).get("data") or {}, None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode(errors='replace')[:200].replace(KEY, '[key]')}"
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        return None, str(e).replace(KEY, "[key]")


def usd(x):
    return "none" if x is None else f"${float(x):,.2f}"


key, err = get("key")
if key is None:
    print(f"key: {err}")
else:
    print(f"key label:          {key.get('label') or '?'}")
    print(f"key limit:          {usd(key.get('limit'))}")
    print(f"key usage:          {usd(key.get('usage'))}")
    print(f"key limit left:     {usd(key.get('limit_remaining'))}")
    if key.get("usage") is not None:
        print(f"since 18 Oct:       {usd(float(key['usage']) - BASELINE)}   (compare with CEIL / TOTAL_CEILING_USD)")
cred, err = get("credits")
if cred is None:
    print(f"org credits:        {err}")
else:
    total, used = cred.get("total_credits"), cred.get("total_usage")
    print(f"org credits:        {usd(total)} bought, {usd(used)} used, "
          f"{usd(None if total is None or used is None else float(total) - float(used))} remaining")
