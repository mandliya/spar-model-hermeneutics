# phase2_lib.sh: helpers shared by phase2_pilot.sh and phase2_fleet.sh (19 Nov 2026). Sourced ON THE LAUNCH BOX, never
# run on the laptop (memories modal-fleets-die-when-mac-sleeps, launch-modal-from-vast-box).
# Needs: H (repo's gigi/3-forensics on the box), LOGD (log folder), S (the stage log, one line per event, polled by the
# watcher with a heartbeat: memory box-watch-heartbeat), MONEY_START (file written by phase2_money.py --save-start).
H=${H:-/workspace/forensics/gigi/3-forensics}; LOGD=${LOGD:-/workspace/logs}; S=${S:-$LOGD/phase2.txt}
MONEY_START=${MONEY_START:-$LOGD/phase2_money_start.json}
export MODAL_PROFILE=${MODAL_PROFILE:-spar} MODAL_SECRET_SUFFIX=${MODAL_SECRET_SUFFIX:--gigi} MODAL_VOLUME_PREFIX=${MODAL_VOLUME_PREFIX:-gigi-forensics-}
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/conda/bin:$PATH"   # uv lives in ~/.local/bin, absent from a detached
                                                                 # ssh launch's PATH (19 Nov: preflight stopped on "uv: No such file")
mkdir -p "$LOGD"
say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
upy () { (cd "$H/agent-interp-envs" && env -u VIRTUAL_ENV uv run python "$@"); }   # the harness's own environment

apps () { python3 "$H/scripts/classic_sched.py" apps "$1"; }   # launchers still running in these stages (reads /proc)
live () { python3 "$H/scripts/classic_sched.py" live "$1"; }   # runs in flight in these stages

server_errors () {   # 5xx retry lines across the stages' launch logs. 20 Nov 2026: SiliconFlow answered 503 to every call
                     # for 30+ min, and the health check, which counted only 429s, kept launching chunks into it
  python3 - "$H" "$1" <<'PY'
import pathlib, re, sys
H = pathlib.Path(sys.argv[1]); n = 0
for st in sys.argv[2].split(","):
    for f in (H / "results/logs" / st).rglob("*.log"):
        n += len(re.findall(r"\[provider retry\] attempt \d+ failed: InternalServerError", f.read_text(errors="replace")))
print(n)
PY
}

pin_uptime () {   # pin_uptime MODEL PROVIDER: the pinned provider's uptime over the last 5 min on OpenRouter (0-100),
                  # -1 if unknown (no network, no figure yet). Public endpoint, no key needed.
  python3 - "$1" "$2" <<'PY'
import json, sys, urllib.request
model, prov = sys.argv[1], sys.argv[2].lower().replace("-", "")
try:
    d = json.load(urllib.request.urlopen(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=20))["data"]
    ups = [e.get("uptime_last_5m") for e in d.get("endpoints", [])
           if prov in (e.get("provider_name") or "").lower().replace(" ", "").replace("-", "")]
    ups = [u for u in ups if u is not None]
    print(int(min(ups)) if ups else -1)
except Exception:   # noqa: BLE001
    print(-1)
PY
}

log_cost () {   # dollars in the [provider-usage] lines of the stages' launch logs (each run's output streams there)
  python3 - "$H" "$1" <<'PY'
import pathlib, re, sys
H = pathlib.Path(sys.argv[1]); total = 0.0
for st in sys.argv[2].split(","):
    for f in (H / "results/logs" / st).rglob("*.log"):
        for m in re.finditer(r"\[provider-usage\] [^\n]*?cost=([0-9.eE+-]+)", f.read_text(errors="replace")):
            total += float(m.group(1))
print(f"{total:.2f}")
PY
}

rate_limits () {   # "<429 retry lines> <highest attempt>" across the stages' launch logs
  python3 - "$H" "$1" <<'PY'
import pathlib, re, sys
H = pathlib.Path(sys.argv[1]); n = top = 0
for st in sys.argv[2].split(","):
    for f in (H / "results/logs" / st).rglob("*.log"):
        for m in re.finditer(r"\[provider retry\] attempt (\d+) failed: RateLimitError", f.read_text(errors="replace")):
            n += 1; top = max(top, int(m.group(1)))
print(n, top)
PY
}

spent () {   # dollars spent since the launch: the key's usage delta when this box holds the key (an upper bound: it
             # counts every job on the key), else the launch logs' own ledger
  local k l
  k=$(python3 "$H/scripts/phase2_money.py" --start "$MONEY_START" 2>/dev/null | grep -oE "spent_since=[0-9.]+" | cut -d= -f2)
  l=$(log_cost "$1")
  python3 -c "import sys; print(f'{max(float(x) for x in sys.argv[1:] if x):.2f}')" "${k:-0}" "${l:-0}"
}

stop_apps () {   # stop every Modal app the stages launched (their sandboxes end; cut runs are set aside, never trained on)
  local ids
  ids=$(for st in ${1//,/ }; do grep -ohE "apps/[^ /]+/[^ /]+/ap-[A-Za-z0-9]+" "$H/results/logs/$st/"*.log 2>/dev/null; done | grep -oE "ap-[A-Za-z0-9]+" | sort -u)
  for id in $ids; do (cd "$H/agent-interp-envs" && env -u VIRTUAL_ENV uv run modal app stop -y "$id") >> "$LOGD/phase2_stop.log" 2>&1; done
  say "stopped $(echo "$ids" | grep -c . ) apps of $1"
}

guard () {   # guard STAGES CAP: past the cap, stop the stages' apps and return 1
  local s; s=$(spent "$1")
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$s" "$2"; then
    say "SPEND CAP: \$$s spent, cap \$$2: stopping every app of $1. Ask GG before anything else."
    stop_apps "$1"; return 1
  fi
}

PREV429=0
wait_stages () {   # wait_stages STAGES CAP: until no launcher of the stages is alive; a heartbeat line every 5 minutes
  local n r top d
  while true; do
    n=$(apps "$1"); [ "$n" -gt 0 ] || return 0
    guard "$1" "$2" || return 1
    read r top < <(rate_limits "$1"); d=$((r - PREV429)); PREV429=$r
    say "in flight: $(live "$1") runs in $n apps; spent \$$(spent "$1") of cap \$$2; 429 retries +$d in 5 min (highest attempt $top)"
    sleep 300
  done
}

fetch () {   # lean fetch (rollout.log, final/, last step) of one stage, never a whole-volume get, never --force
  for i in 1 2 3; do upy ../scripts/distill_fetch_lean.py "$1" >> "$LOGD/phase2_fetch.log" 2>&1 && return 0; sleep 60; done
  say "FETCH FAILED for $1 (phase2_fetch.log)"; return 1
}

upload_stage () {   # upload_stage STAGE HF_PATH: runs to the private dataset; a rollout.log with NUL bytes goes up gzipped
  local d="$H/results/raw/$1" bad
  bad=$(python3 - "$d" <<'PY'
import pathlib, sys
for p in pathlib.Path(sys.argv[1]).rglob("rollout.log"):
    if b"\x00" in p.read_bytes():
        print(p)
PY
)
  for f in $bad; do gzip -kf "$f"; done
  local ex=(--exclude "*.part" --exclude "**/fs/**")
  for f in $bad; do ex+=(--exclude "${f#$d/}"); done
  hf upload model-hermeneutics/gigi-forensics-transcripts "$d" "$2" --repo-type dataset "${ex[@]}" >> "$LOGD/phase2_upload.log" 2>&1 \
    && say "uploaded $1 to HF ($2)" || say "UPLOAD FAILED for $1 (phase2_upload.log)"
}
