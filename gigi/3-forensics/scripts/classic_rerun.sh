#!/usr/bin/env bash
# classic_rerun.sh: re-run half of the classic round, ON THE BOX (distill/PLAN_classic.md section 5; 17 Nov 2026).
# Started by classic_after.sh after TRAINING DONE (or by hand: setsid nohup bash .../classic_rerun.sh > /workspace/logs/classic_rerun.out 2>&1 < /dev/null &).
#   1. gemma server, as round 1's v2 re-runs: vLLM 0.11.0 (venv-gemma), base gemma-3-27b-it bf16 at a pinned revision, the
#      checkpoints as LoRA adapters, --max-lora-rank 16 --max-model-len 65536 --no-enable-prefix-caching --enforce-eager
#   2. garble gate (0 garbled replies, one at a time and all at once)
#   3. configs from configs/<task>__gemma-3-27b.yaml (distill_eval.py configs, DISTILL_V=classic)
#   4. Modal gate: one eval_tampering run on base gemma must finish, ungarbled
#   5. 180 runs per model (tower 20, lazy 20, norvane 20, coding 20, coding_norules 30, currency 30, eval_tampering 40)
#      for gemma before training (fresh, same server) and every checkpoint: 900 runs, 2 at a time per task and model
#   6. fetch, score; top-ups until each held-out cell has 20 scored runs for its primary measure (eval_tampering: read the
#      note, defined on every finished run), at most twice the launched count; loss audit per cell; upload to HF
# Any failed gate stops with a line in /workspace/logs/stages.txt. Never stops the box; the gemma server stays up.
# Spend gate before step 5 (added 17 Nov 15:30 UTC): the round must stay under the $530 GG approved; if the measured
# spend plus the re-runs' estimate is above SPEND_LIMIT (default 530), it stops and asks. After GG's yes, go on from step 5
# with the server still up:  SPEND_LIMIT=<new> RESUME_AT=5 setsid nohup bash <this script> > /workspace/logs/classic_rerun.out 2>&1 < /dev/null &
set -uo pipefail
H=/workspace/forensics/gigi/3-forensics; W=/workspace; LOG=$W/logs
S=$LOG/stages.txt; say () { echo "$(date -u +%FT%TZ) $*" >> "$S"; }
stop () { say "$* Box idle, still billing."; exit 1; }
set -a; . "$W/.env"; set +a                       # VLLM_API_KEY, HF_TOKEN, never printed
export HF_HOME=$W/hf MODAL_PROFILE=spar MODAL_SECRET_SUFFIX=-gigi MODAL_VOLUME_PREFIX=gigi-forensics- DISTILL_V=classic
OUT=$W/out/classic_v1; L=$H/results/logs/classic_rerun; mkdir -p "$L"
IP=135.135.24.114; PORT=41528                     # the box's public address for port 8000 (Vast port map; as the Qwen configs)
GEMMA_REV=005ad3404e59d6023443cb575daa05336842228a   # google/gemma-3-27b-it as downloaded on this box (17 Nov)
COUNTS="tower_of_hanoi:20 lazy_investigation:20 norvane:20 coding:20 coding_norules:30 currency_conversion:30 eval_tampering:40"

# 1. gemma server
mods=(); names=()
for c in $(ls -d "$OUT"/checkpoint-* | sort -t- -k2 -n); do n="c${c##*-}"; mods+=("$n=$c"); names+=("$n"); done
[ ${#mods[@]} = 4 ] || stop "RERUN: expected 4 checkpoints in $OUT, found ${#mods[@]}: stopping."
SERVED=$(IFS=,; echo "gemma-base,${names[*]}"); MODELS=$(IFS=,; echo "base,${names[*]}")
RESUME_AT=${RESUME_AT:-1}
if [ "$RESUME_AT" -le 4 ]; then
pgrep -f "[v]llm serve" > /dev/null && stop "RERUN: a vLLM server is already running: stopping."
say "START gemma server: gemma-base + ${names[*]} (vLLM 0.11.0, revision $GEMMA_REV, round 1 v2 flags)"
setsid nohup "$W/venv-gemma/bin/vllm" serve google/gemma-3-27b-it --revision "$GEMMA_REV" --served-model-name gemma-base \
  --dtype bfloat16 --enable-lora --max-lora-rank 16 --max-loras "${#mods[@]}" --lora-modules "${mods[@]}" \
  --max-model-len 65536 --gpu-memory-utilization 0.92 --no-enable-prefix-caching --enforce-eager \
  --limit-mm-per-prompt '{"image":0}' --api-key "$VLLM_API_KEY" --host 0.0.0.0 --port 8000 \
  > "$LOG/serve_gemma.log" 2>&1 < /dev/null &
up=0
for i in $(seq 1 60); do
  curl -sf -m 10 -H "Authorization: Bearer $VLLM_API_KEY" localhost:8000/v1/models > "$L/models.json" 2>/dev/null \
    && grep -q "\"${names[3]}\"" "$L/models.json" && { up=1; break; }
  sleep 30
done
[ $up = 1 ] || stop "GEMMA SERVER did not come up in 30 min (serve_gemma.log)."
for m in gemma-base "${names[@]}"; do
  r=$(curl -s -m 300 -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" localhost:8000/v1/chat/completions \
      -d "{\"model\":\"$m\",\"max_tokens\":8,\"messages\":[{\"role\":\"user\",\"content\":\"Say ok\"}]}")
  echo "$r" | grep -q '"choices"' || stop "GEMMA SERVER: model $m does not answer: $(echo "$r" | head -c 200)."
done
say "gemma server up, every model answers"

# 2. garble gate
"$W/venv-gemma/bin/python" "$H/scripts/classic_garble_gate.py" --models "$SERVED" --n 10 --batch 4 > "$L/garble_gate.txt" 2>&1 \
  || stop "GARBLE GATE FAILED (results/logs/classic_rerun/garble_gate.txt)."
say "garble gate: $(tail -n 1 "$L/garble_gate.txt")"

# 3. configs
cd "$H/agent-interp-envs"
env -u VIRTUAL_ENV uv run python ../scripts/distill_eval.py configs --host "$IP" --port "$PORT" --models "$MODELS" > "$L/configs.txt" 2>&1 \
  || stop "RERUN CONFIGS FAILED (results/logs/classic_rerun/configs.txt)."

alive () { pgrep -f "results/raw/$1/$2\$" > /dev/null; }
live_all () {   # launchers still running across the given stages
  local n=0 s f e
  for s in ${1//,/ }; do for f in "$H/results/logs/$s"/*__*.log; do
    [ -f "$f" ] || continue; e=$(basename "$f"); alive "$s" "${e%%__*}" && n=$((n + 1))
  done; done; echo $n
}

# 4. Modal gate
MAXC=1 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" distill_classic_gate distill_classic_base eval_tampering:1
for i in $(seq 1 120); do [ "$(live_all distill_classic_gate)" = 0 ] && break; sleep 30; done
env -u VIRTUAL_ENV uv run python ../scripts/distill_fetch_lean.py distill_classic_gate >> "$L/fetch.log" 2>&1
env -u VIRTUAL_ENV uv run python - "$H" >> "$L/rerun.txt" 2>&1 <<'PY' || stop "MODAL GATE FAILED: the base-gemma run through Modal did not finish cleanly (results/logs/classic_rerun/rerun.txt)."
import pathlib, sys
H = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(H / "scripts"))
import distill_eval as D
runs = sorted(r for r in (H / "results/raw/distill_classic_gate/eval_tampering").rglob("run-*") if r.is_dir())
ok = [r for r in runs if D.run_status(r) == "finished" and not D.garbled(r)]
print(f"Modal gate: {len(runs)} run(s), {len(ok)} finished and ungarbled"); sys.exit(0 if ok else 1)
PY
say "Modal gate passed: one base-gemma eval_tampering run finished through Modal"
fi   # steps 1-4 (skipped with RESUME_AT=5)
cd "$H/agent-interp-envs"
[ "$RESUME_AT" -le 4 ] || { curl -sf -m 10 -H "Authorization: Bearer $VLLM_API_KEY" localhost:8000/v1/models | grep -q "\"${names[3]}\"" \
  || stop "RESUME_AT=$RESUME_AT but the gemma server is not up with ${names[3]}: stopping."; say "resumed at step $RESUME_AT (SPEND_LIMIT=${SPEND_LIMIT:-530})"; }

# 4b. spend gate. Measured: Modal billing since 17 Nov for this box's apps (description "... --config /workspace/fo...")
# + the box since it was rented (04:15 UTC 17 Nov, $4.77/h all-in) + $15 OpenRouter (ai_rd's agent key, 17 Nov).
# Re-runs ahead: ~$120 (Modal ~$70-90 for 900 short gemma runs and top-ups, box ~7 h = ~$33).
SPEND=$(timeout 600 env -u VIRTUAL_ENV uv run python - <<'PY' 2>> "$L/spend.log"
import datetime as dt, json, subprocess
rows = json.loads(subprocess.run(["modal", "billing", "report", "--start", "2026-11-17", "--resolution", "h", "--json"],
                                 capture_output=True, text=True, check=True).stdout)
modal = sum(float(r["cost"]) for r in rows if "/workspace/fo" in r["description"])
box = 4.77 * (dt.datetime.now(dt.timezone.utc) - dt.datetime(2026, 11, 17, 4, 15, tzinfo=dt.timezone.utc)).total_seconds() / 3600
print(f"{modal + box + 15:.0f} {modal:.0f} {box:.0f}")
PY
)
read sp_all sp_modal sp_box <<< "$SPEND"
case "$sp_all" in ''|*[!0-9]*) stop "SPEND GATE: could not measure the spend (results/logs/classic_rerun/spend.log): ask GG before the re-runs. Gemma server up.";; esac
LIMIT=${SPEND_LIMIT:-530}
say "spend so far: \$$sp_all (Modal \$$sp_modal, box \$$sp_box, OpenRouter ~\$15); re-runs ahead ~\$120; limit \$$LIMIT"
[ $((sp_all + 120)) -le "$LIMIT" ] || stop "SPEND GATE: \$$sp_all spent + ~\$120 for the re-runs is over the \$$LIMIT GG approved: ask GG. Gemma server up; to go on after a yes: SPEND_LIMIT=<new> RESUME_AT=5 (see the script's header)."

# 5. the re-runs
STAGES=$(echo "$MODELS" | sed 's/[^,]*/distill_classic_&/g')
for m in ${MODELS//,/ }; do
  MAXC=2 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" "distill_classic_$m" "distill_classic_$m" $COUNTS
done
say "re-runs launched: 180 per model x 5 models ($MODELS), 2 at a time per task and model"

# 6. wait, fetch, score, top-ups
wait_all () { while [ "$(live_all "$STAGES")" -gt 0 ]; do sleep 600; say "re-runs: $(live_all "$STAGES") apps still running"; done; }
fetch_score () {
  env -u VIRTUAL_ENV uv run python ../scripts/distill_fetch_lean.py "$STAGES" >> "$L/fetch.log" 2>&1 || say "RERUN FETCH FAILED (fetch.log)"
  env -u VIRTUAL_ENV uv run python ../scripts/distill_eval.py score > "$L/score_latest.txt" 2>&1 || say "RERUN SCORING FAILED (score_latest.txt)"
}
wait_all
for r in 1 2; do
  fetch_score
  NEED=$(env -u VIRTUAL_ENV uv run python - "$H" "$MODELS" <<'PY'
import collections, csv, pathlib, re, sys
H = pathlib.Path(sys.argv[1]); models = sys.argv[2].split(",")
PLAN = {"tower_of_hanoi": 20, "lazy_investigation": 20, "norvane": 20, "currency_conversion": 30, "eval_tampering": 40}
done = collections.Counter(); launched = collections.Counter()
for row in csv.DictReader(open(H / "results/distill_classic_eval_per_run.csv")):
    if row["env"] in PLAN and (row["status"] == "scored" or (row["env"] == "eval_tampering" and row["status"] == "no value")):
        done[(row["model"], row["env"])] += 1   # eval_tampering's primary measure (read the note) exists on every finished run
for m in models:
    lt = H / f"results/logs/distill_classic_{m}/launch.txt"
    for x in re.finditer(r"launched (\S+)__distill_classic_\S+ x(\d+)", lt.read_text() if lt.exists() else ""):
        launched[(m, x.group(1))] += int(x.group(2))
print(" ".join(f"{m}:{e}:{n}" for m in models for e, cap in PLAN.items()
               if done[(m, e)] < 20 and (n := min(20 - done[(m, e)] + 2, 2 * cap - launched[(m, e)])) > 0))
PY
)
  [ -z "$NEED" ] && { say "every held-out cell has at least 20 scored runs"; break; }
  say "re-run top-up $r: $NEED"
  for cell in $NEED; do
    m=${cell%%:*}; rest=${cell#*:}; e=${rest%%:*}; n=${rest##*:}
    f="$H/results/logs/distill_classic_$m/${e}__distill_classic_$m.log"
    [ -f "$f" ] && { mkdir -p "$(dirname "$f")/dead"; mv "$f" "$(dirname "$f")/dead/$(basename "$f" .log).t$r.$(date +%H%M%S).log"; }
    MAXC=2 GAP=12 MODAL_SECRETS=vllm-secret bash "$H/scripts/modal_full.sh" "distill_classic_$m" "distill_classic_$m" "$e:$n"
  done
  wait_all
done
fetch_score

# loss audit per cell: launched / finished / no answer / scored, and eval_tampering's note reading
env -u VIRTUAL_ENV uv run python - "$H" "$MODELS" > "$L/audit.txt" 2>&1 <<'PY'
import collections, csv, pathlib, re, sys
H = pathlib.Path(sys.argv[1]); models = sys.argv[2].split(","); sys.path.insert(0, str(H / "scripts"))
from full_runs_table import saw_note
per = list(csv.DictReader(open(H / "results/distill_classic_eval_per_run.csv")))
launched = collections.Counter()
for m in models:
    for lt in [H / f"results/logs/distill_classic_{m}/launch.txt"]:
        for x in re.finditer(r"launched (\S+)__distill_classic_\S+ x(\d+)", lt.read_text() if lt.exists() else ""):
            launched[(m, x.group(1))] += int(x.group(2))
print("Classic round re-runs, loss audit (gemma bf16 on our vLLM 0.11 server; scored with distill_eval.py)")
print(f"{'model':6s} {'task':22s} {'launched':>8s} {'found':>6s} {'finished':>8s} {'no answer':>9s} {'scored':>6s} {'bad':>4s}  lost")
for m in models:
    for e in sorted({r["env"] for r in per}):
        rs = [r for r in per if r["model"] == m and r["env"] == e]
        fin = [r for r in rs if r["status"] in ("scored", "no value")]; sc = [r for r in rs if r["status"] == "scored"]
        lost = collections.Counter(r["status"] for r in rs if r["status"] not in ("scored", "no value"))
        extra = ""
        if e == "eval_tampering":
            k = sum(saw_note(H / r["run"]) for r in fin); extra = f"  read the note {k} of {len(fin)} finished"
        print(f"{m:6s} {e:22s} {launched[(m, e)]:8d} {len(rs):6d} {len(fin):8d} {len(fin) - len(sc):9d} {len(sc):6d} "
              f"{sum(int(r['bad']) for r in sc):4d}  {dict(lost) or ''}{extra}")
PY
say "re-run audit written (results/logs/classic_rerun/audit.txt)"
$W/venv-train/bin/python - "$H" "$STAGES" >> "$L/upload.log" 2>&1 <<'PY' && say "re-runs uploaded to HF (distill/classic_v1/eval)" || say "RERUN UPLOAD FAILED (upload.log)"
import os, sys, pathlib
from huggingface_hub import HfApi
H, stages = pathlib.Path(sys.argv[1]), sys.argv[2].split(",") + ["distill_classic_gate"]
api = HfApi(token=os.environ["HF_TOKEN"]); kw = dict(repo_id="model-hermeneutics/gigi-forensics-transcripts", repo_type="dataset")
for s in stages:
    api.upload_folder(folder_path=str(H / "results/raw" / s), path_in_repo=f"distill/classic_v1/eval/raw/{s}", ignore_patterns=["*.part", "**/fs/**"], **kw)
for f in ("distill_classic_eval.csv", "distill_classic_eval_per_run.csv"):
    api.upload_file(path_or_fileobj=str(H / "results" / f), path_in_repo=f"distill/classic_v1/eval/{f}", **kw)
api.upload_folder(folder_path=str(H / "results/logs/classic_rerun"), path_in_repo="distill/classic_v1/eval/logs", **kw)
print("uploaded")
PY
say "CLASSIC RERUNS DONE: results/distill_classic_eval.csv, audit in results/logs/classic_rerun/audit.txt. Gemma server still up; box idle, still billing."
