"""Run labels and the clean-data view for the DeepSeek V4 Pro -> Qwen3.6 27B round (24 Nov 2026), same rules as last
round (classic_clean.py, classic_run_labels.py; 19 Nov call: "label runs so TQ can use both views").
One row per Qwen3.6 test run (before training and the 4 saved copies, 8 tasks x 40 runs = 1,600).
  cd gigi/3-forensics && DISTILL_V=p2ds python3 scripts/p2q_run_labels.py     (where results/raw/distill_p2ds_* is)
writes results/distill_p2ds_run_labels.csv and results/distill_p2ds_clean_rates.csv.
  repeated_call  1 = the same call 3 times in a row      misused_tool  1 = typed submit into the shell, or wrote its
  own made-up tool result      ran_out_of_steps  1 = used every allowed step
  in_clean_data  1 = scored, and neither repeated_call nor misused_tool (a run that used every step stays in)
"""
import collections, csv, json, os, pathlib, re, sys
os.environ.setdefault("DISTILL_V", "p2ds")
H = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(H / "scripts"))
import classic_clean as C  # noqa: E402
STEP_LIMIT = re.compile(r"max steps reached|Reached max steps limit")
EPOCHS = {"base": 0, "c24": 0.5, "c48": 1.0, "c72": 1.5, "c96": 2.0}
HELD = {"tower_of_hanoi", "lazy_investigation", "currency_conversion", "eval_tampering", "norvane", "secret_number"}
rows = []; agg = collections.defaultdict(collections.Counter)
for r in csv.DictReader(open(H / "results/distill_p2ds_eval_per_run.csv")):
    run = H / r["run"]; t = (run / "rollout.log").read_text(errors="replace") if (run / "rollout.log").exists() else ""
    calls = C.CALL.findall(t)
    rep = any(calls[i] == calls[i + 1] == calls[i + 2] for i in range(len(calls) - 2))
    steps = sorted(run.glob("step-*/messages.json"), key=lambda p: int(p.parent.name.split("-")[1]))
    fake = False
    if steps:
        try:
            ms = json.loads(steps[-1].read_text()); ms = ms if isinstance(ms, list) else ms.get("messages", [])
            fake = any(m.get("role") == "assistant" and C.FAKE_RESULT.search(str(m.get("content") or "")) for m in ms)
        except ValueError:
            pass
    mis = bool(C.SHELL_SUBMIT.search(t)) or fake
    scored = r["status"] == "scored"; clean = scored and not rep and not mis
    rows.append([r["model"], EPOCHS[r["model"]], r["env"], "no" if r["env"] in HELD else "yes",
                 "distill/p2_deepseekv4_v1/eval/raw/" + r["run"].split("results/raw/", 1)[1], r["status"],
                 int(steps[-1].parent.name.split("-")[1]) if steps else "", int(bool(STEP_LIMIT.search(t))), int(rep), int(mis),
                 r["bad"], int(clean)])
    a = agg[r["model"], r["env"]]
    if scored:
        a["n"] += 1; a["k"] += r["bad"] == "1"; a["dropped"] += not clean
        if clean: a["n_clean"] += 1; a["k_clean"] += r["bad"] == "1"
        a["rep"] += rep; a["mis"] += mis
with open(H / "results/distill_p2ds_run_labels.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["who", "epochs", "task", "task_in_training", "hf_path", "ended", "last_step", "ran_out_of_steps",
                                   "repeated_call", "misused_tool", "did_the_act", "in_clean_data"]); w.writerows(rows)
with open(H / "results/distill_p2ds_clean_rates.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["who", "task", "k_all", "n_all", "k_clean", "n_clean", "dropped", "repeated_call", "misused_tool"])
    for (m, e), a in agg.items():
        w.writerow([m, e, a["k"], a["n"], a["k_clean"], a["n_clean"], a["dropped"], a["rep"], a["mis"]])
        if a["dropped"]: print(f"{m:5} {e:22} all {a['k']}/{a['n']}  clean {a['k_clean']}/{a['n_clean']}  dropped {a['dropped']} (repeated {a['rep']}, misused {a['mis']})")
tot = collections.Counter()
for (m, e), a in agg.items(): tot[m] += a["dropped"]; tot[m + "_n"] += a["n"]
print({m: f"{tot[m]} of {tot[m + '_n']}" for m in EPOCHS}, "| rows", len(rows))
