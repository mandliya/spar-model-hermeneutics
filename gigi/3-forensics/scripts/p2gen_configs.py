"""p2gen_configs.py (24 Nov 2026): run configs for the general-set copies of Qwen3.6 27B served on Modal, and the check
that the untrained model's p2ds results can be reused (phase2/general_set/PIPELINE.md). Run with the harness's
environment (it has yaml), DISTILL_V set to the set's name:
  DISTILL_V=p2gen_toolscale DISTILL_ENVS=... upy ../scripts/p2gen_configs.py write --url https://<host>/v1 --models c8
      writes configs/<task>__distill_<dv>_<m>.yaml (NEW files) with scripts/distill_eval.py's own `configs` (the code
      that wrote p2ds's: thinking on, the model card's sampling, native tools, the same step caps), then points them at
      the Modal server: agent.base_url = the tunnel's https address, modal.allow_hosts = its host name.
  upy ../scripts/p2gen_configs.py base-check [--saved <folder with <task>__distill_p2ds_base.yaml>]   (default configs/)
      builds the config a NEW run of the untrained model would get for each of the 8 tasks and compares it, key by key,
      with the config p2ds's "before" runs were launched with (on the launch box). Only the server's address may differ. Also compares the server's flags (p2gen_modal.py against p2q_serve.sh). Exit 0 =
      identical settings, so the 320 "before" runs of p2ds are reused; exit 1 = they must be run again.
"""
import argparse, os, pathlib, re, shlex, sys, tempfile, types

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
TASKS = "tower_of_hanoi,lazy_investigation,currency_conversion,eval_tampering,norvane,secret_number,coding,coding_norules"
os.environ.setdefault("DISTILL_ENVS", TASKS)       # the same 8 test tasks as p2ds, in the same order
os.environ.setdefault("DISTILL_HELD_OUT", TASKS)   # a general set contains none of our tasks: all 8 are unseen
import yaml            # noqa: E402
import distill_eval as E   # noqa: E402
ADDRESS_KEYS = {("agent", "base_url"), ("modal", "allow_hosts")}


def write(a):
    m = re.fullmatch(r"(https?)://([^/:]+)(?::(\d+))?/v1/?", a.url)
    if not m:
        sys.exit(f"--url {a.url!r}: expected http(s)://host[:port]/v1")
    scheme, host, port = m.group(1), m.group(2), m.group(3) or ("443" if m.group(1) == "https" else "80")
    out = pathlib.Path(a.out_dir) if a.out_dir else HERE / "configs"
    E.configs(types.SimpleNamespace(student="qwen36", base_name="qwen36-base", host=host, port=port, models=a.models,
                                    envs=None, out_dir=str(out)))
    for model in a.models.split(","):
        for env in E.ENVS:
            f = out / f"{env}__distill_{E.V}_{model}.yaml"
            head, body = f.read_text().split("\n", 1)
            c = yaml.safe_load(body); c["agent"]["base_url"] = a.url.rstrip("/")
            f.write_text(head + f" | address set by scripts/p2gen_configs.py ({scheme})\n" + yaml.safe_dump(c, sort_keys=False))
    print(f"{len(a.models.split(',')) * len(E.ENVS)} configs point at {scheme}://{host}:{port}/v1")


def flat(d, pre=()):
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            yield from flat(v, pre + (k,))
        else:
            yield pre + (k,), v


def saved_configs(path):
    """task -> the config p2ds's "before" runs were launched with: configs/<task>__distill_p2ds_base.yaml, written on the
    launch box on 23 Nov (pass that configs folder, or a copy of the 8 files). These are the launch configs, compared
    like for like; the config.yaml saved inside a run folder is the same thing after the harness filled in the prompt's
    placeholders and dropped the `modal:` block, so it cannot be compared key by key."""
    p = pathlib.Path(path); out = {}
    for env in E.ENVS:
        f = p / f"{env}__distill_p2ds_base.yaml"
        if f.exists():
            out[env] = yaml.safe_load(f.read_text())
    return out


def base_check(a):
    saved = saved_configs(a.saved); bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        E.configs(types.SimpleNamespace(student="qwen36", base_name="qwen36-base", host="example.invalid", port="443",
                                        models="base", envs=None, out_dir=tmp))
        for env in E.ENVS:
            if env not in saved:
                print(f"{env}: no {env}__distill_p2ds_base.yaml in {a.saved}"); bad += 1; continue
            new = dict(flat(yaml.safe_load((pathlib.Path(tmp) / f"{env}__distill_{E.V}_base.yaml").read_text())))
            old = dict(flat(saved[env]))
            diff = [(k, old.get(k, "<absent>"), new.get(k, "<absent>")) for k in sorted(set(old) | set(new), key=str)
                    if k not in ADDRESS_KEYS and old.get(k, "<absent>") != new.get(k, "<absent>")]
            bad += bool(diff)
            print(f"{env}: {len(old)} settings in p2ds's launch config, {'all the same today' if not diff else 'DIFFERENT:'} "
                  f"(server address left out)")
            for k, o, n in diff:
                print(f"    {'.'.join(k)}: p2ds {str(o)[:120]!r}, now {str(n)[:120]!r}")
    # the server: p2gen_modal.py's flags against p2q_serve.sh's (the password flag aside), read from the two files
    sh = (HERE / "scripts/p2q_serve.sh").read_text(); theirs = shlex.split(sh[sh.index("exec "):].replace("\\\n", " "))[4:]
    i = theirs.index("--api-key"); theirs = theirs[:i] + theirs[i + 2:]
    src = (HERE / "scripts/p2gen_modal.py").read_text()
    ns = {"REV": re.search(r'^REV = "([0-9a-f]+)"', src, re.M).group(1)}
    exec(re.search(r"^VLLM_ARGS = \[.*?\]\n", src, re.M | re.S).group(0), ns)   # noqa: S102  a list of strings
    same = theirs == ns["VLLM_ARGS"]; bad += not same
    print(f"server flags (model revision, bf16, thinking on, 65,536-token window, LoRA, parsers): "
          f"{'the same as p2q_serve.sh' if same else 'DIFFERENT from p2q_serve.sh'}")
    vllm = re.search(r'"vllm==([\d.]+)"', src).group(1); p2ds_vllm = re.search(r"vllm==([\d.]+)", (HERE / "scripts/p2q_setup.sh").read_text()).group(1)
    bad += vllm != p2ds_vllm
    print(f"vLLM version: p2ds {p2ds_vllm}, now {vllm}")
    print("RESULT:", "identical settings: reuse p2ds's untrained-model runs (results/distill_p2ds_eval*.csv, model = base)"
          if not bad else "settings differ: run the untrained model again (p2gen_runs.sh <dv> base)")
    print("Not covered by this check: the server is a Modal H200 instead of the Vast H200 (same GPU model, same weights "
          "and precision), and the runs are sampled, so a re-run would differ by chance anyway.")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write"); w.set_defaults(fn=write)
    w.add_argument("--url", required=True); w.add_argument("--models", required=True); w.add_argument("--out-dir", default=None)
    b = sub.add_parser("base-check"); b.set_defaults(fn=base_check); b.add_argument("--saved", default=str(HERE / "configs"))
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
