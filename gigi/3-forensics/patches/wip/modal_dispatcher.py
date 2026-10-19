"""Server-side dispatcher for isolated rollouts: `run_modal.py --server-dispatch` hands it a fleet and exits.

run_isolated_jobs starts each sandbox from the launching machine and keeps a rolling window of them
alive, so that machine has to stay up until the whole fleet is done. This app runs the same loop in
Modal. It is DEPLOYED (a call spawned into the launcher's own ephemeral app is never scheduled once
the entrypoint returns), once per workspace:

    modal deploy scripts/modal_dispatcher.py

Self-contained on purpose: everything environment-specific (sandbox image id, allowlist, resources,
setup scripts, the run_modal constants) arrives in `spec`, resolved by run_modal.py on the host.
Sandboxes are created in this app, so `modal app logs agent-interp-envs-dispatch` shows every launch.
"""
import random
import time

import modal

app = modal.App("agent-interp-envs-dispatch")


@app.function(image=modal.Image.debian_slim(), timeout=86400)
def dispatch(spec: dict) -> list:
    from concurrent.futures import ThreadPoolExecutor
    iso_image = modal.Image.from_id(spec["image_id"])
    volume = modal.Volume.from_name(spec["volume"], version=2)
    secrets = [modal.Secret.from_name(name) for name in spec["secrets"]]
    sandbox_app = modal.App.lookup(app.name)

    def _one(job):
        cfg_yaml, run_dir = job
        try:
            for draw in range(spec["gate_draws"]):
                for attempt in range(spec["create_tries"]):
                    try:
                        sb = modal.Sandbox.create(
                            "bash", "-lc", spec["argv_cmd"],
                            app=sandbox_app, image=iso_image, timeout=spec["timeout"],
                            **spec["resources"], **{spec["allowlist_param"]: spec["cidrs"]},
                            volumes={"/mnt/results": volume}, secrets=secrets,
                            env={"CONTAINER_INIT": spec["setup_cmd"], "CONTAINER_RESTORE": spec["recovery_setup"],
                                 "ROLLOUT_CONFIG_YAML": cfg_yaml, "ROLLOUT_RUN_DIR": run_dir,
                                 "ROLLOUT_HOSTS": spec["hosts_pin"]},
                        )
                        break
                    except (modal.exception.ResourceExhaustedError, TimeoutError) as exc:
                        if attempt == spec["create_tries"] - 1:
                            raise
                        delay = min(spec["create_max_delay_s"], random.uniform(2, 10) * (attempt + 1))
                        print(f"  … {run_dir}: sandbox creation rate-limited ({type(exc).__name__}), "
                              f"retrying in {delay:.0f}s", flush=True)
                        time.sleep(delay)
                print(f"  ▶ {run_dir} (sandbox {sb.object_id})", flush=True)
                sb.wait()
                if sb.returncode != spec["gate_exit"]:
                    break
                print(f"  … {run_dir}: machine draw rejected by the gate (draw {draw + 1}), redrawing", flush=True)
            print(f"  {'✓' if sb.returncode == 0 else '✗'} {run_dir} (sandbox {sb.object_id}, exit {sb.returncode})",
                  flush=True)
            return {"run_dir": run_dir, "returncode": sb.returncode}
        except Exception as exc:   # one rollout failing to start must not stop the others
            print(f"  ✗ {run_dir} (launch failed: {type(exc).__name__}: {exc})", flush=True)
            return {"run_dir": run_dir, "returncode": -1}

    jobs = [tuple(j) for j in spec["jobs"]]
    print(f"dispatching {len(jobs)} runs to {spec['volume']}, at most {spec['max_concurrent'] or len(jobs)} at once",
          flush=True)
    with ThreadPoolExecutor(max_workers=min(spec["max_concurrent"] or len(jobs), len(jobs))) as pool:
        results = list(pool.map(_one, jobs))
    print(f"Succeeded: {sum(r['returncode'] == 0 for r in results)}/{len(results)}", flush=True)
    return results
