#!/usr/bin/env python3
"""Run one native AISBench window with idle gates and raw server sampling."""
import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["di", "vllm"], required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config")
    ap.add_argument("--bootstrap-warm", action="store_true")
    ap.add_argument("--server-log", default="logs/e27-vllm-8192-16-hccl1024.log")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.run_id):
        raise ValueError("Invalid run ID")
    if a.bootstrap_warm and ("-W" not in a.run_id or "-R" in a.run_id):
        raise ValueError("Bootstrap applies only to an untimed warm run")
    out = ROOT / "evidence-e27" / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    base = "http://127.0.0.1:" + ("18121" if a.framework == "di" else "18122")
    stop = threading.Event()
    sampler = None
    jit_root = ROOT.parent / "runtime-20261021/dg-jit"

    def jit_files():
        return {str(p.relative_to(jit_root)): p.stat().st_mtime
                for p in jit_root.rglob("meta.json")} if a.framework == "di" else {}

    def save(phase, **extra):
        (out / "progress.json").write_text(json.dumps(dict(
            phase=phase, run_id=a.run_id, framework=a.framework,
            observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            **extra), indent=2) + "\n")

    def fetch(endpoint):
        with urllib.request.urlopen(base + "/" + endpoint, timeout=8) as f:
            return f.read().decode("utf-8")

    def snapshot(label):
        for ep in (["status", "diag/requests"] if a.framework == "di" else ["metrics"]):
            (out / (label + "." + ep.replace("/", "_"))).write_text(fetch(ep))

    def gate(label):
        save(label)
        if label == "postflight":
            # A crashed service cannot become idle; avoid a ten-minute wait.
            for attempt in range(3):
                try:
                    fetch("status" if a.framework == "di" else "health")
                    break
                except Exception:
                    if attempt == 2:
                        raise RuntimeError("Service unavailable after AISBench; reject window")
                    time.sleep(0.5)
        p = subprocess.run([sys.executable, str(ROOT / "tools/e26_wait_idle.py"),
                            "--framework", a.framework, "--base-url", base,
                            "--timeout", "600", "--json-out", str(out / (label + ".json"))],
                           capture_output=True, text=True, timeout=620)
        if p.returncode:
            raise RuntimeError(label + ": " + (p.stderr or p.stdout)[-1200:])

    def sample():
        ep = "status" if a.framework == "di" else "metrics"
        with (out / "server-samples.jsonl").open("w") as f:
            while not stop.is_set():
                row = dict(wall=time.time(), monotonic=time.monotonic(), endpoint=ep)
                try:
                    row["raw"] = fetch(ep)
                except Exception as e:
                    row["error"] = repr(e)
                f.write(json.dumps(row) + "\n")
                f.flush()
                stop.wait(1.0)

    try:
        if a.bootstrap_warm:
            if a.framework != "vllm":
                raise ValueError("Only fresh vLLM gauges need bootstrap")
            save("bootstrap_wait_native_init")
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                try:
                    fetch("health")
                    log = (ROOT / a.server_log).read_text(errors="replace")
                    initialized = {int(m.group(1)) for line in log.splitlines()
                                   if "init engine (profile, create kv cache, warmup model) took" in line
                                   for m in [re.search(r"EngineCore_DP(\d+)", line)] if m}
                    if initialized == set(range(8)):
                        break
                except Exception:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Native eight-engine initialization/health not ready")
            (out / "preflight.json").write_text(json.dumps(dict(
                bootstrap_warm_only=True, formal_window=False,
                health200=True, all8_native_engine_init=sorted(initialized)), indent=2))
        else:
            gate("preflight")
        snapshot("before")
        jit_before = jit_files()
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        relative_config = a.config or "configs/e27/" + a.run_id + ".py"
        path = (ROOT / relative_config).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.exists():
            raise ValueError("Config must exist inside project root")
        container_root = "/work/performance-20261021"
        cmd = ["docker", "exec", "-e", "TORCH_DEVICE_BACKEND_AUTOLOAD=0", "-e",
               "PYTHONPATH=" + container_root + "/e24-aisbench:" + container_root +
               "/e24-aisbench/plugins/prefix_cache:" + container_root + "/tools",
               "performance-20261022-aisbench-client", "timeout", "--signal=TERM",
               "--kill-after=15s", str(a.timeout),
               container_root + "/e24-aisbench-venv/bin/ais_bench",
               container_root + "/" + str(path.relative_to(ROOT.resolve())),
               "--mode", "perf", "--num-warmups", "0", "--debug"]
        save("aisbench_running", command=cmd, parent_monotonic=time.monotonic())
        with (out / "aisbench.log").open("w") as f:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout + 40)
        if p.returncode:
            raise RuntimeError("AISBench exit " + str(p.returncode))
        gate("postflight")
        snapshot("after")
        if a.framework == "di":
            jit_after = jit_files()
            changed = {p: t for p, t in jit_after.items() if jit_before.get(p) != t}
            (out / "native-jit.json").write_text(json.dumps(dict(
                changed_metadata_count=len(changed), changed=changed,
                source="Native DG-JIT meta.json snapshots outside AIS timing"), indent=2) + "\n")
        save("completed", aisbench_exit=0, bootstrap_warm=a.bootstrap_warm)
    except BaseException as e:
        try:
            snapshot("failure")
        except Exception:
            pass
        save("failed", error=type(e).__name__ + ": " + str(e))
        raise
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=10)


if __name__ == "__main__":
    main()
