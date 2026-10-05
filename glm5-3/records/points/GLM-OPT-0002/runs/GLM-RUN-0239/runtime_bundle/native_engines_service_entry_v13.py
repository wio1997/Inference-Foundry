"""Run one GLM gateway using a provenance-checked rendered service config.

The controller performs live API/NPU ownership checks before launch. This entry
checks immutable evidence and configuration only; it does not start native models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import sys

from native_engines_service_config import checked_config


def _exit_after_shutdown(signum, frame):
    # Uvicorn restores and replays SIGTERM after its async shutdown. Raising
    # SystemExit lets an enclosing native ACL lifecycle run its finally block.
    raise SystemExit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config, state_dir = checked_config(args.config)
    state_dir.mkdir(parents=True, exist_ok=True)
    for key in ["GLM_ROUTER_AUDIT_DIR", "GLM_PD_AUDIT_DIR"]:
        Path(config["environment"][key]).mkdir(parents=True, exist_ok=True)
    # This optional override must not replace the compiled persistent fault journal.
    os.environ.pop("GLM_GROUP_FAULT_STATE_PATH", None)
    os.environ.update(config["environment"])
    proc = Path("/proc/self/stat").read_text()
    ticks = proc[proc.rfind(")") + 2:].split()[19]
    print("GLM_SERVICE_ENTRY_INSTALLED " + json.dumps(dict(
        pid=os.getpid(), boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        start_ticks=ticks, config_path=str(args.config),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        native_domains=config["native_domains"], port=args.port, gateway_version="V13",
        token_memo_byte_budget=int(os.environ.get("GLM_TOKEN_MEMO_MAX_BYTES","33554432")),
        token_memo_trace_path=os.environ.get("GLM_TOKEN_MEMO_TRACE_PATH"),
        live_native_identity_check="controller responsibility",
        response_owner_state_path=config["environment"]["GLM_RESPONSE_OWNER_STATE_PATH"],
    )), flush=True)
    gateway = Path(__file__).with_name("response_affinity_gateway_v12.py")
    sys.argv = [str(gateway), "--host", args.host, "--port", str(args.port)]
    previous = signal.signal(signal.SIGTERM, _exit_after_shutdown)
    try:
        runpy.run_path(str(gateway), run_name="__main__")
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
