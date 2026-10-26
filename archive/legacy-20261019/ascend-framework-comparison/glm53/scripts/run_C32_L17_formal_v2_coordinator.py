"""DRAFT_PENDING_PRIMARY_REVIEW: local durable coordinator for L17 formal v2.

The detached wrapper starts this file exactly once.  It source-pins the actual
runner, contract and controller, executes the runner through ``runpy`` as
``__main__`` so every runner/controller local guard is traversed, and records
the real process exit from a ``try/finally``.  It performs no SSH, HTTP, Docker,
NPU, SDK or model action itself and creates no retry or polling path.
"""
import hashlib
import json
import os
import runpy
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = "scripts/run_C32_L17_native_formal_v2.py"
RUNNER_SHA256 = "7c65c3f27ebc959090efb8c63912cbeb50afe6505ab6972bd74031cf8be330a4"
CONTRACT = "scripts/glm53_C32_L17_pipeline_contract_v3.py"
CONTRACT_SHA256 = "c2f03fd0ae43ff3cc88b98138d4a408fca23845d55c0bd992ec07909d77bbe74"
CONTROLLER = "scripts/glm53_C32_L17_formal_resource_controller_v2.py"
CONTROLLER_SHA256 = "ffe433832a056d7368bb8d131a94a899dafbe9a67a2d27435708da3bea5c2290"
RUN = ROOT / "runs" / "20261019T-L17-formal-v2-detached-executor-v1"
EXIT_RECORD = RUN / "coordinator-exit.json"
RUNNER_RECEIPT = ROOT / "runs" / "20261019T-L17-native-formal-launch-v2" / "receipt.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_source(relative, expected):
    path = ROOT / relative
    assert path.is_file() and not path.is_symlink(), "missing or symlinked source: " + relative
    actual = sha256_file(path)
    assert actual == expected, "%s sha256 %s != %s" % (relative, actual, expected)
    return path


def write_exit(payload):
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(str(EXIT_RECORD), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, raw)
        os.fsync(fd)
    finally:
        os.close(fd)


def main():
    assert RUN.is_dir() and not RUN.is_symlink(), "one-shot wrapper run dir missing: " + str(RUN)
    assert not EXIT_RECORD.exists(), "coordinator exit record already exists; one-shot only"
    runner = pinned_source(RUNNER, RUNNER_SHA256)
    pinned_source(CONTRACT, CONTRACT_SHA256)
    pinned_source(CONTROLLER, CONTROLLER_SHA256)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    scripts = str(ROOT / "scripts")
    if sys.path[0] != scripts:
        sys.path.insert(0, scripts)

    started_utc = utc_now()
    started = datetime.now(timezone.utc)
    phase = "RUNNER_RUNNING"
    exit_code = 0
    error = None
    error_traceback = None
    try:
        runpy.run_path(str(runner), run_name="__main__")
        phase = "RUNNER_RETURNED"
    except SystemExit as exc:
        code = exc.code
        exit_code = 0 if code is None else (code if isinstance(code, int) else 1)
        phase = "RUNNER_SYSTEM_EXIT"
        raise
    except BaseException as exc:
        exit_code = 1
        phase = "RUNNER_EXCEPTION"
        error = repr(exc)
        error_traceback = traceback.format_exc()
        raise
    finally:
        finished_utc = utc_now()
        receipt_sha256 = sha256_file(RUNNER_RECEIPT) if RUNNER_RECEIPT.is_file() else None
        payload = {
            "project_id": "glm53",
            "run_id": RUN.name,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "argv": sys.argv,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "duration_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "phase": phase,
            "exit_code": exit_code,
            "error": error,
            "traceback": error_traceback,
            "runner": RUNNER,
            "runner_sha256": RUNNER_SHA256,
            "contract": CONTRACT,
            "contract_sha256": CONTRACT_SHA256,
            "controller": CONTROLLER,
            "controller_sha256": CONTROLLER_SHA256,
            "runner_launch_receipt": str(RUNNER_RECEIPT.relative_to(ROOT)),
            "runner_launch_receipt_sha256": receipt_sha256,
            "executes_actual_main": True,
            "runpy_run_name": "__main__",
            "source_pin_enforced": True,
            "SIGHUP_ignored": True,
            "stdin": "inherited DEVNULL from detached wrapper",
            "additional_network_actions_outside_actual_runner": 0,
            "retries": 0,
            "polling": False,
            "completion_claimed": False,
        }
        write_exit(payload)
        print("GLM53_COORDINATOR_EXIT=" + json.dumps({
            "pid": payload["pid"],
            "phase": phase,
            "exit_code": exit_code,
            "exit_record": str(EXIT_RECORD),
        }, sort_keys=True), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
