import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = "/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41"
OUT = os.path.join(BASE, "whole-scope-20261030", "zcode-contract-ledger-execution.json")
MODEL = "bc10aa1f-6d2d-4158-be40-7017fe6bf933/deepseek-flash"

COMMANDS = [
    "/Users/wio/.local/bin/python3.12 whole-scope-20261030/build_source_ledger_v2.py",
    "/Users/wio/.local/bin/python3.12 whole-scope-20261030/build_contract_review_delta.py",
]

payload = {
    "model": MODEL,
    "harness": "python_subprocess_sequential_once",
    "working_directory": BASE,
    "started_at": datetime.now(timezone.utc).isoformat(),
    "status": "RUNNING",
    "commands": [],
}

def flush():
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUT)

def tail(text, n):
    text = text or ""
    return text[-n:]

flush()

for i, cmd in enumerate(COMMANDS, 1):
    argv = cmd.split()
    entry = {
        "index": i,
        "command": cmd,
        "argv": argv,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "returncode": None,
        "stdout": None,
        "stderr": None,
    }
    payload["commands"].append(entry)
    t0 = time.monotonic()
    try:
        p = subprocess.run(argv, cwd=BASE, capture_output=True)
        rc = p.returncode
        out = p.stdout.decode("utf-8", "replace")
        err = p.stderr.decode("utf-8", "replace")
        out_bytes = len(p.stdout)
        err_bytes = len(p.stderr)
    except Exception as e:  # harness-level failure, recorded and stop
        rc = -999
        out = ""
        err = "HARNESS_EXCEPTION: %r" % (e,)
        out_bytes = 0
        err_bytes = len(err.encode("utf-8"))
    entry["returncode"] = rc
    entry["stdout"] = out
    entry["stderr"] = err
    entry["stdout_chars"] = len(out)
    entry["stderr_chars"] = len(err)
    entry["stdout_bytes"] = out_bytes
    entry["stderr_bytes"] = err_bytes
    entry["finished_at"] = datetime.now(timezone.utc).isoformat()
    entry["duration_seconds"] = round(time.monotonic() - t0, 3)

    if rc != 0:
        payload["status"] = "STOPPED_AFTER_COMMAND_%d_FAILURE" % i
        payload["stopped_at_command"] = i
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        flush()
        print("COMMAND %d FAILED rc=%s duration=%.2fs stdout_bytes=%d stderr_bytes=%d"
              % (i, rc, entry["duration_seconds"], out_bytes, err_bytes))
        print("--- stdout tail ---")
        print(tail(out, 2500))
        print("--- stderr tail ---")
        print(tail(err, 2500))
        print("HARNESS_STOPPED status=%s" % payload["status"])
        sys.exit(1)

    payload["status"] = "COMMAND_%d_OK" % i
    flush()
    print("COMMAND %d OK rc=%s duration=%.2fs stdout_bytes=%d stderr_bytes=%d"
          % (i, rc, entry["duration_seconds"], out_bytes, err_bytes))
    print("--- stdout tail ---")
    print(tail(out, 2500))
    if err.strip():
        print("--- stderr tail ---")
        print(tail(err, 1500))

payload["status"] = "ALL_COMMANDS_OK"
payload["finished_at"] = datetime.now(timezone.utc).isoformat()
flush()
print("HARNESS_COMPLETE status=ALL_COMMANDS_OK")
