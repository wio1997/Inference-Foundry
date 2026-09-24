#!/usr/bin/env python3
"""Run one explicitly selected bounded Zcode subtask and record provenance.

The primary agent makes the routing decision. This script only executes a
bounded task; it does not alter TaskCtl or choose a model by task keywords.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--mode", choices=("plan", "build"), default="plan")
    args = parser.parse_args()

    task = args.task_file.resolve().read_text(encoding="utf-8").strip()
    if not task:
        parser.error("task file is empty")
    cwd = args.cwd.resolve(strict=True)
    record_dir = args.record_dir.resolve()
    record_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path.home() / ".zcode" / "cli" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = config.get("model", {}).get("main")
    if not isinstance(model, str) or not model.startswith("deepseek/"):
        parser.error(f"Zcode main model is not DeepSeek: {model!r}")
    zcode = shutil.which("zcode")
    if zcode is None:
        parser.error("zcode executable unavailable")

    command = [
        zcode, "--cwd", str(cwd), "--mode", args.mode, "--json",
        "--prompt", task,
    ]
    start = datetime.now(timezone.utc).isoformat()
    start_mono = time.monotonic()
    timeout_hit = False
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True,
            timeout=args.timeout, check=False,
        )
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        timeout_hit = True
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        returncode = 124

    (record_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError:
        response = {}
    if not isinstance(response, dict):
        response = {}
    (record_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    observed_model = None
    observed_source = None
    for key in ("model", "providerModel", "modelId"):
        value = response.get(key)
        if isinstance(value, str) and value:
            observed_model, observed_source = value, f"stdout.{key}"
            break
    if observed_model is None:
        match = re.search(r"anthropic\.messages / ([A-Za-z0-9_.-]+)", stderr)
        if match:
            observed_model = match.group(1)
            observed_source = "stderr.provider_warning"
    model_verified = bool(
        observed_model and (
            observed_model == model or observed_model == model.rsplit("/", 1)[-1]
        )
    )
    record = {
        "started_at": start,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - start_mono, 3),
        "route": "zcode",
        "configured_main_model": model,
        "observed_model": observed_model,
        "observed_model_source": observed_source,
        "model_verified": model_verified,
        "executable": zcode,
        "cwd": str(cwd),
        "mode": args.mode,
        "task_file": str(args.task_file.resolve()),
        "task_sha256": hashlib.sha256(task.encode()).hexdigest(),
        "spawn_mechanism": "subprocess.run",
        "session_id": response.get("sessionId"),
        "trace_id": response.get("traceId"),
        "provider_requests": response.get("usage", {}).get("modelRequestCount"),
        "returncode": returncode,
        "timeout": timeout_hit,
        "stdout_file": "stdout.txt",
        "stderr_file": "stderr.txt",
    }
    (record_dir / "execution.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
