#!/usr/bin/env python3
"""CLI wrapper: run the read-only recovered-native analyzer on R247 artifacts.

The parser framework is the main-control ``analyze-recovered-native-profile.py``
adapted only for R247 paths, ``profiler_info*.json`` and control-task exclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ANALYZER = HERE / "analyze_recovered_native_profile.py"


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_identity(root):
    root = pathlib.Path(root)
    wanted = []
    for pattern in ("profiler_info*.json", "profiler_metadata.json"):
        wanted.extend(root.rglob(pattern))
    for name in ("operator_details.csv", "kernel_details.csv", "step_trace_time.csv",
                 "trace_view.json"):
        wanted.extend(root.rglob("ASCEND_PROFILER_OUTPUT/" + name))
    wanted.extend(root.rglob("forward-metadata-*.jsonl"))
    wanted.extend(root.rglob("observer-status-*.json"))
    wanted.extend(root.rglob("observer-sitecustomize-*.json"))
    wanted.extend(root.rglob("observer-install-*.json"))
    rows = {}
    for path in sorted(set(wanted)):
        if path.is_file():
            rows[str(path.relative_to(root))] = {"bytes": path.stat().st_size,
                                                 "sha256": sha(path)}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--windows", default="[]")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    windows = json.loads(args.windows)
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".analyzer.tmp")
    result = {"project_id": "glm53-flash", "unit": "r247",
              "status": "PARTIAL", "artifacts": args.artifacts,
              "windows_requested": windows, "captures": []}
    try:
        q = subprocess.run([sys.executable, str(ANALYZER), args.artifacts,
                            args.artifacts, str(temporary)],
                           capture_output=True, text=True, timeout=90)
        result["analyzer_rc"] = q.returncode
        result["analyzer_stdout_tail"] = q.stdout[-4000:]
        result["analyzer_stderr_tail"] = q.stderr[-4000:]
        if q.returncode == 0 and temporary.is_file():
            parsed = json.loads(temporary.read_text())
            merged = dict(result)
            merged.update(parsed)
            merged["windows_requested"] = windows
            result = merged
        else:
            result["error"] = "analyzer_failed"
    except subprocess.TimeoutExpired as exc:
        result["error"] = "analyzer_timeout: " + repr(exc)
    finally:
        try:
            temporary.unlink()
        except OSError:
            pass
    result["input_files"] = input_identity(args.artifacts)
    result["analyzer"] = {"path": str(ANALYZER),
                          "sha256": sha(ANALYZER),
                          "trace_stats_sha256": sha(HERE / "trace_stats.py"),
                          "categories_sha256": sha(HERE / "categories.json")}
    result["export_or_capture_incomplete"] = result.get("status") != "COMPUTED"
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(output)
    print(json.dumps({"status": result["status"],
                      "captures": len(result.get("captures", [])),
                      "output": str(output)}))
    return 0 if result.get("status") in ("COMPUTED", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
