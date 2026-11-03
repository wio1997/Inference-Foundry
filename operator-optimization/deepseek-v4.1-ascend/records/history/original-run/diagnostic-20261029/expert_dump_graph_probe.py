#!/usr/bin/env python3
"""Guarded launcher for a synthetic CANN dump probe; disabled by default.

The caller must independently verify device ownership and idle service state
immediately before use. This script does NOT establish device ownership.
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


TASK_DIR = Path("/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029")
CANN = Path("/usr/local/Ascend/cann-9.2.0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-isolated", action="store_true")
    parser.add_argument("--checked-idle-device-and-isolated-container", action="store_true")
    parser.add_argument("--checked-own-idle-service-and-no-foreign-processes", action="store_true")
    parser.add_argument("--mode", choices=("eager", "graph"), required=True,
                        help="graph only after decoding and manually verifying eager results")
    parser.add_argument("--eager-values-verified", action="store_true",
                        help="caller confirms all eager dump values were decoded and checked")
    parser.add_argument("--device-id", help="one explicitly checked CANN logical device ID")
    parser.add_argument("--output-dir", help="pre-existing empty child of the task directory")
    args = parser.parse_args()
    if not (args.execute_isolated and (args.checked_idle_device_and_isolated_container or
                                       args.checked_own_idle_service_and_no_foreign_processes)):
        parser.error("disabled: verify device ownership and idle service or an empty device")
    if args.mode == "graph" and not args.eager_values_verified:
        parser.error("graph requires prior eager dump value verification")
    if not args.device_id or not re.fullmatch(r"(?:[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])", args.device_id):
        parser.error("--device-id must be one numeric logical ID")
    if not args.output_dir:
        parser.error("--output-dir is required")
    root = TASK_DIR.resolve(strict=True)
    out = Path(args.output_dir).resolve(strict=True)
    if out == root or root not in out.parents or not out.is_dir():
        parser.error("output must be a pre-existing child directory of the existing task directory")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(out)):
        parser.error("output path contains unsupported characters")
    if any(out.iterdir()):
        parser.error("output directory must be empty; existing files are never removed")
    source = root / "expert_dump_graph_probe.cpp"
    binary = out / "expert_dump_graph_probe"
    if not source.is_file() or not (CANN / "include/aclnn/acl_meta.h").is_file():
        parser.error("source or CANN 9.2.0 headers missing")
    # No inherited ASCEND_*, LD_PRELOAD, dump scene, or other task settings.
    # All known writable runtime paths resolve under this single isolated output.
    env = {
        "PATH": "/usr/bin:/bin",
        "TMPDIR": str(out),
        "ASCEND_DUMP_PATH": str(out),
        "ASCEND_WORK_PATH": str(out),
        "ASCEND_PROCESS_LOG_PATH": str(out),
        "ASCEND_CACHE_PATH": str(out),
        "ASCEND_SLOG_PRINT_TO_STDOUT": "1",
        "NPU_COLLECT_PATH": str(out),
        "ASCEND_HOME_PATH": str(CANN),
        "ASCEND_TOOLKIT_HOME": str(CANN),
        "ASCEND_RT_VISIBLE_DEVICES": args.device_id,
        "PROBE_LAUNCHED_BY_GUARD": "1",
        "LD_LIBRARY_PATH": ":".join((str(CANN / "lib64"), "/usr/local/Ascend/driver/lib64", "/usr/local/Ascend/driver/lib64/common", "/usr/local/Ascend/driver/lib64/driver")),
    }
    command = ["/usr/bin/g++", "-std=c++17", "-O2", "-I" + str(CANN / "include"), str(source),
               "-L" + str(CANN / "lib64"), "-Wl,-rpath," + str(CANN / "lib64"),
               "-lascendcl", "-lnnopbase", "-o", str(binary)]
    print("Compiling in the designated output directory; selected logical device:", args.device_id, flush=True)
    compile_result = subprocess.run(command, env=env, cwd=out, check=False)
    if compile_result.returncode:
        print("Compilation failed; no NPU probe was launched.", file=sys.stderr)
        return compile_result.returncode
    # CANN documents remapping the single visible logical ID to ordinal 0;
    # verify container logical-to-physical mapping and exposure before launch.
    result = subprocess.run([str(binary), "--execute-isolated", args.mode, "0", str(out)], env=env, cwd=out, check=False)
    print("API status only; dump files must be decoded and matched to replay IDs.", flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
