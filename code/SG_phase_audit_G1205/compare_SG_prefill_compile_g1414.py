"""Compare the pinned W3 and W5 Prefill compiler boundary inventories."""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path


def load(path, expected, boundary):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Compiler snapshot SHA mismatch at {boundary}")
    report = json.loads(raw)
    summary = report["summary"]
    if not (
        summary["boundary_label"] == boundary and summary["epoch"] == 1317
        and summary["P_controller_PID"] == 2302205 and summary["guard_FD_verified"]
        and summary["ready"] == 200 and all(values and not any(values) for values in summary["idle"].values())
        and summary["compiler_file_count"] == len(report["compiler_files"])
    ):
        raise ValueError(f"Unexpected or nonidle Prefill snapshot at {boundary}")
    return report


def compare(before, after):
    left, right = before["summary"], after["summary"]
    identity = ("epoch", "P_controller_PID", "P_container_ID", "image_ID", "verified_guard_PID",
                "source_function_SHA256", "selected_env")
    if any(left[key] != right[key] for key in identity):
        raise ValueError("Prefill process, capture source or compiler environment changed")
    if datetime.fromisoformat(right["at"]) <= datetime.fromisoformat(left["at"]):
        raise ValueError("Unexpected order for snapshots from the same Prefill host")
    original, present = before["compiler_files"], after["compiler_files"]
    changes = sorted(path for path in original.keys() | present.keys() if original.get(path) != present.get(path))
    return {
        "same_actual_P_instance": True, "P_container_ID": left["P_container_ID"],
        "persistent_compiler_artifacts_unchanged": not changes,
        "before_label": left["boundary_label"], "after_label": right["boundary_label"],
        "compiler_files_before": len(original), "compiler_files_after": len(present),
        "persistent_changed_paths": len(changes),
        "persistent_changed_path_set_SHA256": sha256(json.dumps(changes).encode()).hexdigest(),
        "separate_Ascend_work_trees_changed":
            before["separate_Ascend_work_files"] != after["separate_Ascend_work_files"],
        "scope": "Persistent compiler inventory compared by path, file size and mtime; "
                 "Ascend runtime work trees are recorded separately.",
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for boundary in ("before", "after"):
        cli.add_argument(f"--{boundary}", type=Path, required=True)
        cli.add_argument(f"--{boundary}-sha256", required=True)
    args = cli.parse_args()
    report = compare(
        load(args.before, args.before_sha256, "after_warmup3_32768"),
        load(args.after, args.after_sha256, "after_warmup5_32768"),
    )
    print(json.dumps({"before_snapshot_SHA256": args.before_sha256,
                      "after_snapshot_SHA256": args.after_sha256, **report}, sort_keys=True))
