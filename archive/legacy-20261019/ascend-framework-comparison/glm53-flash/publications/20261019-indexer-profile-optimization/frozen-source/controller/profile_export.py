# SPDX-License-Identifier: Apache-2.0
"""CANN export/completeness policy for the R247 B windows.

A missing or unnamed native info file is not a pass.  ``profiler_info_0.json``
and every other ``profiler_info*.json`` are recognized.  A truncated host
index or a missing required export product makes the window incomplete and the
run FAIL; collection timeouts are never converted into an empty success.
"""
import fnmatch
import pathlib

WINDOWS = ("early", "mid", "late")
RANK_NAMESPACE = "dp0-rank0"
PROFILE_ROOT = "profile-B"


def _match(files, pattern):
    return sorted(rel for rel in files if fnmatch.fnmatchcase(rel, pattern))


def required_window_files(window, rank_namespace=RANK_NAMESPACE):
    """Patterns that accept both the native timestamp subdir and a direct dir."""
    base = "%s/%s/%s" % (PROFILE_ROOT, rank_namespace, window)
    output = base + "/ASCEND_PROFILER_OUTPUT/"
    nested_output = base + "/*/ASCEND_PROFILER_OUTPUT/"
    return {
        "profiler_info": [base + "/profiler_info*.json",
                          base + "/*/profiler_info*.json"],
        "kernel_details": [output + "kernel_details.csv",
                           nested_output + "kernel_details.csv"],
        "operator_details": [output + "operator_details.csv",
                             nested_output + "operator_details.csv"],
        "step_trace_time": [output + "step_trace_time.csv",
                            nested_output + "step_trace_time.csv"],
        "trace_view": [output + "trace_view.json",
                       nested_output + "trace_view.json"],
    }


def required_shared_files(rank_namespace=RANK_NAMESPACE):
    return {
        "forward_metadata": ["%s/forward-metadata-%s.jsonl" % (PROFILE_ROOT, rank_namespace)],
        "observer_status": ["%s/observer-status-%s.json" % (PROFILE_ROOT, rank_namespace)],
    }


def _matches(files, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    found = set()
    for pattern in patterns:
        found.update(_match(files, pattern))
    return sorted(found)


def evaluate(files, windows=WINDOWS, rank_namespace=RANK_NAMESPACE,
             truncated=False, collection_error=None):
    """Evaluate the host profile index without reading any file body."""
    files = dict(files or {})
    result = {
        "schema": "r247/profile-export/v1",
        "rank_namespace": rank_namespace,
        "windows": {},
        "shared": {},
        "missing": [],
        "complete": False,
        "truncated": bool(truncated),
        "collection_error": collection_error,
    }
    for window in windows:
        row = {"window": window, "required": {}, "missing": []}
        for label, pattern in required_window_files(window, rank_namespace).items():
            matches = _matches(files, pattern)
            row["required"][label] = matches
            if not matches:
                row["missing"].append(label)
                result["missing"].append(window + ":" + label)
        row["complete"] = not row["missing"]
        result["windows"][window] = row
    for label, pattern in required_shared_files(rank_namespace).items():
        matches = _matches(files, pattern)
        result["shared"][label] = matches
        if not matches:
            result["missing"].append("shared:" + label)
    result["shared_complete"] = all(result["shared"].values())
    result["complete"] = (not truncated and collection_error is None
                          and not result["missing"])
    return result
