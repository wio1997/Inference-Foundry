#!/usr/bin/env python3
"""CPU tests for profiler_info_0.json, control-task exclusion and export FAIL.

A synthetic 3-window native export is analyzed twice: through the recovered
raw analyzer and through the run_analysis wrapper.  WAIT/NOTIFY/SQE must stay
out of compute, and a missing/truncated export must never become COMPUTED.
"""
from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R / "controller"))
sys.path.insert(0, str(R / "analysis"))

import profile_export  # noqa: E402
import trace_stats  # noqa: E402

ANALYZER = R / "analysis/analyze_recovered_native_profile.py"
WRAPPER = R / "analysis/run_analysis.py"
WINDOWS = [{"name": "early", "lo": 8192, "hi": 16384, "forwards": 2},
           {"name": "mid", "lo": 32768, "hi": 40960, "forwards": 2},
           {"name": "late", "lo": 61440, "hi": 73728, "forwards": 2}]
CONTROL_DURATION = 900 + 200 + 100
CORE_DURATION = 1000 + 500


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def build_capture(root, window, base_index, context_ends):
    # The native torch_npu exporter appends a timestamp directory below the
    # window, exactly as in the recovered R238 profiles.
    directory = root / "profile-B" / "dp0-rank0" / window / "r247_ascend_pt"
    output = directory / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True, exist_ok=True)
    write_json(directory / "profiler_info_0.json", {
        "config": {
            "common_config": {"activities": ["CPU", "NPU"],
                              "record_shapes": False, "with_stack": False,
                              "profile_memory": False},
            "experimental_config": {"_profiler_level": "Level1"},
        }
    })
    events = [
        {"ph": "M", "name": "process_name", "pid": 10,
         "args": {"name": "Ascend Hardware"}},
        {"ph": "M", "name": "thread_name", "pid": 10, "tid": 1,
         "args": {"name": "stream"}},
        {"ph": "X", "name": "r234_capture_window_dp0-rank0_" + window,
         "ts": 1000, "dur": 10000, "pid": 10, "tid": 1},
        {"ph": "X", "name": "r234_forward_iter_dp0-rank0_" + str(base_index),
         "ts": 1100, "dur": 4000, "pid": 10, "tid": 1},
        {"ph": "X", "name": "r234_forward_iter_dp0-rank0_" + str(base_index + 1),
         "ts": 6000, "dur": 3000, "pid": 10, "tid": 1},
        {"ph": "X", "name": "MatMul", "ts": 1200, "dur": 1000,
         "pid": 10, "tid": 1},
        {"ph": "X", "name": "WAIT", "ts": 1300, "dur": 900,
         "pid": 10, "tid": 1},
        {"ph": "X", "name": "NOTIFY", "ts": 7000, "dur": 200,
         "pid": 10, "tid": 1},
        {"ph": "X", "name": "SQE", "ts": 7100, "dur": 100,
         "pid": 10, "tid": 1},
        {"ph": "X", "name": "MatMul", "ts": 7200, "dur": 500,
         "pid": 10, "tid": 1},
        {"ph": "X", "name": "MatMul", "ts": 50, "dur": 10,
         "cat": "cpu_op", "pid": 1, "tid": 2},
    ]
    write_json(output / "trace_view.json", events)
    with (output / "kernel_details.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Device", "Name", "Type", "Start Time(us)",
                         "Duration(us)", "Wait Time(us)", "Accelerator Core"])
        writer.writerow(["NPU:0", "MatMul", "AI Core", "1200", "1000", "0", "AI CORE"])
        writer.writerow(["NPU:0", "WAIT", "AI CPU", "1300", "900", "0", "AI CPU"])
        writer.writerow(["NPU:0", "NOTIFY", "AI CPU", "7000", "200", "0", "AI CPU"])
        writer.writerow(["NPU:0", "SQE", "AI CPU", "7100", "100", "0", "AI CPU"])
        writer.writerow(["NPU:0", "MatMul", "AI Core", "7200", "500", "0", "AI CORE"])
    with (output / "operator_details.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Host Self Duration(us)", "Host Total Duration(us)",
                         "Device Self Duration(us)", "Device Total Duration(us)"])
        writer.writerow(["MatMul", "100", "120", "1500", "1600"])
    with (output / "step_trace_time.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Duration(us)"])
    with (root / "profile-B" / "forward-metadata-dp0-rank0.jsonl").open("a") as stream:
        for offset, end in enumerate(context_ends):
            row = {"forward_index": base_index + offset,
                   "rank_namespace": "dp0-rank0",
                   "capture": window,
                   "query_tokens": 4096,
                   "requests": [{"rid": "r", "context_start": end - 4096,
                                 "context_end": end, "query_tokens": 4096}]}
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(root / "profile-B" / "observer-status-dp0-rank0.json",
               {"rank_namespace": "dp0-rank0", "policy": {"complete": True}})


def build_captures(tmp):
    root = tmp / "captures"
    for index, window in enumerate(WINDOWS):
        start = window["lo"]
        build_capture(root, window["name"], index * 10 + 1,
                      [start + 4096, start + 8192])
    return root


def test_control_flags():
    assert trace_stats.is_control_task("WAIT") is True
    assert trace_stats.is_control_task("NOTIFY") is True
    assert trace_stats.is_control_task("SQE") is True
    assert trace_stats.is_control_task("WAIT_TIME") is True
    assert trace_stats.is_control_task("MatMul") is False
    assert trace_stats.is_control_task("FlashAttention") is False


def test_export_policy(tmp):
    windows = ("early", "mid", "late")
    root = "profile-B/dp0-rank0"
    files = {}
    for window in windows:
        base = root + "/" + window
        files[base + "/profiler_info_0.json"] = {"bytes": 1, "sha256": "x"}
        for name in ("kernel_details.csv", "operator_details.csv",
                     "step_trace_time.csv", "trace_view.json"):
            files[base + "/ASCEND_PROFILER_OUTPUT/" + name] = {"bytes": 1, "sha256": "x"}
    files["profile-B/forward-metadata-dp0-rank0.jsonl"] = {"bytes": 1, "sha256": "x"}
    files["profile-B/observer-status-dp0-rank0.json"] = {"bytes": 1, "sha256": "x"}
    result = profile_export.evaluate(files)
    assert result["complete"] is True, result
    assert result["windows"]["early"]["required"]["profiler_info"] == [
        "profile-B/dp0-rank0/early/profiler_info_0.json"]
    del files["profile-B/dp0-rank0/mid/profiler_info_0.json"]
    assert profile_export.evaluate(files)["complete"] is False
    files["profile-B/dp0-rank0/mid/profiler_info.json"] = {"bytes": 1, "sha256": "x"}
    assert profile_export.evaluate(files)["complete"] is True
    assert profile_export.evaluate(files, truncated=True)["complete"] is False
    assert profile_export.evaluate(files, collection_error="timeout")["complete"] is False


def test_synthetic_analysis(tmp):
    root = build_captures(tmp)
    out = tmp / "analysis.json"
    q = subprocess.run([sys.executable, str(ANALYZER), str(root), str(root), str(out)],
                       capture_output=True, text=True, timeout=60)
    assert q.returncode == 0, q.stderr
    value = json.loads(out.read_text())
    assert value["status"] == "COMPUTED", value
    assert value["captures_complete"] is True
    assert len(value["captures"]) == 3
    for capture in value["captures"]:
        assert capture["profiler_info_file"].endswith("profiler_info_0.json")
        assert capture["native_CSV_control_tasks"] == 3
        assert capture["control_task_count"] == 3
        assert capture["control_task_duration_sum_us"] == CONTROL_DURATION
        assert capture["core_compute_union_us"] == CORE_DURATION
        assert capture["native_kernel_busy_union_us"] == CORE_DURATION
        assert capture["core_category_union_us"].get("matmul") == CORE_DURATION
        assert "control" not in capture["core_category_union_us"]
        names = {row["name"] for row in capture["kernel_duration_sums_not_critical_path"]}
        assert names == {"MatMul"}, names
        assert capture["cpu_enqueue_is_device_forward"] is False
        assert capture["per_forward_device_attribution"] is None
        assert capture["native_profiler_config_valid"] is True
    wrapper_out = tmp / "wrapper.json"
    q = subprocess.run([sys.executable, str(WRAPPER), "--artifacts", str(root),
                        "--windows", json.dumps(WINDOWS), "--output", str(wrapper_out)],
                       capture_output=True, text=True, timeout=90)
    assert q.returncode == 0, q.stderr
    wrapped = json.loads(wrapper_out.read_text())
    assert wrapped["status"] == "COMPUTED", wrapped
    assert any(name.endswith("profiler_info_0.json") for name in wrapped["input_files"])
    assert wrapped["export_or_capture_incomplete"] is False


def main():
    test_control_flags()
    with tempfile.TemporaryDirectory(prefix="r247-profile-") as path:
        tmp = pathlib.Path(path)
        test_export_policy(tmp)
        test_synthetic_analysis(tmp)
    print(json.dumps({"passed": True, "unit": "r247",
                      "profiler_info_names": ["profiler_info.json", "profiler_info_0.json"],
                      "control_tasks": ["WAIT", "NOTIFY", "SQE"],
                      "control_excluded_from_compute": True,
                      "cpu_enqueue_is_device_forward": False,
                      "export_timeout_is_fail": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
