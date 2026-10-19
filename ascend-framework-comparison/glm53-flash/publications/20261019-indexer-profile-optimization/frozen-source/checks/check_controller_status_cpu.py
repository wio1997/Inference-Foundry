#!/usr/bin/env python3
"""CPU tests for the R247 controller PASS/Fail composition.

A profile export or download timeout, a non-COMPUTED analyzer, an extra formal
request, or a non-PASS framework result must fail the run.  There is no empty
success path.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R / "controller"))

import controller  # noqa: E402

PLAN = {"requests": {"generation_POST_max": 4, "cold_pairs": 2}}
BASE = {"generation_POST_attempted": 4, "HTTP_terminal": 4, "cold_pairs": 2,
        "formal_requests": 0, "status": "PASS",
        "profile_export": {"complete": True},
        "download": {"complete": True}}
ANALYSIS = {"status": "COMPUTED"}


def expect_fail(result, analysis, reason):
    status, reasons = controller.compute_status(result, analysis, PLAN)
    assert status == "FAIL", (status, reasons)
    assert reason in reasons, (reason, reasons)


def test_status():
    assert controller.compute_status(copy.deepcopy(BASE), dict(ANALYSIS), PLAN) == ("PASS", [])
    for field, value, reason in (
            ("generation_POST_attempted", 3, "generation_POST"),
            ("HTTP_terminal", 3, "HTTP_terminal"),
            ("cold_pairs", 1, "cold_pairs"),
            ("formal_requests", 1, "formal_requests"),
            ("status", "FAIL", "framework_status")):
        broken = copy.deepcopy(BASE)
        broken[field] = value
        expect_fail(broken, dict(ANALYSIS), reason)
    broken = copy.deepcopy(BASE)
    broken["profile_export"]["complete"] = False
    expect_fail(broken, dict(ANALYSIS), "profile_export")
    broken = copy.deepcopy(BASE)
    broken["download"]["complete"] = False
    expect_fail(broken, dict(ANALYSIS), "profile_download")
    expect_fail(copy.deepcopy(BASE), {"status": "PARTIAL"}, "analysis_not_computed")
    expect_fail(copy.deepcopy(BASE), {"status": "TIMEOUT"}, "analysis_not_computed")


def test_precheck_and_paths():
    plan = json.loads((R / "controller/profile_plan.json").read_text())
    precheck = controller.preflight_sitecustomize(plan)
    assert precheck["image"] == controller.SGLANG_IMAGE
    assert precheck["decision"] == "inject_observer"
    assert precheck["native_present"] is False
    try:
        controller.load_review()
    except SystemExit as exc:
        assert "NOT_FROZEN" in str(exc)
    else:
        raise AssertionError("review must be absent in source-only state")
    text = (R / "controller/controller.py").read_text()
    for marker in ("profile_export_incomplete", "profile_download_incomplete",
                   "sdma_native_transfer_unproven", "analysis_not_computed",
                   "writer_stop_unproven_decode_retained",
                   "R246_ACTUAL_PASS_IDLE8_SIGNOFF"):
        assert marker in text, marker
    assert "compute_status" in text
    required = controller._required_downloads({
        "profile_export": {
            "windows": {"early": {"required": {"profiler_info": ["a"],
                                                "kernel_details": ["b"]}}},
            "shared": {"forward_metadata": ["c"], "observer_status": ["d"]},
        }
    })
    assert required == ["a", "b", "c", "d"]


def main():
    test_status()
    test_precheck_and_paths()
    print(json.dumps({"passed": True, "unit": "r247",
                      "pass_counts": {"POST": 4, "HTTP": 4, "cold_pairs": 2,
                                      "formal_requests": 0},
                      "export_incomplete_fails": True,
                      "download_incomplete_fails": True,
                      "non_computed_analysis_fails": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
