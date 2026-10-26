#!/usr/bin/env python3
"""CPU tests for the actual R246 PASS/idle8 gate.

The gate is not a source-hash check.  It requires a main-control signoff that
binds the node-side R246 host receipt SHA, and the launcher/gateway re-reads
that receipt and both nodes' live idle8 before staging R247.
"""
from __future__ import annotations

import copy
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R / "controller"))
sys.path.insert(0, str(R))

import controller  # noqa: E402
import freeze_policy  # noqa: E402
import launcher  # noqa: E402

REVIEW_SHA = "a" * 64
RECEIPT_SHA = "b" * 64
GATE = {
    "project_id": "glm53-flash", "unit": "r247",
    "kind": "R246_ACTUAL_PASS_IDLE8_SIGNOFF",
    "actual": True, "passed": True, "r246_unit": "r246",
    "r246_host_receipt_sha256": RECEIPT_SHA,
    "r246_launch_result_sha256": "c" * 64,
    "idle8": {"node46": True, "node47": True},
    "r247_review_sha256": REVIEW_SHA,
    "signed_by": "main-control", "signed_at": "2026-10-19T00:00:00Z",
    "authorization": {"r247_dispatch_allowed": True, "execute_once": True,
                      "exact_command": "python3 dispatch.py"},
}
RECEIPT = {
    "unit": "r246", "passed": True, "rank_passed": 16,
    "stage_a": {"passed": True}, "owned_container_verified": True,
    "bounded_wait_timeout": False,
    "post_inspect": {"rc": 0, "stdout": "false 0"},
    "post_npu": {"rc": 0,
                 "stdout": "\n".join(["No running processes found"] * 8)},
    "container": "if-glm53-flash-r246-rowstride-component",
    "image": "sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263",
    "archive": {"files": {"rank0.json": {"sha256": "d" * 64}}},
}


def expect_fail(call, contains=None):
    try:
        call()
    except SystemExit as exc:
        if contains:
            assert contains in str(exc), (contains, str(exc))
        return
    raise AssertionError("expected SystemExit")


def test_gate():
    assert controller.verify_r246_gate_local(copy.deepcopy(GATE), REVIEW_SHA) == RECEIPT_SHA
    for field, value, marker in (
            ("passed", False, "IDENTITY_INVALID"),
            ("actual", False, "IDENTITY_INVALID"),
            ("r246_unit", "r247", "UNIT_INVALID"),
            ("r246_host_receipt_sha256", "bad", "RECEIPT_SHA_INVALID"),
            ("r247_review_sha256", "d" * 64, "REVIEW_SHA_MISMATCH")):
        broken = copy.deepcopy(GATE)
        broken[field] = value
        expect_fail(lambda broken=broken: controller.verify_r246_gate_local(broken, REVIEW_SHA), marker)
    broken = copy.deepcopy(GATE)
    broken["idle8"]["node47"] = False
    expect_fail(lambda: controller.verify_r246_gate_local(broken, REVIEW_SHA), "IDLE8_UNPROVEN")
    broken = copy.deepcopy(GATE)
    broken["authorization"]["execute_once"] = False
    expect_fail(lambda: controller.verify_r246_gate_local(broken, REVIEW_SHA), "NOT_AUTHORIZED")
    assert controller.verify_r246_receipt(copy.deepcopy(RECEIPT), GATE)["rank_passed"] == 16
    for field, value, marker in (
            ("passed", False, "NOT_PASSED"),
            ("rank_passed", 15, "RANK_PASSED_INVALID"),
            ("owned_container_verified", False, "OWNED_CONTAINER_UNVERIFIED"),
            ("bounded_wait_timeout", True, "BOUNDED_WAIT_TIMEOUT"),
            ("container", "wrong", "CONTAINER_INVALID"),
            ("image", "wrong", "IMAGE_INVALID")):
        broken = copy.deepcopy(RECEIPT)
        broken[field] = value
        expect_fail(lambda broken=broken: controller.verify_r246_receipt(broken, GATE), marker)
    broken = copy.deepcopy(RECEIPT)
    broken["stage_a"]["passed"] = False
    expect_fail(lambda: controller.verify_r246_receipt(broken, GATE), "STAGE_A_NOT_PASSED")
    broken = copy.deepcopy(RECEIPT)
    broken["post_inspect"]["stdout"] = "true 0"
    expect_fail(lambda: controller.verify_r246_receipt(broken, GATE), "CONTAINER_NOT_STOPPED")
    broken = copy.deepcopy(RECEIPT)
    broken["post_npu"]["stdout"] = "\n".join(["No running processes found"] * 7)
    expect_fail(lambda: controller.verify_r246_receipt(broken, GATE), "IDLE8_UNPROVEN")


def test_method_is_actual():
    assert (R / "r246-gate.json").exists() is False
    expect_fail(lambda: launcher.load_r246_gate_local(R), "MISSING_R246_ACTUAL_PASS_IDLE8_SIGNOFF")
    text = (R / "controller/controller.py").read_text()
    assert controller.R246_RECEIPT.endswith("/host-receipt.json")
    assert "R246_ACTUAL_PASS_IDLE8_SIGNOFF" in text
    assert "R246_RECEIPT_READ_FAILED" in text
    assert "npu-smi" in text and "No running processes found" in text
    assert "stage_a" in text and "rank_passed" in text
    gateway = (R / "launcher.py").read_text()
    assert "host-receipt.json" in gateway
    assert "r246_host_receipt_sha256" in gateway
    assert "npu-smi" in gateway and "No running processes found" in gateway
    assert "host_code" in gateway and "R246 gate read failed" in gateway
    assert "r246-gate.json" in freeze_policy.MUTABLE_FILES
    dispatch_text = (R / "dispatch.py").read_text()
    assert "load_review_and_authorization" in dispatch_text
    assert "r246_gate_sha256" in dispatch_text


def main():
    test_gate()
    test_method_is_actual()
    print(json.dumps({"passed": True, "unit": "r247",
                      "gate_kind": GATE["kind"],
                      "receipt_path": controller.R246_RECEIPT,
                      "live_idle_nodes": [46, 47],
                      "gate_present_now": (R / "r246-gate.json").exists()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
