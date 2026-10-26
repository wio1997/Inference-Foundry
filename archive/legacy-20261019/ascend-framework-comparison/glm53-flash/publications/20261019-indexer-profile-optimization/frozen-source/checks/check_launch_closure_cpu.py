#!/usr/bin/env python3
"""CPU drill of the full R247 review -> freeze -> gate -> payload closure.

The unit is copied to a temporary directory, a synthetic independent review and
R246 signoff are written there, and the real dispatch/launcher code is
exercised.  No SSH, Docker, NPU, HTTP or model call is made.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent
sys.path.insert(0, str(R / "controller"))
sys.path.insert(0, str(R))

import dispatch  # noqa: E402
import freeze_policy  # noqa: E402
import launcher  # noqa: E402


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def copy_source(target):
    ignore = shutil.ignore_patterns(
        "results", "artifacts", "collected", "evidence", "logs", "__pycache__",
        "*.pyc", "review.json", "review-result.json", "freeze-input.json",
        "r246-gate.json", "dispatch.once.lock", "once.lock",
        "controller.once.lock", "runner.log", "dispatch.json",
        "remote-submit.json", ".DS_Store")
    shutil.copytree(R, target, ignore=ignore)


def write_authorization(root, gate=True):
    immutable = freeze_policy.immutable_sha256(root)
    package = freeze_policy.package_sha256(root)
    freeze = {"project_id": "glm53-flash", "unit": "r247",
              "kind": "FREEZE_INPUT_NOT_A_REVIEW",
              "file_count": len(immutable), "files_sha256": immutable,
              "package_file_count": len(package),
              "package_files_sha256": package,
              "mutable_files": sorted(set(package) - set(immutable))}
    freeze_path = root / "freeze-input.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    review = {"project_id": "glm53-flash", "unit": "r247",
              "kind": "R247_INDEPENDENT_FROZEN_SOURCE_REVIEW",
              "reviewed_freeze_input_sha256": sha(freeze_path),
              "files_sha256": immutable}
    review_path = root / "review.json"
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    auth = {"project_id": "glm53-flash", "unit": "r247",
            "kind": "R247_INDEPENDENT_REVIEW_RESULT_AUTHORIZATION",
            "passed": True, "review_sha256": sha(review_path),
            "freeze_input_sha256": sha(freeze_path),
            "authorization": {
                "dispatch_allowed": True, "execute_once": True,
                "exact_command": "python3 dispatch.py",
                "controller_entry": "python3 controller/controller.py",
                "sitecustomize_precheck_sha256": sha(
                    root / "sitecustomize-precheck.json")}}
    (root / "review-result.json").write_text(json.dumps(auth, indent=2) + "\n")
    if gate:
        signoff = {"project_id": "glm53-flash", "unit": "r247",
                   "kind": "R246_ACTUAL_PASS_IDLE8_SIGNOFF",
                   "actual": True, "passed": True, "r246_unit": "r246",
                   "r246_host_receipt_sha256": "b" * 64,
                   "r246_launch_result_sha256": "c" * 64,
                   "idle8": {"node46": True, "node47": True},
                   "r247_review_sha256": sha(review_path),
                   "signed_by": "main-control",
                   "signed_at": "2026-10-19T00:00:00Z",
                   "authorization": {"r247_dispatch_allowed": True,
                                     "execute_once": True,
                                     "exact_command": "python3 dispatch.py"}}
        (root / "r246-gate.json").write_text(json.dumps(signoff, indent=2) + "\n")
    return review, auth


def test_closure(tmp):
    package = tmp / "package"
    copy_source(package)
    review, auth = write_authorization(package)
    loaded = launcher.load_review_and_authorization(package)
    assert loaded[0]["unit"] == "r247" and loaded[4]["passed"] is True
    payload = launcher.build_payload(package)
    assert payload["immutable_count"] == len(freeze_policy.immutable_sha256(package))
    assert payload["decoded_bytes"] < 16 * 1024 * 1024
    assert "r246-gate.json" in payload["files"]
    assert "review.json" in payload["files"] and "review-result.json" in payload["files"]
    assert "freeze-input.json" in payload["files"]
    assert not any("172.16.12.14" in name or "172.16.12.15" in name
                   for name in payload["files"])
    code = launcher.gateway_code(payload)
    compile(code, "rendered-gateway-full", "exec")
    ready = dispatch.check_ready(package)
    assert ready["gate"]["kind"] == "R246_ACTUAL_PASS_IDLE8_SIGNOFF"
    assert ready["r246_gate_sha256"] == sha(package / "r246-gate.json")

    class Process:
        pid = 4242

    calls = []

    def popen(args, **kwargs):
        calls.append((args, kwargs))
        return Process()

    dispatch.main(package, popen=popen)
    assert len(calls) == 1
    assert (package / "dispatch.once.lock").exists()
    assert json.loads((package / "dispatch.json").read_text())["unit"] == "r247"
    try:
        dispatch.main(package, popen=popen)
    except SystemExit as exc:
        assert "DISPATCH_ALREADY_STARTED" in str(exc)
    else:
        raise AssertionError("second dispatch must refuse")

    missing = tmp / "missing-gate"
    copy_source(missing)
    write_authorization(missing, gate=False)
    try:
        dispatch.check_ready(missing)
    except SystemExit as exc:
        assert "MISSING_R246_ACTUAL_PASS_IDLE8_SIGNOFF" in str(exc)
    else:
        raise AssertionError("missing R246 gate must refuse")

    stale = tmp / "stale"
    copy_source(stale)
    write_authorization(stale)
    (stale / "controller/profile_plan.json").write_text("{\"tampered\": true}\n")
    try:
        launcher.load_review_and_authorization(stale)
    except SystemExit as exc:
        assert any(token in str(exc) for token in (
            "FREEZE_INPUT_STALE", "SHA_MISMATCH",
            "REVIEW_IMMUTABLE_SET_MISMATCH")), str(exc)
    else:
        raise AssertionError("tampered source must refuse")


def main():
    with tempfile.TemporaryDirectory(prefix="r247-closure-") as path:
        test_closure(pathlib.Path(path))
    print(json.dumps({"passed": True, "unit": "r247",
                      "full_closure": True, "dispatch_once": True,
                      "missing_r246_gate_refused": True,
                      "tampered_source_refused": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
