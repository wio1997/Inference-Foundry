"""DRAFT_PENDING_PRIMARY_REVIEW: one-shot fresh pre-formal v2 native baseline.

This draft does not execute.  It source-pins the v4 capture core, the
interrupted-warmup ledger, the independent original-failure review, the
read-only recovery receipt and the existing native readiness record.  After the
local gates, ``main()`` calls only
``core.execute(contract.FORMAL_BASELINE_CAPTURE,
contract.FORMAL_BASELINE_REVIEW, "postfunctional")``.

The captured run is four native nodes only: health, metrics, source/SDK logs and
NPU state are read without model requests, SDK initialization, profiling or
service changes.  There is no CPU client collection, no reuse of the original
run and no replay of the interrupted execution.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v3 as contract

ROOT = contract.ROOT
CORE = "scripts/glm53_C32_L17_native_capture_v4.py"
CORE_SHA256 = "93508d4b7a60a76ea0be2fff83d06f5f8a29ce648078e416f4b240a73dc96390"
LEDGER = "results/model-request-ledger-through-L17-interrupted-warmup-v1.json"
LEDGER_SHA256 = "4c2cd04961806c217b0c8be3b892ad61c88cfda816f44f4174ea8bc8bad2f32a"
ORIGINAL_FAILURE_REVIEW = "results/C32-L17-original-formal-failure-independent-review-v1.json"
ORIGINAL_FAILURE_REVIEW_SHA256 = "e9bb41f13c3504bbc38eb8aa55bbdae38b2a407ce621c1745064bfb5ce760b93"
RECOVERY_RECEIPT = "runs/20261019T-L17-formal-interruption-readonly-recovery-v4/receipt.json"
RECOVERY_RECEIPT_SHA256 = "3e56587f756e189e1960a5fc3fa2e36c09ec403935b97dd37c6e5ccb21b1e722"
READINESS = "results/sglang-C32-L17-native-readiness-v1.json"
READINESS_SHA256 = "a233245e6daf23f2756932765c0e3f5c92d8361ce23c16235803fc361b14b9a4"
ORIGINAL_RUN = "20261019T-L17-sglang-formal-C32-native-v1"
FRESH_CONTROL_SOCKET = "/Users/wio/work/sglang-glm53-pd/ssh-control-l17-recovery-20261019"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_json(relative, expected):
    path = ROOT / relative
    assert path.is_file() and not path.is_symlink(), "missing or symlinked source: " + relative
    actual = sha256_file(path)
    assert actual == expected, "%s sha256 %s != %s" % (relative, actual, expected)
    return json.loads(path.read_text(encoding="utf-8"))


def load_core():
    path = ROOT / CORE
    assert path.is_file() and not path.is_symlink(), "missing v4 capture core"
    assert sha256_file(path) == CORE_SHA256, "v4 capture core SHA mismatch"
    spec = importlib.util.spec_from_file_location("l17_native_capture_v4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preconditions():
    ledger = pinned_json(LEDGER, LEDGER_SHA256)
    assert ledger["project_id"] == contract.PROJECT
    assert ledger["classification"] == "WARMUP32_ACTUAL_PENDING_NATIVE_64_FORMAL0_NO_PERFORMANCE"
    pending = ledger["new_pending_native_actual"]
    assert pending["run_id"] == ORIGINAL_RUN
    assert pending["warmup_logical_requests"] == 32
    assert pending["warmup_pass_logical_requests"] == 32
    assert pending["backend_HTTP_attempts"] == 64
    assert pending["P_attempts"] == 32 and pending["D_attempts"] == 32
    assert pending["formal_requests"] == 0 and pending["formal_backend_attempts"] == 0
    assert pending["pending_native_backend_inferences"] == 64
    assert pending["pending_native_successful_output_tokens"] == 64
    assert ledger["totals"]["pending_native_backend_inferences"] == 64
    evidence = pending["evidence"]
    assert evidence["run_root"] == "runs/20261019T-L17-formal-interruption-readonly-recovery-v4"
    assert evidence["original_docker_client_pid_exists"] is False
    assert evidence["owned_container_lookup"]["rc"] != 0
    assert "No such object" in evidence["owned_container_lookup"]["stderr"]
    progress = "runs/" + ORIGINAL_RUN + "/controller-progress.json"
    assert progress not in ledger["source_sha256"]
    inherited = ledger["inherited_historical_source_sha256"][progress]
    assert inherited["provenance"] == "PRIOR_LEDGER_HISTORICAL_OBSERVATION_ONLY"
    assert inherited["current_file_matches"] is False

    review = pinned_json(ORIGINAL_FAILURE_REVIEW, ORIGINAL_FAILURE_REVIEW_SHA256)
    assert review["project_id"] == contract.PROJECT
    assert review["decision"] == "ACCEPTED_ORIGINAL_TERMINAL_FAILURE_AND_ACTUAL_WARMUP_COUNTS_ONLY"
    assert review["original_run_id"] == ORIGINAL_RUN
    assert review["original_CPU_container_absent"] is True
    assert review["original_docker_client_PID_absent"] is True
    assert review["formal_requests_dispatched"] == 0
    assert review["no_client_replay"] is True
    assert review["no_performance_measurement"] is True
    assert review["actual_backend_attempts"] == 64

    recovery = pinned_json(RECOVERY_RECEIPT, RECOVERY_RECEIPT_SHA256)
    assert recovery["original_run"] == ORIGINAL_RUN and recovery["read_only"] is True
    assert recovery["model_requests"] == 0 and recovery["NPU_calls"] == 0
    assert recovery["SDK_calls"] == 0 and recovery["remote_files_written"] == 0
    assert recovery["original_launcher_replayed"] is False
    assert recovery["formal_counts_booked"] is False
    assert recovery["control_socket"] == FRESH_CONTROL_SOCKET

    readiness = pinned_json(READINESS, READINESS_SHA256)
    assert readiness["project_id"] == contract.PROJECT
    assert readiness["pass"] is True and readiness["formal_ready"] is True
    assert readiness["all32_P_actual_native_packed_MTP_firstuse_verified"] is True
    assert readiness["native_cache_and_SDMA_verified"] is True

    core = load_core()
    assert core.MODE_CAPTURE["postfunctional"] == contract.FORMAL_BASELINE_CAPTURE
    assert core.MODE_REVIEW["postfunctional"] == contract.FORMAL_BASELINE_REVIEW
    assert core.MODE_CAPTURE["formal"] == contract.FORMAL_CAPTURE
    assert core.MODE_REVIEW["formal"] == contract.FORMAL_CAPTURE_REVIEW
    assert "startup" not in core.MODES
    assert core.HISTORICAL_STARTUP["policy"] == "HISTORICAL_NOT_EXECUTED"
    assert core.FRESH_CONTROL_SOCKET == FRESH_CONTROL_SOCKET
    assert contract.FORMAL_BASELINE_CAPTURE not in (
        contract.FUNCTIONAL_CAPTURE,
        contract.FORMAL_CAPTURE,
        core.HISTORICAL_STARTUP["capture"],
    )

    run = ROOT / "runs" / contract.FORMAL_BASELINE_CAPTURE
    assert not run.exists() and not run.is_symlink(), "pre-formal baseline run already exists; one-shot only"
    review_path = ROOT / contract.FORMAL_BASELINE_REVIEW
    assert review_path.is_file() and not review_path.is_symlink(), (
        "primary APPROVED capture review missing; not executable: " + contract.FORMAL_BASELINE_REVIEW
    )
    baseline_review = json.loads(review_path.read_text(encoding="utf-8"))
    assert baseline_review.get("project_id") == contract.PROJECT
    assert str(baseline_review.get("decision", "")).startswith("APPROVED"), (
        "capture review is not APPROVED; not executable"
    )
    assert baseline_review.get("model_requests") in (0, None)
    return core


def main():
    core = _preconditions()
    receipt = core.execute(
        contract.FORMAL_BASELINE_CAPTURE,
        contract.FORMAL_BASELINE_REVIEW,
        "postfunctional",
    )
    assert set(receipt["nodes"]) == {str(node) for node in contract.ALL_NODES}
    assert receipt["cpu_client_collection"] is False
    assert receipt["model_requests"] == 0 and receipt["SDK_init_calls"] == 0
    assert receipt["profiling"] is False and receipt["service_changes"] == 0
    assert receipt["max_SSH"] == 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
