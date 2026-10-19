"""DRAFT_PENDING_PRIMARY_REVIEW: once-only startup native capture wrapper.

Requires the actual D model-start/observe completion record, then captures all
four current native nodes before any external client.  The core factory pins the
approved capture review and writes the raw wire before decoding.  No model/SDK
request, profile, retry or Git action is performed by this draft.
"""
import hashlib
import importlib.util
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v1 as contract

ROOT = Path(__file__).resolve().parents[1]
CORE = "scripts/glm53_C32_L17_native_capture_v1.py"
CORE_SHA = "229893e45c8f0778699889446cd5f7d9097f7c8d1b6873612cc90eb679c793f2"
D_START_REVIEW = "results/C32-L17-D-model-start-actual-review-v1.json"
D_START_RECEIPT = "runs/20261019T-L17-D-models-and-observe-v1/receipt.json"
STARTUP_PHASE = "STARTUP_OBSERVED_AWAITING_INDEPENDENT_CAPTURE_AND_FUNCTIONAL"
START_KEYS = ("start-36", "start-41")
OBSERVE_KEYS = ("observe-36", "observe-41")
RUNTIME_CIDS = {
    "34": contract.P34_CID,
    "35": contract.P35_CID,
    "36": contract.D36_CID,
    "41": contract.D41_CID,
}


def _rc0(mapping, key):
    return isinstance(mapping, dict) and type(mapping.get(key)) is int and mapping[key] == 0


def startup_receipt_gate(row):
    """Require the actual model-coordinator observe-complete schema.

    The coordinator writes start-36/start-41 and observe-36/observe-41 (not
    numeric node keys), never a top-level startup_pass.  Bool is not an rc.
    """
    if not isinstance(row, dict):
        return False
    if row.get("project_id") != "glm53":
        return False
    if row.get("pass") is not True or row.get("phase") != STARTUP_PHASE:
        return False
    if row.get("capacity_sha256") != contract.D_CAPACITY_SHA:
        return False
    starts = row.get("starts")
    if not isinstance(starts, dict) or set(starts) != set(START_KEYS):
        return False
    if not all(_rc0(starts, key) for key in START_KEYS):
        return False
    observe = row.get("observe")
    if not isinstance(observe, dict) or set(observe) != set(OBSERVE_KEYS):
        return False
    if not all(_rc0(observe, key) for key in OBSERVE_KEYS):
        return False
    summaries = row.get("native_startup_summaries")
    if not isinstance(summaries, dict) or set(summaries) != set(OBSERVE_KEYS):
        return False
    for key in OBSERVE_KEYS:
        summary = summaries[key]
        if not isinstance(summary, dict):
            return False
        if summary.get("startup_pass") is not True or summary.get("formal_ready") is not False:
            return False
    if type(row.get("external_model_requests")) is not int or row["external_model_requests"] != 0:
        return False
    if row.get("cleanup_or_rollback_executed") is not False:
        return False
    if row.get("runtime_CIDs") != RUNTIME_CIDS:
        return False
    return True


def _load_core():
    path = ROOT / CORE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA
    spec = importlib.util.spec_from_file_location("l17_native_capture_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    review = contract.derived_require(
        D_START_REVIEW,
        lambda row: str(row.get("review_status", "")).startswith("APPROVED")
        and row.get("model_requests_authorized") == 0
        and row.get("cleanup_execution_authorized") is False,
    )
    contract.derived_require(D_START_RECEIPT, startup_receipt_gate)
    assert review["capacity_sha256"] == contract.D_CAPACITY_SHA
    core = _load_core()
    result = core.execute(contract.STARTUP_CAPTURE, contract.STARTUP_CAPTURE_REVIEW, "startup")
    print(result)
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
