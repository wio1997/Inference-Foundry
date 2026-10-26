"""DRAFT_PENDING_PRIMARY_REVIEW: post-formal native capture wrapper.

Requires the new L17 readiness certificate and the completed formal RPC, then
captures all four nodes plus the CPU43 formal metric archive.  The core factory
preserves raw wire before decoding, full logs, typed SDMA audit, Graph counters
and D eager-input evidence.  It starts no profile and performs no retry.
"""
import hashlib
import importlib.util
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v2 as contract

ROOT = Path(__file__).resolve().parents[1]
CORE = "scripts/glm53_C32_L17_native_capture_v3.py"
CORE_SHA = "f83c374a7c4b75d2af2c19b74cc6e913b5be5f049764c7f9e04671a44a24ef3f"


def _load_core():
    path = ROOT / CORE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == CORE_SHA
    spec = importlib.util.spec_from_file_location("l17_native_capture_core", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    readiness = contract.derived_require(
        contract.READINESS_RESULT,
        lambda row: row.get("pass") is True
        and row.get("all32_P_actual_native_packed_MTP_firstuse_verified") is True
        and row.get("native_cache_and_SDMA_verified") is True,
    )
    rpc = contract.derived_require(
        "runs/" + contract.FORMAL_RUN + "/controller-rpc-state.json",
        lambda row: row.get("done_received") is True and row.get("connection_rc") is not None,
    )
    assert (ROOT / "runs" / contract.FORMAL_RUN / "formal-result.remote.json").is_file()
    assert readiness["runtime_CIDs"]["36"] == contract.D36_CID
    assert readiness["runtime_CIDs"]["41"] == contract.D41_CID
    core = _load_core()
    result = core.execute(contract.FORMAL_CAPTURE, contract.FORMAL_CAPTURE_REVIEW, "formal")
    print(result)
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
