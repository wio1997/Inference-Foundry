"""DRAFT_PENDING_PRIMARY_REVIEW: post-functional all-four native capture wrapper.

Runs only after the client-only short count/response audit and the two-80K long
functional guard passed.  The capture requires all32 D eager-input bs2
draft-DECODE rows.  After the four node captures, it records the
already-produced CPU43 long-client artifact referenced by the accepted
collectors; it does not run a client or contact a model.
"""
import hashlib
import importlib.util
import json
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


def _collect_cpu43(capture):
    source = ROOT / "runs" / contract.LONG_RUN / "node-43.json"
    raw = source.read_bytes()
    wire = json.loads(raw)
    assert wire["rc"] == 0
    artifacts = None
    client_exit = None
    for line in wire["stdout"].splitlines():
        if line.startswith("GLM53_LONG_ARTIFACTS="):
            artifacts = json.loads(line.split("=", 1)[1])
        elif line.startswith("GLM53_CLIENT_EXIT="):
            client_exit = json.loads(line.split("=", 1)[1])
    assert artifacts is not None and client_exit is not None
    record = {
        "scope": "Existing CPU43 long-client artifact referenced by source-qualified collectors; no client rerun and no external call.",
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw),
        "client_exit": client_exit,
        "artifacts": artifacts,
        "if_required_by_old_collectors": True,
        "external_calls": 0,
    }
    target = ROOT / "runs" / capture / "cpu43-long-client-artifact.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return target, record


def main():
    client = contract.derived_require(
        contract.CLIENT_RESULT,
        lambda row: row.get("label") == contract.CLIENT_LABEL
        and row.get("pass") is True
        and row.get("native_qualified") is False
        and row.get("formal_ready") is False
        and row.get("successful_backend_inferences") == contract.SHORT_BACKEND
        and row.get("successful_output_tokens") == contract.SHORT_OUTPUT_IF_ALL,
    )
    long_receipt = contract.derived_require(
        "runs/" + contract.LONG_RUN + "/receipt.json",
        lambda row: row.get("rc") == 0
        and row.get("logical_client_cases_planned") == contract.LONG_LOGICAL
        and row.get("backend_model_calls_planned") == contract.LONG_BACKEND,
    )
    startup = contract.derived_require(
        contract.STARTUP_RESULT,
        lambda row: row.get("pass") is True
        and row.get("all64_actual_SDMA_and_BF16_layout_verified") is True
        and row.get("D_KV_capacity_floor") == contract.D_KV_FLOOR,
    )
    expected_cids = {contract.P34_CID, contract.P35_CID, contract.D36_CID, contract.D41_CID}
    assert set(client["runtime_CIDs"].values()) == expected_cids
    assert startup["runtime_CIDs"] == client["runtime_CIDs"]
    core = _load_core()
    result = core.execute(contract.FUNCTIONAL_CAPTURE, contract.FUNCTIONAL_CAPTURE_REVIEW, "postfunctional")
    target, cpu43 = _collect_cpu43(contract.FUNCTIONAL_CAPTURE)
    receipt_path = ROOT / "runs" / contract.FUNCTIONAL_CAPTURE / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["cpu43_client_artifact"] = {
        "path": str(target.relative_to(ROOT)),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_sha256": cpu43["source_sha256"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    result["cpu43_client_artifact"] = receipt["cpu43_client_artifact"]
    print(json.dumps(result, indent=2))
    return 0 if result.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
