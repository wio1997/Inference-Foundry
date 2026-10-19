"""DRAFT_PENDING_PRIMARY_REVIEW: client-only L17 short count/response audit v2.

v2 fixes only precheck_identity_rows for the actual two-record precheck stdout:
the mixed identity renderer emits one assertion record with model_identity and
staged_files, then render_check appends one final idle record with
actual_model_identity, supervisor_phase, provider_phase and idle_gauges.  Both
must be exactly one, agree on node/CID/parent identity, and the idle record must
be MODEL_RUNNING/READY with non-empty all-zero gauges.  Both records and the raw
stdout SHA are kept as evidence; unrelated stdout lines are never a pass basis.

Breaks the long/short dependency cycle.  This audit consumes only the actual
short client receipt/wire plus the startup result, review and current CIDs.  It
reuses the strict 32 logical / 64 backend / 19232 output, prompt SHA, P/D
global-DP route, unique-ID, response-success and count checks from the first
part of the full short functional audit, but it does not read the
post-functional native capture and never claims native qualification.

Output is explicitly CLIENT_COUNTS_AND_RESPONSE_ONLY, native_qualified=false,
formal_ready=false, with no all32 bs2/Graph/SDMA claim.  It never overwrites
results/sglang-C32-L17-native-functional-v1.json (the later full native short
audit owns that path).
"""
import collections
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v2 as contract

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
SHORT_RUNNER = "scripts/run_C32_L17_native_P_functional_v1.py"


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def source_sha(relative):
    return hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()


def load_short_runner():
    path = PROJECT_ROOT / SHORT_RUNNER
    spec = importlib.util.spec_from_file_location("l17_short_runner_for_client_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IDENTITY_KEYS = ("model_identity", "staged_files")
IDLE_KEYS = ("actual_model_identity", "idle_gauges", "supervisor_phase", "provider_phase")


def precheck_identity_rows(run):
    rows = {}
    for node in contract.ALL_NODES:
        relative = "runs/" + contract.SHORT_RUN + "/precheck-" + str(node) + ".json"
        doc = json.loads((ROOT / relative).read_text())
        assert type(doc.get("rc")) is int and doc["rc"] == 0
        identities = []
        idles = []
        ignored = 0
        for line in doc["stdout"].splitlines():
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            is_identity = any(key in row for key in IDENTITY_KEYS)
            is_idle = any(key in row for key in IDLE_KEYS)
            if not (is_identity or is_idle):
                ignored += 1
                continue
            assert not (is_identity and is_idle), "ambiguous identity/idle record"
            if is_identity:
                assert all(key in row for key in IDENTITY_KEYS), "incomplete identity record"
                assert row.get("node") == node, "wrong-node identity record"
                identities.append(row)
            else:
                assert all(key in row for key in IDLE_KEYS), "incomplete idle record"
                assert row.get("node") == node, "wrong-node idle record"
                idles.append(row)
        assert len(identities) == 1, "exactly one identity assertion record per node"
        assert len(idles) == 1, "exactly one final idle record per node"
        identity, idle = identities[0], idles[0]
        assert type(identity.get("model_requests")) is int and identity["model_requests"] == 0
        assert type(idle.get("model_requests")) is int and idle["model_requests"] == 0
        assert isinstance(identity.get("container_id"), str) and identity["container_id"], "missing identity CID"
        assert identity["container_id"] == idle.get("container_id"), "node/CID identity mismatch"
        parent_identity = identity["model_identity"]
        assert isinstance(parent_identity, dict) and parent_identity, "missing parent identity"
        assert parent_identity == idle["actual_model_identity"], "parent identity mismatch"
        assert idle["supervisor_phase"] == "MODEL_RUNNING", "supervisor is not MODEL_RUNNING"
        assert idle["provider_phase"] == "READY", "provider is not READY"
        gauges = idle["idle_gauges"]
        assert isinstance(gauges, list) and gauges, "idle gauges must be non-empty"
        assert all(type(value) in (int, float) and value == 0 for value in gauges), "idle gauges are not all zero"
        rows[str(node)] = {
            **identity,
            "actual_model_identity": idle["actual_model_identity"],
            "supervisor_phase": idle["supervisor_phase"],
            "provider_phase": idle["provider_phase"],
            "idle_gauges": gauges,
            "identity_record": identity,
            "idle_record": idle,
            "raw_stdout_sha256": hashlib.sha256(doc["stdout"].encode()).hexdigest(),
            "ignored_json_stdout_lines": ignored,
        }
    return rows


def main():
    target = ROOT / contract.CLIENT_RESULT
    assert not target.exists(), "Preserve existing client-only short audit result"
    full_short = ROOT / contract.SHORT_RESULT
    full_short_sha_before = hashlib.sha256(full_short.read_bytes()).hexdigest() if full_short.is_file() else None

    p_capacity, d_capacity = contract.check_capacity_pair()
    expected_cids = {
        str(node): (p_capacity if node in contract.P_NODES else d_capacity)["providers"][str(node)]["container_id"]
        for node in contract.ALL_NODES
    }
    startup = json.loads((ROOT / contract.STARTUP_RESULT).read_text())
    assert startup["pass"] is True
    assert startup["all64_actual_SDMA_and_BF16_layout_verified"] is True
    assert startup["all32_D_nativeOps_fullmesh_and_targetGraph_retained"] is True
    assert startup["all32_P_native_IO_and_existing_MC_mapping_reuse_verified"] is True
    assert startup["runtime_CIDs"] == expected_cids

    run = ROOT / "runs" / contract.SHORT_RUN
    receipt = json.loads((run / "receipt.json").read_text())
    assert type(receipt.get("rc")) is int and receipt["rc"] == 0
    assert receipt["logical_client_cases_planned"] == contract.SHORT_LOGICAL
    assert receipt["backend_model_calls_planned"] == contract.SHORT_BACKEND
    assert receipt["successful_output_tokens_if_all_pass"] == contract.SHORT_OUTPUT_IF_ALL
    assert type(receipt.get("formal_requests")) is int and receipt["formal_requests"] == 0
    assert type(receipt.get("retries")) is int and receipt["retries"] == 0
    assert len(receipt["prechecks"]) == 4
    assert set(receipt["prechecks"]) == {str(node) for node in contract.ALL_NODES}
    assert all(type(value) is int and value == 0 for value in receipt["prechecks"].values())
    prechecks = precheck_identity_rows(run)
    assert {key: row["container_id"] for key, row in prechecks.items()} == expected_cids

    raw_relative = "runs/" + contract.SHORT_RUN + "/node-34.json"
    wire = json.loads((ROOT / raw_relative).read_text())
    assert type(wire.get("rc")) is int and wire["rc"] == 0
    frames = [
        line.split("=", 1)[1] for line in wire["stdout"].splitlines()
        if line.startswith("GLM53_SG_MATRIX_ARTIFACTS=")
    ]
    assert len(frames) == 1
    result = json.loads(frames[0])
    assert type(result.get("client_rc")) is int and result["client_rc"] == 0
    report = result["result"]
    assert report["pass"] and report["offered_concurrency"] == 32
    assert len(report["cases"]) == 32
    assert len(report["backend_attempts"]) == report["model_requests_attempted"] == 64
    assert report["successful_inferences"] == 64
    assert report["actual_successful_output_tokens"] == report["successful_output_tokens"] == 19232
    assert report["input_tokens"] == 512 and report["input_sha256_le_u32"] == contract.INPUT512_SHA
    assert report["D_output_tokens"] == 600 and report["D_min_tokens"] == 600
    identifiers = set()
    dp = collections.Counter()
    for row in report["cases"]:
        assert row["pass"]
        prefill = row["prefill"]
        decode = row["decode"]
        match = re.search(r"-p([0-9]{2})-functional\.P$", prefill["rid"])
        assert match
        prefix_id = int(match[1])
        assert prefix_id not in identifiers and prefix_id in range(32)
        identifiers.add(prefix_id)
        assert prefill["rid"][:-1] == decode["rid"][:-1]
        assert row["P_DP"] == prefill["global_dp"] == prefix_id % 4
        assert row["D_DP"] == decode["global_dp"] == prefix_id % 16
        dp[prefix_id % 16] += 1
        for response, outputs in [(prefill, 1), (decode, 600)]:
            assert response["pass"] and response["status"] == 200
            assert response["terminal_monotonic_ns"] >= response["dispatch_monotonic_ns"]
            native = response["response"]
            assert native["usage"]["prompt_tokens"] == 512
            assert native["usage"]["completion_tokens"] == outputs
            assert native["choices"][0]["finish_reason"] == "length"
            assert native["choices"][0]["text"]
            assert hashlib.sha256(native["choices"][0]["text"].encode()).hexdigest() == response["text_sha256"]
            assert response["retry"] is False
            assert isinstance(response["vllm_short_reference_text_equal"], bool)
            assert "not" in response["vllm_short_reference_text_scope"].lower()
            assert response["vllm_short_reference_text_scope"].lower().find("whole") >= 0
            assert isinstance(response["vllm_short_reference_prefix16_text_equal"], bool)
            assert "diagnostic" in response["vllm_short_reference_prefix16_scope"].lower()
    assert identifiers == set(range(32))
    assert dp == collections.Counter({index: 2 for index in range(16)})

    startup_review_relative = contract.STARTUP_CAPTURE_REVIEW
    startup_review = contract.read_json(ROOT / startup_review_relative)
    assert startup_review.get("project_id") == contract.PROJECT
    assert str(startup_review.get("decision", "")).startswith("APPROVED")
    assert startup_review.get("model_requests") in (0, None)
    assert startup_review.get("startup_qualified") is not False
    assert startup_review.get("runtime_model_dispatch_authorized") is not False
    short_review_relative = contract.SHORT_REVIEW
    startup_result_sha256 = sha(contract.STARTUP_RESULT)
    startup_receipt_relative = "runs/" + contract.STARTUP_CAPTURE + "/receipt.json"
    startup_receipt_sha256 = sha(startup_receipt_relative)
    runner = load_short_runner()
    expected_host_hashes = {
        name: hashlib.sha256(code.encode()).hexdigest() for name, code in runner.plan().items()
    }
    contract.inference_review_check(
        ROOT / short_review_relative,
        expected_host_hashes,
        launcher_sha256=source_sha(SHORT_RUNNER),
        files_sha256=runner.FROZEN,
        logical_client_cases=contract.SHORT_LOGICAL,
        backend_model_calls=contract.SHORT_BACKEND,
        successful_output_tokens_if_all_pass=contract.SHORT_OUTPUT_IF_ALL,
        provenance={
            "startup_result_sha256": startup_result_sha256,
            "startup_capture_receipt_sha256": startup_receipt_sha256,
        },
    )

    sources = {
        contract.STARTUP_RESULT: startup_result_sha256,
        startup_receipt_relative: startup_receipt_sha256,
        "runs/" + contract.SHORT_RUN + "/receipt.json": sha("runs/" + contract.SHORT_RUN + "/receipt.json"),
        raw_relative: sha(raw_relative),
        short_review_relative: sha(short_review_relative),
        startup_review_relative: sha(startup_review_relative),
        contract.P_CAPACITY: contract.P_CAPACITY_SHA,
        contract.D_CAPACITY: contract.D_CAPACITY_SHA,
        SHORT_RUNNER: source_sha(SHORT_RUNNER),
        contract.SHORT_CLIENT_AUDIT: source_sha(contract.SHORT_CLIENT_AUDIT),
    }
    for node in contract.ALL_NODES:
        relative = "runs/" + contract.SHORT_RUN + "/precheck-" + str(node) + ".json"
        sources[relative] = sha(relative)
    for item in contract.old_source_provenance().values():
        sources[item["path"]] = item["sha256"]

    output = {
        "project_id": "glm53",
        "framework": "sglang",
        "status": "L17_SHORT_CLIENT_COUNTS_AND_RESPONSE_ONLY",
        "label": contract.CLIENT_LABEL,
        "pass": True,
        "native_qualified": False,
        "formal_ready": False,
        "all32_bs2_claim": False,
        "graph_delta_claim": False,
        "sdma_claim": False,
        "qualification": "CLIENT_COUNTS_AND_RESPONSE_ONLY; no native, bs2, Graph, SDMA or cache proof is claimed here.",
        "scope": (
            "Actual short client receipt/wire and startup current CIDs only. "
            "The later full short native audit still requires all32 actual bs2 draft-DECODE rows and per-rank Graph deltas."
        ),
        "logical_pairs": contract.SHORT_LOGICAL,
        "backend_attempts": contract.SHORT_BACKEND,
        "successful_backend_inferences": contract.SHORT_BACKEND,
        "successful_output_tokens": contract.SHORT_OUTPUT_IF_ALL,
        "D_output_tokens_per_case": 600,
        "P_output_tokens_per_case": 1,
        "D_DP_logical_cases": dict(dp),
        "runtime_CIDs": expected_cids,
        "precheck_rcs": {str(node): receipt["prechecks"][str(node)] for node in contract.ALL_NODES},
        "precheck_wire_evidence": {
            str(node): {
                "identity_record": prechecks[str(node)]["identity_record"],
                "idle_record": prechecks[str(node)]["idle_record"],
                "raw_stdout_sha256": prechecks[str(node)]["raw_stdout_sha256"],
                "ignored_json_stdout_lines": prechecks[str(node)]["ignored_json_stdout_lines"],
            }
            for node in contract.ALL_NODES
        },
        "startup_current_CIDs_verified": True,
        "short_review_sha256": sha(short_review_relative),
        "startup_review_sha256": sha(startup_review_relative),
        "startup_result_sha256": startup_result_sha256,
        "startup_capture_receipt_sha256": startup_receipt_sha256,
        "raw_short_wire": raw_relative,
        "source_evidence_sha256": sources,
        "external_model_requests": contract.SHORT_BACKEND,
        "formal_requests": 0,
        "retries": 0,
        "new_profiles": 0,
    }
    target.write_text(json.dumps(output, indent=2) + "\n")
    full_short_sha_after = hashlib.sha256(full_short.read_bytes()).hexdigest() if full_short.is_file() else None
    assert full_short_sha_before == full_short_sha_after, "Full short native result must not be overwritten"
    print(json.dumps({
        "pass": True,
        "label": contract.CLIENT_LABEL,
        "logical_pairs": contract.SHORT_LOGICAL,
        "backend_attempts": contract.SHORT_BACKEND,
        "successful_output_tokens": contract.SHORT_OUTPUT_IF_ALL,
        "native_qualified": False,
        "formal_ready": False,
        "output": contract.CLIENT_RESULT,
        "sha256": sha(contract.CLIENT_RESULT),
    }))


if __name__ == "__main__":
    main()
