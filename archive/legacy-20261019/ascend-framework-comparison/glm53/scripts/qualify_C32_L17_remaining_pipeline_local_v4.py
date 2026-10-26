"""DRAFT_PENDING_PRIMARY_REVIEW: local-only v4 qualification of identity-v2 wiring.

Mechanically qualifies the still-unexecuted L17 drafts after the actual identity
v1 staged-loop type defect, where the path->SHA string mapping was read as
``item['sha256']``:

1. the active pipeline contract pins mixed identity v2 and the v2 diagnostic
   startup capture/review/result;
2. post-functional, formal-native and independent-metrics wrappers pin capture
   core v2;
3. the formal settings identity helper/SHA and startup qualification path follow
   the same v2 sources;
4. the old observation entry and the failed post-timeout v1 entry are recorded
   as frozen/superseded and are not treated as the current entry;
5. the full local stdlib entry fixtures, actual codec encoder/decoder fixtures,
   identity rendering and staged-loop typed positive/negative fixtures run,
   including the v1-loop TypeError regression;
6. capture core v2 startup plan hashes are re-derived locally and must still
   equal the already-approved review-v2 host hashes, proving the live diagnostic
   capture is unaffected by the contract wire change;
7. the external v2 startup capture (nodes 36/41 rc=1, 13 native errors pending
   primary localization) is never consumed as startup evidence, and this script
   never declares startup-pass.

This script makes no SSH/network/Docker/NPU/SDK/model/Git call, starts no
capture or client, reads no credential/provider file, writes no APPROVED review
and preserves all v1/v2/v3 results, receipts, verification files and frozen
sources.  Synthetic fixtures are local qualification only; they are not native,
model or formal evidence.
"""
import ast
import builtins
import collections
import contextlib
import copy
import gzip
import hashlib
import importlib.util
import io
import json
import py_compile
import re
import symtable
import tempfile
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v1 as contract

ROOT = Path(__file__).resolve().parents[1]
CWD = ROOT / "runs" / "20261019T-L17-pipeline-identity-v2-wire-local-v1"
RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v4.json"
V1_RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v1.json"
V2_RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v2.json"
V3_RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v3.json"
V1_CWD = ROOT / "runs/20261019T-L17-functional-formal-local-draft-executor-v1"
V2_CWD = ROOT / "runs/20261019T-L17-pipeline-wire-fix-local-v1"
V3_CWD = ROOT / "runs/20261019T-L17-pipeline-entry-fix-local-v1"
VERIFICATION = CWD / "verification.json"
HOST_HASHES = CWD / "host-code-hashes.json"
ENTRY_GATE = CWD / "entry-gate-qualification.json"
CODEC_GATE = CWD / "codec-identity-qualification.json"
STAGED_GATE = CWD / "identity-staged-loop-qualification.json"
DEPENDENCY_GATE = CWD / "dependency-graph-qualification.json"
FIXTURE_SUMMARY = CWD / "fixture-summary.json"
RECEIPT = CWD / "receipt.json"

CONTRACT = "scripts/glm53_C32_L17_pipeline_contract_v1.py"
CAPTURE = "scripts/glm53_C32_L17_native_capture_v2.py"
CAPTURE_V1 = "scripts/glm53_C32_L17_native_capture_v1.py"
IDENTITY = "scripts/glm53_C32_L17_mixed_model_identity_v2.py"
IDENTITY_V1 = "scripts/glm53_C32_L17_mixed_model_identity_v1.py"
STARTUP_ENTRY = "scripts/inspect_C32_L17_post_timeout_native_startup_v2.py"
STARTUP_ENTRY_V1_FROZEN = "scripts/inspect_C32_L17_native_startup_v1.py"
STARTUP_ENTRY_V1_FAILED = "scripts/inspect_C32_L17_post_timeout_native_startup_v1.py"
STARTUP_CAPTURE_V1 = "20261019T-L17-sglang-native-startup-capture-v1"
STARTUP_REVIEW_V1 = "results/C32-L17-native-startup-capture-review-v1.json"
POST_WRAPPER = "scripts/inspect_C32_L17_native_post_functional_v1.py"
FORMAL_WRAPPER = "scripts/glm53_C32_L17_formal_native_capture_v1.py"
METRICS_INSPECTOR = "scripts/inspect_C32_L17_independent_formal_metrics_v1.py"
PROBE = "scripts/probe_C32_L17_native_pd_payload_v1.py"
SHORT_RUNNER = "scripts/run_C32_L17_native_P_functional_v1.py"
LONG_CLIENT = "scripts/glm53_C32_L17_native_long_client_v1.py"
SEMANTIC_PROBE = "scripts/glm53_C32_L17_native_semantic_probe_v1.py"
LONG_RUNNER = "scripts/run_C32_L17_native_long_v1.py"
STARTUP_AUDIT = "scripts/audit_C32_L17_native_startup_local_v1.py"
SHORT_CLIENT_AUDIT = "scripts/audit_C32_L17_native_short_client_local_v1.py"
FUNCTIONAL_AUDIT = "scripts/audit_C32_L17_native_functional_local_v1.py"
READINESS_AUDIT = "scripts/audit_C32_L17_native_readiness_local_v1.py"
FORMAL_RUNNER = "scripts/run_C32_L17_native_formal_v1.py"
FORMAL_CONTROLLER = "scripts/glm53_C32_L17_formal_resource_controller_v1.py"
FORMAL_CLIENT_AUDIT = "scripts/audit_C32_L17_native_formal_client_local_v1.py"
FORMAL_NATIVE_AUDIT = "scripts/audit_C32_L17_native_formal_native_local_v1.py"
QUALIFIER = "scripts/qualify_C32_L17_remaining_pipeline_local_v4.py"
FORMAL_SETTINGS = "config/C32-L17-native-runtime-v1/formal-C32-native-v1.json"
D_START_RECEIPT = "runs/20261019T-L17-D-models-and-observe-v1/receipt.json"

CHANGED_FILES = [CONTRACT, POST_WRAPPER, FORMAL_WRAPPER, METRICS_INSPECTOR, FORMAL_SETTINGS]
NEW_FILES = [
    CONTRACT,
    CAPTURE,
    IDENTITY,
    STARTUP_ENTRY,
    POST_WRAPPER,
    FORMAL_WRAPPER,
    METRICS_INSPECTOR,
    PROBE,
    SHORT_RUNNER,
    LONG_CLIENT,
    SEMANTIC_PROBE,
    LONG_RUNNER,
    STARTUP_AUDIT,
    SHORT_CLIENT_AUDIT,
    FUNCTIONAL_AUDIT,
    READINESS_AUDIT,
    FORMAL_RUNNER,
    FORMAL_CONTROLLER,
    FORMAL_CLIENT_AUDIT,
    FORMAL_NATIVE_AUDIT,
    QUALIFIER,
    FORMAL_SETTINGS,
]
FROZEN_REFERENCES = [
    CAPTURE_V1,
    IDENTITY_V1,
    STARTUP_ENTRY_V1_FROZEN,
    STARTUP_ENTRY_V1_FAILED,
]
ACTIVE_CONSUMERS = [
    SHORT_RUNNER,
    LONG_RUNNER,
    POST_WRAPPER,
    FORMAL_WRAPPER,
    METRICS_INSPECTOR,
    STARTUP_AUDIT,
    SHORT_CLIENT_AUDIT,
    FUNCTIONAL_AUDIT,
    READINESS_AUDIT,
    FORMAL_RUNNER,
    FORMAL_CONTROLLER,
    FORMAL_CLIENT_AUDIT,
    FORMAL_NATIVE_AUDIT,
    STARTUP_ENTRY,
    FORMAL_SETTINGS,
]

PIPELINE_ORDER = [
    "startup_capture",
    "startup_audit",
    "short_client",
    "short_client_audit",
    "long_client",
    "postfunctional_capture",
    "full_short_native_audit",
    "readiness",
    "final_functional_acceptance",
    "formal",
]
PIPELINE_EDGES = {
    "startup_audit": ["startup_capture"],
    "short_client": ["startup_audit"],
    "short_client_audit": ["short_client", "startup_audit"],
    "long_client": ["short_client_audit", "startup_audit"],
    "postfunctional_capture": ["long_client", "short_client_audit", "startup_audit"],
    "full_short_native_audit": ["short_client", "postfunctional_capture", "startup_audit"],
    "readiness": ["full_short_native_audit", "long_client", "postfunctional_capture", "startup_audit"],
    "final_functional_acceptance": ["readiness"],
    "formal": ["final_functional_acceptance", "readiness"],
}


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def import_file(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_tree(path):
    return {
        str(item.relative_to(ROOT)): sha_path(item)
        for item in sorted(Path(path).rglob("*")) if item.is_file()
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def symtable_module_names(code, label):
    table = symtable.symtable(code, label, "exec")
    return {symbol.get_name() for symbol in table.get_symbols() if symbol.is_assigned() or symbol.is_imported()}


def unbound_globals(code, label, extra_names=()):
    table = symtable.symtable(code, label, "exec")
    module_names = symtable_module_names(code, label) | set(extra_names)
    unresolved = []

    def walk(current):
        for symbol in current.get_symbols():
            if symbol.is_global() and symbol.get_name() not in module_names and not hasattr(builtins, symbol.get_name()):
                unresolved.append(symbol.get_name())
        for child in current.get_children():
            walk(child)

    walk(table)
    return sorted(set(unresolved))


def contract_attrs(relative):
    tree = ast.parse((ROOT / relative).read_text())
    return sorted({
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "contract"
    })


def identity_literal(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "compile":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        text = node.args[0].value
        if isinstance(text, str) and "NODE = " in text and "CID = " in text:
            return text
    return None


def metrics_core_pin():
    tree = ast.parse((ROOT / METRICS_INSPECTOR).read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "load":
            continue
        if len(node.args) < 3:
            continue
        relative, digest = node.args[1], node.args[2]
        if isinstance(relative, ast.Constant) and relative.value == CAPTURE:
            found.append(digest.value if isinstance(digest, ast.Constant) else None)
    assert len(found) == 1, found
    return found[0]


def cids():
    p_capacity, d_capacity = contract.check_capacity_pair()
    return {
        str(node): (p_capacity if node in contract.P_NODES else d_capacity)["providers"][str(node)]["container_id"]
        for node in contract.ALL_NODES
    }


CID_MAP = cids()


def startup_fixture():
    return {
        "project_id": "glm53",
        "fixture": True,
        "not_actual_evidence": True,
        "pass": True,
        "all64_actual_SDMA_and_BF16_layout_verified": True,
        "all32_D_nativeOps_fullmesh_and_targetGraph_retained": True,
        "all32_P_native_IO_and_existing_MC_mapping_reuse_verified": True,
        "D_KV_capacity_floor": contract.D_KV_FLOOR,
        "runtime_CIDs": dict(CID_MAP),
    }


def startup_receipt_fixture():
    return {
        "project_id": "glm53",
        "fixture": True,
        "pass": True,
        "mode": "startup",
        "nodes": {"34": 0, "35": 0, "36": 0, "41": 0},
    }


def startup_attempt_state(capture_id):
    relative = "runs/" + capture_id
    run = ROOT / relative
    receipt_path = run / "receipt.json"
    state = {
        "capture_id": capture_id,
        "path": relative,
        "present": run.is_dir(),
        "is_contract_startup_capture": capture_id == contract.STARTUP_CAPTURE,
        "receipt_sha256": None,
        "nodes": {},
        "pass": False,
        "all_four_rc0": False,
        "valid_actual_evidence": False,
        "startup_pass_declared": False,
        "consumed_as_evidence": False,
        "created_by_this_run": False,
    }
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        nodes = receipt.get("nodes") if isinstance(receipt.get("nodes"), dict) else {}
        all_rc0 = len(nodes) == 4 and all(type(value) is int and value == 0 for value in nodes.values())
        state.update({
            "receipt_sha256": sha_path(receipt_path),
            "nodes": nodes,
            "pass": receipt.get("pass") is True,
            "all_four_rc0": all_rc0,
            "valid_actual_evidence": receipt.get("pass") is True and all_rc0,
        })
    if state["present"] and not state["valid_actual_evidence"]:
        state["reason"] = "present but not qualified: receipt has no passing startup result and/or every node rc is not 0"
    return state


def startup_review_state(relative):
    path = ROOT / relative
    state = {
        "path": relative,
        "present": path.is_file(),
        "sha256": sha_path(path) if path.is_file() else None,
        "decision": None,
        "startup_qualified": None,
        "runtime_model_dispatch_authorized": None,
        "valid_for_startup_gate": False,
        "startup_pass_declared": False,
        "consumed_as_evidence": False,
        "created_by_this_run": False,
    }
    if path.is_file():
        review = json.loads(path.read_text())
        state.update({
            "decision": review.get("decision"),
            "startup_qualified": review.get("startup_qualified"),
            "runtime_model_dispatch_authorized": review.get("runtime_model_dispatch_authorized"),
        })
        state["valid_for_startup_gate"] = (
            review.get("decision") == "APPROVED"
            and review.get("startup_qualified") is True
            and review.get("runtime_model_dispatch_authorized") is True
        )
        if not state["valid_for_startup_gate"]:
            state["reason"] = "diagnostic/limited approval only; does not qualify startup"
    return state


def d_start_state():
    path = ROOT / D_START_RECEIPT
    state = {
        "path": D_START_RECEIPT,
        "present": path.is_file(),
        "sha256": sha_path(path) if path.is_file() else None,
        "phase": None,
        "pass": False,
        "capacity_sha256": None,
        "starts": {},
        "observe": {},
        "native_startup_summaries_present": False,
        "external_model_requests": None,
        "cleanup_or_rollback_executed": None,
        "runtime_CIDs": {},
    }
    if path.is_file():
        row = json.loads(path.read_text())
        state.update({
            "phase": row.get("phase"),
            "pass": row.get("pass") is True,
            "capacity_sha256": row.get("capacity_sha256"),
            "starts": row.get("starts") if isinstance(row.get("starts"), dict) else {},
            "observe": row.get("observe") if isinstance(row.get("observe"), dict) else {},
            "native_startup_summaries_present": isinstance(row.get("native_startup_summaries"), dict),
            "external_model_requests": row.get("external_model_requests"),
            "cleanup_or_rollback_executed": row.get("cleanup_or_rollback_executed"),
            "runtime_CIDs": row.get("runtime_CIDs") if isinstance(row.get("runtime_CIDs"), dict) else {},
        })
    return state


def startup_receipt_gate(row):
    """Local copy of the frozen coordinator receipt schema gate.

    Reads fields only; it neither imports nor executes the superseded startup
    entry.  Bool is not accepted as an rc.
    """
    if not isinstance(row, dict):
        return False
    if row.get("project_id") != "glm53":
        return False
    if row.get("pass") is not True or row.get("phase") != "STARTUP_OBSERVED_AWAITING_INDEPENDENT_CAPTURE_AND_FUNCTIONAL":
        return False
    if row.get("capacity_sha256") != contract.D_CAPACITY_SHA:
        return False
    starts = row.get("starts")
    if not isinstance(starts, dict) or set(starts) != {"start-36", "start-41"}:
        return False
    if not all(type(starts.get(key)) is int and starts[key] == 0 for key in starts):
        return False
    observe = row.get("observe")
    if not isinstance(observe, dict) or set(observe) != {"observe-36", "observe-41"}:
        return False
    if not all(type(observe.get(key)) is int and observe[key] == 0 for key in observe):
        return False
    summaries = row.get("native_startup_summaries")
    if not isinstance(summaries, dict) or set(summaries) != set(observe):
        return False
    for key in observe:
        summary = summaries[key]
        if not isinstance(summary, dict):
            return False
        if summary.get("startup_pass") is not True or summary.get("formal_ready") is not False:
            return False
    if type(row.get("external_model_requests")) is not int or row["external_model_requests"] != 0:
        return False
    if row.get("cleanup_or_rollback_executed") is not False:
        return False
    if row.get("runtime_CIDs") != CID_MAP:
        return False
    return True


def short_review_fixture(expected_host_hashes, runner, temp):
    startup_path = temp / contract.STARTUP_RESULT
    receipt_path = temp / ("runs/" + contract.STARTUP_CAPTURE + "/receipt.json")
    return {
        "project_id": "glm53",
        "fixture": True,
        "not_actual_evidence": True,
        "decision": "APPROVED",
        "runtime_authorized": True,
        "zero_retries": True,
        "retries": 0,
        "new_profiles": 0,
        "formal_requests": 0,
        "logical_client_cases": contract.SHORT_LOGICAL,
        "backend_model_calls": contract.SHORT_BACKEND,
        "successful_output_tokens_if_all_pass": contract.SHORT_OUTPUT_IF_ALL,
        "launcher_sha256": sha_path(ROOT / SHORT_RUNNER),
        "files_sha256": dict(runner.FROZEN),
        "host_codes_sha256": dict(expected_host_hashes),
        "p_capacity_sha256": contract.P_CAPACITY_SHA,
        "d_capacity_sha256": contract.D_CAPACITY_SHA,
        "startup_result_sha256": sha_path(startup_path),
        "startup_capture_receipt_sha256": sha_path(receipt_path),
    }


def long_review_fixture(expected_host_hashes, runner, temp):
    startup_path = temp / contract.STARTUP_RESULT
    client_path = temp / contract.CLIENT_RESULT
    return {
        "project_id": "glm53",
        "fixture": True,
        "not_actual_evidence": True,
        "decision": "APPROVED",
        "runtime_authorized": True,
        "zero_retries": True,
        "retries": 0,
        "new_profiles": 0,
        "formal_requests": 0,
        "logical_client_cases": contract.LONG_LOGICAL,
        "backend_model_calls": contract.LONG_BACKEND,
        "successful_output_tokens_if_all_pass": contract.LONG_OUTPUT_IF_ALL,
        "launcher_sha256": sha_path(ROOT / LONG_RUNNER),
        "files_sha256": dict(runner.FROZEN),
        "host_codes_sha256": dict(expected_host_hashes),
        "p_capacity_sha256": contract.P_CAPACITY_SHA,
        "d_capacity_sha256": contract.D_CAPACITY_SHA,
        "startup_result_sha256": sha_path(startup_path),
        "short_client_result_sha256": sha_path(client_path),
        "short_client_audit_sha256": sha_path(ROOT / SHORT_CLIENT_AUDIT),
    }


def fake_transport(calls, fail_node=None):
    def invoke(node, code, timeout=100):
        calls.append({"node": node, "timeout": timeout, "client": timeout > 1000})
        if fail_node is not None and node == fail_node and timeout <= 100:
            return {"rc": 1, "stdout": "", "stderr": "synthetic precheck failure"}
        return {"rc": 0, "stdout": "{}", "stderr": ""}

    return invoke


def prepare_short_root(temp, runner, real_identity, review_mutation=None):
    runner.ROOT = temp
    runner.RUN = temp / "runs" / contract.SHORT_RUN
    runner._identity = lambda: real_identity
    write_json(temp / "runs/20261017T1400-sglang-gateway-fixture-v2/receipt.json", {
        "rc": 0, "model_requests": 0, "fixture": True, "not_actual_evidence": True,
    })
    write_json(temp / contract.STARTUP_RESULT, startup_fixture())
    write_json(temp / ("runs/" + contract.STARTUP_CAPTURE + "/receipt.json"), startup_receipt_fixture())
    expected = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in runner.plan().items()}
    review = short_review_fixture(expected, runner, temp)
    if review_mutation is not None:
        review_mutation(review)
    write_json(temp / contract.SHORT_REVIEW, review)
    return expected


def run_short_case(runner, real_identity, fail_node=None, review_mutation=None):
    with tempfile.TemporaryDirectory(prefix="l17-v3-short-") as directory:
        temp = Path(directory)
        prepare_short_root(temp, runner, real_identity, review_mutation)
        calls = []
        runner.invoke = fake_transport(calls, fail_node)
        captured = io.StringIO()
        error = None
        with contextlib.redirect_stdout(captured):
            try:
                runner.main()
            except AssertionError as exc:
                error = repr(exc)
        receipt_path = temp / "runs" / contract.SHORT_RUN / "receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
        return {
            "error": error,
            "calls": calls,
            "client_calls": [call for call in calls if call["client"]],
            "identity_calls": [call for call in calls if not call["client"]],
            "receipt": receipt,
            "run_created": (temp / "runs" / contract.SHORT_RUN).is_dir(),
            "node_34_written": (temp / "runs" / contract.SHORT_RUN / "node-34.json").is_file(),
        }


def short_entry_fixture(runner):
    original = {"ROOT": runner.ROOT, "RUN": runner.RUN, "_identity": runner._identity, "invoke": runner.invoke}
    real_identity = runner._identity()
    try:
        runner.invoke = lambda node, code, timeout=100: (_ for _ in ()).throw(AssertionError("plan must not invoke"))
        plan = runner.plan()
        assert len(plan) == 5 and set(plan) == {"identity-34", "identity-35", "identity-36", "identity-41", "client"}
        pure_plan = True

        positive = run_short_case(runner, real_identity)
        assert positive["error"] is None
        assert [call["node"] for call in positive["identity_calls"]] == [34, 35, 36, 41]
        assert len(positive["client_calls"]) == 1
        assert positive["client_calls"][0]["node"] == 34
        assert positive["client_calls"][0]["timeout"] == 3220
        assert positive["receipt"]["rc"] == 0
        assert positive["receipt"]["prechecks"] == {"34": 0, "35": 0, "36": 0, "41": 0}
        assert positive["node_34_written"]

        precheck_failures = {}
        for index, node in enumerate([34, 35, 36, 41]):
            case = run_short_case(runner, real_identity, fail_node=node)
            assert case["error"] is not None
            assert len(case["client_calls"]) == 0, node
            assert len(case["identity_calls"]) == index + 1, (node, case["identity_calls"])
            assert case["receipt"]["rc"] is None
            assert case["receipt"]["prechecks"][str(node)] == 1
            assert not case["node_34_written"]
            precheck_failures[str(node)] = {
                "client_calls": 0,
                "identity_calls": len(case["identity_calls"]),
                "receipt_rc": case["receipt"]["rc"],
                "client_started": False,
            }

        review_mutations = {
            "decision_denied": lambda row: row.__setitem__("decision", "DENIED"),
            "decision_missing": lambda row: row.pop("decision"),
            "runtime_authorized_false": lambda row: row.__setitem__("runtime_authorized", False),
            "runtime_authorized_string": lambda row: row.__setitem__("runtime_authorized", "true"),
            "zero_retries_false": lambda row: row.__setitem__("zero_retries", False),
            "retries_one": lambda row: row.__setitem__("retries", 1),
            "new_profiles_one": lambda row: row.__setitem__("new_profiles", 1),
            "formal_requests_one": lambda row: row.__setitem__("formal_requests", 1),
            "logical_31": lambda row: row.__setitem__("logical_client_cases", 31),
            "backend_63": lambda row: row.__setitem__("backend_model_calls", 63),
            "output_19231": lambda row: row.__setitem__("successful_output_tokens_if_all_pass", 19231),
            "launcher_wrong": lambda row: row.__setitem__("launcher_sha256", "0" * 64),
            "files_wrong": lambda row: row["files_sha256"].__setitem__(
                "glm53_sglang_pd_gateway_v2.py", "0" * 64
            ),
            "host_wrong": lambda row: row["host_codes_sha256"].__setitem__(
                "identity-34", "0" * 64
            ),
            "p_capacity_wrong": lambda row: row.__setitem__("p_capacity_sha256", "0" * 64),
            "d_capacity_wrong": lambda row: row.__setitem__("d_capacity_sha256", "0" * 64),
            "startup_result_wrong": lambda row: row.__setitem__("startup_result_sha256", "0" * 64),
            "startup_receipt_wrong": lambda row: row.__setitem__("startup_capture_receipt_sha256", "0" * 64),
        }
        review_failures = {}
        for name, mutation in review_mutations.items():
            case = run_short_case(runner, real_identity, review_mutation=mutation)
            assert case["error"] is not None, name
            assert case["calls"] == [], (name, case["calls"])
            assert case["run_created"] is False, name
            review_failures[name] = {"rejected": True, "invoke_calls": 0, "run_created": False}

        return {
            "module": SHORT_RUNNER,
            "plan": {"entries": sorted(plan), "pure_local": pure_plan, "invoke_calls": 0},
            "positive": {
                "identity_calls": [call["node"] for call in positive["identity_calls"]],
                "identity_before_client": True,
                "client_calls": positive["client_calls"],
                "client_count": len(positive["client_calls"]),
                "receipt_rc": positive["receipt"]["rc"],
                "prechecks": positive["receipt"]["prechecks"],
                "node_34_written": positive["node_34_written"],
            },
            "precheck_failure_cases": precheck_failures,
            "precheck_failure_count": len(precheck_failures),
            "review_failure_cases": review_failures,
            "review_failure_count": len(review_failures),
            "no_real_ssh": True,
            "verdict": "PASS",
        }
    finally:
        runner.ROOT = original["ROOT"]
        runner.RUN = original["RUN"]
        runner._identity = original["_identity"]
        runner.invoke = original["invoke"]


def prepare_long_root(temp, runner, real_identity, client_result, review_mutation=None):
    runner.ROOT = temp
    runner.RUN = temp / "runs" / contract.LONG_RUN
    runner._identity = lambda: real_identity
    write_json(temp / contract.STARTUP_RESULT, startup_fixture())
    write_json(temp / contract.CLIENT_RESULT, client_result)
    expected = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in runner.plan().items()}
    review = long_review_fixture(expected, runner, temp)
    if review_mutation is not None:
        review_mutation(review)
    write_json(temp / contract.LONG_REVIEW, review)
    return expected


def run_long_case(runner, real_identity, client_result, fail_node=None, review_mutation=None,
                  client_mutation=None):
    with tempfile.TemporaryDirectory(prefix="l17-v3-long-") as directory:
        temp = Path(directory)
        client = copy.deepcopy(client_result)
        if client_mutation is not None:
            client_mutation(client)
        prepare_long_root(temp, runner, real_identity, client, review_mutation)
        calls = []
        runner.invoke = fake_transport(calls, fail_node)
        captured = io.StringIO()
        error = None
        with contextlib.redirect_stdout(captured):
            try:
                runner.main()
            except AssertionError as exc:
                error = repr(exc)
        receipt_path = temp / "runs" / contract.LONG_RUN / "receipt.json"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
        return {
            "error": error,
            "calls": calls,
            "client_calls": [call for call in calls if call["client"]],
            "identity_calls": [call for call in calls if not call["client"]],
            "receipt": receipt,
            "run_created": (temp / "runs" / contract.LONG_RUN).is_dir(),
            "short_result_absent": not (temp / contract.SHORT_RESULT).exists(),
        }


def long_entry_fixture(runner, client_result):
    original = {"ROOT": runner.ROOT, "RUN": runner.RUN, "_identity": runner._identity, "invoke": runner.invoke}
    real_identity = runner._identity()
    try:
        runner.invoke = lambda node, code, timeout=100: (_ for _ in ()).throw(AssertionError("plan must not invoke"))
        plan = runner.plan()
        assert len(plan) == 5 and set(plan) == {"identity-34", "identity-35", "identity-36", "identity-41", "client"}
        pure_plan = True

        positive = run_long_case(runner, real_identity, client_result)
        assert positive["error"] is None
        assert [call["node"] for call in positive["identity_calls"]] == [34, 35, 36, 41]
        assert len(positive["client_calls"]) == 1
        assert positive["client_calls"][0]["node"] == 43
        assert positive["client_calls"][0]["timeout"] == 4120
        assert positive["receipt"]["rc"] == 0
        assert positive["receipt"]["model_precheck"] == {"34": 0, "35": 0, "36": 0, "41": 0}
        assert positive["short_result_absent"]

        precheck_failures = {}
        for index, node in enumerate([34, 35, 36, 41]):
            case = run_long_case(runner, real_identity, client_result, fail_node=node)
            assert case["error"] is not None
            assert len(case["client_calls"]) == 0, node
            assert len(case["identity_calls"]) == index + 1, (node, case["identity_calls"])
            assert case["receipt"]["rc"] is None
            assert case["receipt"]["model_precheck"][str(node)] == 1
            precheck_failures[str(node)] = {
                "client_calls": 0,
                "identity_calls": len(case["identity_calls"]),
                "receipt_rc": case["receipt"]["rc"],
                "client_started": False,
            }

        review_mutations = {
            "decision_denied": lambda row: row.__setitem__("decision", "DENIED"),
            "runtime_authorized_false": lambda row: row.__setitem__("runtime_authorized", False),
            "zero_retries_false": lambda row: row.__setitem__("zero_retries", False),
            "logical_5": lambda row: row.__setitem__("logical_client_cases", 5),
            "backend_11": lambda row: row.__setitem__("backend_model_calls", 11),
            "output_2407": lambda row: row.__setitem__("successful_output_tokens_if_all_pass", 2407),
            "launcher_wrong": lambda row: row.__setitem__("launcher_sha256", "0" * 64),
            "files_wrong": lambda row: row["files_sha256"].__setitem__(
                "glm53_C32_L17_native_long_client_v1.py", "0" * 64
            ),
            "host_wrong": lambda row: row["host_codes_sha256"].__setitem__("client", "0" * 64),
            "p_capacity_wrong": lambda row: row.__setitem__("p_capacity_sha256", "0" * 64),
            "d_capacity_wrong": lambda row: row.__setitem__("d_capacity_sha256", "0" * 64),
            "startup_result_wrong": lambda row: row.__setitem__("startup_result_sha256", "0" * 64),
            "client_result_wrong": lambda row: row.__setitem__("short_client_result_sha256", "0" * 64),
            "client_audit_wrong": lambda row: row.__setitem__("short_client_audit_sha256", "0" * 64),
        }
        review_failures = {}
        for name, mutation in review_mutations.items():
            case = run_long_case(runner, real_identity, client_result, review_mutation=mutation)
            assert case["error"] is not None, name
            assert case["calls"] == [], (name, case["calls"])
            review_failures[name] = {"rejected": True, "invoke_calls": 0}

        client_mutations = {
            "label_wrong": lambda row: row.__setitem__("label", "FULL_SHORT_NATIVE"),
            "native_qualified_true": lambda row: row.__setitem__("native_qualified", True),
            "formal_ready_true": lambda row: row.__setitem__("formal_ready", True),
            "backend_63": lambda row: row.__setitem__("successful_backend_inferences", 63),
            "output_19231": lambda row: row.__setitem__("successful_output_tokens", 19231),
            "runtime_cid_wrong": lambda row: row["runtime_CIDs"].__setitem__("34", contract.D41_CID),
        }
        client_failures = {}
        for name, mutation in client_mutations.items():
            case = run_long_case(runner, real_identity, client_result, client_mutation=mutation)
            assert case["error"] is not None, name
            assert case["calls"] == [], (name, case["calls"])
            client_failures[name] = {"rejected": True, "invoke_calls": 0}

        return {
            "module": LONG_RUNNER,
            "plan": {"entries": sorted(plan), "pure_local": pure_plan, "invoke_calls": 0},
            "positive": {
                "identity_calls": [call["node"] for call in positive["identity_calls"]],
                "identity_before_client": True,
                "client_calls": positive["client_calls"],
                "client_count": len(positive["client_calls"]),
                "receipt_rc": positive["receipt"]["rc"],
                "prechecks": positive["receipt"]["model_precheck"],
                "short_result_absent": positive["short_result_absent"],
            },
            "precheck_failure_cases": precheck_failures,
            "precheck_failure_count": len(precheck_failures),
            "review_failure_cases": review_failures,
            "review_failure_count": len(review_failures),
            "client_result_failure_cases": client_failures,
            "client_result_failure_count": len(client_failures),
            "no_real_ssh": True,
            "verdict": "PASS",
        }
    finally:
        runner.ROOT = original["ROOT"]
        runner.RUN = original["RUN"]
        runner._identity = original["_identity"]
        runner.invoke = original["invoke"]


def synthetic_short_report():
    cases = []
    for prefix_id in range(32):
        base = "glm53-l17-fixture-case%d" % prefix_id
        prefill_text = "P"
        decode_text = "D" * 600
        cases.append({
            "pass": True,
            "prefill": {
                "rid": base + "-p%02d-functional.P" % prefix_id,
                "global_dp": prefix_id % 4,
                "pass": True,
                "status": 200,
                "dispatch_monotonic_ns": 1000,
                "terminal_monotonic_ns": 2000,
                "response": {
                    "usage": {"prompt_tokens": 512, "completion_tokens": 1},
                    "choices": [{"finish_reason": "length", "text": prefill_text}],
                },
                "text_sha256": hashlib.sha256(prefill_text.encode()).hexdigest(),
                "retry": False,
                "vllm_short_reference_text_equal": False,
                "vllm_short_reference_text_scope": "diagnostic only; not whole output equality",
                "vllm_short_reference_prefix16_text_equal": False,
                "vllm_short_reference_prefix16_scope": "diagnostic prefix16 only",
            },
            "decode": {
                "rid": base + "-p%02d-functional.D" % prefix_id,
                "global_dp": prefix_id % 16,
                "pass": True,
                "status": 200,
                "dispatch_monotonic_ns": 3000,
                "terminal_monotonic_ns": 4000,
                "response": {
                    "usage": {"prompt_tokens": 512, "completion_tokens": 600},
                    "choices": [{"finish_reason": "length", "text": decode_text}],
                },
                "text_sha256": hashlib.sha256(decode_text.encode()).hexdigest(),
                "retry": False,
                "vllm_short_reference_text_equal": False,
                "vllm_short_reference_text_scope": "diagnostic only; not whole output equality",
                "vllm_short_reference_prefix16_text_equal": False,
                "vllm_short_reference_prefix16_scope": "diagnostic prefix16 only",
            },
            "P_DP": prefix_id % 4,
            "D_DP": prefix_id % 16,
        })
    return {
        "pass": True,
        "offered_concurrency": 32,
        "cases": cases,
        "backend_attempts": [{"logical": index // 2} for index in range(64)],
        "model_requests_attempted": 64,
        "successful_inferences": 64,
        "actual_successful_output_tokens": 19232,
        "successful_output_tokens": 19232,
        "input_tokens": 512,
        "input_sha256_le_u32": contract.INPUT512_SHA,
        "D_output_tokens": 600,
        "D_min_tokens": 600,
    }


def prepare_client_audit_root(temp, audit, review_mutation=None):
    current = cids()
    audit.ROOT = temp
    write_json(temp / contract.STARTUP_RESULT, startup_fixture())
    write_json(temp / ("runs/" + contract.STARTUP_CAPTURE + "/receipt.json"), startup_receipt_fixture())
    write_json(temp / contract.STARTUP_CAPTURE_REVIEW, {
        "project_id": "glm53",
        "fixture": True,
        "not_actual_evidence": True,
        "decision": "APPROVED",
        "model_requests": 0,
    })
    report = synthetic_short_report()
    receipt = {
        "project_id": "glm53",
        "fixture": True,
        "run_id": contract.SHORT_RUN,
        "rc": 0,
        "logical_client_cases_planned": contract.SHORT_LOGICAL,
        "backend_model_calls_planned": contract.SHORT_BACKEND,
        "successful_output_tokens_if_all_pass": contract.SHORT_OUTPUT_IF_ALL,
        "prechecks": {"34": 0, "35": 0, "36": 0, "41": 0},
        "formal_requests": 0,
        "retries": 0,
    }
    write_json(temp / ("runs/" + contract.SHORT_RUN + "/receipt.json"), receipt)
    for node in contract.ALL_NODES:
        write_json(temp / ("runs/" + contract.SHORT_RUN + "/precheck-" + str(node) + ".json"), {
            "rc": 0,
            "stderr": "",
            "stdout": json.dumps({
                "node": node,
                "container_id": current[str(node)],
                "model_requests": 0,
                "fixture": True,
            }) + "\n",
        })
    wire = {
        "rc": 0,
        "stdout": "GLM53_SG_MATRIX_ARTIFACTS=" + json.dumps({
            "client_rc": 0,
            "result": report,
            "artifacts": {"fixture-report.json": {"sha256": "1" * 64, "bytes": 1}},
        }) + "\n",
        "stderr": "",
    }
    write_json(temp / ("runs/" + contract.SHORT_RUN + "/node-34.json"), wire)
    runner = import_file("l17_short_runner_client_audit_fixture", SHORT_RUNNER)
    expected = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in runner.plan().items()}
    review = short_review_fixture(expected, runner, temp)
    if review_mutation is not None:
        review_mutation(review)
    write_json(temp / contract.SHORT_REVIEW, review)
    return report, receipt


def run_client_audit_case(audit, review_mutation=None, report_mutation=None, receipt_mutation=None,
                          precheck_mutation=None, startup_mutation=None):
    with tempfile.TemporaryDirectory(prefix="l17-v3-client-audit-") as directory:
        temp = Path(directory)
        report, receipt = prepare_client_audit_root(temp, audit, review_mutation)
        if startup_mutation is not None:
            startup = json.loads((temp / contract.STARTUP_RESULT).read_text())
            startup_mutation(startup)
            write_json(temp / contract.STARTUP_RESULT, startup)
        if report_mutation is not None:
            report_mutation(report)
        if receipt_mutation is not None:
            receipt_mutation(receipt)
        if report_mutation is not None:
            wire = json.loads((temp / ("runs/" + contract.SHORT_RUN + "/node-34.json")).read_text())
            frame = json.loads(wire["stdout"].split("=", 1)[1])
            frame["result"] = report
            wire["stdout"] = "GLM53_SG_MATRIX_ARTIFACTS=" + json.dumps(frame) + "\n"
            write_json(temp / ("runs/" + contract.SHORT_RUN + "/node-34.json"), wire)
        if receipt_mutation is not None:
            write_json(temp / ("runs/" + contract.SHORT_RUN + "/receipt.json"), receipt)
        if precheck_mutation is not None:
            relative = "runs/" + contract.SHORT_RUN + "/precheck-34.json"
            precheck = json.loads((temp / relative).read_text())
            precheck_mutation(precheck)
            write_json(temp / relative, precheck)
        captured = io.StringIO()
        error = None
        with contextlib.redirect_stdout(captured):
            try:
                audit.main()
            except AssertionError as exc:
                error = repr(exc)
        target = temp / contract.CLIENT_RESULT
        output = json.loads(target.read_text()) if target.is_file() else None
        return {"error": error, "output": output, "written": target.is_file(), "stdout_tail": captured.getvalue()[-2000:]}


def client_audit_fixture(audit):
    original = {"ROOT": audit.ROOT, "PROJECT_ROOT": audit.PROJECT_ROOT}
    try:
        positive = run_client_audit_case(audit)
        assert positive["error"] is None
        output = positive["output"]
        assert output["label"] == contract.CLIENT_LABEL
        assert output["pass"] is True
        assert output["native_qualified"] is False
        assert output["formal_ready"] is False
        assert output["all32_bs2_claim"] is False
        assert output["graph_delta_claim"] is False
        assert output["sdma_claim"] is False
        assert output["logical_pairs"] == 32
        assert output["backend_attempts"] == 64
        assert output["successful_backend_inferences"] == 64
        assert output["successful_output_tokens"] == 19232
        assert output["D_DP_logical_cases"] == {str(index): 2 for index in range(16)}
        assert output["runtime_CIDs"] == cids()

        def cases_31(report):
            report["cases"].pop()

        def backend_63(report):
            report["backend_attempts"].pop()
            report["model_requests_attempted"] = 63

        def output_19231(report):
            report["successful_output_tokens"] = 19231

        def precheck_rc(receipt):
            receipt["prechecks"]["34"] = 1

        def wrong_startup(startup):
            startup["runtime_CIDs"]["34"] = contract.D41_CID

        negatives = {
            "cases_31": {"report_mutation": cases_31},
            "backend_63": {"report_mutation": backend_63},
            "output_19231": {"report_mutation": output_19231},
            "precheck_rc1": {"precheck_mutation": lambda row: row.__setitem__("rc", 1)},
            "startup_cid_wrong": {"startup_mutation": wrong_startup},
            "review_backend_63": {"review_mutation": lambda row: row.__setitem__("backend_model_calls", 63)},
            "review_decision_denied": {"review_mutation": lambda row: row.__setitem__("decision", "DENIED")},
        }
        negative_results = {}
        for name, kwargs in negatives.items():
            case = run_client_audit_case(audit, **kwargs)
            assert case["error"] is not None, name
            assert case["written"] is False, name
            negative_results[name] = {"rejected": True, "result_written": False}

        return {
            "module": SHORT_CLIENT_AUDIT,
            "positive": {
                "label": output["label"],
                "logical_pairs": output["logical_pairs"],
                "backend_attempts": output["backend_attempts"],
                "successful_output_tokens": output["successful_output_tokens"],
                "native_qualified": output["native_qualified"],
                "formal_ready": output["formal_ready"],
                "all32_bs2_claim": output["all32_bs2_claim"],
                "graph_delta_claim": output["graph_delta_claim"],
                "sdma_claim": output["sdma_claim"],
                "result_written_to_ephemeral_temp": True,
                "full_short_written": False,
            },
            "negative_cases": negative_results,
            "negative_case_count": len(negative_results),
            "full_short_result_untouched": True,
            "output": output,
            "verdict": "PASS",
        }
    finally:
        audit.ROOT = original["ROOT"]
        audit.PROJECT_ROOT = original["PROJECT_ROOT"]


class StubCore:
    def __init__(self, temp):
        self.temp = temp
        self.calls = []

    def execute(self, *args):
        self.calls.append(args)
        run = self.temp / "runs" / contract.FUNCTIONAL_CAPTURE
        run.mkdir(parents=True)
        write_json(run / "receipt.json", {
            "project_id": "glm53",
            "fixture": True,
            "pass": True,
            "mode": "postfunctional",
        })
        return {"pass": True, "mode": "postfunctional", "fixture": True}


def prepare_post_root(temp, client_result):
    write_json(temp / contract.CLIENT_RESULT, client_result)
    write_json(temp / contract.STARTUP_RESULT, startup_fixture())
    write_json(temp / ("runs/" + contract.LONG_RUN + "/receipt.json"), {
        "project_id": "glm53",
        "fixture": True,
        "rc": 0,
        "logical_client_cases_planned": contract.LONG_LOGICAL,
        "backend_model_calls_planned": contract.LONG_BACKEND,
    })
    write_json(temp / ("runs/" + contract.LONG_RUN + "/node-43.json"), {
        "rc": 0,
        "stderr": "",
        "stdout": "GLM53_LONG_ARTIFACTS=" + json.dumps({"fixture.json": {"sha256": "2" * 64, "bytes": 2, "text": "{}"}}) + "\n"
                  + "GLM53_CLIENT_EXIT=" + json.dumps({"client_rc": 0, "gates": {}}) + "\n",
    })


def post_wrapper_fixture(post_wrapper, client_result):
    original = {"ROOT": post_wrapper.ROOT, "_load_core": post_wrapper._load_core}
    original_contract_root = contract.ROOT
    try:
        with tempfile.TemporaryDirectory(prefix="l17-v3-post-") as directory:
            temp = Path(directory)
            prepare_post_root(temp, client_result)
            post_wrapper.ROOT = temp
            contract.ROOT = temp
            stub = StubCore(temp)
            post_wrapper._load_core = lambda: stub
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                rc = post_wrapper.main()
            assert rc == 0
            assert stub.calls == [(contract.FUNCTIONAL_CAPTURE, contract.FUNCTIONAL_CAPTURE_REVIEW, "postfunctional")]
            receipt = json.loads((temp / "runs" / contract.FUNCTIONAL_CAPTURE / "receipt.json").read_text())
            assert receipt["pass"] is True and receipt["mode"] == "postfunctional"
            assert "cpu43_client_artifact" in receipt
            assert not (temp / contract.SHORT_RESULT).exists()

            negatives = {}
            for name, mutate in {
                "missing_client_result": lambda root, client: (root / contract.CLIENT_RESULT).unlink(),
                "native_qualified_true": lambda root, client: write_json(
                    root / contract.CLIENT_RESULT, {**client, "native_qualified": True}
                ),
                "runtime_cids_wrong": lambda root, client: write_json(
                    root / contract.CLIENT_RESULT,
                    {**client, "runtime_CIDs": {**client["runtime_CIDs"], "34": contract.D41_CID}},
                ),
            }.items():
                with tempfile.TemporaryDirectory(prefix="l17-v3-post-negative-") as negative_dir:
                    negative_temp = Path(negative_dir)
                    prepare_post_root(negative_temp, client_result)
                    post_wrapper.ROOT = negative_temp
                    contract.ROOT = negative_temp
                    local_stub = StubCore(negative_temp)
                    post_wrapper._load_core = lambda: local_stub
                    mutate(negative_temp, client_result)
                    rejected = False
                    try:
                        with contextlib.redirect_stdout(io.StringIO()):
                            post_wrapper.main()
                    except AssertionError:
                        rejected = True
                    assert rejected, name
                    assert local_stub.calls == [], (name, local_stub.calls)
                    negatives[name] = {"rejected": True, "capture_calls": 0}

            return {
                "positive": {
                    "rc": rc,
                    "capture_calls": 1,
                    "capture_is_local_stub": True,
                    "client_result_used": True,
                    "short_result_needed": False,
                    "short_result_absent": True,
                },
                "negative_cases": negatives,
                "negative_case_count": len(negatives),
                "no_real_capture_or_client": True,
                "verdict": "PASS",
            }
    finally:
        post_wrapper.ROOT = original["ROOT"]
        post_wrapper._load_core = original["_load_core"]
        contract.ROOT = original_contract_root


def codec_identity_fixture(startup_audit):
    codec = contract.load_module("l17_v3_codec_fixture", contract.CODEC, contract.CODEC_SHA)
    out = {
        "project_id": "glm53",
        "fixture": True,
        "not_actual_evidence": True,
        "rows": [{"index": index, "value": "x" * 32} for index in range(8)],
    }
    with tempfile.TemporaryDirectory(prefix="l17-v3-codec-") as directory:
        archive = Path(directory)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            scope = {"archive": archive, "out": out, "json": json, "gzip": gzip, "hashlib": hashlib}
            exec(codec.ENCODER, scope)
        encoder_stdout = capture.getvalue()
        assert encoder_stdout.count("GLM53_NATIVE_GZIP=") == 1
        identity = {
            "project_id": "glm53",
            "node": 34,
            "role": "P",
            "selected_P_variant": "candidate",
            "selected_D_variant": None,
            "selected_variant_scope": None,
            "container_id": contract.P34_CID,
            "name": "glm53-L17-fixture-34",
            "run_id": "20261019T-L17-fixture",
            "model_identity": {"host_pid": 1234},
            "optional_extension_mounted": True,
            "staged_files": 7,
            "model_requests": 0,
            "fixture": True,
        }
        identity_line = json.dumps(identity) + "\n"
        raw_wire = {"rc": 0, "stdout": identity_line + encoder_stdout, "stderr": ""}
        decoded = codec.decode(raw_wire)
        frame = decoded["native_output_transport"]
        assert hashlib.sha256(raw_wire["stdout"].encode()).hexdigest() == frame["wire_stdout_sha256"]
        assert json.loads(decoded["stdout"]) == out
        record, evidence = startup_audit.identity_assertion_record(raw_wire, 34, contract.P34_CID, "P", frame, codec)
        assert record["container_id"] == contract.P34_CID and record["node"] == 34
        assert evidence["identity_assertion_record_count"] == 1

        positive_unrelated = {"rc": 0, "stdout": '{"message":"unrelated"}\n' + raw_wire["stdout"], "stderr": ""}
        unrelated_frame = codec.decode(positive_unrelated)["native_output_transport"]
        unrelated_record, _ = startup_audit.identity_assertion_record(
            positive_unrelated, 34, contract.P34_CID, "P", unrelated_frame, codec
        )
        assert unrelated_record["container_id"] == contract.P34_CID

        wrong_frame = copy.deepcopy(frame)
        wrong_frame["raw_sha256"] = "0" * 64
        wrong_wire_sha = copy.deepcopy(frame)
        wrong_wire_sha["wire_stdout_sha256"] = "0" * 64
        duplicate_identity = {"rc": 0, "stdout": identity_line + raw_wire["stdout"], "stderr": ""}
        wrong_cid = {**identity, "container_id": contract.D36_CID}
        wrong_node = {**identity, "node": 35}
        wrong_role = {**identity, "role": "D"}
        duplicate_wire = duplicate_identity
        wrong_cid_wire = {"rc": 0, "stdout": json.dumps(wrong_cid) + "\n" + encoder_stdout, "stderr": ""}
        wrong_node_wire = {"rc": 0, "stdout": json.dumps(wrong_node) + "\n" + encoder_stdout, "stderr": ""}
        wrong_role_wire = {"rc": 0, "stdout": json.dumps(wrong_role) + "\n" + encoder_stdout, "stderr": ""}
        negatives = {
            "duplicate_identity": (
                duplicate_wire, 34, contract.P34_CID, "P", codec.decode(duplicate_wire)["native_output_transport"]
            ),
            "wrong_cid": (
                wrong_cid_wire, 34, contract.P34_CID, "P", codec.decode(wrong_cid_wire)["native_output_transport"]
            ),
            "wrong_node": (
                wrong_node_wire, 34, contract.P34_CID, "P", codec.decode(wrong_node_wire)["native_output_transport"]
            ),
            "wrong_role": (
                wrong_role_wire, 34, contract.P34_CID, "P", codec.decode(wrong_role_wire)["native_output_transport"]
            ),
            "wire_sha_mismatch": ({"rc": 0, "stdout": raw_wire["stdout"] + " ", "stderr": ""}, 34, contract.P34_CID, "P", frame),
            "frame_raw_sha_wrong": (raw_wire, 34, contract.P34_CID, "P", wrong_frame),
            "frame_wire_sha_wrong": (raw_wire, 34, contract.P34_CID, "P", wrong_wire_sha),
            "rc_nonzero": ({"rc": 1, "stdout": raw_wire["stdout"], "stderr": ""}, 34, contract.P34_CID, "P", frame),
            "missing_frame": ({"rc": 0, "stdout": identity_line, "stderr": ""}, 34, contract.P34_CID, "P", frame),
            "duplicate_frame": ({"rc": 0, "stdout": raw_wire["stdout"] + encoder_stdout, "stderr": ""}, 34, contract.P34_CID, "P", frame),
        }
        negative_results = {}
        for name, args in negatives.items():
            rejected = False
            message = None
            try:
                startup_audit.identity_assertion_record(*args, codec)
            except (AssertionError, ValueError, KeyError, TypeError, AttributeError) as exc:
                rejected = True
                message = str(exc)[:240]
            assert rejected, name
            negative_results[name] = {"rejected": True, "error": message}
        truncated = raw_wire["stdout"]
        marker = "GLM53_NATIVE_GZIP="
        at = truncated.index(marker)
        payload = truncated[at + len(marker):]
        frame_obj = json.loads(payload)
        frame_obj["gzip_base64"] = frame_obj["gzip_base64"][:-8]
        truncated_wire = {"rc": 0, "stdout": truncated[:at] + marker + json.dumps(frame_obj) + "\n", "stderr": ""}
        rejected = False
        message = None
        try:
            codec.decode(truncated_wire)
        except (AssertionError, ValueError, KeyError, TypeError) as exc:
            rejected = True
            message = str(exc)[:240]
        assert rejected
        negative_results["truncated_base64"] = {"rejected": True, "error": message}
        packed_frame = json.loads(payload)
        packed_frame["packed_sha256"] = "0" * 64
        direct_codec_cases = {
            "codec_missing_frame": {"rc": 0, "stdout": identity_line, "stderr": ""},
            "codec_duplicate_frame": {"rc": 0, "stdout": encoder_stdout + encoder_stdout, "stderr": ""},
            "codec_packed_sha_wrong": {
                "rc": 0, "stdout": identity_line + marker + json.dumps(packed_frame) + "\n", "stderr": ""
            },
        }
        for name, wire in direct_codec_cases.items():
            rejected = False
            message = None
            try:
                codec.decode(wire)
            except (AssertionError, ValueError, KeyError, TypeError) as exc:
                rejected = True
                message = str(exc)[:240]
            assert rejected, name
            negative_results[name] = {"rejected": True, "error": message}

        return {
            "codec_path": contract.CODEC,
            "codec_sha256": contract.CODEC_SHA,
            "encoder_source_is_pinned_codec": True,
            "decoder_source_is_pinned_codec": True,
            "positive": {
                "identity_records": 1,
                "raw_stdout_sha256": frame["wire_stdout_sha256"],
                "frame_raw_bytes": frame["raw_bytes"],
                "frame_raw_sha256": frame["raw_sha256"],
                "frame_packed_bytes": frame["packed_bytes"],
                "frame_packed_sha256": frame["packed_sha256"],
                "decoded_json_matches": True,
                "unrelated_json_line_ignored": True,
            },
            "negative_cases": negative_results,
            "negative_case_count": len(negative_results),
            "verdict": "PASS",
        }


def _staged_loop_block(host):
    lines = host.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "for relative,item in STAGED_FILES.items():":
            assert index + 1 < len(lines)
            return "\n".join(lines[index:index + 2])
    raise AssertionError("staged-loop block not found")


def _run_staged_loop(block, mapping, root):
    source = "import pathlib,hashlib\n" + block
    ast.parse(source)
    scope = {
        "pathlib": __import__("pathlib"),
        "hashlib": hashlib,
        "STAGED_FILES": mapping,
        "STAGED_ROOT": str(root),
    }
    exec(compile(source, "<staged-loop>", "exec"), scope)


def _staged_failure(block, mapping, root):
    try:
        _run_staged_loop(block, mapping, root)
    except Exception as error:
        return type(error).__name__ + ": " + str(error)[:160]
    raise AssertionError("staged loop unexpectedly accepted the fixture")


def identity_staged_loop_fixture():
    """Execute the actual rendered staged-file loop with typed fixtures.

    v2 must accept a path->lowercase-SHA256-string mapping, reject tampered,
    dict-typed, uppercase and missing-path mappings, and the frozen v1 loop must
    reproduce the original TypeError on the correct string mapping.
    """
    p_capacity, d_capacity = contract.check_capacity_pair()
    identity = import_file("l17_v4_identity_v2", IDENTITY)
    identity_v1 = import_file("l17_v4_identity_v1_reference", IDENTITY_V1)
    assert sha(IDENTITY) == contract.MIXED_IDENTITY_SHA
    v2_block = _staged_loop_block(identity.HOST)
    v1_block = _staged_loop_block(identity_v1.HOST)
    assert "==item,relative" in v2_block
    assert "item['sha256']" not in v2_block
    assert "item['sha256']" in v1_block

    rendered = {}
    for node in contract.ALL_NODES:
        code = identity.render(node, p_capacity, d_capacity, "candidate")
        ast.parse(code)
        match = re.search(r"^STAGED_FILES = (.+)$", code, re.MULTILINE)
        assert match, node
        mapping = ast.literal_eval(match.group(1))
        assert isinstance(mapping, dict) and mapping
        assert all(
            isinstance(relative, str)
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            for relative, digest in mapping.items()
        ), node
        role = re.search(r"^ROLE = '([PD])'$", code, re.MULTILINE).group(1)
        cid = ast.literal_eval(re.search(r"^CID = (.+)$", code, re.MULTILINE).group(1))
        assert cid == CID_MAP[str(node)]
        assert role == ("P" if node in contract.P_NODES else "D")
        assert "item['sha256']" not in code
        rendered[str(node)] = {
            "role": role,
            "container_id": cid,
            "staged_files": len(mapping),
            "mapping_values_are_sha256_strings": True,
        }

    with tempfile.TemporaryDirectory(prefix="l17-v4-staged-") as directory:
        root = Path(directory)
        files = {"python/a.py": b"alpha\n", "config/b.json": b'{"b":1}\n'}
        for relative, raw in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        mapping = {relative: hashlib.sha256(raw).hexdigest() for relative, raw in files.items()}
        _run_staged_loop(v2_block, mapping, root)

        tampered = dict(mapping)
        first = sorted(tampered)[0]
        tampered[first] = hashlib.sha256(b"tampered").hexdigest()
        dict_typed = {relative: {"sha256": digest} for relative, digest in mapping.items()}
        uppercase = {relative: digest.upper() for relative, digest in mapping.items()}
        missing = dict(mapping)
        missing["python/missing.py"] = "0" * 64
        negatives = {
            "tampered_source": _staged_failure(v2_block, tampered, root),
            "dict_typed_mapping": _staged_failure(v2_block, dict_typed, root),
            "uppercase_sha_string": _staged_failure(v2_block, uppercase, root),
            "missing_path": _staged_failure(v2_block, missing, root),
        }
        v1_on_strings = _staged_failure(v1_block, mapping, root)
        assert v1_on_strings.startswith("TypeError"), v1_on_strings
        negatives["v1_loop_on_correct_string_mapping"] = v1_on_strings
        _run_staged_loop(v1_block, dict_typed, root)
        v1_accepts_dict = True

    return {
        "identity_path": IDENTITY,
        "identity_sha256": sha(IDENTITY),
        "identity_v1_reference_path": IDENTITY_V1,
        "identity_v1_reference_sha256": sha(IDENTITY_V1),
        "rendered_nodes": rendered,
        "rendered_count": len(rendered),
        "staged_files_header_types": "path->lowercase_SHA256_string",
        "positive": {"files": len(mapping), "accepted": True},
        "negative_cases": negatives,
        "negative_case_count": len(negatives),
        "v1_loop_rejects_correct_string_mapping": True,
        "v1_loop_accepts_dict_mapping": v1_accepts_dict,
        "v1_identity_preserved_unmodified": True,
        "verdict": "PASS",
    }


def core_pin_fixture():
    expected = sha(CAPTURE)
    assert expected == "d21502d853892799273b4fe25df8a101a0aa35031c165d8a67a6757350f2bdbd"
    pins = {}
    for name, relative in [
        ("startup_entry", STARTUP_ENTRY),
        ("post_wrapper", POST_WRAPPER),
        ("formal_wrapper", FORMAL_WRAPPER),
    ]:
        module = import_file("l17_v4_pin_" + name, relative)
        assert module.CORE_SHA == expected, name
        assert str(module.CORE).endswith(CAPTURE), name
        pins[name] = {"path": relative, "core": CAPTURE, "core_sha256": module.CORE_SHA}
    startup_entry = import_file("l17_v4_startup_entry_pin", STARTUP_ENTRY)
    assert startup_entry.REVIEW.name == contract.STARTUP_CAPTURE_REVIEW.split("/")[-1]
    assert metrics_core_pin() == expected

    pin_negatives = {}
    with tempfile.TemporaryDirectory(prefix="l17-v4-core-pin-") as directory:
        temp = Path(directory)
        for name, relative in [("post_wrapper", POST_WRAPPER), ("formal_wrapper", FORMAL_WRAPPER)]:
            module = import_file("l17_v4_pin_negative_" + name, relative)
            original_root = module.ROOT
            try:
                module.ROOT = temp
                target = temp / CAPTURE
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"tampered\n")
                rejected = False
                try:
                    module._load_core()
                except AssertionError:
                    rejected = True
                assert rejected, name
                target.write_bytes((ROOT / CAPTURE).read_bytes())
                loaded = module._load_core()
                assert loaded.MODE_CAPTURE["startup"] == contract.STARTUP_CAPTURE
                pin_negatives[name] = {"tampered_core_rejected": True, "actual_core_loaded": True}
            finally:
                module.ROOT = original_root

    return {
        "capture_core": CAPTURE,
        "capture_core_sha256": expected,
        "wrapper_pins": pins,
        "metrics_inspector_pin": metrics_core_pin(),
        "pin_negative_cases": pin_negatives,
        "all_active_pins_match": True,
        "verdict": "PASS",
    }


def dependency_fixture():
    attrs = {
        "long_runner": contract_attrs(LONG_RUNNER),
        "post_wrapper": contract_attrs(POST_WRAPPER),
        "short_client_audit": contract_attrs(SHORT_CLIENT_AUDIT),
        "full_short_audit": contract_attrs(FUNCTIONAL_AUDIT),
        "readiness": contract_attrs(READINESS_AUDIT),
        "formal_runner": contract_attrs(FORMAL_RUNNER),
    }
    assert contract.MIXED_IDENTITY == IDENTITY
    assert contract.MIXED_IDENTITY_SHA == sha(IDENTITY)
    assert contract.STARTUP_CAPTURE == "20261019T-L17-sglang-native-startup-capture-v2"
    assert contract.STARTUP_CAPTURE_REVIEW == "results/C32-L17-native-startup-capture-review-v2.json"
    assert contract.STARTUP_RESULT == "results/sglang-C32-L17-native-startup-v2.json"
    settings = json.loads((ROOT / FORMAL_SETTINGS).read_text())
    assert settings["model_identity_helper"] == contract.MIXED_IDENTITY
    assert settings["model_identity_sha256"] == contract.MIXED_IDENTITY_SHA
    assert settings["startup_qualification_path"] == contract.STARTUP_RESULT
    stale = {}
    for relative in ACTIVE_CONSUMERS:
        text = (ROOT / relative).read_text()
        hits = [
            token for token in (
                "glm53_C32_L17_mixed_model_identity_v1.py",
                "glm53_C32_L17_native_capture_v1.py",
                "C32-L17-native-startup-capture-review-v1.json",
                "sglang-C32-L17-native-startup-v1.json",
            ) if token in text
        ]
        stale[relative] = hits
    assert all(not hits for hits in stale.values()), stale
    frozen = {
        relative: {
            "sha256": sha(relative),
            "active": False,
            "reason": "frozen/superseded; preserved and not a current entry",
        }
        for relative in FROZEN_REFERENCES
    }
    startup_entry_module = import_file("l17_v4_dependency_startup_entry", STARTUP_ENTRY)
    assert startup_entry_module.CORE_SHA == sha(CAPTURE)
    assert startup_entry_module.REVIEW.name == contract.STARTUP_CAPTURE_REVIEW.split("/")[-1]
    assert "CLIENT_RESULT" in attrs["long_runner"] and "STARTUP_RESULT" in attrs["long_runner"]
    assert "SHORT_RESULT" not in attrs["long_runner"], "long runner must not require full short result"
    assert "CLIENT_RESULT" in attrs["post_wrapper"] and "STARTUP_RESULT" in attrs["post_wrapper"]
    assert "SHORT_RESULT" not in attrs["post_wrapper"], "post-functional wrapper must not require full short result"
    assert "CLIENT_RESULT" in attrs["short_client_audit"]
    assert "SHORT_RESULT" in attrs["short_client_audit"], "client audit may hash but not require full short result"
    client_source = (ROOT / SHORT_CLIENT_AUDIT).read_text()
    assert "target = ROOT / contract.CLIENT_RESULT" in client_source
    assert "full_short_sha_before == full_short_sha_after" in client_source
    assert "target = ROOT / contract.SHORT_RESULT" not in client_source
    assert "SHORT_RESULT" in attrs["full_short_audit"] and "FUNCTIONAL_CAPTURE" in attrs["full_short_audit"]
    assert "SHORT_RESULT" in attrs["readiness"] and "STARTUP_RESULT" in attrs["readiness"]
    assert "FORMAL_SETTINGS" in attrs["formal_runner"]

    positions = {name: index for index, name in enumerate(PIPELINE_ORDER)}
    edges = {}
    for node, dependencies in PIPELINE_EDGES.items():
        for dependency in dependencies:
            assert positions[dependency] < positions[node], (node, dependency)
        edges[node] = dependencies

    visiting = set()
    visited = set()

    def visit(node):
        assert node not in visiting, "dependency cycle at " + node
        if node in visited:
            return
        visiting.add(node)
        for dependency in edges.get(node, []):
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in PIPELINE_ORDER:
        visit(node)
    assert positions["short_client_audit"] < positions["long_client"]
    assert positions["long_client"] < positions["postfunctional_capture"] < positions["full_short_native_audit"]
    assert positions["full_short_native_audit"] < positions["readiness"] < positions["formal"]
    assert "full_short_native_audit" not in edges["long_client"]
    assert "full_short_native_audit" not in edges["postfunctional_capture"]
    return {
        "order": PIPELINE_ORDER,
        "edges": {key: list(value) for key, value in PIPELINE_EDGES.items()},
        "acyclic": True,
        "long_requires": ["CLIENT_RESULT", "STARTUP_RESULT"],
        "post_functional_requires": ["CLIENT_RESULT", "STARTUP_RESULT", "LONG receipt"],
        "full_short_requires": ["SHORT_RESULT", "FUNCTIONAL_CAPTURE", "all32 bs2", "Graph deltas"],
        "readiness_requires": ["SHORT_RESULT", "FUNCTIONAL_CAPTURE", "LONG", "native cache/SDMA/semantic proofs"],
        "contract_attributes": attrs,
        "identity_wire": {
            "path": contract.MIXED_IDENTITY,
            "sha256": contract.MIXED_IDENTITY_SHA,
        },
        "startup_constants": {
            "capture": contract.STARTUP_CAPTURE,
            "review": contract.STARTUP_CAPTURE_REVIEW,
            "result": contract.STARTUP_RESULT,
        },
        "formal_settings_identity_and_startup": {
            "model_identity_helper": settings["model_identity_helper"],
            "model_identity_sha256": settings["model_identity_sha256"],
            "startup_qualification_path": settings["startup_qualification_path"],
        },
        "stale_identity_references": stale,
        "stale_reference_count": sum(len(hits) for hits in stale.values()),
        "frozen_references": frozen,
        "active_consumers": ACTIVE_CONSUMERS,
        "current_startup_entry": STARTUP_ENTRY,
        "superseded_startup_entries": [STARTUP_ENTRY_V1_FROZEN, STARTUP_ENTRY_V1_FAILED],
        "verdict": "PASS",
    }



def main():
    assert not RESULT.exists(), "Preserve existing v4 result"
    assert not RECEIPT.exists(), "Preserve existing v4 receipt"
    assert not VERIFICATION.exists() and not HOST_HASHES.exists() and not ENTRY_GATE.exists()
    assert not CODEC_GATE.exists() and not STAGED_GATE.exists()
    assert not DEPENDENCY_GATE.exists() and not FIXTURE_SUMMARY.exists()
    assert V1_RESULT.is_file() and V2_RESULT.is_file() and V3_RESULT.is_file()
    v1_result_sha_before = sha_path(V1_RESULT)
    v2_result_sha_before = sha_path(V2_RESULT)
    v3_result_sha_before = sha_path(V3_RESULT)
    v3 = json.loads(V3_RESULT.read_text())
    v3_files = {relative: item["sha256"] for relative, item in v3["new_files"].items()}
    v3_files.update({relative: item["v3_sha256"] for relative, item in v3["changed_files"].items()})
    v1_tree_before = snapshot_tree(V1_CWD)
    v2_tree_before = snapshot_tree(V2_CWD)
    v3_tree_before = snapshot_tree(V3_CWD)
    protected = [
        "STATE.json",
        "HANDOFF.md",
        "scripts/start_C32_L17_selected_D_models_and_observe_v1.py",
        contract.P_CAPACITY,
        contract.D_CAPACITY,
        "diagnostics/model-profile/loops/L17-D-native-eager-live-inputs-v1.json",
        "diagnostics/model-profile/L17-primary-pipeline-entry-review-v2.md",
        str(V1_RESULT.relative_to(ROOT)),
        str(V2_RESULT.relative_to(ROOT)),
        str(V3_RESULT.relative_to(ROOT)),
    ]
    protected += sorted(str(path.relative_to(ROOT)) for path in (ROOT / "results").glob("*review*.json"))
    protected += sorted(str(path.relative_to(ROOT)) for path in (ROOT / "results").glob("*decision*.json"))
    protected_before = {relative: sha(relative) for relative in protected if (ROOT / relative).is_file()}
    old_sources_before = {item["path"]: sha(item["path"]) for item in contract.old_source_provenance().values()}
    frozen_before = {relative: sha(relative) for relative in FROZEN_REFERENCES}

    compile_results = {}
    ast_results = {}
    with tempfile.TemporaryDirectory(prefix="l17-v4-pyc-") as pyc_dir:
        for index, relative in enumerate(NEW_FILES + FROZEN_REFERENCES):
            path = ROOT / relative
            assert path.is_file(), relative
            if path.suffix == ".py":
                py_compile.compile(str(path), cfile=str(Path(pyc_dir) / ("%02d.pyc" % index)), doraise=True)
                ast.parse(path.read_text())
                compile_results[relative] = "safe_compile"
                ast_results[relative] = "ast_parse_ok"
            else:
                json.loads(path.read_text())
                compile_results[relative] = "json_parse_ok"
                ast_results[relative] = "not_python"
    provenance = contract.verify_old_sources()

    capture = import_file("l17_v4_capture", CAPTURE)
    identity = import_file("l17_v4_identity", IDENTITY)
    short_runner = import_file("l17_v4_short_runner", SHORT_RUNNER)
    long_runner = import_file("l17_v4_long_runner", LONG_RUNNER)
    startup_audit = import_file("l17_v4_startup_audit", STARTUP_AUDIT)
    short_client_audit = import_file("l17_v4_short_client_audit", SHORT_CLIENT_AUDIT)
    post_wrapper = import_file("l17_v4_post_wrapper", POST_WRAPPER)
    formal_wrapper = import_file("l17_v4_formal_wrapper", FORMAL_WRAPPER)
    startup_entry = import_file("l17_v4_startup_entry", STARTUP_ENTRY)
    readiness_audit = import_file("l17_v4_readiness", READINESS_AUDIT)
    functional_audit = import_file("l17_v4_full_short_audit", FUNCTIONAL_AUDIT)
    formal_runner = import_file("l17_v4_formal_runner", FORMAL_RUNNER)
    metrics_inspector = import_file("l17_v4_metrics_inspector", METRICS_INSPECTOR)

    d_start_receipt = json.loads((ROOT / D_START_RECEIPT).read_text())
    d_start_gate = startup_receipt_gate(d_start_receipt)
    d_start = d_start_state()
    assert d_start["present"] and d_start["sha256"] == sha(D_START_RECEIPT)

    settings = json.loads((ROOT / FORMAL_SETTINGS).read_text())
    assert settings["resource_controller_sha256"] == sha(FORMAL_CONTROLLER)
    assert settings["model_identity_helper"] == contract.MIXED_IDENTITY
    assert settings["model_identity_sha256"] == contract.MIXED_IDENTITY_SHA
    assert settings["startup_qualification_path"] == contract.STARTUP_RESULT
    assert short_runner.PROBE_SHA == sha(PROBE)
    assert short_runner.FROZEN["glm53_sglang_pd_gateway_v2.py"] == contract.GATEWAY_V2_SHA
    assert long_runner.FROZEN["glm53_C32_L17_native_long_client_v1.py"] == sha(LONG_CLIENT)
    assert long_runner.FROZEN["glm53_C32_L17_native_semantic_probe_v1.py"] == sha(SEMANTIC_PROBE)
    assert startup_audit.main is not None and short_client_audit.main is not None
    assert readiness_audit.main is not None and functional_audit.main is not None
    assert formal_runner.main is not None and metrics_inspector.main is not None
    assert capture.MODE_CAPTURE["startup"] == contract.STARTUP_CAPTURE
    assert capture.MODE_REVIEW["startup"] == contract.STARTUP_CAPTURE_REVIEW
    assert startup_entry.CORE_SHA == sha(CAPTURE)
    assert startup_entry.REVIEW.name == contract.STARTUP_CAPTURE_REVIEW.split("/")[-1]

    startup_review_v2_doc = json.loads((ROOT / contract.STARTUP_CAPTURE_REVIEW).read_text())
    short_plan = short_runner.plan()
    long_plan = long_runner.plan()
    capture_hashes = {
        mode: capture.plan_hashes(capture_id, mode)
        for mode, capture_id in capture.MODE_CAPTURE.items()
    }
    assert capture_hashes["startup"] == startup_review_v2_doc["host_codes_sha256"]
    host_codes = {
        "capture": capture_hashes,
        "startup_plan_matches_review_v2": True,
        "short": {name: hashlib.sha256(code.encode()).hexdigest() for name, code in short_plan.items()},
        "long": {name: hashlib.sha256(code.encode()).hexdigest() for name, code in long_plan.items()},
        "capture_core_sha256": sha(CAPTURE),
        "capture_core_v1_sha256": sha(CAPTURE_V1),
        "identity_sha256": sha(IDENTITY),
        "identity_v1_sha256": sha(IDENTITY_V1),
        "wrappers": {
            STARTUP_ENTRY: {"core_sha256": startup_entry.CORE_SHA},
            POST_WRAPPER: {"core_sha256": post_wrapper.CORE_SHA},
            FORMAL_WRAPPER: {"core_sha256": formal_wrapper.CORE_SHA},
            METRICS_INSPECTOR: {"core_sha256": metrics_core_pin()},
        },
        "short_client_audit_sha256": sha(SHORT_CLIENT_AUDIT),
        "contract_sha256": sha(CONTRACT),
        "formal_settings_sha256": sha(FORMAL_SETTINGS),
        "frozen_references": {relative: sha(relative) for relative in FROZEN_REFERENCES},
    }
    assert all(item["core_sha256"] == sha(CAPTURE) for item in host_codes["wrappers"].values())

    symtable_results = {}
    for mode, capture_id in capture.MODE_CAPTURE.items():
        _, codes, cpu = capture.plans(capture_id, mode)
        for node, code in codes.items():
            label = "capture-%s-%s" % (mode, node)
            identity_code = identity_literal(code)
            assert identity_code is not None
            identity_unbound = unbound_globals(identity_code, label + "-identity")
            outer_unbound = unbound_globals(
                code, label, symtable_module_names(identity_code, label + "-identity-extra")
            )
            assert not identity_unbound and not outer_unbound, (label, identity_unbound, outer_unbound)
            symtable_results[label] = {"identity": identity_unbound, "outer": outer_unbound}
        if cpu is not None:
            label = "capture-%s-43" % mode
            symtable_results[label] = {"identity": [], "outer": unbound_globals(cpu, label)}
            assert not symtable_results[label]["outer"]
    for name, code in short_plan.items():
        symtable_results["short-" + name] = unbound_globals(code, "short-" + name)
        assert not symtable_results["short-" + name]
    for name, code in long_plan.items():
        symtable_results["long-" + name] = unbound_globals(code, "long-" + name)
        assert not symtable_results["long-" + name]

    client_audit_result = client_audit_fixture(short_client_audit)
    assert client_audit_result["verdict"] == "PASS"
    client_output = client_audit_result.pop("output")
    entry = {
        "short": short_entry_fixture(short_runner),
        "long": long_entry_fixture(long_runner, client_output),
        "fixtures_are_simulated": True,
        "not_native_or_model_evidence": True,
    }
    assert entry["short"]["verdict"] == "PASS" and entry["long"]["verdict"] == "PASS"
    post_gate = post_wrapper_fixture(post_wrapper, client_output)
    assert post_gate["verdict"] == "PASS"
    codec_gate = codec_identity_fixture(startup_audit)
    staged_gate = identity_staged_loop_fixture()
    assert staged_gate["verdict"] == "PASS"
    pin_gate = core_pin_fixture()
    assert pin_gate["verdict"] == "PASS"
    dependency_gate = dependency_fixture()
    assert dependency_gate["verdict"] == "PASS"

    counts = {
        "functional_backend": contract.FUNCTIONAL_BACKEND,
        "functional_output": contract.FUNCTIONAL_OUTPUT_IF_ALL,
        "formal_offered_backend": contract.FORMAL_OFFERED_BACKEND,
        "formal_output": contract.FORMAL_OUTPUT_IF_ALL,
        "total_backend": contract.TOTAL_BACKEND_IF_ALL,
        "total_output": contract.TOTAL_OUTPUT_IF_ALL,
        "short_logical": contract.SHORT_LOGICAL,
        "short_backend": contract.SHORT_BACKEND,
        "short_output": contract.SHORT_OUTPUT_IF_ALL,
        "long_logical": contract.LONG_LOGICAL,
        "long_backend": contract.LONG_BACKEND,
        "long_output": contract.LONG_OUTPUT_IF_ALL,
        "new_profiles": 0,
        "retries": 0,
        "ledger_increment": False,
    }
    assert counts["short_logical"] == 32 and counts["short_backend"] == 64 and counts["short_output"] == 19232
    assert counts["long_logical"] == 6 and counts["long_backend"] == 12 and counts["long_output"] == 2408
    assert counts["functional_backend"] == 76 and counts["functional_output"] == 21640
    assert counts["formal_offered_backend"] == 320 and counts["formal_output"] == 76992
    assert counts["total_backend"] == 396 and counts["total_output"] == 98632
    assert counts["new_profiles"] == counts["retries"] == 0 and counts["ledger_increment"] is False

    changed = {}
    for relative in CHANGED_FILES:
        changed[relative] = {
            "v3_sha256": v3_files.get(relative),
            "v4_sha256": sha(relative),
            "bytes": (ROOT / relative).stat().st_size,
        }
    changed[QUALIFIER] = {
        "v3_sha256": None,
        "v4_sha256": sha(QUALIFIER),
        "bytes": (ROOT / QUALIFIER).stat().st_size,
    }
    new_sources = {}
    for relative, creator in [
        (IDENTITY, "primary_created_before_this_run"),
        (CAPTURE, "primary_created_before_this_run"),
        (STARTUP_ENTRY, "primary_created_before_this_run"),
        (QUALIFIER, "this_v4_run"),
    ]:
        new_sources[relative] = {
            "sha256": sha(relative),
            "bytes": (ROOT / relative).stat().st_size,
            "created_by": creator,
            "active": True,
        }
    unmodified_v3 = {}
    for relative, digest in v3_files.items():
        if relative in changed:
            continue
        actual = sha(relative)
        unmodified_v3[relative] = {
            "v3_sha256": digest,
            "actual_sha256": actual,
            "matches_v3": actual == digest,
        }
        assert actual == digest, relative

    startup_attempt_v1 = startup_attempt_state(STARTUP_CAPTURE_V1)
    startup_attempt_v2 = startup_attempt_state(contract.STARTUP_CAPTURE)
    startup_review_v1 = startup_review_state(STARTUP_REVIEW_V1)
    startup_review_v2 = startup_review_state(contract.STARTUP_CAPTURE_REVIEW)
    unqualified = []
    if not d_start_gate:
        unqualified.append({
            "path": D_START_RECEIPT,
            "reason": "current coordinator receipt does not pass startup_receipt_gate",
            "phase": d_start["phase"],
            "pass": d_start["pass"],
            "starts": d_start["starts"],
            "observe": d_start["observe"],
            "native_startup_summaries_present": d_start["native_startup_summaries_present"],
            "startup_pass_declared": False,
            "consumed_as_evidence": False,
            "modified_by_this_run": False,
        })
    for label, state in (("v1", startup_attempt_v1), ("v2", startup_attempt_v2)):
        if state["present"] and not state["valid_actual_evidence"]:
            unqualified.append({
                "path": state["path"],
                "capture_id": state["capture_id"],
                "reason": state.get("reason"),
                "nodes": state["nodes"],
                "startup_pass_declared": False,
                "consumed_as_evidence": False,
                "modified_by_this_run": False,
            })
    for label, state in (("v1", startup_review_v1), ("v2", startup_review_v2)):
        if state["present"] and not state["valid_for_startup_gate"]:
            unqualified.append({
                "path": state["path"],
                "reason": state.get("reason"),
                "decision": state["decision"],
                "startup_qualified": state["startup_qualified"],
                "runtime_model_dispatch_authorized": state["runtime_model_dispatch_authorized"],
                "startup_pass_declared": False,
                "consumed_as_evidence": False,
                "modified_by_this_run": False,
            })

    pending = [
        D_START_RECEIPT,
        contract.STARTUP_RESULT,
        "runs/" + contract.STARTUP_CAPTURE,
        contract.STARTUP_CAPTURE_REVIEW,
        "runs/" + contract.SHORT_RUN,
        contract.SHORT_REVIEW,
        contract.CLIENT_RESULT,
        "runs/" + contract.LONG_RUN,
        contract.LONG_REVIEW,
        "runs/" + contract.FUNCTIONAL_CAPTURE,
        contract.FUNCTIONAL_CAPTURE_REVIEW,
        contract.SHORT_RESULT,
        contract.READINESS_RESULT,
        "results/C32-L17-native-functional-decision-v1.json",
        "results/C32-L17-native-functional-actual-review-v1.json",
        contract.FORMAL_REVIEW,
        "runs/" + contract.FORMAL_RUN,
        "runs/" + contract.FORMAL_CAPTURE,
        contract.FORMAL_CAPTURE_REVIEW,
        contract.INDEPENDENT_METRICS_REVIEW,
        "runs/" + contract.INDEPENDENT_METRICS_RUN,
        contract.FORMAL_CLIENT_AUDIT,
        contract.FORMAL_NATIVE_AUDIT,
        contract.FORMAL_MEASUREMENT,
    ]

    def pending_valid(item):
        if item == D_START_RECEIPT:
            return d_start_gate
        if item == "runs/" + contract.STARTUP_CAPTURE:
            return startup_attempt_v2["valid_actual_evidence"]
        if item == contract.STARTUP_CAPTURE_REVIEW:
            return startup_review_v2["valid_for_startup_gate"]
        return (ROOT / item).exists()

    missing = [item for item in pending if not pending_valid(item)]
    startup_route = {
        "current_diagnostic_entry": STARTUP_ENTRY,
        "current_diagnostic_entry_sha256": sha(STARTUP_ENTRY),
        "current_diagnostic_capture": contract.STARTUP_CAPTURE,
        "current_diagnostic_capture_review": contract.STARTUP_CAPTURE_REVIEW,
        "independent_startup_audit_entry": STARTUP_AUDIT,
        "independent_startup_audit_output": contract.STARTUP_RESULT,
        "superseded_observation_entry": STARTUP_ENTRY_V1_FROZEN,
        "failed_post_timeout_entry_v1": STARTUP_ENTRY_V1_FAILED,
        "superseded_entries_active": False,
        "functional_gate": "BLOCKED_PENDING_PRIMARY_ACCEPT_OF_V2_STARTUP_EVIDENCE_AND_NATIVE_ERROR_LOCALIZATION",
        "startup_pass_declared": False,
        "startup_qualified": False,
        "model_requests": 0,
    }
    actual_evidence_state = {
        "d_model_start_observe": {**d_start, "startup_receipt_gate_accepts_current": d_start_gate},
        "startup_capture_attempts": {"v1": startup_attempt_v1, "v2": startup_attempt_v2},
        "startup_capture_reviews": {"v1": startup_review_v1, "v2": startup_review_v2},
        "pending_native_errors": {
            "count": 13,
            "source": "primary handoff; review-v2 reason reports 7+6 errors",
            "primary_localization_complete": False,
            "consumed_as_evidence": False,
        },
        "unqualified_actual_evidence": unqualified,
        "startup_pass_declared": False,
        "startup_qualified": False,
        "external_artifacts_consumed_as_evidence": False,
        "note": "Both preserved startup captures fail: v1 has four rc=1 nodes; v2 has nodes 36/41 rc=1 and 13 native errors pending primary localization. Neither receipt nor review is startup evidence.",
    }

    assert sha_path(V1_RESULT) == v1_result_sha_before
    assert sha_path(V2_RESULT) == v2_result_sha_before
    assert sha_path(V3_RESULT) == v3_result_sha_before
    assert snapshot_tree(V1_CWD) == v1_tree_before
    assert snapshot_tree(V2_CWD) == v2_tree_before
    assert snapshot_tree(V3_CWD) == v3_tree_before
    assert {relative: sha(relative) for relative in frozen_before} == frozen_before
    assert {relative: sha(relative) for relative in old_sources_before} == old_sources_before
    assert {relative: sha(relative) for relative in protected_before} == protected_before

    result = {
        "project_id": "glm53",
        "result_id": "C32-L17-remaining-pipeline-local-draft-v4",
        "status": "LOCAL_DRAFT_V4_COMPLETE_PRIMARY_REVIEW_PENDING",
        "review_status": "PENDING_PRIMARY_REVIEW",
        "runtime_authorized": False,
        "deployment_authorized": False,
        "formal_execution_authorized": False,
        "model_start_authorized": False,
        "objective_complete": False,
        "startup_pass_declared": False,
        "scope": "Local stdlib/source v4 of the remaining L17 SGLang C32 identity-v2 pipeline wiring only. No SSH/network/Docker/NPU/SDK/model/Git, no launcher execution, no client/capture start, no review approval.",
        "supersedes_for_review_only": "results/C32-L17-remaining-pipeline-local-draft-v3.json",
        "v1_original_result": {
            "path": str(V1_RESULT.relative_to(ROOT)),
            "sha256": v1_result_sha_before,
            "modified": False,
            "tree": v1_tree_before,
        },
        "v2_original_result": {
            "path": str(V2_RESULT.relative_to(ROOT)),
            "sha256": v2_result_sha_before,
            "modified": False,
            "receipt_tree": v2_tree_before,
        },
        "v3_original_result": {
            "path": str(V3_RESULT.relative_to(ROOT)),
            "sha256": v3_result_sha_before,
            "modified": False,
            "receipt_tree": v3_tree_before,
        },
        "changed_files": changed,
        "changed_file_count": len(changed),
        "new_sources": new_sources,
        "unmodified_v3_files": unmodified_v3,
        "old_source_provenance": provenance,
        "actual_evidence_state": actual_evidence_state,
        "startup_route": startup_route,
        "capture_core": {
            "path": CAPTURE,
            "sha256": sha(CAPTURE),
            "bytes": (ROOT / CAPTURE).stat().st_size,
            "v1_frozen": {"path": CAPTURE_V1, "sha256": sha(CAPTURE_V1)},
        },
        "identity": {
            "path": contract.MIXED_IDENTITY,
            "sha256": contract.MIXED_IDENTITY_SHA,
            "v1_frozen": {"path": IDENTITY_V1, "sha256": sha(IDENTITY_V1)},
            "staged_loop_fix": "hash hexdigest compared to the path->SHA string directly, not item['sha256']",
        },
        "formal_settings": {
            "path": FORMAL_SETTINGS,
            "sha256": sha(FORMAL_SETTINGS),
            "model_identity_helper": settings["model_identity_helper"],
            "model_identity_sha256": settings["model_identity_sha256"],
            "startup_qualification_path": settings["startup_qualification_path"],
            "resource_controller_sha256": settings["resource_controller_sha256"],
        },
        "local_qualification": {
            "python_compile": compile_results,
            "ast_parse": ast_results,
            "symtable_unbound_globals": symtable_results,
            "symtable_all_resolved": all(
                not any(value.values()) if isinstance(value, dict) else not value
                for value in symtable_results.values()
            ),
            "source_provenance_all_match": True,
            "wrapper_core_pins_match": True,
            "formal_config_identity_and_startup_match": True,
            "startup_plan_hashes": capture_hashes["startup"],
            "startup_plan_matches_review_v2": True,
            "entry_gate": entry,
            "codec_identity_gate": codec_gate,
            "identity_staged_loop_gate": staged_gate,
            "core_pin_gate": pin_gate,
            "client_only_audit": client_audit_result,
            "post_functional_gate": post_gate,
            "dependency_graph": dependency_gate,
            "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
            "fixture_reviews_written_to_ephemeral_temp_and_deleted": True,
            "approved_review_files_created": False,
            "external_diagnostic_startup_attempt_consumed_as_evidence": False,
            "external_diagnostic_startup_attempt_modified": False,
            "remote_or_capture_or_client_main_executed": False,
            "framework_torch_or_SDK_import": False,
        },
        "host_codes": host_codes,
        "counts": counts,
        "gates": {
            "selected_variants": {"P": "candidate", "D": "candidate", "scope": "D_ONLY_NOT_P"},
            "max_SSH_future_execution_per_launcher": 2,
            "capacity_combination": "IN_MEMORY_FROM_IMMUTABLE_P_AND_D_CAPACITY_FILES",
            "raw_capacity_files_unchanged": True,
            "P_KV_capacity_floor": contract.P_KV_FLOOR,
            "D_KV_capacity_floor": contract.D_KV_FLOOR,
            "old_D_reported_KV_capacity": contract.OLD_D_KV,
            "layout": "237 kv_item_lens + 11 aux_item_lens; BF16 target78/draft1; MTP5/6",
            "short_counts": {"logical": 32, "backend": 64, "successful_output_if_all": 19232},
            "long_counts": {"logical": 6, "backend": 12, "successful_output_if_all": 2408},
            "functional_counts": {"backend": 76, "output_if_all": 21640},
            "formal_counts": {"warm_logical": 32, "formal": 128, "offered_backend": 320, "output_if_all": 76992},
            "proposed_total": {"backend": 396, "output_if_all": 98632, "new_profiles": 0, "retries": 0, "ledger_increment": False},
            "qualification_order": " -> ".join(PIPELINE_ORDER),
            "full_short_native_audit_still_requires_all32_bs2_and_graph_deltas": True,
            "readiness_still_requires_full_short_audit_and_native_cache_SDMA_semantic_proofs": True,
            "startup_wire": {
                "capture": contract.STARTUP_CAPTURE,
                "review": contract.STARTUP_CAPTURE_REVIEW,
                "result": contract.STARTUP_RESULT,
                "current_entry": STARTUP_ENTRY,
            },
            "stale_identity_references": dependency_gate["stale_identity_references"],
            "stale_reference_count": dependency_gate["stale_reference_count"],
            "startup_plan_hash_stable_across_contract_wire": True,
        },
        "pending_reviews": {
            "status": "PENDING_PRIMARY_REVIEW",
            "created": False,
            "approved_created": False,
            "external_diagnostic_review_present": startup_review_v2["present"],
            "external_diagnostic_review_created_by_this_run": False,
            "external_diagnostic_review_startup_qualified": startup_review_v2["startup_qualified"],
            "paths": [
                contract.STARTUP_CAPTURE_REVIEW,
                contract.SHORT_REVIEW,
                contract.FUNCTIONAL_CAPTURE_REVIEW,
                contract.LONG_REVIEW,
                "results/C32-L17-native-functional-decision-v1.json",
                "results/C32-L17-native-functional-actual-review-v1.json",
                contract.FORMAL_REVIEW,
                contract.FORMAL_CAPTURE_REVIEW,
                contract.INDEPENDENT_METRICS_REVIEW,
            ],
        },
        "missing_actual_evidence": missing,
        "pending_actual_evidence": pending,
        "actual_reviews_created": False,
        "boundaries": {
            "SSH": 0,
            "network": 0,
            "HTTP_external": 0,
            "Docker": 0,
            "NPU": 0,
            "SDK_init": 0,
            "model_requests": 0,
            "local_simulated_client_calls": 0,
            "new_profiles": 0,
            "remote_writes": 0,
            "Git": 0,
            "capture_executed": False,
            "client_started": False,
            "framework_torch_or_SDK_import": False,
            "existing_frozen_artifact_edits": 0,
            "old_source_edits": 0,
            "state_handoff_loop_capacity_reviews_edits": 0,
            "approved_reviews_created": 0,
            "external_artifacts_consumed_as_evidence": 0,
            "external_artifacts_modified": 0,
            "running_v2_startup_capture_touched": False,
            "ledger_incremented": False,
        },
        "verification": str(VERIFICATION.relative_to(ROOT)),
        "receipt": str(RECEIPT.relative_to(ROOT)),
        "created_at": "2026-10-19",
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result_sha = sha_path(RESULT)
    verification = {
        "project_id": "glm53",
        "run_id": CWD.name,
        "mode": "LOCAL_ONLY_PIPELINE_IDENTITY_V2_WIRE",
        "result": {"path": str(RESULT.relative_to(ROOT)), "sha256": result_sha},
        "changed_files": changed,
        "new_sources": new_sources,
        "unmodified_v3_files": unmodified_v3,
        "entry_gate": entry,
        "codec_identity_gate": codec_gate,
        "identity_staged_loop_gate": staged_gate,
        "core_pin_gate": pin_gate,
        "client_only_audit": client_audit_result,
        "post_functional_gate": post_gate,
        "dependency_graph": dependency_gate,
        "counts": counts,
        "startup_route": startup_route,
        "actual_evidence_state": actual_evidence_state,
        "missing_actual_evidence": missing,
        "runtime_authorized": False,
        "startup_pass_declared": False,
        "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
    }
    write_json(VERIFICATION, verification)
    write_json(HOST_HASHES, host_codes)
    write_json(ENTRY_GATE, entry)
    write_json(CODEC_GATE, codec_gate)
    write_json(STAGED_GATE, staged_gate)
    write_json(DEPENDENCY_GATE, dependency_gate)
    write_json(FIXTURE_SUMMARY, {
        "project_id": "glm53",
        "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
        "short_entry": entry["short"],
        "long_entry": entry["long"],
        "client_only_audit": client_audit_result,
        "post_functional_gate": post_gate,
        "codec_identity": codec_gate,
        "identity_staged_loop": staged_gate,
        "core_pins": pin_gate,
        "dependency_graph": dependency_gate,
        "approved_review_files_created": False,
        "ephemeral_temp_roots_removed": True,
    })
    receipt = {
        "project_id": "glm53",
        "run_id": CWD.name,
        "mode": "LOCAL_ONLY_PIPELINE_IDENTITY_V2_WIRE",
        "status": "LOCAL_DRAFT_V4_COMPLETE_PRIMARY_REVIEW_PENDING",
        "runtime_authorized": False,
        "deployment_authorized": False,
        "formal_execution_authorized": False,
        "model_start_authorized": False,
        "objective_complete": False,
        "startup_pass_declared": False,
        "scope": "Local v4 identity-v2 wiring, staged-loop typed fixtures, actual codec encoder/decoder, entry fixtures and dependency acyclicity; no remote/client/capture execution.",
        "result": {"path": str(RESULT.relative_to(ROOT)), "sha256": result_sha},
        "verification": {"path": str(VERIFICATION.relative_to(ROOT)), "sha256": sha_path(VERIFICATION)},
        "host_code_hashes": {"path": str(HOST_HASHES.relative_to(ROOT)), "sha256": sha_path(HOST_HASHES)},
        "entry_gate_qualification": {"path": str(ENTRY_GATE.relative_to(ROOT)), "sha256": sha_path(ENTRY_GATE)},
        "codec_identity_qualification": {"path": str(CODEC_GATE.relative_to(ROOT)), "sha256": sha_path(CODEC_GATE)},
        "identity_staged_loop_qualification": {"path": str(STAGED_GATE.relative_to(ROOT)), "sha256": sha_path(STAGED_GATE)},
        "dependency_graph_qualification": {"path": str(DEPENDENCY_GATE.relative_to(ROOT)), "sha256": sha_path(DEPENDENCY_GATE)},
        "fixture_summary": {"path": str(FIXTURE_SUMMARY.relative_to(ROOT)), "sha256": sha_path(FIXTURE_SUMMARY)},
        "changed_files": changed,
        "new_sources": new_sources,
        "pins": {
            "contract_sha256": sha(CONTRACT),
            "mixed_identity": {"path": contract.MIXED_IDENTITY, "sha256": contract.MIXED_IDENTITY_SHA},
            "capture_core": {"path": CAPTURE, "sha256": sha(CAPTURE)},
            "startup_capture": contract.STARTUP_CAPTURE,
            "startup_capture_review": contract.STARTUP_CAPTURE_REVIEW,
            "startup_result": contract.STARTUP_RESULT,
            "formal_model_identity_helper": settings["model_identity_helper"],
            "formal_model_identity_sha256": settings["model_identity_sha256"],
            "wrappers": host_codes["wrappers"],
            "current_startup_entry": STARTUP_ENTRY,
            "superseded_startup_entries": [STARTUP_ENTRY_V1_FROZEN, STARTUP_ENTRY_V1_FAILED],
        },
        "startup_route": startup_route,
        "actual_evidence_state": actual_evidence_state,
        "counts": counts,
        "entry_gate": {
            "short_positive_identity_calls": 4,
            "short_positive_client_calls": 1,
            "long_positive_identity_calls": 4,
            "long_positive_client_calls": 1,
            "precheck_failure_cases": 8,
            "client_started_on_precheck_failure": 0,
            "short_review_negative_cases": entry["short"]["review_failure_count"],
            "long_review_negative_cases": entry["long"]["review_failure_count"],
            "long_client_result_negative_cases": entry["long"]["client_result_failure_count"],
            "identity_staged_loop_negative_cases": staged_gate["negative_case_count"],
            "core_pin_negative_cases": len(pin_gate["pin_negative_cases"]),
            "no_real_ssh": True,
        },
        "client_only_audit": {
            "path": contract.CLIENT_RESULT,
            "label": contract.CLIENT_LABEL,
            "simulated_fixture_pass": True,
            "negative_case_count": client_audit_result["negative_case_count"],
            "native_qualified": False,
            "formal_ready": False,
            "does_not_overwrite_full_short_result": True,
        },
        "startup_identity": {
            "raw_wire_source": True,
            "codec_sha256": contract.CODEC_SHA,
            "negative_case_count": codec_gate["negative_case_count"],
            "identity_staged_loop_negative_case_count": staged_gate["negative_case_count"],
            "v1_loop_type_error_reproduced": True,
            "verdict": codec_gate["verdict"] + "/" + staged_gate["verdict"],
        },
        "dependency_acyclic": dependency_gate["acyclic"],
        "stale_identity_references": dependency_gate["stale_identity_references"],
        "stale_reference_count": dependency_gate["stale_reference_count"],
        "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
        "approved_review_files_created": False,
        "external_execution": {
            "SSH": 0, "network": 0, "HTTP": 0, "Docker": 0, "NPU": 0,
            "SDK_init": 0, "model_requests": 0, "new_profiles": 0, "Git": 0,
            "capture": 0, "client": 0,
        },
        "v1_result_preserved": {"path": str(V1_RESULT.relative_to(ROOT)), "sha256": v1_result_sha_before},
        "v2_result_preserved": {"path": str(V2_RESULT.relative_to(ROOT)), "sha256": v2_result_sha_before},
        "v3_result_preserved": {"path": str(V3_RESULT.relative_to(ROOT)), "sha256": v3_result_sha_before},
        "v1_v2_v3_receipt_verification_preserved": True,
        "state_handoff_loop_capacity_reviews_edited": False,
        "model_coordinator_or_old_source_edited": False,
        "model_runtime_edited": False,
        "running_v2_startup_capture_modified_or_consumed": False,
        "ledger_incremented": False,
        "maximum_SSH_future_execution_per_launcher": 2,
        "created_at": "2026-10-19",
    }
    write_json(RECEIPT, receipt)
    print(json.dumps({
        "result": str(RESULT.relative_to(ROOT)),
        "result_sha256": result_sha,
        "receipt": str(RECEIPT.relative_to(ROOT)),
        "receipt_sha256": sha_path(RECEIPT),
        "entry_gate": entry["short"]["verdict"] + "/" + entry["long"]["verdict"],
        "codec_gate": codec_gate["verdict"],
        "identity_staged_loop_gate": staged_gate["verdict"],
        "core_pin_gate": pin_gate["verdict"],
        "dependency_acyclic": dependency_gate["acyclic"],
        "stale_reference_count": dependency_gate["stale_reference_count"],
        "startup_plan_matches_review_v2": True,
        "startup_pass_declared": False,
        "counts": counts,
        "missing_actual_evidence": len(missing),
        "runtime_authorized": False,
        "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
    }, indent=2))


if __name__ == "__main__":
    main()
