"""DRAFT_PENDING_PRIMARY_REVIEW: local-only v2 qualification of corrected L17 drafts.

Mechanically re-qualifies the remaining L17 local drafts after two corrections:
the startup wrapper now consumes the actual observe-complete model coordinator
receipt, and per-node D eager coverage is labelled LOCAL16 rather than global32.
This script compiles every new draft, verifies all pinned old-source SHAs,
renders the plan-only host programs, checks AST/symtable globals, exercises the
actual startup-entry prefix with synthetic receipts before any core load or
mkdir, and writes a new v2 result plus this run's own receipt.  It does not
modify the preserved v1 result or execute any remote/capture/client main.

No SSH/network/Docker/NPU/SDK/model/Git action is performed.
"""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import py_compile
import symtable
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v1 as contract

ROOT = Path(__file__).resolve().parents[1]
CWD = ROOT / "runs" / "20261019T-L17-pipeline-wire-fix-local-v1"
RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v2.json"
V1_RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v1.json"
V1_RECEIPT = ROOT / "runs/20261019T-L17-functional-formal-local-draft-executor-v1/receipt.json"
V1_VERIFICATION = ROOT / "runs/20261019T-L17-functional-formal-local-draft-executor-v1/verification.json"
VERIFICATION = CWD / "verification.json"
HOST_HASHES = CWD / "host-code-hashes.json"
ENTRY_GATE = CWD / "entry-gate-qualification.json"
SYNTHETIC_POSITIVE = CWD / "synthetic-startup-receipt-positive.json"
RECEIPT = CWD / "receipt.json"

CAPTURE = "scripts/glm53_C32_L17_native_capture_v1.py"
STARTUP_WRAPPER = "scripts/inspect_C32_L17_native_startup_v1.py"
POST_WRAPPER = "scripts/inspect_C32_L17_native_post_functional_v1.py"
FORMAL_WRAPPER = "scripts/glm53_C32_L17_formal_native_capture_v1.py"
METRICS_INSPECTOR = "scripts/inspect_C32_L17_independent_formal_metrics_v1.py"
FUNCTIONAL_AUDIT = "scripts/audit_C32_L17_native_functional_local_v1.py"
READINESS_AUDIT = "scripts/audit_C32_L17_native_readiness_local_v1.py"
FORMAL_NATIVE_AUDIT = "scripts/audit_C32_L17_native_formal_native_local_v1.py"
QUALIFIER = "scripts/qualify_C32_L17_remaining_pipeline_local_v2.py"

CHANGED_FILES = [
    CAPTURE,
    STARTUP_WRAPPER,
    POST_WRAPPER,
    FORMAL_WRAPPER,
    METRICS_INSPECTOR,
    FUNCTIONAL_AUDIT,
    READINESS_AUDIT,
    FORMAL_NATIVE_AUDIT,
]

NEW_FILES = [
    "scripts/glm53_C32_L17_pipeline_contract_v1.py",
    CAPTURE,
    STARTUP_WRAPPER,
    POST_WRAPPER,
    FORMAL_WRAPPER,
    "scripts/probe_C32_L17_native_pd_payload_v1.py",
    "scripts/run_C32_L17_native_P_functional_v1.py",
    "scripts/glm53_C32_L17_native_long_client_v1.py",
    "scripts/glm53_C32_L17_native_semantic_probe_v1.py",
    "scripts/run_C32_L17_native_long_v1.py",
    "scripts/audit_C32_L17_native_startup_local_v1.py",
    FUNCTIONAL_AUDIT,
    READINESS_AUDIT,
    "scripts/run_C32_L17_native_formal_v1.py",
    "scripts/glm53_C32_L17_formal_resource_controller_v1.py",
    "scripts/audit_C32_L17_native_formal_client_local_v1.py",
    FORMAL_NATIVE_AUDIT,
    METRICS_INSPECTOR,
    QUALIFIER,
    "config/C32-L17-native-runtime-v1/formal-C32-native-v1.json",
]

SAFE_IMPORTS = [
    ("scripts/glm53_C32_L17_pipeline_contract_v1.py", "l17_contract"),
    (CAPTURE, "l17_capture"),
    ("scripts/run_C32_L17_native_P_functional_v1.py", "l17_short_runner"),
    ("scripts/run_C32_L17_native_long_v1.py", "l17_long_runner"),
    ("scripts/glm53_C32_L17_formal_resource_controller_v1.py", "l17_formal_controller"),
    ("scripts/run_C32_L17_native_formal_v1.py", "l17_formal_runner"),
    (METRICS_INSPECTOR, "l17_metrics_inspector"),
    ("scripts/audit_C32_L17_native_startup_local_v1.py", "l17_startup_audit"),
    (FUNCTIONAL_AUDIT, "l17_functional_audit"),
    (READINESS_AUDIT, "l17_readiness_audit"),
    ("scripts/audit_C32_L17_native_formal_client_local_v1.py", "l17_formal_client_audit"),
    (FORMAL_NATIVE_AUDIT, "l17_formal_native_audit"),
]

DERIVATIONS = {
    CAPTURE: {
        "old": [contract.OLD_CAPTURE_FACTORY, contract.OLD_CAPTURE_BASE],
        "mechanical": "Reuses the accepted v3 HOST/STREAM/CPU_HOST/codec and old mixed-root wrapper pattern; changes root/CIDs to the new P+D capacity records, uses the L17 mixed renderer, adds D eager-input rows/env/identity and mode wiring. V2 relabels the per-node coverage status LOCAL16 and states that global32 requires a composite node+namespace_pid join.",
    },
    STARTUP_WRAPPER: {
        "old": ["scripts/inspect_C32_L12C_native_startup_v1.py"],
        "mechanical": "New capture/review IDs; requires the actual L17 D model-start/observe completion before the four-node startup capture. V2 consumes the actual coordinator schema (pass/phase, start-36/start-41, observe-36/observe-41, native_startup_summaries, external_model_requests, cleanup flag and four runtime CIDs) and rejects missing keys, bool rc and failed rc before loading the capture core or creating a run directory.",
    },
    POST_WRAPPER: {
        "old": ["scripts/inspect_C32_L12C_native_post_functional_v1.py"],
        "mechanical": "New capture/short/long IDs; requires D eager bs2 coverage; appends the existing CPU43 long-client artifact without rerunning it. V2 pins the corrected capture core SHA.",
    },
    FORMAL_WRAPPER: {
        "old": [contract.OLD_FORMAL_CAPTURE],
        "mechanical": "New readiness/formal run/capture IDs; calls the L17 capture core and adds the CPU43 formal archive. V2 pins the corrected capture core SHA.",
    },
    METRICS_INSPECTOR: {
        "old": ["scripts/inspect_C32_L12D_independent_formal_metrics_v1.py"],
        "mechanical": "New formal/client-audit/run IDs; same completed CPU43 archive re-derivation, one SSH, no model/SDK. V2 pins the corrected capture core SHA used only for its raw-wire helper.",
    },
    FUNCTIONAL_AUDIT: {
        "old": [contract.OLD_FUNCTIONAL_AUDIT, contract.OLD_STARTUP_AUDIT],
        "mechanical": "Same C32 count/DP logic extended to 600-output D; adds all32 D bs2 draft-DECODE/out_cache_loc evidence and per-rank Graph deltas. V2 records that global32 is a composite node+namespace_pid join, never a PID-only pool.",
    },
    READINESS_AUDIT: {
        "old": [contract.OLD_READINESS_AUDIT],
        "mechanical": "New captures/IDs; strict SDK begin==good/bad0/inflight0 plus 11th TOTAL column, D eager all32, Graph deltas, all32 typed SDMA destinations, pinned optional-D-storage helper. V2 requires 32 unique composite D worker pairs.",
    },
    FORMAL_NATIVE_AUDIT: {
        "old": [contract.OLD_FORMAL_NATIVE_AUDIT],
        "mechanical": "New run/captures/IDs; adds D eager 32-worker prerequisite, pinned optional-D-storage null preservation and strict SDK/Graph/MC/SDMA gates. V2 requires 32 unique composite D worker pairs.",
    },
}


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def import_file(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def cpu_gateway_fixture():
    source_path = ROOT / contract.GATEWAY_V2
    source = source_path.read_text()
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("route", "bootstrap_room"):
            selected.append(node)
    assert {node.name for node in selected} == {"route", "bootstrap_room"}
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"hashlib": hashlib}
    exec(compile(module, contract.GATEWAY_V2, "exec"), scope)
    route = scope["route"]
    room = scope["bootstrap_room"]
    rooms = [room(f"glm53-l17-cpu-room-{index}", index % 4) for index in range(256)]
    assert len(set(rooms)) == 256
    assert all(room_value >= 0 and room_value % 4 == index % 4 for index, room_value in enumerate(rooms))
    for prefix in range(64):
        selected_route = route(prefix)
        assert selected_route["P"]["global_dp"] == prefix % 4
        assert selected_route["D"]["global_dp"] == prefix % 16
        assert selected_route["P"]["api_host"] == "172.16.12.34"
        assert selected_route["D"]["api_host"] == "172.16.12.36"
    receipt = contract.read_json("runs/20261017T1400-sglang-gateway-fixture-v2/receipt.json")
    assert receipt["rc"] == 0 and receipt["model_requests"] == 0
    assert receipt["files_sha256"][contract.GATEWAY_V2.split("/")[-1]] == contract.GATEWAY_V2_SHA
    return {
        "gateway_source_sha256": contract.GATEWAY_V2_SHA,
        "route_prefixes_tested": 64,
        "bootstrap_rooms_tested": 256,
        "existing_fixture_receipt_rc": receipt["rc"],
        "existing_fixture_receipt_sha256": sha("runs/20261017T1400-sglang-gateway-fixture-v2/receipt.json"),
        "external_network": 0,
    }


def synthetic_startup_receipt(wrapper):
    return {
        "project_id": "glm53",
        "synthetic": True,
        "not_actual_evidence": True,
        "phase": wrapper.STARTUP_PHASE,
        "pass": True,
        "capacity_sha256": contract.D_CAPACITY_SHA,
        "review_sha256": "0" * 64,
        "launcher_sha256": "1" * 64,
        "precheck": {
            "health-34": 0,
            "health-35": 0,
            "meta-identity": 0,
            "prestart-36": 0,
            "prestart-41": 0,
        },
        "starts": {"start-36": 0, "start-41": 0},
        "observe": {"observe-36": 0, "observe-41": 0},
        "max_SSH": 2,
        "external_model_requests": 0,
        "cleanup_or_rollback_executed": False,
        "runtime_CIDs": {
            "34": contract.P34_CID,
            "35": contract.P35_CID,
            "36": contract.D36_CID,
            "41": contract.D41_CID,
        },
        "native_startup_summaries": {
            "observe-36": {
                "project_id": "glm53",
                "node": 36,
                "startup_pass": True,
                "formal_ready": False,
                "external_model_requests": 0,
                "actual_SDK_worker_PIDs": list(range(1000, 1016)),
            },
            "observe-41": {
                "project_id": "glm53",
                "node": 41,
                "startup_pass": True,
                "formal_ready": False,
                "external_model_requests": 0,
                "actual_SDK_worker_PIDs": list(range(2000, 2016)),
            },
        },
    }


def negative_gate_cases(positive):
    cases = {}

    def add(name, mutate):
        doc = copy.deepcopy(positive)
        mutate(doc)
        cases[name] = doc

    add("missing_pass", lambda doc: doc.pop("pass"))
    add("pass_false", lambda doc: doc.__setitem__("pass", False))
    add("pass_truthy_string", lambda doc: doc.__setitem__("pass", "True"))
    add("missing_phase", lambda doc: doc.pop("phase"))
    add("phase_observing", lambda doc: doc.__setitem__("phase", "OBSERVING_NATIVE_D_STARTUP"))
    add("phase_failed", lambda doc: doc.__setitem__("phase", "FAILED_OR_PARTIAL_STATE_PRESERVED"))
    add("capacity_wrong", lambda doc: doc.__setitem__("capacity_sha256", "0" * 64))
    add("starts_missing_41", lambda doc: doc["starts"].pop("start-41"))
    add("starts_extra_numeric_key", lambda doc: doc["starts"].__setitem__("36", 0))
    add("starts_bool_rc", lambda doc: doc["starts"].__setitem__("start-36", False))
    add("starts_failed_rc", lambda doc: doc["starts"].__setitem__("start-41", 1))
    add("observe_numeric_keys", lambda doc: doc.__setitem__("observe", {"36": 0, "41": 0}))
    add("observe_hybrid_numeric_key", lambda doc: doc["observe"].__setitem__("36", 0))
    add("observe_missing_41", lambda doc: doc["observe"].pop("observe-41"))
    add("observe_bool_rc", lambda doc: doc["observe"].__setitem__("observe-36", False))
    add("observe_failed_rc", lambda doc: doc["observe"].__setitem__("observe-41", 1))
    add("summaries_missing", lambda doc: doc.pop("native_startup_summaries"))
    add("summaries_numeric_keys", lambda doc: doc.__setitem__("native_startup_summaries", {"36": doc["native_startup_summaries"]["observe-36"], "41": doc["native_startup_summaries"]["observe-41"]}))
    add("summary_startup_false", lambda doc: doc["native_startup_summaries"]["observe-36"].__setitem__("startup_pass", False))
    add("summary_formal_ready_true", lambda doc: doc["native_startup_summaries"]["observe-41"].__setitem__("formal_ready", True))
    add("summary_missing_formal_ready", lambda doc: doc["native_startup_summaries"]["observe-36"].pop("formal_ready"))
    add("external_requests_missing", lambda doc: doc.pop("external_model_requests"))
    add("external_requests_bool_false", lambda doc: doc.__setitem__("external_model_requests", False))
    add("external_requests_one", lambda doc: doc.__setitem__("external_model_requests", 1))
    add("cleanup_missing", lambda doc: doc.pop("cleanup_or_rollback_executed"))
    add("cleanup_true", lambda doc: doc.__setitem__("cleanup_or_rollback_executed", True))
    add("cleanup_null", lambda doc: doc.__setitem__("cleanup_or_rollback_executed", None))
    add("runtime_cids_missing_41", lambda doc: doc["runtime_CIDs"].pop("41"))
    add("runtime_cid_wrong", lambda doc: doc["runtime_CIDs"].__setitem__("36", contract.D41_CID))
    add("runtime_cids_extra", lambda doc: doc["runtime_CIDs"].__setitem__("43", contract.D41_CID))
    add("top_level_startup_pass_only", lambda doc: (
        doc.__setitem__("startup_pass", True),
        doc.pop("native_startup_summaries"),
        doc.__setitem__("observe", {}),
    ))
    return cases


def entry_prefix_qualification(wrapper):
    positive = synthetic_startup_receipt(wrapper)
    SYNTHETIC_POSITIVE.write_text(json.dumps(positive, indent=2) + "\n", encoding="utf-8")
    synthetic_relative = str(SYNTHETIC_POSITIVE.relative_to(ROOT))
    real_positive = contract.derived_require(synthetic_relative, wrapper.startup_receipt_gate)
    assert real_positive["synthetic"] is True and real_positive["not_actual_evidence"] is True

    current = json.loads((ROOT / wrapper.D_START_RECEIPT).read_text())
    current_gate = wrapper.startup_receipt_gate(current)
    assert current_gate is False, "Current incomplete observe receipt must be rejected"

    gate_cases = {}
    for name, doc in negative_gate_cases(positive).items():
        assert wrapper.startup_receipt_gate(doc) is False, name
        gate_cases[name] = "REJECTED"

    review = json.loads((ROOT / wrapper.D_START_REVIEW).read_text())
    docs = {wrapper.D_START_REVIEW: review, wrapper.D_START_RECEIPT: positive}
    original_derived = wrapper.contract.derived_require
    original_core = wrapper._load_core
    original_mkdir = Path.mkdir
    calls = {"core": 0, "mkdir": 0}

    def fake_derived(relative, extra=None):
        doc = docs[relative]
        if extra is not None:
            assert extra(doc), "actual evidence gate failed: " + relative
        return doc

    def forbidden_core():
        calls["core"] += 1
        raise AssertionError("capture core loaded before the startup receipt gate passed")

    def forbidden_mkdir(self, *args, **kwargs):
        calls["mkdir"] += 1
        raise AssertionError("mkdir reached before the startup receipt gate passed")

    prefix_negatives = {
        "actual_current_incomplete_receipt": current,
        "observing_incomplete": {**copy.deepcopy(positive), "phase": "OBSERVING_NATIVE_D_STARTUP", "pass": False, "observe": {}},
        "failed_receipt": {**copy.deepcopy(positive), "phase": "FAILED_OR_PARTIAL_STATE_PRESERVED", "pass": False},
        "numeric_observe_keys": {**copy.deepcopy(positive), "observe": {"36": 0, "41": 0}},
        "bool_observe_rc": {**copy.deepcopy(positive), "observe": {"observe-36": False, "observe-41": 0}},
        "failed_observe_rc": {**copy.deepcopy(positive), "observe": {"observe-36": 0, "observe-41": 1}},
        "missing_summaries": {**copy.deepcopy(positive), "native_startup_summaries": {}},
    }
    prefix_negative_results = {}
    try:
        wrapper.contract.derived_require = fake_derived
        wrapper._load_core = forbidden_core
        Path.mkdir = forbidden_mkdir
        for name, doc in prefix_negatives.items():
            docs[wrapper.D_START_RECEIPT] = doc
            calls["core"] = calls["mkdir"] = 0
            rejected = False
            try:
                wrapper.main()
            except AssertionError:
                rejected = True
            assert rejected, name
            assert calls["core"] == 0 and calls["mkdir"] == 0, (name, calls)
            prefix_negative_results[name] = {
                "rejected": True,
                "capture_core_loaded": calls["core"],
                "mkdir_calls": calls["mkdir"],
            }

        core_loads = []
        execute_calls = []

        class StubCore:
            def execute(self, *args):
                execute_calls.append(args)
                return {"project_id": "glm53", "synthetic": True, "pass": True}

        docs[wrapper.D_START_RECEIPT] = positive
        calls["core"] = calls["mkdir"] = 0
        wrapper._load_core = lambda: (core_loads.append(1) or StubCore())
        rc = wrapper.main()
        assert rc == 0
        assert core_loads == [1]
        assert execute_calls == [(contract.STARTUP_CAPTURE, contract.STARTUP_CAPTURE_REVIEW, "startup")]
        assert calls["mkdir"] == 0
    finally:
        wrapper.contract.derived_require = original_derived
        wrapper._load_core = original_core
        Path.mkdir = original_mkdir

    return {
        "module": STARTUP_WRAPPER,
        "gate_function": "startup_receipt_gate",
        "actual_receipt": {
            "path": wrapper.D_START_RECEIPT,
            "sha256": sha(wrapper.D_START_RECEIPT),
            "phase": current.get("phase"),
            "pass": current.get("pass"),
            "observe_keys": sorted(current.get("observe", {})),
            "has_native_startup_summaries": "native_startup_summaries" in current,
            "gate_result": False,
        },
        "synthetic_positive": {
            "path": synthetic_relative,
            "sha256": sha(synthetic_relative),
            "not_actual_evidence": True,
            "gate_result": True,
            "real_derived_require": True,
        },
        "negative_gate_cases": gate_cases,
        "negative_gate_case_count": len(gate_cases),
        "entry_prefix_main": {
            "negative_cases": prefix_negative_results,
            "positive": {
                "result_rc": 0,
                "capture_core_loaded": core_loads[0],
                "execute_calls": [list(args) for args in execute_calls],
                "mkdir_calls": calls["mkdir"],
            },
            "capture_execute_is_stub": True,
            "remote_or_capture_or_client_main_executed": False,
            "mkdir_before_gate": False,
        },
        "verdict": "PASS",
    }


def main():
    assert not RESULT.exists(), "Preserve existing v2 result"
    assert not RECEIPT.exists(), "Preserve existing v2 receipt"
    assert not VERIFICATION.exists() and not HOST_HASHES.exists() and not ENTRY_GATE.exists()
    assert V1_RESULT.is_file() and V1_RECEIPT.is_file() and V1_VERIFICATION.is_file()
    v1_result_sha_before = sha_path(V1_RESULT)
    v1 = json.loads(V1_RESULT.read_text())
    v1_files = {relative: item["sha256"] for relative, item in v1["new_files"].items()}
    for relative in CHANGED_FILES:
        assert relative in v1_files, relative
    assert QUALIFIER not in v1_files

    compile_results = {}
    ast_results = {}
    for relative in NEW_FILES:
        path = ROOT / relative
        assert path.is_file(), relative
        if path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)
            ast.parse(path.read_text())
            compile_results[relative] = "safe_compile"
            ast_results[relative] = "ast_parse_ok"
        else:
            json.loads(path.read_text())
            compile_results[relative] = "json_parse_ok"
            ast_results[relative] = "not_python"
    provenance = contract.verify_old_sources()
    modules = {name: import_file(name, relative) for relative, name in SAFE_IMPORTS}
    capture = modules["l17_capture"]
    short_runner = modules["l17_short_runner"]
    long_runner = modules["l17_long_runner"]
    controller = modules["l17_formal_controller"]
    settings = json.loads((ROOT / contract.FORMAL_SETTINGS).read_text())
    assert settings["resource_controller_sha256"] == sha("scripts/glm53_C32_L17_formal_resource_controller_v1.py"), "Update formal config controller SHA"
    assert modules["l17_formal_runner"].CONTROLLER == "scripts/glm53_C32_L17_formal_resource_controller_v1.py"
    assert short_runner.PROBE_SHA == sha("scripts/probe_C32_L17_native_pd_payload_v1.py")
    assert short_runner.FROZEN["glm53_sglang_pd_gateway_v2.py"] == contract.GATEWAY_V2_SHA
    assert long_runner.FROZEN["glm53_C32_L17_native_long_client_v1.py"] == sha("scripts/glm53_C32_L17_native_long_client_v1.py")
    assert long_runner.FROZEN["glm53_C32_L17_native_semantic_probe_v1.py"] == sha("scripts/glm53_C32_L17_native_semantic_probe_v1.py")
    assert modules["l17_startup_audit"].main is not None
    assert modules["l17_functional_audit"].main is not None
    assert modules["l17_readiness_audit"].main is not None
    assert modules["l17_formal_native_audit"].audit is not None

    capture_sha = sha(CAPTURE)
    wrapper = import_file("l17_startup_wrapper_v2", STARTUP_WRAPPER)
    post_wrapper = import_file("l17_post_wrapper_v2", POST_WRAPPER)
    formal_wrapper = import_file("l17_formal_capture_wrapper_v2", FORMAL_WRAPPER)
    assert wrapper.CORE_SHA == capture_sha
    assert post_wrapper.CORE_SHA == capture_sha
    assert formal_wrapper.CORE_SHA == capture_sha
    assert metrics_core_pin() == capture_sha

    capture_source = (ROOT / CAPTURE).read_text()
    for relative in NEW_FILES:
        if relative.endswith(".py") and relative != QUALIFIER:
            assert "COMPLETE_32_BS2_DRAFT_DECODE" not in (ROOT / relative).read_text(), relative
    assert "COMPLETE_32_BS2_DRAFT_DECODE" not in capture_source
    assert capture_source.count("COMPLETE_LOCAL16_BS2_DRAFT_DECODE") == 2
    assert "'coverage_status':'COMPLETE_LOCAL16_BS2_DRAFT_DECODE' if coverage else 'PENDING_OR_INCOMPLETE_LOCAL16_BS2_DRAFT_DECODE'" in capture_source
    assert "NODE_LOCAL_16_ACTUAL_SDK_WORKERS" in capture_source
    assert "COMPOSITE_NODE_PLUS_NAMESPACE_PID_ONLY" in capture_source
    functional_source = (ROOT / FUNCTIONAL_AUDIT).read_text()
    readiness_source = (ROOT / READINESS_AUDIT).read_text()
    formal_native_source = (ROOT / FORMAL_NATIVE_AUDIT).read_text()
    assert "COMPOSITE_NODE_PLUS_NAMESPACE_PID" in functional_source
    assert "COMPOSITE_NODE_PLUS_NAMESPACE_PID" in readiness_source
    assert '{(row["node"], row["namespace_pid"]) for row in worker_rows}' in functional_source
    assert '{(row["node"], row["namespace_pid"]) for row in eager["worker_rows"]}' in formal_native_source
    assert '{(row["node"], row["namespace_pid"]) for row in short["D_eager_input_acceptance"]["worker_rows"]}' in readiness_source

    entry_gate = entry_prefix_qualification(wrapper)
    assert entry_gate["verdict"] == "PASS"

    capture_hashes = {
        mode: capture.plan_hashes(capture_id, mode)
        for mode, capture_id in capture.MODE_CAPTURE.items()
    }
    short_plan = short_runner.plan()
    long_plan = long_runner.plan()
    controller_plan = controller.plans(settings, contract.FORMAL_SETTINGS)
    host_codes = {
        "capture": capture_hashes,
        "short": {name: hashlib.sha256(code.encode()).hexdigest() for name, code in short_plan.items()},
        "long": {name: hashlib.sha256(code.encode()).hexdigest() for name, code in long_plan.items()},
        "formal_controller_precheck": controller_plan["precheck_hashes"],
        "formal_settings_sha256": controller_plan["settings_sha256"],
        "wrappers": {
            STARTUP_WRAPPER: {"core_sha256": wrapper.CORE_SHA},
            POST_WRAPPER: {"core_sha256": post_wrapper.CORE_SHA},
            FORMAL_WRAPPER: {"core_sha256": formal_wrapper.CORE_SHA},
            METRICS_INSPECTOR: {"core_sha256": metrics_core_pin()},
        },
        "capture_core_sha256": capture_sha,
    }
    assert all(item["core_sha256"] == capture_sha for item in host_codes["wrappers"].values())

    symtable_results = {}
    for mode, capture_id in capture.MODE_CAPTURE.items():
        _, codes, cpu = capture.plans(capture_id, mode)
        for node, code in codes.items():
            identity_code = identity_literal(code)
            assert identity_code is not None
            identity_unbound = unbound_globals(identity_code, f"identity-{mode}-{node}")
            outer_unbound = unbound_globals(code, f"capture-{mode}-{node}", symtable_module_names(identity_code, "identity-extra"))
            symtable_results[f"capture-{mode}-{node}"] = {"identity": identity_unbound, "outer": outer_unbound}
            assert not identity_unbound and not outer_unbound
        if cpu is not None:
            symtable_results[f"capture-{mode}-43"] = {"identity": [], "outer": unbound_globals(cpu, f"capture-{mode}-43")}
            assert not symtable_results[f"capture-{mode}-43"]["outer"]
    for name, code in short_plan.items():
        symtable_results["short-" + name] = {"identity": [], "outer": unbound_globals(code, "short-" + name)}
        assert not symtable_results["short-" + name]["outer"]
    for name, code in long_plan.items():
        symtable_results["long-" + name] = {"identity": [], "outer": unbound_globals(code, "long-" + name)}
        assert not symtable_results["long-" + name]["outer"]
    for node, code in controller_plan["precheck_codes"].items():
        symtable_results["formal-controller-" + str(node)] = {"identity": [], "outer": unbound_globals(code, "formal-controller-" + str(node))}
        assert not symtable_results["formal-controller-" + str(node)]["outer"]
    fixture = cpu_gateway_fixture()
    result_symtable_ok = all(not value["identity"] and not value["outer"] for value in symtable_results.values())
    coordinator = "scripts/start_C32_L17_selected_D_models_and_observe_v1.py"
    d_start_review = json.loads((ROOT / wrapper.D_START_REVIEW).read_text())
    assert sha(coordinator) == d_start_review["launcher_sha256"]
    assert sha(contract.P_CAPACITY) == contract.P_CAPACITY_SHA
    assert sha(contract.D_CAPACITY) == contract.D_CAPACITY_SHA

    counts = {
        "functional_backend": contract.FUNCTIONAL_BACKEND,
        "functional_output": contract.FUNCTIONAL_OUTPUT_IF_ALL,
        "formal_offered_backend": contract.FORMAL_OFFERED_BACKEND,
        "formal_output": contract.FORMAL_OUTPUT_IF_ALL,
        "total_backend": contract.TOTAL_BACKEND_IF_ALL,
        "total_output": contract.TOTAL_OUTPUT_IF_ALL,
        "new_profiles": 0,
        "retries": 0,
        "ledger_increment": False,
    }
    assert counts["functional_backend"] == 76 and counts["functional_output"] == 21640
    assert counts["formal_offered_backend"] == 320 and counts["formal_output"] == 76992
    assert counts["total_backend"] == 396 and counts["total_output"] == 98632

    pending = [
        "D model-start/observe completion (current receipt is OBSERVING_NATIVE_D_STARTUP, pass=false, observe empty)",
        "results/sglang-C32-L17-native-startup-v1.json",
        "runs/" + contract.STARTUP_CAPTURE,
        contract.STARTUP_CAPTURE_REVIEW,
        contract.SHORT_RESULT,
        "runs/" + contract.SHORT_RUN,
        contract.SHORT_REVIEW,
        "runs/" + contract.FUNCTIONAL_CAPTURE,
        contract.FUNCTIONAL_CAPTURE_REVIEW,
        "runs/" + contract.LONG_RUN,
        contract.LONG_REVIEW,
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
    missing = [item for item in pending if not (ROOT / item).exists()]

    changed = {}
    for relative in CHANGED_FILES:
        changed[relative] = {
            "v1_sha256": v1_files[relative],
            "v2_sha256": sha(relative),
            "bytes": (ROOT / relative).stat().st_size,
        }
    changed[QUALIFIER] = {
        "v1_sha256": None,
        "v2_sha256": sha(QUALIFIER),
        "bytes": (ROOT / QUALIFIER).stat().st_size,
    }
    unmodified = {}
    for relative, digest in v1_files.items():
        if relative in CHANGED_FILES:
            continue
        actual = sha(relative)
        unmodified[relative] = {"v1_sha256": digest, "actual_sha256": actual, "matches_v1": actual == digest}
        assert actual == digest, relative
    assert len(unmodified) == len(v1_files) - len(CHANGED_FILES)

    assert sha_path(V1_RESULT) == v1_result_sha_before
    result = {
        "project_id": "glm53",
        "result_id": "C32-L17-remaining-pipeline-local-draft-v2",
        "status": "LOCAL_DRAFT_V2_COMPLETE_PRIMARY_REVIEW_PENDING",
        "review_status": "PENDING_PRIMARY_REVIEW",
        "runtime_authorized": False,
        "deployment_authorized": False,
        "formal_execution_authorized": False,
        "model_start_authorized": False,
        "objective_complete": False,
        "scope": "Local stdlib/source v2 of the remaining L17 SGLang C32 pipeline only. No SSH/network/Docker/NPU/SDK/model/Git, no launcher execution, no review approval.",
        "supersedes_for_review_only": "results/C32-L17-remaining-pipeline-local-draft-v1.json",
        "v1_original_result": {
            "path": "results/C32-L17-remaining-pipeline-local-draft-v1.json",
            "sha256": v1_result_sha_before,
            "modified": False,
            "receipt_sha256": sha_path(V1_RECEIPT),
            "verification_sha256": sha_path(V1_VERIFICATION),
        },
        "changed_files": changed,
        "changed_file_count": len(CHANGED_FILES),
        "new_files": {relative: {"sha256": sha(relative), "bytes": (ROOT / relative).stat().st_size} for relative in sorted(NEW_FILES)},
        "new_file_count": len(NEW_FILES),
        "unmodified_v1_files": unmodified,
        "capture_core": {"path": CAPTURE, "sha256": capture_sha, "bytes": (ROOT / CAPTURE).stat().st_size},
        "coverage_correction": {
            "per_node_status": "COMPLETE_LOCAL16_BS2_DRAFT_DECODE",
            "pending_status": "PENDING_OR_INCOMPLETE_LOCAL16_BS2_DRAFT_DECODE",
            "per_node_scope": "NODE_LOCAL_16_ACTUAL_SDK_WORKERS",
            "global32_join": "COMPOSITE_NODE_PLUS_NAMESPACE_PID_ONLY",
            "namespace_pid_overlap_across_nodes_allowed": True,
            "pid_only_global_pool_rejected": True,
            "functional_audit_unique_pairs": 32,
            "readiness_requires_composite_pairs": True,
            "formal_native_requires_composite_pairs": True,
        },
        "startup_receipt_correction": {
            "source_schema": "runs/20261019T-L17-D-models-and-observe-v1/receipt.json (start_C32_L17_selected_D_models_and_observe_v1.py)",
            "required_pass": True,
            "required_phase": wrapper.STARTUP_PHASE,
            "required_start_keys": list(wrapper.START_KEYS),
            "required_observe_keys": list(wrapper.OBSERVE_KEYS),
            "required_summary_keys": list(wrapper.OBSERVE_KEYS),
            "summary_startup_pass": True,
            "summary_formal_ready": False,
            "external_model_requests": 0,
            "cleanup_or_rollback_executed": False,
            "runtime_CIDs": {key: value for key, value in wrapper.RUNTIME_CIDS.items()},
            "rejects_missing_keys": True,
            "rejects_bool_rc": True,
            "rejects_failed_rc": True,
            "gate_runs_before_capture_core_load_and_mkdir": True,
        },
        "schema_audit": {
            "checked_scope": "scripts/*.py local text search; no execution of consumers",
            "model_start_receipt_consumers": [STARTUP_WRAPPER],
            "l17_numeric_observe_key_consumers": [],
            "l17_top_level_startup_pass_consumers": [],
            "old_l12b_l12c_top_level_startup_pass_consumers_untouched": [
                "scripts/glm53_C32_L12B_native_capture_v1.py",
                "scripts/glm53_C32_L12B_restored_native_capture_v1.py",
                "scripts/glm53_C32_L12B_restored_native_capture_v2.py",
                "scripts/glm53_C32_L12C_native_capture_v1.py",
                "scripts/glm53_C32_L12D_formal_native_capture_v1.py",
            ],
            "global32_pid_only_join_found_in_l17_drafts": False,
            "capture_per_node_old_status": "COMPLETE_32_BS2_DRAFT_DECODE",
            "capture_per_node_new_status": "COMPLETE_LOCAL16_BS2_DRAFT_DECODE",
            "composite_join_sites": [
                FUNCTIONAL_AUDIT + ":216",
                READINESS_AUDIT + ":282",
                FORMAL_NATIVE_AUDIT + ":96",
            ],
            "modified_only_unexecuted_drafts": True,
            "approved_reviews_created": False,
        },
        "pure_cpu_validation": {
            "python_files_compiled": sum(1 for relative in NEW_FILES if relative.endswith(".py")),
            "json_files_parsed": sum(1 for relative in NEW_FILES if not relative.endswith(".py")),
            "ast_parsed": sum(1 for value in ast_results.values() if value == "ast_parse_ok"),
            "symtable_entries": len(symtable_results),
            "symtable_all_resolved": result_symtable_ok,
            "capture_plans_rendered": len(capture.MODE_CAPTURE),
            "entry_gate_negative_cases": entry_gate["negative_gate_case_count"],
            "entry_prefix_negative_cases": len(entry_gate["entry_prefix_main"]["negative_cases"]),
            "synthetic_real_schema_positive_accepted": True,
            "old_source_provenance_all_match": True,
            "capacity_pair_sha_match": True,
            "coordinator_sha_matches_approved_launcher": True,
            "network_ssh_docker_npu_sdk_model_git": 0,
            "capture_or_client_main_executed": False,
        },
        "old_source_provenance": provenance,
        "allowed_mechanical_diffs": DERIVATIONS,
        "local_qualification": {
            "python_compile": compile_results,
            "ast_parse": ast_results,
            "symtable_unbound_globals": symtable_results,
            "symtable_all_resolved": result_symtable_ok,
            "source_provenance_all_match": True,
            "wrapper_core_pins_match": True,
            "metrics_inspector_core_pin_matches": True,
            "short_probe_pin_matches": True,
            "long_client_probe_pins_match": True,
            "formal_config_controller_pin_matches": True,
            "capture_coverage_static_checks": True,
            "entry_gate_qualification": entry_gate,
            "cpu_gateway_fixture": fixture,
            "installed_scope": "Compiled/AST/symtable/hashes and a synthetic-receipt entry-prefix gate only; no framework, torch or SDK import; no installed remote source was executed.",
        },
        "host_codes": host_codes,
        "gates": {
            "selected_variants": {"P": "candidate", "D": "candidate", "scope": "D_ONLY_NOT_P"},
            "max_SSH_future_execution_per_launcher": 2,
            "capacity_combination": "IN_MEMORY_FROM_IMMUTABLE_P_AND_D_CAPACITY_FILES",
            "raw_capacity_files_unchanged": True,
            "new_P_CIDs": {"34": contract.P34_CID, "35": contract.P35_CID},
            "new_D_CIDs": {"36": contract.D36_CID, "41": contract.D41_CID},
            "D_service_ids": {"36": contract.D36_SID, "41": contract.D41_SID},
            "P_KV_capacity_floor": contract.P_KV_FLOOR,
            "D_KV_capacity_floor": contract.D_KV_FLOOR,
            "old_D_reported_KV_capacity": contract.OLD_D_KV,
            "layout": "237 kv_item_lens + 11 aux_item_lens; BF16 target78/draft1; MTP5/6",
            "D_eager_input": "All32 actual workers require native flag true, clone distinct, is_draft_worker true, forward_mode DECODE, batch_size 2 and real out_cache_loc shape/dtype/stride after functional; global32 is a composite node+namespace_pid join.",
            "short_counts": {"logical": 32, "backend": 64, "D_output_per_case": 600, "success_output_if_all": 19232},
            "long_counts": {"logical": 6, "backend": 12, "output_if_all": 2408},
            "functional_counts": {"backend": 76, "output_if_all": 21640},
            "formal_counts": {"warm_logical": 32, "formal": 128, "offered_backend": 320, "output_if_all": 76992, "formal_D_output": 76800},
            "proposed_total": {"backend": 396, "output_if_all": 98632, "new_profiles": 0, "retries": 0, "ledger_increment": False},
            "qualification_order": "startup capture/audit -> short functional -> post-functional capture -> long functional -> readiness -> functional decision/review -> formal -> independent metrics/client/native audit",
        },
        "pending_reviews": {
            "status": "PENDING_PRIMARY_REVIEW",
            "created": False,
            "approved_created": False,
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
            "actual_review_files_created_now": False,
            "new_review_files_never_APPROVED": True,
        },
        "missing_actual_evidence": missing,
        "pending_actual_evidence": pending,
        "actual_reviews_created": False,
        "formal_settings_actual_sha_fields": {
            key: settings[key] for key in [
                "native_qualification_sha256",
                "startup_qualification_sha256",
                "functional_decision_sha256",
                "functional_actual_review_sha256",
                "actual_review_sha256",
            ]
        },
        "counts": counts,
        "boundaries": {
            "SSH": 0,
            "network": 0,
            "HTTP_external": 0,
            "Docker": 0,
            "NPU": 0,
            "SDK_init": 0,
            "model_requests": 0,
            "new_profiles": 0,
            "remote_writes": 0,
            "Git": 0,
            "capture_executed": False,
            "framework_torch_or_SDK_import": False,
            "existing_frozen_artifact_edits": 0,
            "old_source_edits": 0,
            "state_handoff_loop_capacity_reviews_edits": 0,
            "max_SSH_future_execution_per_launcher": 2,
        },
        "activation": {
            "requires": [
                "actual D model-start/observe completion and approved startup capture review",
                "new startup auditor result",
                "new short functional result and approved short review",
                "new post-functional capture with all32 D bs2 draft-DECODE rows joined by node+namespace_pid",
                "new long functional result",
                "new readiness result",
                "new functional decision/actual review",
                "new formal actual review and frozen settings SHA values",
            ],
            "next_action": "Primary review/freeze of the corrected local drafts; no execution follows from this result.",
        },
        "verification": str(VERIFICATION.relative_to(ROOT)),
        "receipt": str(RECEIPT.relative_to(ROOT)),
        "created_at": "2026-10-19",
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result_sha = sha(str(RESULT.relative_to(ROOT)))
    verification = {
        "project_id": "glm53",
        "run_id": CWD.name,
        "mode": "LOCAL_ONLY",
        "result": {"path": str(RESULT.relative_to(ROOT)), "sha256": result_sha},
        "changed_files": changed,
        "new_files": result["new_files"],
        "unmodified_v1_files": unmodified,
        "capture_core": result["capture_core"],
        "local_qualification": result["local_qualification"],
        "host_codes": result["host_codes"],
        "missing_actual_evidence": result["missing_actual_evidence"],
        "runtime_authorized": False,
    }
    VERIFICATION.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    HOST_HASHES.write_text(json.dumps(result["host_codes"], indent=2) + "\n", encoding="utf-8")
    ENTRY_GATE.write_text(json.dumps(entry_gate, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "project_id": "glm53",
        "run_id": CWD.name,
        "mode": "LOCAL_ONLY_PIPELINE_DRAFT_V2",
        "status": "LOCAL_DRAFT_V2_COMPLETE_PRIMARY_REVIEW_PENDING",
        "runtime_authorized": False,
        "deployment_authorized": False,
        "formal_execution_authorized": False,
        "model_start_authorized": False,
        "objective_complete": False,
        "scope": "Mechanical L17 remaining-pipeline v2 drafts, old-source provenance, compile/AST/symtable/hash qualification, synthetic entry-prefix receipt gate and existing CPU gateway fixture only.",
        "result": {"path": str(RESULT.relative_to(ROOT)), "sha256": result_sha},
        "verification": {"path": str(VERIFICATION.relative_to(ROOT)), "sha256": sha(str(VERIFICATION.relative_to(ROOT)))},
        "host_code_hashes": {"path": str(HOST_HASHES.relative_to(ROOT)), "sha256": sha(str(HOST_HASHES.relative_to(ROOT)))},
        "entry_gate_qualification": {"path": str(ENTRY_GATE.relative_to(ROOT)), "sha256": sha(str(ENTRY_GATE.relative_to(ROOT)))},
        "changed_files": changed,
        "capture_core_sha256": capture_sha,
        "counts": counts,
        "entry_gate": {
            "verdict": entry_gate["verdict"],
            "current_receipt_rejected": entry_gate["actual_receipt"]["gate_result"] is False,
            "synthetic_real_schema_positive_accepted": entry_gate["synthetic_positive"]["gate_result"] is True,
            "negative_case_count": entry_gate["negative_gate_case_count"],
            "mkdir_or_capture_core_before_rejection": False,
            "remote_or_capture_or_client_main_executed": False,
        },
        "pure_cpu_validation": result["pure_cpu_validation"],
        "schema_audit": result["schema_audit"],
        "missing_actual_evidence": result["missing_actual_evidence"],
        "pending_reviews": result["pending_reviews"],
        "external_execution": {
            "SSH": 0,
            "network": 0,
            "HTTP": 0,
            "Docker": 0,
            "NPU": 0,
            "SDK_init": 0,
            "model_requests": 0,
            "new_profiles": 0,
            "Git": 0,
            "capture": 0,
            "client": 0,
        },
        "v1_result_preserved": {"path": "results/C32-L17-remaining-pipeline-local-draft-v1.json", "sha256": v1_result_sha_before},
        "v1_receipt_verification_preserved": True,
        "state_handoff_loop_capacity_reviews_edited": False,
        "model_coordinator_or_old_source_edited": False,
        "model_runtime_edited": False,
        "ledger_incremented": False,
        "maximum_SSH_future_execution_per_launcher": 2,
        "created_at": "2026-10-19",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(RESULT.relative_to(ROOT)),
        "result_sha256": result_sha,
        "receipt": str(RECEIPT.relative_to(ROOT)),
        "receipt_sha256": sha(str(RECEIPT.relative_to(ROOT))),
        "entry_gate_verdict": entry_gate["verdict"],
        "negative_gate_cases": entry_gate["negative_gate_case_count"],
        "changed_files": len(CHANGED_FILES),
        "files": len(NEW_FILES),
        "symtable_all_resolved": result["local_qualification"]["symtable_all_resolved"],
        "missing_actual_evidence": len(missing),
        "runtime_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
