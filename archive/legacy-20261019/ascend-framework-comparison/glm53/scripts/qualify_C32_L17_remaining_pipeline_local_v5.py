"""DRAFT_PENDING_PRIMARY_REVIEW: local-only v5 qualification of the v3 startup wire.

Mechanically qualifies the still-unexecuted L17 drafts after the preserved
diagnostic capture-v2 failure and the approved diagnostic capture-v3 collector:

1. the active pipeline contract pins capture-v3, review-v3 and startup-result-v3;
2. post-functional, formal-native and independent-metrics wrappers pin capture
   core v3 (SHA f83c374a7c4b75d2af2c19b74cc6e913b5be5f049764c7f9e04671a44a24ef3f);
3. the formal settings startup qualification path follows startup-result-v3;
4. the startup auditor keeps the parent NoCopy identity and the D Ops
   source-pinned gate, reports worker /proc exec-initial values as-is with the
   expected-match mask, and re-derives per-worker live first-use getter/clone
   evidence from actual rows only;
5. a local CPU fixture proves that missing worker initial flags plus actual
   native first-use True rows are accepted as getter evidence, while a false
   getter, a missing worker row, a wrong source SHA or clone_distinct false is
   rejected; progress and environment claims are not consulted;
6. capture core v3 startup plan hashes are re-derived locally and must equal the
   already-approved review-v3 host hashes, proving the contract wire change does
   not alter the diagnostic capture;
7. the terminal diagnostic capture-v3 receipt and review are recorded but never
   consumed as startup evidence; the 13 native errors remain pending primary raw
   localization and this script never declares startup-pass.

This script makes no SSH/network/Docker/NPU/SDK/model/Git call, starts no
capture, client or native auditor, reads no credential/provider file, writes no
APPROVED review, and preserves all v1/v2/v3/v4 results, receipts, verification
files, reviews, coordinator source, capture cores and identity.  Synthetic
fixtures are local qualification only; they are not native, model or formal
evidence.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import py_compile
import tempfile
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v1 as contract

ROOT = Path(__file__).resolve().parents[1]
CWD = ROOT / "runs/20261019T-L17-pipeline-initial-vs-live-env-local-v1"
RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v5.json"
V1_RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v1.json"
V2_RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v2.json"
V3_RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v3.json"
V4_RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v4.json"
V1_CWD = ROOT / "runs/20261019T-L17-functional-formal-local-draft-executor-v1"
V2_CWD = ROOT / "runs/20261019T-L17-pipeline-wire-fix-local-v1"
V3_CWD = ROOT / "runs/20261019T-L17-pipeline-entry-fix-local-v1"
V4_CWD = ROOT / "runs/20261019T-L17-pipeline-identity-v2-wire-local-v1"
VERIFICATION = CWD / "verification.json"
HOST_HASHES = CWD / "host-code-hashes.json"
DEPENDENCY_GATE = CWD / "dependency-graph-qualification.json"
FIRST_USE_GATE = CWD / "startup-first-use-qualification.json"
FIXTURE_SUMMARY = CWD / "fixture-summary.json"
RECEIPT = CWD / "receipt.json"

CONTRACT = "scripts/glm53_C32_L17_pipeline_contract_v1.py"
CAPTURE = "scripts/glm53_C32_L17_native_capture_v3.py"
CAPTURE_V2 = "scripts/glm53_C32_L17_native_capture_v2.py"
CAPTURE_V1 = "scripts/glm53_C32_L17_native_capture_v1.py"
CAPTURE_SHA = "f83c374a7c4b75d2af2c19b74cc6e913b5be5f049764c7f9e04671a44a24ef3f"
CAPTURE_V2_SHA = "d21502d853892799273b4fe25df8a101a0aa35031c165d8a67a6757350f2bdbd"
CAPTURE_V1_SHA = "229893e45c8f0778699889446cd5f7d9097f7c8d1b6873612cc90eb679c793f2"
IDENTITY = "scripts/glm53_C32_L17_mixed_model_identity_v2.py"
IDENTITY_V1 = "scripts/glm53_C32_L17_mixed_model_identity_v1.py"
STARTUP_ENTRY = "scripts/inspect_C32_L17_post_timeout_native_startup_v3.py"
STARTUP_ENTRY_V2 = "scripts/inspect_C32_L17_post_timeout_native_startup_v2.py"
STARTUP_ENTRY_V1_FROZEN = "scripts/inspect_C32_L17_native_startup_v1.py"
STARTUP_ENTRY_V1_FAILED = "scripts/inspect_C32_L17_post_timeout_native_startup_v1.py"
STARTUP_CAPTURE_V1 = "20261019T-L17-sglang-native-startup-capture-v1"
STARTUP_CAPTURE_V2 = "20261019T-L17-sglang-native-startup-capture-v2"
STARTUP_REVIEW_V1 = "results/C32-L17-native-startup-capture-review-v1.json"
STARTUP_REVIEW_V2 = "results/C32-L17-native-startup-capture-review-v2.json"
STARTUP_REVIEW_V3 = "results/C32-L17-native-startup-capture-review-v3.json"
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
FORMAL_SETTINGS = "config/C32-L17-native-runtime-v1/formal-C32-native-v1.json"
COORDINATOR = "scripts/start_C32_L17_selected_D_models_and_observe_v1.py"
QUALIFIER = "scripts/qualify_C32_L17_remaining_pipeline_local_v5.py"
V4_QUALIFIER = "scripts/qualify_C32_L17_remaining_pipeline_local_v4.py"
D_START_RECEIPT = "runs/20261019T-L17-D-models-and-observe-v1/receipt.json"

CHANGED_FILES = [CONTRACT, STARTUP_AUDIT, POST_WRAPPER, FORMAL_WRAPPER, METRICS_INSPECTOR, FORMAL_SETTINGS]
COMPILE_FILES = [
    CONTRACT,
    STARTUP_AUDIT,
    POST_WRAPPER,
    FORMAL_WRAPPER,
    METRICS_INSPECTOR,
    FORMAL_SETTINGS,
    CAPTURE,
    STARTUP_ENTRY,
    QUALIFIER,
]
FROZEN_REFERENCES = [
    CAPTURE_V1,
    CAPTURE_V2,
    IDENTITY_V1,
    STARTUP_ENTRY_V1_FROZEN,
    STARTUP_ENTRY_V1_FAILED,
    STARTUP_ENTRY_V2,
    V4_QUALIFIER,
]
ACTIVE_CONSUMERS = [
    CONTRACT,
    STARTUP_ENTRY,
    POST_WRAPPER,
    FORMAL_WRAPPER,
    METRICS_INSPECTOR,
    FORMAL_SETTINGS,
    STARTUP_AUDIT,
    SHORT_RUNNER,
    LONG_RUNNER,
    SHORT_CLIENT_AUDIT,
    FUNCTIONAL_AUDIT,
    READINESS_AUDIT,
    FORMAL_RUNNER,
    FORMAL_CONTROLLER,
    FORMAL_CLIENT_AUDIT,
    FORMAL_NATIVE_AUDIT,
]

STALE_TOKENS = (
    "glm53_C32_L17_mixed_model_identity_v1.py",
    "glm53_C32_L17_native_capture_v1.py",
    "glm53_C32_L17_native_capture_v2.py",
    "C32-L17-native-startup-capture-review-v1.json",
    "C32-L17-native-startup-capture-review-v2.json",
    "sglang-C32-L17-native-startup-v1.json",
    "sglang-C32-L17-native-startup-v2.json",
    "20261019T-L17-sglang-native-startup-capture-v1",
    "20261019T-L17-sglang-native-startup-capture-v2",
)

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


def import_file(name, relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def snapshot_tree(path):
    return {
        str(item.relative_to(ROOT)): sha_path(item)
        for item in sorted(Path(path).rglob("*")) if item.is_file()
    }


def cids():
    p_capacity, d_capacity = contract.check_capacity_pair()
    return {
        str(node): (p_capacity if node in contract.P_NODES else d_capacity)["providers"][str(node)]["container_id"]
        for node in contract.ALL_NODES
    }


CID_MAP = cids()


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


def startup_fixture():
    return {
        "project_id": "glm53",
        "fixture": True,
        "not_actual_evidence": True,
        "pass": True,
        "all64_actual_SDMA_and_BF16_layout_verified": True,
        "all64_actual_source_weights_image_CLI_and_role_specific_execution_settings_verified": True,
        "all32_D_nativeOps_fullmesh_and_targetGraph_retained": True,
        "all32_P_native_IO_and_existing_MC_mapping_reuse_verified": True,
        "D_KV_capacity_floor": contract.D_KV_FLOOR,
        "runtime_CIDs": dict(CID_MAP),
        "capture_id": contract.STARTUP_CAPTURE,
        "role_specific_execution_settings_scope": "D only TASK_QUEUE_ENABLE=0/HCCL_BUFFSIZE=256; P original values remain empty.",
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
        "receipt_pass": False,
        "all_four_rc0": False,
        "terminal": False,
        "valid_actual_evidence": False,
        "startup_evidence_consumed": False,
        "startup_pass_declared": False,
        "modified_by_this_run": False,
        "touched_by_this_run": False,
    }
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        nodes = receipt.get("nodes") if isinstance(receipt.get("nodes"), dict) else {}
        all_rc0 = len(nodes) == 4 and all(type(value) is int and value == 0 for value in nodes.values())
        state.update({
            "receipt_sha256": sha_path(receipt_path),
            "nodes": nodes,
            "receipt_pass": receipt.get("pass") is True,
            "all_four_rc0": all_rc0,
            "terminal": True,
        })
        state["reason"] = (
            "Diagnostic capture receipt only. Startup qualification additionally requires "
            "the independent startup audit and primary raw localization of the 13 native errors."
        )
    else:
        state["reason"] = "capture directory present without a terminal receipt" if state["present"] else "capture absent"
    return state


def startup_review_state(relative):
    path = ROOT / relative
    state = {
        "path": relative,
        "present": path.is_file(),
        "sha256": sha_path(path) if path.is_file() else None,
        "decision": None,
        "core_sha256": None,
        "launcher_sha256": None,
        "model_requests": None,
        "SDK_init_calls": None,
        "startup_qualified": None,
        "runtime_model_dispatch_authorized": None,
        "worker_environment_scope": None,
        "valid_for_startup_gate": False,
        "startup_pass_declared": False,
        "consumed_as_evidence": False,
        "created_by_this_run": False,
        "modified_by_this_run": False,
    }
    if path.is_file():
        review = json.loads(path.read_text())
        state.update({
            "decision": review.get("decision"),
            "core_sha256": review.get("core_sha256"),
            "launcher_sha256": review.get("launcher_sha256"),
            "model_requests": review.get("model_requests"),
            "SDK_init_calls": review.get("SDK_init_calls"),
            "startup_qualified": review.get("startup_qualified"),
            "runtime_model_dispatch_authorized": review.get("runtime_model_dispatch_authorized"),
            "worker_environment_scope": review.get("worker_environment_scope"),
        })
        state["valid_for_startup_gate"] = (
            review.get("decision") == "APPROVED"
            and review.get("startup_qualified") is True
            and review.get("runtime_model_dispatch_authorized") is True
        )
        if not state["valid_for_startup_gate"]:
            state["reason"] = "diagnostic approval only; does not qualify startup"
    return state


def first_use_fixture(startup_audit):
    """CPU-only proof of the live getter/clone join and its rejection cases."""
    node = 36
    pids = list(range(100, 116))
    expected = startup_audit.worker_environment_expectation(node)
    workers = [
        {
            "namespace_pid": pid,
            "host_pid": 2000 + index,
            "device_id": index,
            "selected_environment": {},
            "initial_environment_expected_flags_match": {key: False for key in expected},
            "alive": True,
            "cgroup_contains_current_cid": True,
            "other_environment_retained": False,
            "full_environment_hashed": False,
        }
        for index, pid in enumerate(pids)
    ]
    rows = []
    for pid in pids:
        row = {
            "schema": "glm53.L17.eager_input_audit.v2",
            "audit_version": "v2",
            "project_id": "glm53",
            "source_sha256": contract.EAGER_SOURCE_SHA,
            "eager_source_sha256": contract.EAGER_SOURCE_SHA,
            "environ_source_sha256": contract.ENVIRON_SOURCE_SHA,
            "node": node,
            "pid": pid,
            "native_no_copy_get": True,
            "clone_distinct": True,
            "is_draft_worker": False,
            "batch_size": 1,
            "forward_mode": "DECODE",
            "dataclass_field_count": 2,
            "field_verdicts": {"present": 2},
            "field_names_sha256": "a" * 64,
            "allowlisted": {name: None for name in contract.AUDIT_ALLOWLIST_FIELDS},
        }
        rows.append({"path": "/tmp/pid-%d.jsonl" % pid, "line": 1, "row": row})
    coverage = {
        "workers": 16,
        "flag_true_workers": 16,
        "clone_ok_workers": 16,
        "draft_decode_bs2_workers": 0,
        "out_cache_loc_workers": 0,
        "bs2_decode_rows": 0,
        "out_cache_loc_rows": 0,
        "all_16_workers": True,
        "all_flag_true": True,
        "all_clone_ok": True,
        "all_bs2_draft_decode": False,
        "all_out_cache_loc_tensor_metadata": False,
        "no_row_errors": True,
    }
    files = [
        {
            "path": item["path"],
            "rows": 1,
            "bytes": 1,
            "sha256": "0" * 64,
            "complete_line_ending": True,
        }
        for item in rows
    ]
    d_eager = {
        "applicable": True,
        "files": files,
        "records": rows,
        "file_errors": [],
        "coverage": coverage,
        "worker_environment_and_identity": workers,
        "progress": {
            "eager_status": "COMPLETE_ALL16",
            "eager_bs2_draft_decode_pids": 16,
        },
        "coverage_status": "PENDING_OR_INCOMPLETE_LOCAL16_BS2_DRAFT_DECODE",
        "coverage_pass": False,
    }
    positive = startup_audit.first_use_coverage(d_eager, node, set(pids))
    mask_rows, mask = startup_audit.worker_initial_environment(workers, node)
    assert positive["actual_sdk_workers"] == 16
    assert positive["workers_with_native_no_copy_get"] == 16
    assert positive["workers_with_clone_distinct"] == 16
    assert positive["all_flag_true"] is True and positive["all_clone_ok"] is True
    assert positive["no_row_errors"] is True
    assert positive["bs2_observed_workers"] == 0 and positive["bs2_observed_rows"] == 0
    assert positive["bs2_required_at_startup"] is False
    assert mask == {
        "true": 0,
        "false": 96,
        "mask_keys_per_worker": 6,
        "mask_missing_forced_to_true": False,
    }
    assert len(mask_rows) == 16
    assert all(row["initial_environment_raw"] == {} for row in mask_rows)
    assert all(row["missing_initial_keys"] == sorted(expected) for row in mask_rows)
    assert all(all(value is False for value in row["initial_environment_expected_flags_match"].values()) for row in mask_rows)

    def expect_rejected(name, mutate):
        candidate = copy.deepcopy(d_eager)
        mutate(candidate)
        try:
            startup_audit.first_use_coverage(candidate, node, set(pids))
        except AssertionError as error:
            return {"rejected": True, "error": str(error)[:240]}
        raise AssertionError("first-use gate accepted negative case: " + name)

    negatives = {
        "actual_getter_false": expect_rejected(
            "actual_getter_false", lambda item: item["records"][0]["row"].__setitem__("native_no_copy_get", False)
        ),
        "clone_distinct_false": expect_rejected(
            "clone_distinct_false", lambda item: item["records"][0]["row"].__setitem__("clone_distinct", False)
        ),
        "wrong_source_sha": expect_rejected(
            "wrong_source_sha", lambda item: item["records"][0]["row"].__setitem__("source_sha256", "0" * 64)
        ),
        "missing_one_worker_rows": expect_rejected(
            "missing_one_worker_rows", lambda item: item["records"].pop(0)
        ),
        "reported_flag_false": expect_rejected(
            "reported_flag_false", lambda item: item["coverage"].__setitem__("all_flag_true", False)
        ),
        "row_error_present": expect_rejected(
            "row_error_present", lambda item: item.__setitem__("file_errors", [{"path": "fixture"}])
        ),
    }
    env_negative = copy.deepcopy(workers)
    env_negative[0]["initial_environment_expected_flags_match"]["SGLANG_EAGER_INPUT_NO_COPY"] = True
    env_rejected = False
    env_error = None
    try:
        startup_audit.worker_initial_environment(env_negative, node)
    except AssertionError as error:
        env_rejected = True
        env_error = str(error)[:240]
    assert env_rejected
    negatives["missing_initial_flag_claimed_true"] = {"rejected": True, "error": env_error}

    return {
        "module": STARTUP_AUDIT,
        "positive": {
            "workers": positive["actual_sdk_workers"],
            "workers_with_native_no_copy_get": positive["workers_with_native_no_copy_get"],
            "workers_with_clone_distinct": positive["workers_with_clone_distinct"],
            "all_flag_true": positive["all_flag_true"],
            "all_clone_ok": positive["all_clone_ok"],
            "no_row_errors": positive["no_row_errors"],
            "bs2_observed_workers": positive["bs2_observed_workers"],
            "bs2_required_at_startup": positive["bs2_required_at_startup"],
            "coverage_pass_as_reported": d_eager["coverage_pass"],
            "coverage_status_as_reported": d_eager["coverage_status"],
            "coverage_pass_required_at_startup": False,
            "evidence_source": positive["evidence_source"],
            "per_worker": positive["per_worker"],
            "initial_environment_mask": mask,
            "initial_environment_rows": len(mask_rows),
            "initial_environment_missing_keys_preserved": True,
            "progress_claim_present_but_ignored": True,
        },
        "negative_cases": negatives,
        "negative_case_count": len(negatives),
        "fixture_is_cpu_only": True,
        "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
        "verdict": "PASS",
    }


def core_pin_fixture():
    expected = sha(CAPTURE)
    assert expected == CAPTURE_SHA
    pins = {}
    for name, relative in [
        ("startup_entry", STARTUP_ENTRY),
        ("post_wrapper", POST_WRAPPER),
        ("formal_wrapper", FORMAL_WRAPPER),
    ]:
        module = import_file("l17_v5_pin_" + name, relative)
        assert module.CORE_SHA == expected, name
        assert str(module.CORE).endswith(CAPTURE), name
        pins[name] = {"path": relative, "core": CAPTURE, "core_sha256": module.CORE_SHA}
    startup_entry = import_file("l17_v5_startup_entry_pin", STARTUP_ENTRY)
    assert startup_entry.REVIEW.name == STARTUP_REVIEW_V3.split("/")[-1]
    assert metrics_core_pin() == expected

    pin_negatives = {}
    with tempfile.TemporaryDirectory(prefix="l17-v5-core-pin-") as directory:
        temp = Path(directory)
        for name, relative in [("post_wrapper", POST_WRAPPER), ("formal_wrapper", FORMAL_WRAPPER)]:
            module = import_file("l17_v5_pin_negative_" + name, relative)
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
        entry = import_file("l17_v5_startup_entry_negative", STARTUP_ENTRY)
        original_core = entry.CORE
        try:
            target = temp / CAPTURE
            target.write_bytes(b"tampered\n")
            entry.CORE = target
            rejected = False
            try:
                entry.plan()
            except AssertionError:
                rejected = True
            assert rejected
            pin_negatives["startup_entry"] = {"tampered_core_rejected": True}
        finally:
            entry.CORE = original_core

    return {
        "capture_core": CAPTURE,
        "capture_core_sha256": expected,
        "wrapper_pins": pins,
        "metrics_inspector_pin": metrics_core_pin(),
        "pin_negative_cases": pin_negatives,
        "all_active_pins_match": True,
        "verdict": "PASS",
    }


def dependency_fixture(v4):
    attrs = {
        "long_runner": v4.contract_attrs(LONG_RUNNER),
        "post_wrapper": v4.contract_attrs(POST_WRAPPER),
        "short_client_audit": v4.contract_attrs(SHORT_CLIENT_AUDIT),
        "full_short_audit": v4.contract_attrs(FUNCTIONAL_AUDIT),
        "readiness": v4.contract_attrs(READINESS_AUDIT),
        "formal_runner": v4.contract_attrs(FORMAL_RUNNER),
    }
    assert contract.MIXED_IDENTITY == IDENTITY
    assert contract.MIXED_IDENTITY_SHA == sha(IDENTITY)
    assert contract.STARTUP_CAPTURE == "20261019T-L17-sglang-native-startup-capture-v3"
    assert contract.STARTUP_CAPTURE_REVIEW == "results/C32-L17-native-startup-capture-review-v3.json"
    assert contract.STARTUP_RESULT == "results/sglang-C32-L17-native-startup-v3.json"
    settings = json.loads((ROOT / FORMAL_SETTINGS).read_text())
    assert settings["model_identity_helper"] == contract.MIXED_IDENTITY
    assert settings["model_identity_sha256"] == contract.MIXED_IDENTITY_SHA
    assert settings["startup_qualification_path"] == contract.STARTUP_RESULT
    assert settings["startup_qualification_sha256"] is None
    stale = {}
    for relative in ACTIVE_CONSUMERS:
        text = (ROOT / relative).read_text()
        hits = [token for token in STALE_TOKENS if token in text]
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
    startup_entry_module = import_file("l17_v5_dependency_startup_entry", STARTUP_ENTRY)
    assert startup_entry_module.CORE_SHA == sha(CAPTURE)
    assert startup_entry_module.REVIEW.name == STARTUP_REVIEW_V3.split("/")[-1]
    assert "CLIENT_RESULT" in attrs["long_runner"] and "STARTUP_RESULT" in attrs["long_runner"]
    assert "SHORT_RESULT" not in attrs["long_runner"], "long runner must not require full short result"
    assert "CLIENT_RESULT" in attrs["post_wrapper"] and "STARTUP_RESULT" in attrs["post_wrapper"]
    assert "SHORT_RESULT" not in attrs["post_wrapper"], "post-functional wrapper must not require full short result"
    assert "CLIENT_RESULT" in attrs["short_client_audit"]
    assert "SHORT_RESULT" in attrs["short_client_audit"], "client audit may hash but not require full short result"
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
            "startup_qualification_sha256": settings["startup_qualification_sha256"],
        },
        "stale_identity_references": stale,
        "stale_reference_count": sum(len(hits) for hits in stale.values()),
        "frozen_references": frozen,
        "active_consumers": ACTIVE_CONSUMERS,
        "current_startup_entry": STARTUP_ENTRY,
        "superseded_startup_entries": [STARTUP_ENTRY_V1_FROZEN, STARTUP_ENTRY_V1_FAILED, STARTUP_ENTRY_V2],
        "verdict": "PASS",
    }


def startup_plan_hash_fixture(capture):
    hashes = {
        mode: capture.plan_hashes(capture_id, mode)
        for mode, capture_id in capture.MODE_CAPTURE.items()
    }
    review = json.loads((ROOT / STARTUP_REVIEW_V3).read_text())
    assert review["core_sha256"] == CAPTURE_SHA
    assert review["host_codes_sha256"] == hashes["startup"]
    assert review["decision"].startswith("APPROVED_DIAGNOSTIC")
    assert review["startup_qualified"] is False
    assert review["runtime_model_dispatch_authorized"] is False
    assert review["model_requests"] == 0 and review["SDK_init_calls"] == 0
    assert capture.MODE_CAPTURE["startup"] == contract.STARTUP_CAPTURE
    assert capture.MODE_REVIEW["startup"] == contract.STARTUP_CAPTURE_REVIEW
    return {
        "capture_id": capture.MODE_CAPTURE,
        "review_paths": capture.MODE_REVIEW,
        "plan_hashes": hashes,
        "startup_plan_matches_review_v3": True,
        "review_decision": review["decision"],
        "review_startup_qualified": review["startup_qualified"],
        "review_runtime_model_dispatch_authorized": review["runtime_model_dispatch_authorized"],
    }


def previous_sha_map(v4_doc):
    out = {}
    for relative, item in v4_doc.get("changed_files", {}).items():
        out[relative] = item.get("v4_sha256")
    for relative, item in v4_doc.get("new_sources", {}).items():
        out.setdefault(relative, item.get("sha256"))
    for relative, item in v4_doc.get("unmodified_v3_files", {}).items():
        out.setdefault(relative, item.get("actual_sha256"))
    return out


def main():
    assert not RESULT.exists(), "Preserve existing v5 result"
    assert CWD.is_dir(), "v5 cwd must exist"
    assert not RECEIPT.exists(), "Preserve existing v5 receipt"
    assert not VERIFICATION.exists() and not HOST_HASHES.exists() and not DEPENDENCY_GATE.exists()
    assert not FIRST_USE_GATE.exists() and not FIXTURE_SUMMARY.exists()
    for path in (V1_RESULT, V2_RESULT, V3_RESULT, V4_RESULT):
        assert path.is_file(), str(path)
    for path in (V1_CWD, V2_CWD, V3_CWD, V4_CWD):
        assert path.is_dir(), str(path)

    v1_result_sha_before = sha_path(V1_RESULT)
    v2_result_sha_before = sha_path(V2_RESULT)
    v3_result_sha_before = sha_path(V3_RESULT)
    v4_result_sha_before = sha_path(V4_RESULT)
    v4_doc = json.loads(V4_RESULT.read_text())
    previous = previous_sha_map(v4_doc)
    v1_tree_before = snapshot_tree(V1_CWD)
    v2_tree_before = snapshot_tree(V2_CWD)
    v3_tree_before = snapshot_tree(V3_CWD)
    v4_tree_before = snapshot_tree(V4_CWD)
    protected = ["STATE.json", "HANDOFF.md", COORDINATOR]
    protected += [
        contract.P_CAPACITY,
        contract.D_CAPACITY,
        "diagnostics/model-profile/loops/L17-D-native-eager-live-inputs-v1.json",
        "diagnostics/model-profile/L17-primary-pipeline-entry-review-v2.md",
        str(V1_RESULT.relative_to(ROOT)),
        str(V2_RESULT.relative_to(ROOT)),
        str(V3_RESULT.relative_to(ROOT)),
        str(V4_RESULT.relative_to(ROOT)),
        CAPTURE_V1,
        CAPTURE_V2,
        CAPTURE,
        IDENTITY_V1,
        IDENTITY,
        STARTUP_ENTRY,
    ]
    protected += sorted(str(path.relative_to(ROOT)) for path in (ROOT / "results").glob("*review*.json"))
    protected += sorted(str(path.relative_to(ROOT)) for path in (ROOT / "results").glob("*decision*.json"))
    protected = sorted(set(protected))
    protected_before = {relative: sha(relative) for relative in protected if (ROOT / relative).is_file()}
    old_sources_before = {item["path"]: sha(item["path"]) for item in contract.old_source_provenance().values()}
    frozen_before = {relative: sha(relative) for relative in FROZEN_REFERENCES}

    compile_results = {}
    ast_results = {}
    with tempfile.TemporaryDirectory(prefix="l17-v5-pyc-") as pyc_dir:
        for index, relative in enumerate(sorted(set(COMPILE_FILES + FROZEN_REFERENCES))):
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

    v4 = import_file("l17_v5_v4_fixture_library", V4_QUALIFIER)
    capture = import_file("l17_v5_capture_v3", CAPTURE)
    identity = import_file("l17_v5_identity_v2", IDENTITY)
    short_runner = import_file("l17_v5_short_runner", SHORT_RUNNER)
    long_runner = import_file("l17_v5_long_runner", LONG_RUNNER)
    startup_audit = import_file("l17_v5_startup_audit", STARTUP_AUDIT)
    short_client_audit = import_file("l17_v5_short_client_audit", SHORT_CLIENT_AUDIT)
    post_wrapper = import_file("l17_v5_post_wrapper", POST_WRAPPER)
    formal_wrapper = import_file("l17_v5_formal_wrapper", FORMAL_WRAPPER)
    startup_entry = import_file("l17_v5_startup_entry", STARTUP_ENTRY)
    readiness_audit = import_file("l17_v5_readiness", READINESS_AUDIT)
    functional_audit = import_file("l17_v5_full_short_audit", FUNCTIONAL_AUDIT)
    formal_runner = import_file("l17_v5_formal_runner", FORMAL_RUNNER)
    metrics_inspector = import_file("l17_v5_metrics_inspector", METRICS_INSPECTOR)

    d_start_receipt = json.loads((ROOT / D_START_RECEIPT).read_text())
    d_start_gate = v4.startup_receipt_gate(d_start_receipt)
    d_start = v4.d_start_state()
    assert d_start["present"] and d_start["sha256"] == sha(D_START_RECEIPT)

    settings = json.loads((ROOT / FORMAL_SETTINGS).read_text())
    assert settings["resource_controller_sha256"] == sha(FORMAL_CONTROLLER)
    assert settings["model_identity_helper"] == contract.MIXED_IDENTITY
    assert settings["model_identity_sha256"] == contract.MIXED_IDENTITY_SHA
    assert settings["startup_qualification_path"] == contract.STARTUP_RESULT
    assert settings["startup_qualification_sha256"] is None
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

    startup_plan = startup_plan_hash_fixture(capture)
    core_pins = core_pin_fixture()
    dependency_gate = dependency_fixture(v4)
    first_use_gate = first_use_fixture(startup_audit)
    assert first_use_gate["verdict"] == "PASS"

    short_plan = short_runner.plan()
    long_plan = long_runner.plan()
    host_codes = {
        "capture": startup_plan["plan_hashes"],
        "startup_plan_matches_review_v3": True,
        "short": {name: hashlib.sha256(code.encode()).hexdigest() for name, code in short_plan.items()},
        "long": {name: hashlib.sha256(code.encode()).hexdigest() for name, code in long_plan.items()},
        "capture_core_sha256": sha(CAPTURE),
        "capture_core_v2_sha256": sha(CAPTURE_V2),
        "capture_core_v1_sha256": sha(CAPTURE_V1),
        "identity_sha256": sha(IDENTITY),
        "identity_v1_sha256": sha(IDENTITY_V1),
        "wrappers": {
            STARTUP_ENTRY: {"core_sha256": startup_entry.CORE_SHA},
            POST_WRAPPER: {"core_sha256": post_wrapper.CORE_SHA},
            FORMAL_WRAPPER: {"core_sha256": formal_wrapper.CORE_SHA},
            METRICS_INSPECTOR: {"core_sha256": metrics_core_pin()},
        },
        "startup_audit_sha256": sha(STARTUP_AUDIT),
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
            identity_code = v4.identity_literal(code)
            assert identity_code is not None
            identity_unbound = v4.unbound_globals(identity_code, label + "-identity")
            outer_unbound = v4.unbound_globals(
                code, label, v4.symtable_module_names(identity_code, label + "-identity-extra")
            )
            assert not identity_unbound and not outer_unbound, (label, identity_unbound, outer_unbound)
            symtable_results[label] = {"identity": identity_unbound, "outer": outer_unbound}
        if cpu is not None:
            label = "capture-%s-43" % mode
            symtable_results[label] = {"identity": [], "outer": v4.unbound_globals(cpu, label)}
            assert not symtable_results[label]["outer"]
    for name, code in short_plan.items():
        symtable_results["short-" + name] = v4.unbound_globals(code, "short-" + name)
        assert not symtable_results["short-" + name]
    for name, code in long_plan.items():
        symtable_results["long-" + name] = v4.unbound_globals(code, "long-" + name)
        assert not symtable_results["long-" + name]

    client_audit_result = v4.client_audit_fixture(short_client_audit)
    assert client_audit_result["verdict"] == "PASS"
    client_output = client_audit_result.pop("output")
    entry = {
        "short": v4.short_entry_fixture(short_runner),
        "long": v4.long_entry_fixture(long_runner, client_output),
        "fixtures_are_simulated": True,
        "not_native_or_model_evidence": True,
        "fixture_library": {"path": V4_QUALIFIER, "sha256": sha(V4_QUALIFIER)},
    }
    assert entry["short"]["verdict"] == "PASS" and entry["long"]["verdict"] == "PASS"
    post_gate = v4.post_wrapper_fixture(post_wrapper, client_output)
    assert post_gate["verdict"] == "PASS"
    codec_gate = v4.codec_identity_fixture(startup_audit)
    staged_gate = v4.identity_staged_loop_fixture()
    assert staged_gate["verdict"] == "PASS"

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
            "previous_v4_sha256": previous.get(relative),
            "v5_sha256": sha(relative),
            "bytes": (ROOT / relative).stat().st_size,
        }
    changed[QUALIFIER] = {
        "previous_v4_sha256": None,
        "v5_sha256": sha(QUALIFIER),
        "bytes": (ROOT / QUALIFIER).stat().st_size,
    }
    new_sources = {}
    for relative, creator in [
        (CAPTURE, "primary_created_before_this_run"),
        (STARTUP_ENTRY, "primary_created_before_this_run"),
        (QUALIFIER, "this_v5_run"),
    ]:
        new_sources[relative] = {
            "sha256": sha(relative),
            "bytes": (ROOT / relative).stat().st_size,
            "created_by": creator,
            "active": True,
        }

    startup_attempt_v1 = startup_attempt_state(STARTUP_CAPTURE_V1)
    startup_attempt_v2 = startup_attempt_state(STARTUP_CAPTURE_V2)
    startup_attempt_v3 = startup_attempt_state(contract.STARTUP_CAPTURE)
    startup_review_v1 = startup_review_state(STARTUP_REVIEW_V1)
    startup_review_v2 = startup_review_state(STARTUP_REVIEW_V2)
    startup_review_v3 = startup_review_state(STARTUP_REVIEW_V3)
    assert startup_review_v3["core_sha256"] == CAPTURE_SHA
    assert startup_review_v3["valid_for_startup_gate"] is False
    assert startup_review_v3["startup_qualified"] is False
    assert startup_review_v3["runtime_model_dispatch_authorized"] is False
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
    for label, state in (
        ("v1", startup_attempt_v1),
        ("v2", startup_attempt_v2),
        ("v3", startup_attempt_v3),
    ):
        if state["present"]:
            unqualified.append({
                "path": state["path"],
                "capture_id": state["capture_id"],
                "reason": state.get("reason"),
                "nodes": state["nodes"],
                "receipt_pass": state["receipt_pass"],
                "all_four_rc0": state["all_four_rc0"],
                "terminal": state["terminal"],
                "startup_evidence_consumed": False,
                "startup_pass_declared": False,
                "modified_by_this_run": False,
            })
    for label, state in (
        ("v1", startup_review_v1),
        ("v2", startup_review_v2),
        ("v3", startup_review_v3),
    ):
        if state["present"]:
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
            return False
        if item == contract.STARTUP_CAPTURE_REVIEW:
            return startup_review_v3["valid_for_startup_gate"]
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
        "superseded_v2_entry": STARTUP_ENTRY_V2,
        "failed_post_timeout_entry_v1": STARTUP_ENTRY_V1_FAILED,
        "superseded_entries_active": False,
        "functional_gate": "BLOCKED_PENDING_PRIMARY_ACCEPT_OF_V3_STARTUP_EVIDENCE_AND_NATIVE_ERROR_LOCALIZATION",
        "startup_pass_declared": False,
        "startup_qualified": False,
        "model_requests": 0,
    }
    actual_evidence_state = {
        "d_model_start_observe": {**d_start, "startup_receipt_gate_accepts_current": d_start_gate},
        "startup_capture_attempts": {
            "v1": startup_attempt_v1,
            "v2": startup_attempt_v2,
            "v3": startup_attempt_v3,
        },
        "startup_capture_reviews": {
            "v1": startup_review_v1,
            "v2": startup_review_v2,
            "v3": startup_review_v3,
        },
        "pending_native_errors": {
            "count": 13,
            "source": "primary handoff; review-v3 reason reports 7+6 errors",
            "primary_localization_complete": False,
            "raw_localization_owner": "primary",
            "consumed_as_evidence": False,
        },
        "unqualified_actual_evidence": unqualified,
        "startup_pass_declared": False,
        "startup_qualified": False,
        "external_artifacts_consumed_as_evidence": False,
        "capture_v3_receipt_consumed_as_evidence": False,
        "capture_v3_review_consumed_as_startup_evidence": False,
        "actual_native_auditor_executed": False,
        "note": (
            "Capture-v1 failed with four rc=1 nodes; capture-v2 failed on nodes 36/41 with 13 native errors; "
            "capture-v3 is a terminal diagnostic receipt (four rc=0) whose review is diagnostic-only. "
            "None is startup evidence until the independent startup audit passes and the primary localizes the 13 native errors."
        ),
    }

    assert sha_path(V1_RESULT) == v1_result_sha_before
    assert sha_path(V2_RESULT) == v2_result_sha_before
    assert sha_path(V3_RESULT) == v3_result_sha_before
    assert sha_path(V4_RESULT) == v4_result_sha_before
    assert snapshot_tree(V1_CWD) == v1_tree_before
    assert snapshot_tree(V2_CWD) == v2_tree_before
    assert snapshot_tree(V3_CWD) == v3_tree_before
    assert snapshot_tree(V4_CWD) == v4_tree_before
    assert {relative: sha(relative) for relative in frozen_before} == frozen_before
    assert {relative: sha(relative) for relative in old_sources_before} == old_sources_before
    assert {relative: sha(relative) for relative in protected_before} == protected_before

    result = {
        "project_id": "glm53",
        "result_id": "C32-L17-remaining-pipeline-local-draft-v5",
        "status": "LOCAL_DRAFT_V5_COMPLETE_PRIMARY_REVIEW_PENDING",
        "review_status": "PENDING_PRIMARY_REVIEW",
        "runtime_authorized": False,
        "deployment_authorized": False,
        "formal_execution_authorized": False,
        "model_start_authorized": False,
        "objective_complete": False,
        "startup_pass_declared": False,
        "scope": "Local stdlib/source v5 of the remaining L17 SGLang C32 v3 startup wire and initial-vs-live worker-environment correction only. No SSH/network/Docker/NPU/SDK/model/Git, no launcher execution, no client/capture/auditor start, no review approval.",
        "supersedes_for_review_only": str(V4_RESULT.relative_to(ROOT)),
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
        "v4_original_result": {
            "path": str(V4_RESULT.relative_to(ROOT)),
            "sha256": v4_result_sha_before,
            "modified": False,
            "receipt_tree": v4_tree_before,
        },
        "changed_files": changed,
        "changed_file_count": len(changed),
        "new_sources": new_sources,
        "unmodified_v1_v4": {
            "v1_result": {"path": str(V1_RESULT.relative_to(ROOT)), "sha256": v1_result_sha_before},
            "v2_result": {"path": str(V2_RESULT.relative_to(ROOT)), "sha256": v2_result_sha_before},
            "v3_result": {"path": str(V3_RESULT.relative_to(ROOT)), "sha256": v3_result_sha_before},
            "v4_result": {"path": str(V4_RESULT.relative_to(ROOT)), "sha256": v4_result_sha_before},
            "capture_cores": {
                CAPTURE_V1: sha(CAPTURE_V1),
                CAPTURE_V2: sha(CAPTURE_V2),
                CAPTURE: sha(CAPTURE),
            },
            "identity": {
                IDENTITY_V1: sha(IDENTITY_V1),
                IDENTITY: sha(IDENTITY),
            },
            "coordinator_source": {COORDINATOR: sha(COORDINATOR)},
            "review_and_decision_sha256": {
                relative: digest for relative, digest in protected_before.items()
                if "review" in relative or "decision" in relative
            },
        },
        "old_source_provenance": provenance,
        "actual_evidence_state": actual_evidence_state,
        "startup_route": startup_route,
        "capture_core": {
            "path": CAPTURE,
            "sha256": sha(CAPTURE),
            "bytes": (ROOT / CAPTURE).stat().st_size,
            "v2_frozen": {"path": CAPTURE_V2, "sha256": sha(CAPTURE_V2)},
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
            "startup_qualification_sha256": settings["startup_qualification_sha256"],
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
            "startup_plan_hashes": startup_plan["plan_hashes"],
            "startup_plan_matches_review_v3": True,
            "startup_first_use_gate": first_use_gate,
            "entry_gate": entry,
            "codec_identity_gate": codec_gate,
            "identity_staged_loop_gate": staged_gate,
            "core_pin_gate": core_pins,
            "client_only_audit": client_audit_result,
            "post_functional_gate": post_gate,
            "dependency_graph": dependency_gate,
            "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
            "fixture_library": {"path": V4_QUALIFIER, "sha256": sha(V4_QUALIFIER)},
            "fixture_reviews_written_to_ephemeral_temp_and_deleted": True,
            "approved_review_files_created": False,
            "external_diagnostic_startup_attempt_consumed_as_evidence": False,
            "capture_v3_receipt_consumed_as_evidence": False,
            "actual_native_auditor_executed": False,
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
            "startup_requires_per_worker_first_use_getter_and_clone": True,
            "startup_requires_all_16_actual_SDK_workers": True,
            "startup_requires_reported_coverage_all_flag_true_all_clone_ok_no_row_errors": True,
            "startup_rows_must_be_schema_and_source_valid": True,
            "startup_rows_may_not_be_filled_from_progress_or_environment": True,
            "worker_initial_environment_is_exec_initial_not_live": True,
            "worker_initial_environment_mask_preserves_missing_as_false": True,
            "bs2_draft_decode_required_at_startup": False,
            "full_short_native_audit_still_requires_all32_bs2_and_graph_deltas": True,
            "readiness_still_requires_full_short_audit_and_native_cache_SDMA_semantic_proofs": True,
            "role_specific_execution_settings_scope": "D only TASK_QUEUE_ENABLE=0/HCCL_BUFFSIZE=256; P retains original empty values.",
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
            "external_diagnostic_review_present": startup_review_v3["present"],
            "external_diagnostic_review_created_by_this_run": False,
            "external_diagnostic_review_startup_qualified": startup_review_v3["startup_qualified"],
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
            "actual_native_auditor_executed": False,
            "startup_result_written": False,
            "framework_torch_or_SDK_import": False,
            "existing_frozen_artifact_edits": 0,
            "old_source_edits": 0,
            "state_handoff_loop_capacity_ledger_reviews_edits": 0,
            "approved_reviews_created": 0,
            "external_artifacts_consumed_as_evidence": 0,
            "external_artifacts_modified": 0,
            "running_or_terminal_capture_v3_touched": False,
            "capture_v3_receipt_consumed_as_evidence": False,
            "capture_v3_review_consumed_as_startup_evidence": False,
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
        "mode": "LOCAL_ONLY_INITIAL_VS_LIVE_ENV_WIRE",
        "result": {"path": str(RESULT.relative_to(ROOT)), "sha256": result_sha},
        "changed_files": changed,
        "new_sources": new_sources,
        "startup_first_use_gate": first_use_gate,
        "core_pin_gate": core_pins,
        "dependency_graph": dependency_gate,
        "entry_gate": entry,
        "codec_identity_gate": codec_gate,
        "identity_staged_loop_gate": staged_gate,
        "client_only_audit": client_audit_result,
        "post_functional_gate": post_gate,
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
    write_json(DEPENDENCY_GATE, dependency_gate)
    write_json(FIRST_USE_GATE, first_use_gate)
    write_json(FIXTURE_SUMMARY, {
        "project_id": "glm53",
        "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
        "startup_first_use": first_use_gate,
        "short_entry": entry["short"],
        "long_entry": entry["long"],
        "client_only_audit": client_audit_result,
        "post_functional_gate": post_gate,
        "codec_identity": codec_gate,
        "identity_staged_loop": staged_gate,
        "core_pins": core_pins,
        "dependency_graph": dependency_gate,
        "approved_review_files_created": False,
        "ephemeral_temp_roots_removed": True,
    })
    receipt = {
        "project_id": "glm53",
        "run_id": CWD.name,
        "mode": "LOCAL_ONLY_INITIAL_VS_LIVE_ENV_WIRE",
        "status": "LOCAL_DRAFT_V5_COMPLETE_PRIMARY_REVIEW_PENDING",
        "runtime_authorized": False,
        "deployment_authorized": False,
        "formal_execution_authorized": False,
        "model_start_authorized": False,
        "objective_complete": False,
        "startup_pass_declared": False,
        "scope": "Local v5 v3 startup wire, per-worker live getter/clone join, initial-vs-live environment separation, core-v3 pins and preservation checks; no remote/client/capture/auditor execution.",
        "result": {"path": str(RESULT.relative_to(ROOT)), "sha256": result_sha},
        "verification": {"path": str(VERIFICATION.relative_to(ROOT)), "sha256": sha_path(VERIFICATION)},
        "host_code_hashes": {"path": str(HOST_HASHES.relative_to(ROOT)), "sha256": sha_path(HOST_HASHES)},
        "dependency_graph_qualification": {"path": str(DEPENDENCY_GATE.relative_to(ROOT)), "sha256": sha_path(DEPENDENCY_GATE)},
        "startup_first_use_qualification": {"path": str(FIRST_USE_GATE.relative_to(ROOT)), "sha256": sha_path(FIRST_USE_GATE)},
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
            "formal_startup_qualification_path": settings["startup_qualification_path"],
            "wrappers": host_codes["wrappers"],
            "current_startup_entry": STARTUP_ENTRY,
            "superseded_startup_entries": [STARTUP_ENTRY_V1_FROZEN, STARTUP_ENTRY_V1_FAILED, STARTUP_ENTRY_V2],
        },
        "startup_route": startup_route,
        "actual_evidence_state": actual_evidence_state,
        "counts": counts,
        "startup_first_use": {
            "positive_workers": first_use_gate["positive"]["workers"],
            "negative_case_count": first_use_gate["negative_case_count"],
            "bs2_required_at_startup": False,
            "evidence_source": first_use_gate["positive"]["evidence_source"],
        },
        "entry_gate": {
            "short_positive_identity_calls": 4,
            "short_positive_client_calls": 1,
            "long_positive_identity_calls": 4,
            "long_positive_client_calls": 1,
            "precheck_failure_cases": 8,
            "client_started_on_precheck_failure": 0,
            "core_pin_negative_cases": len(core_pins["pin_negative_cases"]),
            "no_real_ssh": True,
        },
        "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
        "fixture_library": {"path": V4_QUALIFIER, "sha256": sha(V4_QUALIFIER)},
        "approved_review_files_created": False,
        "external_execution": {
            "SSH": 0, "network": 0, "HTTP": 0, "Docker": 0, "NPU": 0,
            "SDK_init": 0, "model_requests": 0, "new_profiles": 0, "Git": 0,
            "capture": 0, "client": 0, "native_auditor": 0,
        },
        "v1_result_preserved": {"path": str(V1_RESULT.relative_to(ROOT)), "sha256": v1_result_sha_before},
        "v2_result_preserved": {"path": str(V2_RESULT.relative_to(ROOT)), "sha256": v2_result_sha_before},
        "v3_result_preserved": {"path": str(V3_RESULT.relative_to(ROOT)), "sha256": v3_result_sha_before},
        "v4_result_preserved": {"path": str(V4_RESULT.relative_to(ROOT)), "sha256": v4_result_sha_before},
        "v1_v2_v3_v4_receipt_verification_preserved": True,
        "state_handoff_loop_capacity_ledger_reviews_edited": False,
        "model_coordinator_or_old_source_edited": False,
        "capture_cores_or_identity_edited": False,
        "running_or_terminal_capture_v3_modified_or_consumed": False,
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
        "first_use_gate": first_use_gate["verdict"],
        "first_use_negative_cases": first_use_gate["negative_case_count"],
        "core_pin_gate": core_pins["verdict"],
        "dependency_acyclic": dependency_gate["acyclic"],
        "stale_reference_count": dependency_gate["stale_reference_count"],
        "startup_plan_matches_review_v3": True,
        "startup_pass_declared": False,
        "counts": counts,
        "missing_actual_evidence": len(missing),
        "runtime_authorized": False,
        "fixture_evidence_class": "LOCAL_SIMULATED_FIXTURE_NOT_NATIVE_MODEL_OR_FORMAL_EVIDENCE",
    }, indent=2))


if __name__ == "__main__":
    main()
