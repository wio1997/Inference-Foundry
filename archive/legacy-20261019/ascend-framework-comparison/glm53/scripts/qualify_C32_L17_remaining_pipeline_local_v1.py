"""DRAFT_PENDING_PRIMARY_REVIEW: local-only qualification of the L17 drafts.

Compiles every new Python file, verifies all pinned old-source SHAs, renders the
plan-only host programs (without execution), checks AST/symtable globals and the
existing CPU gateway fixture, and writes the draft result plus this run's own
receipt.  No SSH/network/Docker/NPU/SDK/model/Git action is performed.
"""
import ast
import builtins
import hashlib
import importlib.util
import json
import py_compile
import symtable
import sys
from pathlib import Path

import glm53_C32_L17_pipeline_contract_v1 as contract

ROOT = Path(__file__).resolve().parents[1]
CWD = ROOT / "runs" / "20261019T-L17-functional-formal-local-draft-executor-v1"
RESULT = ROOT / "results/C32-L17-remaining-pipeline-local-draft-v1.json"
VERIFICATION = CWD / "verification.json"
HOST_HASHES = CWD / "host-code-hashes.json"
RECEIPT = CWD / "receipt.json"

NEW_FILES = [
    "scripts/glm53_C32_L17_pipeline_contract_v1.py",
    "scripts/glm53_C32_L17_native_capture_v1.py",
    "scripts/inspect_C32_L17_native_startup_v1.py",
    "scripts/inspect_C32_L17_native_post_functional_v1.py",
    "scripts/glm53_C32_L17_formal_native_capture_v1.py",
    "scripts/probe_C32_L17_native_pd_payload_v1.py",
    "scripts/run_C32_L17_native_P_functional_v1.py",
    "scripts/glm53_C32_L17_native_long_client_v1.py",
    "scripts/glm53_C32_L17_native_semantic_probe_v1.py",
    "scripts/run_C32_L17_native_long_v1.py",
    "scripts/audit_C32_L17_native_startup_local_v1.py",
    "scripts/audit_C32_L17_native_functional_local_v1.py",
    "scripts/audit_C32_L17_native_readiness_local_v1.py",
    "scripts/run_C32_L17_native_formal_v1.py",
    "scripts/glm53_C32_L17_formal_resource_controller_v1.py",
    "scripts/audit_C32_L17_native_formal_client_local_v1.py",
    "scripts/audit_C32_L17_native_formal_native_local_v1.py",
    "scripts/inspect_C32_L17_independent_formal_metrics_v1.py",
    "scripts/qualify_C32_L17_remaining_pipeline_local_v1.py",
    "config/C32-L17-native-runtime-v1/formal-C32-native-v1.json",
]

SAFE_IMPORTS = [
    ("scripts/glm53_C32_L17_pipeline_contract_v1.py", "l17_contract"),
    ("scripts/glm53_C32_L17_native_capture_v1.py", "l17_capture"),
    ("scripts/run_C32_L17_native_P_functional_v1.py", "l17_short_runner"),
    ("scripts/run_C32_L17_native_long_v1.py", "l17_long_runner"),
    ("scripts/glm53_C32_L17_formal_resource_controller_v1.py", "l17_formal_controller"),
    ("scripts/run_C32_L17_native_formal_v1.py", "l17_formal_runner"),
    ("scripts/inspect_C32_L17_independent_formal_metrics_v1.py", "l17_metrics_inspector"),
    ("scripts/audit_C32_L17_native_startup_local_v1.py", "l17_startup_audit"),
    ("scripts/audit_C32_L17_native_functional_local_v1.py", "l17_functional_audit"),
    ("scripts/audit_C32_L17_native_readiness_local_v1.py", "l17_readiness_audit"),
    ("scripts/audit_C32_L17_native_formal_client_local_v1.py", "l17_formal_client_audit"),
    ("scripts/audit_C32_L17_native_formal_native_local_v1.py", "l17_formal_native_audit"),
]

DERIVATIONS = {
    "scripts/glm53_C32_L17_native_capture_v1.py": {
        "old": [contract.OLD_CAPTURE_FACTORY, contract.OLD_CAPTURE_BASE],
        "mechanical": "Reuses the accepted v3 HOST/STREAM/CPU_HOST/codec and old mixed-root wrapper pattern; changes root/CIDs to the new P+D capacity records, uses the L17 mixed renderer, adds D eager-input rows/env/identity and mode wiring.",
    },
    "scripts/inspect_C32_L17_native_startup_v1.py": {
        "old": ["scripts/inspect_C32_L12C_native_startup_v1.py"],
        "mechanical": "New capture/review IDs; requires the actual L17 D model-start/observe completion before the four-node startup capture.",
    },
    "scripts/inspect_C32_L17_native_post_functional_v1.py": {
        "old": ["scripts/inspect_C32_L12C_native_post_functional_v1.py"],
        "mechanical": "New capture/short/long IDs; requires D eager bs2 coverage; appends the existing CPU43 long-client artifact without rerunning it.",
    },
    "scripts/glm53_C32_L17_formal_native_capture_v1.py": {
        "old": [contract.OLD_FORMAL_CAPTURE],
        "mechanical": "New readiness/formal run/capture IDs; calls the L17 capture core and adds the CPU43 formal archive.",
    },
    "scripts/probe_C32_L17_native_pd_payload_v1.py": {
        "old": [contract.OLD_SHORT_PROBE],
        "mechanical": "Same 32 logical/64 backend, 512 input SHA, routes/seed and one D DP pair each; D output 16->600 min=max/ignore_eos. Whole-16 reference remains a literal field; a separately scoped prefix16 annotation is diagnostic only.",
    },
    "scripts/run_C32_L17_native_P_functional_v1.py": {
        "old": [contract.OLD_SHORT_RUNNER],
        "mechanical": "New run/namespace/probe SHA, new mixed identity and P/D capacities, all-four prechecks and D eager postconditions.",
    },
    "scripts/glm53_C32_L17_native_semantic_probe_v1.py": {
        "old": [contract.OLD_SEMANTIC_PROBE],
        "mechanical": "Same 581 pages/1743 keys/8253063168 bytes/init_bm=False; namespace only.",
    },
    "scripts/glm53_C32_L17_native_long_client_v1.py": {
        "old": [contract.OLD_LONG_CLIENT],
        "mechanical": "Same six logical/12 backend/2408 output, dataset SHA, AISBench version and semantic/cache guards; namespace only, no extra SDK benchmark.",
    },
    "scripts/run_C32_L17_native_long_v1.py": {
        "old": [contract.OLD_LONG_RUNNER],
        "mechanical": "New run/namespace/files, new mixed identity and P/D capacities; same AISBench assets, metadata init_bm=False gate and no additional native SDK benchmark.",
    },
    "scripts/audit_C32_L17_native_startup_local_v1.py": {
        "old": [contract.OLD_STARTUP_AUDIT],
        "mechanical": "New capture/CIDs/capacities; D floor 161280 vs old 181632, P floor 254848; 237+11 layout, MTP5/6, new D eager worker environment, P buffer_only/9GB/map reuse/extension.",
    },
    "scripts/audit_C32_L17_native_functional_local_v1.py": {
        "old": [contract.OLD_FUNCTIONAL_AUDIT, contract.OLD_STARTUP_AUDIT],
        "mechanical": "Same C32 count/DP logic extended to 600-output D; adds all32 D bs2 draft-DECODE/out_cache_loc evidence and per-rank Graph deltas.",
    },
    "scripts/audit_C32_L17_native_readiness_local_v1.py": {
        "old": [contract.OLD_READINESS_AUDIT],
        "mechanical": "New captures/IDs; strict SDK begin==good/bad0/inflight0 plus 11th TOTAL column, D eager all32, Graph deltas, all32 typed SDMA destinations, pinned optional-D-storage helper.",
    },
    "scripts/run_C32_L17_native_formal_v1.py": {
        "old": [contract.OLD_FORMAL_RUNNER],
        "mechanical": "New settings/readiness/startup/functional decision/review/formal review SHAs and controller source SHA; one C32 execution.",
    },
    "scripts/glm53_C32_L17_formal_resource_controller_v1.py": {
        "old": [contract.OLD_FORMAL_CONTROLLER],
        "mechanical": "Reuses accepted prefix/gate/CPU-stage/RPC orchestration; all prechecks use the L17 mixed renderer and P/D capacity records, no old D IDs/stages; adds actual freeze gates and predecessor checks.",
    },
    "scripts/audit_C32_L17_native_formal_client_local_v1.py": {
        "old": [contract.OLD_FORMAL_CLIENT_AUDIT],
        "mechanical": "New run/settings/manifest/cache-review/output paths; same 128 formal+32 warm, 320 offered backend, 76992 output and full-clock timing rules.",
    },
    "scripts/audit_C32_L17_native_formal_native_local_v1.py": {
        "old": [contract.OLD_FORMAL_NATIVE_AUDIT],
        "mechanical": "New run/captures/IDs; adds D eager 32-worker prerequisite, pinned optional-D-storage null preservation and strict SDK/Graph/MC/SDMA gates.",
    },
    "scripts/inspect_C32_L17_independent_formal_metrics_v1.py": {
        "old": ["scripts/inspect_C32_L12D_independent_formal_metrics_v1.py"],
        "mechanical": "New formal/client-audit/run IDs; same completed CPU43 archive re-derivation, one SSH, no model/SDK.",
    },
    "config/C32-L17-native-runtime-v1/formal-C32-native-v1.json": {
        "old": ["config/C32-L12C-native-runtime-v1/formal-C32-native-v1.json"],
        "mechanical": "New actual P/D CIDs/roots/SIDs, L17 run/namespace and identity helper; actual evidence SHA fields are null pending primary freeze. Same canonical C32 input manifest/client/gate sources, 128 formal/32 warm/no profile/no retry.",
    },
}


def sha(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


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


def main():
    assert not RESULT.exists(), "Preserve existing draft result"
    assert not RECEIPT.exists(), "Preserve existing run receipt"
    CWD.mkdir(parents=True, exist_ok=True)
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
    }
    host_codes["wrappers"] = {
        "inspect_C32_L17_native_startup_v1.py": {"core_sha256": import_file("l17_startup_wrapper", "scripts/inspect_C32_L17_native_startup_v1.py").CORE_SHA},
        "inspect_C32_L17_native_post_functional_v1.py": {"core_sha256": import_file("l17_post_wrapper", "scripts/inspect_C32_L17_native_post_functional_v1.py").CORE_SHA},
        "glm53_C32_L17_formal_native_capture_v1.py": {"core_sha256": import_file("l17_formal_capture_wrapper", "scripts/glm53_C32_L17_formal_native_capture_v1.py").CORE_SHA},
    }
    assert all(item["core_sha256"] == sha("scripts/glm53_C32_L17_native_capture_v1.py") for item in host_codes["wrappers"].values())
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
        "D model-start/observe completion (current receipt is observing, pass=false, observe empty)",
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
    new_file_shas = {relative: sha(relative) for relative in NEW_FILES}
    result = {
        "project_id": "glm53",
        "result_id": "C32-L17-remaining-pipeline-local-draft-v1",
        "status": "LOCAL_DRAFT_COMPLETE_PRIMARY_REVIEW_PENDING",
        "review_status": "PENDING_PRIMARY_REVIEW",
        "runtime_authorized": False,
        "deployment_authorized": False,
        "formal_execution_authorized": False,
        "model_start_authorized": False,
        "objective_complete": False,
        "scope": "Local stdlib/source draft of the remaining L17 SGLang C32 pipeline only. No SSH/network/Docker/NPU/SDK/model/Git, no launcher execution, no review approval.",
        "new_files": {relative: {"sha256": digest, "bytes": (ROOT / relative).stat().st_size} for relative, digest in new_file_shas.items()},
        "new_file_count": len(NEW_FILES),
        "old_source_provenance": provenance,
        "allowed_mechanical_diffs": DERIVATIONS,
        "local_qualification": {
            "python_compile": compile_results,
            "ast_parse": ast_results,
            "symtable_unbound_globals": symtable_results,
            "symtable_all_resolved": all(not value["identity"] and not value["outer"] for value in symtable_results.values()),
            "source_provenance_all_match": True,
            "wrapper_core_pins_match": True,
            "short_probe_pin_matches": True,
            "long_client_probe_pins_match": True,
            "formal_config_controller_pin_matches": True,
            "cpu_gateway_fixture": fixture,
            "installed_scope": "Compiled/AST/symtable/hashes only; no framework, torch or SDK import; no installed remote source was executed.",
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
            "D_eager_input": "All32 actual workers require native flag true, clone distinct, is_draft_worker true, forward_mode DECODE, batch_size 2 and real out_cache_loc shape/dtype/stride after functional.",
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
            "framework_torch_or_SDK_import": False,
            "existing_frozen_artifact_edits": 0,
            "max_SSH_future_execution_per_launcher": 2,
        },
        "activation": {
            "requires": [
                "actual D model-start/observe completion and approved startup capture review",
                "new startup auditor result",
                "new short functional result and approved short review",
                "new post-functional capture with all32 D bs2 draft-DECODE rows",
                "new long functional result",
                "new readiness result",
                "new functional decision/actual review",
                "new formal actual review and frozen settings SHA values",
            ],
            "next_action": "Primary review/freeze of the local drafts; no execution follows from this result.",
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
        "new_files": result["new_files"],
        "local_qualification": result["local_qualification"],
        "host_codes": result["host_codes"],
        "missing_actual_evidence": result["missing_actual_evidence"],
        "runtime_authorized": False,
    }
    VERIFICATION.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    HOST_HASHES.write_text(json.dumps(result["host_codes"], indent=2) + "\n", encoding="utf-8")
    receipt = {
        "project_id": "glm53",
        "run_id": CWD.name,
        "mode": "LOCAL_ONLY_PIPELINE_DRAFT",
        "status": "LOCAL_DRAFT_COMPLETE_PRIMARY_REVIEW_PENDING",
        "runtime_authorized": False,
        "deployment_authorized": False,
        "formal_execution_authorized": False,
        "objective_complete": False,
        "scope": "Mechanical L17 remaining-pipeline drafts, old-source provenance, compile/AST/symtable/hash qualification and existing CPU gateway fixture only.",
        "result": {"path": str(RESULT.relative_to(ROOT)), "sha256": result_sha},
        "verification": {"path": str(VERIFICATION.relative_to(ROOT)), "sha256": sha(str(VERIFICATION.relative_to(ROOT)))},
        "host_code_hashes": {"path": str(HOST_HASHES.relative_to(ROOT)), "sha256": sha(str(HOST_HASHES.relative_to(ROOT)))},
        "new_files": result["new_files"],
        "counts": counts,
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
        },
        "max_SSH_future_execution_per_launcher": 2,
        "state_handoff_loop_capacity_reviews_edited": False,
        "model_runtime_edited": False,
        "ledger_incremented": False,
        "created_at": "2026-10-19",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "result": str(RESULT.relative_to(ROOT)),
        "result_sha256": result_sha,
        "receipt": str(RECEIPT.relative_to(ROOT)),
        "receipt_sha256": sha(str(RECEIPT.relative_to(ROOT))),
        "files": len(NEW_FILES),
        "symtable_all_resolved": result["local_qualification"]["symtable_all_resolved"],
        "missing_actual_evidence": len(missing),
        "runtime_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
