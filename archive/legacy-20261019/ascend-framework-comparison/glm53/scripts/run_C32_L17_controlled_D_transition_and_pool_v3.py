"""DRAFT_PENDING_PRIMARY_REVIEW: D36/41-only stop and candidate-pool transition v2.

This launcher does not run now. It refuses unless:
1. the accepted stage-v3 receipt is pass and the stage review is APPROVED_FOR_STAGE_ONLY;
2. results/C32-L17-D-transition-actual-review-v3.json exists, is
   APPROVED_FOR_D_STOP_AND_CANDIDATE_POOL_ONLY, has runtime_authorized=true,
   model_start_authorized=false, and freezes every host plan SHA below;
3. the retained P/Meta/vLLM capacity identity still matches the accepted file;
4. P is explicitly the L12C native candidate and D is separately selected.

It stops only the exact current D36/D41 containers, prepares candidate roots and
starts candidate pools. It never starts/stops Meta, P, vLLM or a rollback pool,
never starts a model, and never performs a broad cleanup, retry or fallback.
Every raw node result is retained even when a job fails. v1 is unchanged.
"""

import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RID = "20261019T-L17-D-transition-and-pool-v2"
STOP_RID = "20261019T-L17-D-stop-current-v2"
RUN = ROOT / "runs" / RID
LIB = "scripts/glm53_C32_L17_D_transition_library_v2.py"
LIB_SHA = "da4bbcab4ddf1dd4670c45db78a20aace39f12e1429e73aa5da9d27d36b7b8ce"
REVIEW = "results/C32-L17-D-transition-actual-review-v3.json"
CAPACITY_OUT = "results/sglang-C32-L17-selected-D-pool-capacity-v2.json"
STAGE_RECEIPT = "runs/20261019T-L17-D-native-eager-stage-v3/receipt.json"
STAGE_REVIEW = "results/C32-L17-D-native-eager-stage-actual-review-v3.json"
STAGE_LAUNCHER_SHA = "6e9b5dcd486b4977521a280248d8fa77ae3b6593e11dc30184437145ff39978a"
STAGE_RECEIPT_SHA = "d724991b50b03d45eb9c813cdea754c4fd69a42a3a1aabfc85af42f032530cf1"
STAGE_REVIEW_SHA = "c176440c0440b9ea41daf057ebca87dbec1b77a26eea2762b11d3c08678f722b"
CANDIDATE_STAGE_SHA = "a8d6759af5f29a3da15e60b1312bc81eae046e93c99ca3eb366a78bdf055da0c"
ROLLBACK_STAGE_SHA = "f8b49216cd3ac6651cb48f9308a2590d2c0e306b339355e568a522bb882ddcf0"
CANDIDATE_RUN_ID = "20261019T-L17-D-native-eager-v2"
ROLLBACK_RUN_ID = "20261019T-L17-D-original-rollback-v1"
CAPACITY = "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json"
CAPACITY_SHA = "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50"
EXPECTED_PROVIDER_SDK_SETUP_INIT_SEQUENCES = 2
P_NODES = (34, 35)
D_NODES = (36, 41)


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_library():
    path = ROOT / LIB
    assert sha256_file(path) == LIB_SHA, "plan library SHA mismatch"
    spec = importlib.util.spec_from_file_location("glm53_l17_d_library", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_preconditions(lib):
    assert sha256_file(ROOT / STAGE_RECEIPT) == STAGE_RECEIPT_SHA
    assert sha256_file(ROOT / STAGE_REVIEW) == STAGE_REVIEW_SHA
    receipt = read_json(ROOT / STAGE_RECEIPT)
    assert receipt.get("project_id") == "glm53"
    assert receipt.get("pass") is True
    assert receipt.get("CPU_pass") is True
    assert receipt.get("candidate_stage_sha256") == CANDIDATE_STAGE_SHA
    assert receipt.get("rollback_stage_sha256") == ROLLBACK_STAGE_SHA
    assert receipt.get("candidate_bundle_manifest_sha256") == lib.STAGES["candidate"]["bundle_manifest_sha256"]
    assert receipt.get("rollback_bundle_manifest_sha256") == lib.STAGES["rollback"]["bundle_manifest_sha256"]
    assert sorted(receipt["health"]) == ["34", "35", "36", "41"]
    assert all(value == 0 for value in receipt["health"].values())
    assert sorted(receipt["stages"]) == ["0-36", "0-41", "1-36", "1-41"]
    assert all(value == 0 for value in receipt["stages"].values())
    assert receipt.get("staged_file_counts_including_metadata") == [24, 23]
    assert receipt.get("launcher_sha256") == STAGE_LAUNCHER_SHA
    assert receipt.get("review_sha256") == STAGE_REVIEW_SHA
    cpu = receipt.get("CPU_result") or {}
    assert cpu.get("pass") is True and cpu.get("checks") == cpu.get("checks_passed") == 136
    assert cpu.get("SDK_init_calls") == 0 and cpu.get("NPU_compute_calls") == 0
    assert cpu.get("model_requests") == 0
    assert cpu.get("installed_eager_sha") == lib.EAGER_SHA
    assert cpu.get("installed_environ_sha") == lib.ENVIRON_SHA
    review = read_json(ROOT / STAGE_REVIEW)
    assert review.get("review_status") == "APPROVED_FOR_STAGE_ONLY"
    assert review.get("runtime_authorized") is False
    assert review.get("launcher_sha256") == STAGE_LAUNCHER_SHA
    # candidate-v2 and original-rollback-v1 remain the same staged bundles.
    assert stage_source_sha(lib, "candidate") == CANDIDATE_STAGE_SHA
    assert stage_source_sha(lib, "rollback") == ROLLBACK_STAGE_SHA
    assert lib.STAGES["candidate"]["run_id"] == CANDIDATE_RUN_ID
    assert lib.STAGES["rollback"]["run_id"] == ROLLBACK_RUN_ID
    return receipt


def stage_source_sha(lib, kind):
    return sha256_file(ROOT / lib.STAGES[kind]["stage"])


def capacity_preconditions(lib):
    assert sha256_file(ROOT / CAPACITY) == CAPACITY_SHA
    capacity = read_json(ROOT / CAPACITY)
    assert capacity.get("project_id") == "glm53" and capacity.get("pass") is True
    assert lib.p_selection(capacity) == "candidate"
    assert capacity.get("selected_variant") == "candidate"
    assert "P_ONLY" in capacity.get("classification", "")
    assert capacity.get("native_82_kernel_fixture_qualified") is True
    assert capacity.get("Meta_retained") is True
    assert capacity.get("D_identities_and_contexts_retained") is True
    assert capacity.get("native_fixture_actual", {}).get("all4_healthy_idle") is True
    providers = capacity["providers"]
    for node, service_id, dram in ((34, 82, 136), (35, 83, 112)):
        row = providers[str(node)]
        assert row["local_service_id"] == service_id
        assert row["dram_GiB"] == dram
        assert row["name"] == "glm53-L12C-candidate-v1-20261019-%d" % node
        assert row["run_id"] == "20261019T-L12C-candidate-v1"
        assert row["remote_root"] == "/data/wio/inference-foundry/glm53/runs/20261019T-L12C-candidate-v1"
    for node in D_NODES:
        row = providers[str(node)]
        expected = lib.OLD_D[node]
        assert row["container_id"] == expected["container_id"]
        assert row["name"] == expected["name"]
        assert row["local_service_id"] == expected["local_service_id"]
        assert row["dram_GiB"] == 160
        assert row["run_id"] == lib.OLD_D_RID
        assert row["remote_root"] == lib.OLD_D_ROOT
    assert capacity.get("meta_container_id") == lib.META_CID
    return capacity


def reviewed_plans(lib):
    plans = lib.plan_all()
    review = read_json(ROOT / REVIEW)
    assert review.get("review_status") == "APPROVED_FOR_D_STOP_AND_CANDIDATE_POOL_ONLY"
    assert review.get("review_version") == "v2"
    assert review.get("runtime_authorized") is True
    assert review.get("model_start_authorized") is False
    assert review.get("selected_P_variant") == "candidate"
    assert review.get("selected_D_variant") == "candidate"
    assert review.get("capacity_output") == lib.CAPACITY_OUT
    assert review.get("stage_receipt_sha256") == sha256_file(ROOT / STAGE_RECEIPT)
    expected = {name: sha256_text(code) for name, code in plans.items()}
    assert review.get("host_plans_sha256") == expected, "host-plan SHA drift; regenerate review"
    return plans


def parsed(result):
    assert result.get("rc") == 0, result
    return json.loads(result["stdout"])


def health_payload(result):
    assert result.get("rc") == 0, result
    marker = "GLM53_L17_HEALTH="
    assert marker in result["stdout"], result
    return json.loads(result["stdout"].split(marker, 1)[1].strip().splitlines()[0])


def provider_init_evidence(rows):
    evidence = {}
    for node in D_NODES:
        raw = rows.get(str(node), {})
        try:
            payload = json.loads(raw.get('stdout', ''))
            events = payload['provider_status']['events']
            assert isinstance(events, list) and events
            attempts = sum(event.get('event') == 'INIT_START' for event in events)
            successes = sum(event.get('event') == 'INIT_COMPLETE' and event.get('rc') == 0 for event in events)
            evidence[str(node)] = {'known': True, 'init_attempts': attempts, 'successful_init_completions': successes}
        except (ValueError, KeyError, TypeError, AssertionError):
            evidence[str(node)] = {'known': False, 'init_attempts': None, 'successful_init_completions': None}
    unknown = [node for node, row in evidence.items() if not row['known']]
    return {'nodes': evidence, 'unknown_nodes': unknown,
            'known_init_attempts': sum(row['init_attempts'] for row in evidence.values() if row['known']),
            'init_attempts_total_if_known': None if unknown else sum(row['init_attempts'] for row in evidence.values()),
            'known_successful_init_completions': sum(row['successful_init_completions'] for row in evidence.values() if row['known'])}


def main():
    assert not RUN.exists(), "transition run already exists: %s" % RUN
    assert not (ROOT / "runs" / STOP_RID).exists()
    assert not (ROOT / CAPACITY_OUT).exists()
    assert not (ROOT / REVIEW).exists()
    # v1 stays DRAFT: its execution roots/results must never exist.
    assert not (ROOT / "runs/20261019T-L17-D-transition-and-pool-v1").exists()
    assert not (ROOT / "runs/20261019T-L17-D-stop-current-v1").exists()
    assert not (ROOT / "results/sglang-C32-L17-selected-D-pool-capacity-v1.json").exists()
    assert not (ROOT / "results/C32-L17-D-transition-actual-review-v1.json").exists()
    lib = load_library()
    stage_preconditions(lib)
    capacity = capacity_preconditions(lib)
    plans = reviewed_plans(lib)
    base = lib.transport()
    RUN.mkdir(parents=True, exist_ok=False)
    (ROOT / "runs" / STOP_RID).mkdir(parents=True, exist_ok=False)
    (RUN / "execution-once.lock").write_text(str(os.getpid()), encoding="utf-8")

    receipt = {
        "project_id": "glm53",
        "run_id": RID,
        "launcher_sha256": sha256_file(pathlib.Path(__file__)),
        "library_sha256": LIB_SHA,
        "stage_receipt_sha256": sha256_file(ROOT / STAGE_RECEIPT),
        "stage_review_sha256": sha256_file(ROOT / STAGE_REVIEW),
        "candidate_stage_sha256": CANDIDATE_STAGE_SHA,
        "rollback_stage_sha256": ROLLBACK_STAGE_SHA,
        "review_sha256": sha256_file(ROOT / REVIEW),
        "capacity_source_sha256": CAPACITY_SHA,
        "max_SSH": 2,
        "phase": "PREFLIGHT",
        "pass": False,
        "health": {},
        "precheck": {},
        "meta_before": None,
        "stop": {},
        "clean_model_shutdown": {},
        "prepare": {},
        "pool": {},
        "preflight": {},
        "meta_after": None,
        "retained_P_health": {},
        "selected_P_variant": "candidate",
        "selected_D_variant": "candidate",
        "selected_variant": "candidate",
        "selected_variant_scope": "D_ONLY_NOT_P",
        "rollback_staged_not_started": True,
        "models_started": False,
        "model_start_allowed": False,
        "capacity_audit_required": True,
        "capacity_output": CAPACITY_OUT,
        "meta_started_or_stopped": False,
        "P_or_vllm_changed": False,
        "broad_cleanup_performed": False,
        "retry_performed": False,
        "fallback_performed": False,
        "failure_preserved": False,
        "model_requests": 0,
        "SDK_init_calls": 0,
        "provider_SDK_setup_init_sequences": 0,
        "preflight_SDK_init_calls": 0,
        "SDK_count_scope": "provider SETUP_COMPLETE+INIT_COMPLETE+READY once per D node = 2 when both ready; preflight adds 0",
    }

    def save(name, value):
        (RUN / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return value

    def checkpoint():
        save("receipt.json", receipt)

    def invoke(node, key, timeout=150):
        try:
            result = base.invoke(node, plans[key], timeout=timeout)
        except Exception as error:
            result = {"error": repr(error)}
        save(key + ".json", result)
        receipt.setdefault("job_rc", {})[key] = result.get("rc", result.get("error"))
        return result

    def batch(keys, timeout=150):
        rows = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {pool.submit(base.invoke, node, plans[key], timeout): (node, key)
                    for node, key in keys}
            for future in concurrent.futures.as_completed(jobs):
                node, key = jobs[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {"error": repr(error)}
                save(key + ".json", result)
                receipt.setdefault("job_rc", {})[key] = result.get("rc", result.get("error"))
                rows[str(node)] = result
                print(key, result.get("rc", result.get("error")), flush=True)
        return rows

    try:
        receipt["phase"] = "HEALTH_ALL4_OWNERSHIP_AND_IDLE"
        checkpoint()
        rows = batch([(node, "health-%d" % node) for node in (34, 35, 36, 41)], 150)
        for node in (34, 35, 36, 41):
            result = rows[str(node)]
            payload = health_payload(result)
            assert payload["healthy_idle"] is True
            assert payload["container_id"] == capacity["providers"][str(node)]["container_id"]
            assert len(payload["NPU_context_PIDs"]) == 18
            receipt["health"][str(node)] = {"rc": result["rc"], "container_id": payload["container_id"], "NPU_context_PIDs": payload["NPU_context_PIDs"]}
        checkpoint()

        receipt["phase"] = "D_STOP_PRECHECK_AND_BUNDLE_SHA"
        checkpoint()
        prechecks = {}
        for node in D_NODES:
            result = invoke(node, "precheck-%d" % node, 150)
            payload = parsed(result)
            assert payload["preflight_pass"] is True
            assert payload["container_id"] == lib.OLD_D[node]["container_id"]
            assert payload["container_id"] != lib.META_CID
            assert payload["old_service_id"] == lib.OLD_D[node]["local_service_id"]
            prechecks[str(node)] = {"rc": result["rc"], **payload}
        receipt["precheck"] = {node: {"rc": value["rc"], "container_id": value["container_id"], "old_service_id": value["old_service_id"]} for node, value in prechecks.items()}
        checkpoint()

        receipt["phase"] = "META_BEFORE_SNAPSHOT"
        checkpoint()
        before = invoke(34, "meta-before", 180)
        before_payload = parsed(before)
        assert before_payload["container_id"] == lib.META_CID
        assert before_payload["meta_node"] == 34
        assert before_payload["meta_labels"]["project_id"] == "glm53"
        assert before_payload["meta_labels"]["run_id"] == lib.OLD_D_RID
        assert before_payload["meta_after_wait"]["bound_seconds"] == 0
        receipt["meta_before"] = {"rc": before["rc"], "container_id": before_payload["container_id"], "meta_node": 34, "meta_labels": before_payload["meta_labels"], "critical_count": before_payload["critical_count"], "critical_sha256": before_payload["critical_sha256"]}
        checkpoint()

        receipt["phase"] = "STOPPING_EXACT_CURRENT_D"
        checkpoint()
        stopped = batch([(node, "stop-%d" % node) for node in D_NODES], 650)
        for node in D_NODES:
            payload = parsed(stopped[str(node)])
            assert payload["container_id"] == lib.OLD_D[node]["container_id"]
            assert payload["resource_shutdown_complete"] is True
            assert payload["provider_closed0"] is True
            assert payload["all_NPUs_empty"] is True
            assert payload["supervisor"]["phase"] == "STOPPED"
            assert payload["supervisor"]["provider_exit_code"] == 0
            assert payload["provider"]["phase"] == "CLOSE_COMPLETE"
            assert payload["profile_before"] == payload["profile_after"]
            log_prefix = {}
            for stream in ("stdout", "stderr"):
                before = payload["retained_model_logs_before"][stream]
                after = payload["retained_model_logs_after"][stream]
                assert after["prefix_match"] is True
                assert after["prefix_bytes"] == before["bytes"]
                assert after["prefix_sha256"] == before["sha256"]
                assert after["after_full"]["bytes"] >= before["bytes"]
                log_prefix[stream] = {
                    "prefix_match": True,
                    "before_bytes": before["bytes"],
                    "before_sha256": before["sha256"],
                    "after_full_bytes": after["after_full"]["bytes"],
                    "after_full_sha256": after["after_full"]["sha256"],
                    "shutdown_addition_bytes": after["shutdown_addition_bytes"],
                }
            receipt["stop"][str(node)] = {
                "rc": stopped[str(node)]["rc"],
                "container_id": payload["container_id"],
                "resource_shutdown_complete": payload["resource_shutdown_complete"],
                "provider_closed0": payload["provider_closed0"],
                "all_NPUs_empty": payload["all_NPUs_empty"],
                "supervisor_provider_exit_code": payload["supervisor"]["provider_exit_code"],
                "profile_metadata_unchanged": True,
                "model_log_prefix": log_prefix,
                "model_logs_before": payload["retained_model_logs_before"],
                "model_logs_after": payload["retained_model_logs_after"],
            }
            receipt["clean_model_shutdown"][str(node)] = payload["clean_model_shutdown"]
        checkpoint()

        receipt["phase"] = "PREPARING_CANDIDATE_POOLS"
        checkpoint()
        for node in D_NODES:
            result = invoke(node, "candidate-prepare-%d" % node, 150)
            payload = parsed(result)
            assert payload["old_container_closed"] is True
            assert payload["empty_NPU_cards"] == 8
            assert payload["model_weights_files"] == 193
            assert payload["model_weights_bytes"] == 764240829290
            assert payload["MemAvailable"] > 1100 * 1024 ** 3
            assert payload["private_optional_extension_mounted"] is False
            receipt["prepare"][str(node)] = {"rc": result["rc"], "new_runtime_root": payload["new_runtime_root"], "MemAvailable": payload["MemAvailable"]}
        checkpoint()

        receipt["phase"] = "STARTING_CANDIDATE_POOLS"
        receipt["SDK_init_calls"] = None
        receipt["SDK_init_calls_scope"] = "Unknown while provider pool jobs may be starting; qualify native INIT events."
        checkpoint()
        pools = batch([(node, "candidate-pool-%d" % node) for node in D_NODES], 260)
        init_evidence = provider_init_evidence(pools)
        receipt["provider_init_evidence"] = init_evidence
        receipt["SDK_init_calls"] = init_evidence["init_attempts_total_if_known"]
        checkpoint()
        provider_sequences = 0
        pool_service_ids = []
        for node in D_NODES:
            payload = parsed(pools[str(node)])
            provider = payload["provider_status"]
            assert payload["ready"] is True
            assert provider["phase"] == "READY"
            assert payload["supervisor_status"]["phase"] == "POOL_READY_WAITING_MODEL_SIGNAL"
            assert payload["supervisor_status"]["model_started"] is False
            assert provider["configuration"]["dram_size"] == "160GB"
            assert provider["configuration"]["max_dram_size"] == "160GB"
            setup = [event for event in provider["events"] if event["event"] == "SETUP_COMPLETE"]
            init = [event for event in provider["events"] if event["event"] == "INIT_COMPLETE"]
            ready = [event for event in provider["events"] if event["event"] == "READY"]
            assert len(setup) == len(init) == len(ready) == 1, (setup, init, ready)
            assert setup[0].get("rc") == 0 and init[0].get("rc") == 0
            assert type(ready[0].get("local_service_id")) is int and ready[0]["local_service_id"] >= 0
            provider_sequences += 1
            pool_service_ids.append(ready[0]["local_service_id"])
            receipt["pool"][str(node)] = {"rc": pools[str(node)]["rc"], "container_id": payload["container_id"], "name": payload["name"], "ready": True, "local_service_id": ready[0]["local_service_id"], "provider_SETUP_COMPLETE_rc0": True, "provider_INIT_COMPLETE_rc0": True, "provider_READY_events": len(ready)}
        assert provider_sequences == EXPECTED_PROVIDER_SDK_SETUP_INIT_SEQUENCES, provider_sequences
        assert len(set(pool_service_ids)) == len(pool_service_ids), pool_service_ids
        receipt["provider_SDK_setup_init_sequences"] = provider_sequences
        receipt["SDK_init_calls"] = provider_sequences
        receipt["SDK_init_calls_scope"] = "Two actual native provider SETUP_COMPLETE/INIT_COMPLETE rc0 and READY event sets qualified; preflight creates no SDK engine."
        receipt["provider_local_service_ids"] = pool_service_ids
        checkpoint()

        receipt["phase"] = "CANDIDATE_NATIVE_PREFLIGHT"
        checkpoint()
        preflights = batch([(node, "candidate-preflight-%d" % node) for node in D_NODES], 260)
        for node in D_NODES:
            payload = parsed(preflights[str(node)])
            assert payload["query_rc"] == 0, payload
            inner = json.loads(payload["stdout"].split("GLM53_L17_PREFLIGHT=", 1)[1].strip().splitlines()[0])
            assert inner["pass"] is True
            assert inner["tp"] == 32 and inner["dp"] == 16 and inner["ep"] == 32 and inner["attention_tp"] == 2
            assert inner["MTP_steps"] == 5 and inner["MTP_draft_tokens"] == 6
            assert inner["kv_cache_dtype"] == "bfloat16"
            assert inner["no_copy"] == "true" and inner["static_variant_is_candidate"] is True
            assert inner["native_no_copy_get"] is True
            assert inner["eager_audit_v2_marker"] == "v2"
            assert inner["eager_source_sha256"] == lib.EAGER_SHA
            assert inner["environ_source_sha256"] == lib.ENVIRON_SHA
            assert inner["SDK_init_calls"] == 0 and inner["preflight_SDK_init_calls"] == 0
            assert inner["engine_created"] is False and inner["model_constructed"] is False
            assert inner["NPU_compute_calls"] == 0 and inner["NPU_tensor_allocations"] == 0
            assert inner["NPU_allocation_attempted"] is False
            receipt["preflight_SDK_init_calls"] = 0
            receipt["preflight"][str(node)] = {"rc": preflights[str(node)]["rc"], "query_rc": payload["query_rc"], "inner": inner}
        checkpoint()

        receipt["phase"] = "META_AFTER_SNAPSHOT_AND_RETAINED_P"
        checkpoint()
        after = invoke(34, "meta-after", 180)
        after_payload = parsed(after)
        assert after_payload["container_id"] == before_payload["container_id"] == lib.META_CID
        assert after_payload["meta_node"] == 34 and before_payload["meta_node"] == 34
        assert after_payload["meta_labels"] == before_payload["meta_labels"]
        assert after_payload["critical_count"] >= before_payload["critical_count"]
        wait = after_payload["meta_after_wait"]
        assert wait["bound_seconds"] == lib.META_AFTER_WAIT_BOUND_SECONDS
        assert wait["step_seconds"] == lib.META_AFTER_WAIT_STEP_SECONDS
        assert wait["reason"] == lib.META_AFTER_WAIT_REASON
        assert type(wait["attempts"]) is int and wait["attempts"] >= 1
        receipt["meta_after"] = {"rc": after["rc"], "container_id": after_payload["container_id"], "meta_node": 34, "meta_labels": after_payload["meta_labels"], "critical_count": after_payload["critical_count"], "critical_sha256": after_payload["critical_sha256"], "wait": wait}
        retained = {}
        for node in P_NODES:
            plans["retained-P-health-%d" % node] = plans["health-%d" % node]
            result = invoke(node, "retained-P-health-%d" % node, 150)
            payload = health_payload(result)
            assert payload["healthy_idle"] is True
            before_row = receipt["health"][str(node)]
            assert payload["container_id"] == before_row["container_id"]
            assert payload["NPU_context_PIDs"] == before_row["NPU_context_PIDs"]
            retained[str(node)] = {"rc": result["rc"], "container_id": payload["container_id"], "NPU_context_PIDs": payload["NPU_context_PIDs"]}
        receipt["retained_P_health"] = retained
        checkpoint()

        receipt["phase"] = "POOLS_READY_AWAITING_INDEPENDENT_CAPACITY_AUDIT"
        receipt["pass"] = True
        checkpoint()
    except BaseException as error:
        receipt["phase"] = "FAILED_OR_PARTIAL_STATE_PRESERVED"
        receipt["error"] = repr(error)
        receipt["failure_preserved"] = True
        checkpoint()
        raise
    finally:
        checkpoint()
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
