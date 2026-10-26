"""DRAFT_PENDING_PRIMARY_REVIEW: D36/41-only stop and candidate-pool transition.

This launcher does not run now. It refuses unless:
1. the stage-v3 receipt is pass and the stage review is APPROVED_FOR_STAGE_ONLY;
2. results/C32-L17-D-transition-actual-review-v1.json exists, is
   APPROVED_FOR_D_STOP_AND_CANDIDATE_POOL_ONLY, has runtime_authorized=true,
   model_start_authorized=false, and freezes every host plan SHA below;
3. the retained P/Meta/vLLM capacity identity still matches the accepted file.

It stops only the exact current D36/D41 containers, prepares candidate roots and
starts candidate pools. It never starts/stops Meta, P, vLLM or a rollback pool,
never starts a model, and never performs a broad cleanup, retry or fallback.
Every raw node result is retained even when a job fails.
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
RID = "20261019T-L17-D-transition-and-pool-v1"
RUN = ROOT / "runs" / RID
LIB = "scripts/glm53_C32_L17_D_transition_library_v1.py"
LIB_SHA = "95309e2866324a3109e06560e33bdc0b8851e0d2bf99db937da2f05581d23b73"
REVIEW = "results/C32-L17-D-transition-actual-review-v1.json"
STAGE_RECEIPT = "runs/20261019T-L17-D-native-eager-stage-v3/receipt.json"
STAGE_REVIEW = "results/C32-L17-D-native-eager-stage-actual-review-v3.json"
STAGE_LAUNCHER_SHA = "6e9b5dcd486b4977521a280248d8fa77ae3b6593e11dc30184437145ff39978a"
CAPACITY = "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json"
CAPACITY_SHA = "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50"
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
    receipt = read_json(ROOT / STAGE_RECEIPT)
    assert receipt.get("project_id") == "glm53"
    assert receipt.get("pass") is True
    assert receipt.get("CPU_pass") is True
    assert sorted(receipt["health"]) == ["34", "35", "36", "41"]
    assert all(value == 0 for value in receipt["health"].values())
    assert sorted(receipt["stages"]) == ["0-36", "0-41", "1-36", "1-41"]
    assert all(value == 0 for value in receipt["stages"].values())
    assert receipt.get("staged_file_counts_including_metadata") == [24, 23]
    assert receipt.get("launcher_sha256") == STAGE_LAUNCHER_SHA
    cpu = receipt.get("CPU_result") or {}
    assert cpu.get("pass") is True and cpu.get("checks") == cpu.get("checks_passed") == 136
    assert cpu.get("SDK_init_calls") == 0 and cpu.get("NPU_compute_calls") == 0
    assert cpu.get("model_requests") == 0
    review = read_json(ROOT / STAGE_REVIEW)
    assert review.get("review_status") == "APPROVED_FOR_STAGE_ONLY"
    assert review.get("runtime_authorized") is False
    assert review.get("launcher_sha256") == STAGE_LAUNCHER_SHA
    return receipt


def capacity_preconditions(lib):
    assert sha256_file(ROOT / CAPACITY) == CAPACITY_SHA
    capacity = read_json(ROOT / CAPACITY)
    assert capacity.get("project_id") == "glm53" and capacity.get("pass") is True
    assert capacity.get("selected_variant") == "candidate"
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
    assert review.get("runtime_authorized") is True
    assert review.get("model_start_authorized") is False
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


def main():
    assert not RUN.exists(), "transition run already exists: %s" % RUN
    assert not (ROOT / "runs/20261019T-L17-D-stop-current-v1").exists()
    assert not (ROOT / "results/sglang-C32-L17-selected-D-pool-capacity-v1.json").exists()
    lib = load_library()
    stage_preconditions(lib)
    capacity = capacity_preconditions(lib)
    plans = reviewed_plans(lib)
    base = lib.transport()
    RUN.mkdir(parents=True, exist_ok=False)
    (ROOT / "runs/20261019T-L17-D-stop-current-v1").mkdir(parents=True, exist_ok=False)
    (RUN / "execution-once.lock").write_text(str(os.getpid()), encoding="utf-8")

    receipt = {
        "project_id": "glm53",
        "run_id": RID,
        "launcher_sha256": sha256_file(pathlib.Path(__file__)),
        "library_sha256": LIB_SHA,
        "stage_receipt_sha256": sha256_file(ROOT / STAGE_RECEIPT),
        "stage_review_sha256": sha256_file(ROOT / STAGE_REVIEW),
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
        "selected_variant": "candidate",
        "rollback_staged_not_started": True,
        "models_started": False,
        "model_start_allowed": False,
        "capacity_audit_required": True,
        "capacity_output": "results/sglang-C32-L17-selected-D-pool-capacity-v1.json",
        "meta_started_or_stopped": False,
        "P_or_vllm_changed": False,
        "broad_cleanup_performed": False,
        "retry_performed": False,
        "fallback_performed": False,
        "failure_preserved": False,
        "model_requests": 0,
        "SDK_init_calls": 0,
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
            assert payload["old_service_id"] == lib.OLD_D[node]["local_service_id"]
            prechecks[str(node)] = {"rc": result["rc"], **payload}
        receipt["precheck"] = {node: {"rc": value["rc"], "container_id": value["container_id"], "old_service_id": value["old_service_id"]} for node, value in prechecks.items()}
        checkpoint()

        receipt["phase"] = "META_BEFORE_SNAPSHOT"
        checkpoint()
        before = invoke(34, "meta-before", 180)
        before_payload = parsed(before)
        assert before_payload["container_id"] == lib.META_CID
        receipt["meta_before"] = {"rc": before["rc"], "container_id": before_payload["container_id"], "critical_count": before_payload["critical_count"], "critical_sha256": before_payload["critical_sha256"]}
        checkpoint()

        receipt["phase"] = "STOPPING_EXACT_CURRENT_D"
        checkpoint()
        stopped = batch([(node, "stop-%d" % node) for node in D_NODES], 650)
        for node in D_NODES:
            payload = parsed(stopped[str(node)])
            assert payload["resource_shutdown_complete"] is True
            assert payload["provider_closed0"] is True
            assert payload["all_NPUs_empty"] is True
            assert payload["profile_before"] == payload["profile_after"]
            receipt["stop"][str(node)] = {
                "rc": stopped[str(node)]["rc"],
                "container_id": payload["container_id"],
                "resource_shutdown_complete": payload["resource_shutdown_complete"],
                "provider_closed0": payload["provider_closed0"],
                "all_NPUs_empty": payload["all_NPUs_empty"],
                "profile_metadata_unchanged": True,
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
        checkpoint()
        pools = batch([(node, "candidate-pool-%d" % node) for node in D_NODES], 260)
        for node in D_NODES:
            payload = parsed(pools[str(node)])
            assert payload["ready"] is True
            assert payload["provider_status"]["phase"] == "READY"
            assert payload["supervisor_status"]["phase"] == "POOL_READY_WAITING_MODEL_SIGNAL"
            assert payload["supervisor_status"]["model_started"] is False
            assert payload["provider_status"]["configuration"]["dram_size"] == "160GB"
            assert payload["provider_status"]["configuration"]["max_dram_size"] == "160GB"
            receipt["pool"][str(node)] = {"rc": pools[str(node)]["rc"], "container_id": payload["container_id"], "name": payload["name"], "ready": True, "local_service_id": next(event["local_service_id"] for event in payload["provider_status"]["events"] if event["event"] == "READY")}
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
            assert inner["no_copy"] == "true" and inner["candidate_audit_marker"] is True
            assert inner["SDK_init_calls"] == 0 and inner["engine_created"] is False
            assert inner["NPU_compute_calls"] == 0
            receipt["preflight"][str(node)] = {"rc": preflights[str(node)]["rc"], "query_rc": payload["query_rc"], "inner": inner}
        checkpoint()

        receipt["phase"] = "META_AFTER_SNAPSHOT_AND_RETAINED_P"
        checkpoint()
        after = invoke(34, "meta-after", 180)
        after_payload = parsed(after)
        assert after_payload["container_id"] == before_payload["container_id"] == lib.META_CID
        assert after_payload["critical_count"] >= before_payload["critical_count"]
        receipt["meta_after"] = {"rc": after["rc"], "container_id": after_payload["container_id"], "critical_count": after_payload["critical_count"], "critical_sha256": after_payload["critical_sha256"]}
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
