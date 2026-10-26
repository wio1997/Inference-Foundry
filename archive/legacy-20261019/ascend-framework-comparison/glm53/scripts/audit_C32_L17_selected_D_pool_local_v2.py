"""DRAFT_PENDING_PRIMARY_REVIEW: local-only audit of the new selected D pool v2.

Reads only retained transition v2 artifacts and the accepted P capacity evidence;
makes zero SSH/HTTP/NPU/SDK/model calls. Run it only after the controlled D
transition reached POOLS_READY_AWAITING_INDEPENDENT_CAPACITY_AUDIT and a primary
review accepted the transition receipt.

v2 corrections:
- no per-rank native Unmount proof: historical Meta logs only carry aggregate
  Metrics summary unmount_* counters. Old-D resource closure is inferred from the
  owned provider CLOSE_COMPLETE rc0 + supervisor provider_exit_code0 + 8 empty NPUs;
  every per-rank native unmount value is recorded as null with its reason.
- fresh D mounts come only from the Meta before/after critical-line Counter delta,
  exactly one 160 GiB line per actual READY service ID. No guessed 84/85 and no
  historical active-table replay.
- the 568 GiB total and unmount counter delta are read from the actual native
  `Metrics summary: ... dram_used=<used>/<total> ... unmount_success=...` source
  fields, with the after sample timestamp >= the latest fresh D mount timestamp
  on the same Meta log clock. A stale sample leaves capacity PENDING, never PASS.
- preflight checks actual EagerRunner env getter and audit marker, not a static
  constant; provider SETUP/INIT sequences are 2 and the preflight probe adds 0.
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = "scripts/glm53_C32_L17_D_transition_library_v2.py"
LIB_SHA = "da4bbcab4ddf1dd4670c45db78a20aace39f12e1429e73aa5da9d27d36b7b8ce"
DEFAULT_RID = "20261019T-L17-D-transition-and-pool-v2"
DEFAULT_OUT = "results/sglang-C32-L17-selected-D-pool-capacity-v2.json"
P_CAPACITY = "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json"
P_CAPACITY_SHA = "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50"
STAGE_RECEIPT = "runs/20261019T-L17-D-native-eager-stage-v3/receipt.json"
STAGE_RECEIPT_SHA = "d724991b50b03d45eb9c813cdea754c4fd69a42a3a1aabfc85af42f032530cf1"
STAGE_REVIEW = "results/C32-L17-D-native-eager-stage-actual-review-v3.json"
STAGE_REVIEW_SHA = "c176440c0440b9ea41daf057ebca87dbec1b77a26eea2762b11d3c08678f722b"
CAPACITY_FLOOR_TOKENS = 161280
OLD_D_CAPACITY_TOKENS = 181632
FULL_CAPACITY_BYTES = 568 * 1024 ** 3
D_MOUNT_BYTES = 160 * 1024 ** 3
P_MOUNT_BYTES = {82: 136 * 1024 ** 3, 83: 112 * 1024 ** 3}
OLD_D_SERVICE_IDS = {36: 2, 41: 3}
TOTAL_BYTES = 609885356032
MOUNT_RE = re.compile(r"Mount bm on loc\{rank=(\d+),media=DRAM\}, capacity:(\d+)")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
METRICS_PREFIX = "[MMC mmc_meta_service.cpp:539 operator()] Metrics summary: "
METRICS_FIELDS = (
    "mount_req", "mount_success", "mount_fail",
    "unmount_req", "unmount_success", "unmount_fail",
)


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_library():
    path = ROOT / LIB
    assert sha256_file(path) == LIB_SHA, "plan library SHA mismatch"
    spec = importlib.util.spec_from_file_location("glm53_l17_d_audit_library_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_payload(path):
    result = read_json(path)
    assert result.get("rc") == 0, (str(path), result.get("rc"), result.get("stderr"))
    return json.loads(result["stdout"])


def health_payload(path):
    result = read_json(path)
    assert result.get("rc") == 0, (str(path), result.get("rc"), result.get("stderr"))
    marker = "GLM53_L17_HEALTH="
    assert marker in result["stdout"], str(path)
    return json.loads(result["stdout"].split(marker, 1)[1].strip().splitlines()[0])


def line_timestamp(line):
    match = TS_RE.match(line.strip())
    assert match, "native Meta line without timestamp: %r" % line[:200]
    return match.group(1)


def parse_mount_line(line):
    """Return an actual native per-provider Mount record, or None."""
    match = MOUNT_RE.search(line)
    if not match:
        return None
    return {
        "rank": int(match.group(1)),
        "bytes": int(match.group(2)),
        "timestamp": line_timestamp(line),
        "native_line": line,
    }


def parse_metrics_line(line):
    """Parse one actual native Meta aggregate sample from its exact source prefix."""
    if METRICS_PREFIX not in line:
        return None
    match = re.search(r"\bdram_used=(\d+)/(\d+)", line)
    assert match, "metrics summary without dram_used=used/total: %r" % line[:240]
    sample = {
        "timestamp": line_timestamp(line),
        "dram_used": int(match.group(1)),
        "dram_total": int(match.group(2)),
    }
    for name in METRICS_FIELDS:
        field = re.search(r"(?:^| )" + name + r"=(\d+)", line)
        assert field, "metrics summary without %s: %r" % (name, line[:240])
        sample[name] = int(field.group(1))
    sample["source_line_sha256"] = hashlib.sha256(line.encode("utf-8")).hexdigest()
    return sample


def metrics_samples(lines):
    return [parse_metrics_line(line) for line in lines if METRICS_PREFIX in line]


def latest(samples):
    return max(samples, key=lambda item: item["timestamp"]) if samples else None


def critical_prefix_retained(before_lines, after_lines):
    before_counter = collections.Counter(before_lines)
    after_counter = collections.Counter(after_lines)
    removed = before_counter - after_counter
    assert not removed, "Meta critical snapshot lost %d previously observed lines" % sum(removed.values())
    delta_counter = after_counter - before_counter
    return [line for line in after_lines if delta_counter[line] > 0]


def metrics_samples_after(samples, timestamp):
    fresh = [item for item in samples if item["timestamp"] >= timestamp]
    return latest(fresh) if fresh else None


def audit_meta_capacity(before_lines, after_lines, service_ids):
    """Pure Meta capacity decision used by the actual auditor.

    Fresh mounts must be exactly the actual READY service IDs; the aggregate
    sample must be at/after the latest fresh mount on the same log clock.
    """
    delta = critical_prefix_retained(before_lines, after_lines)
    fresh = []
    for line in delta:
        mount = parse_mount_line(line)
        if mount is not None and mount["bytes"] == D_MOUNT_BYTES:
            fresh.append(mount)
    assert sorted(item["rank"] for item in fresh) == sorted(service_ids), (fresh, service_ids)
    assert len(fresh) == len({item["rank"] for item in fresh}) == 2, fresh
    fresh_latest_timestamp = max(item["timestamp"] for item in fresh)
    before_samples = metrics_samples(before_lines)
    after_samples = metrics_samples(after_lines)
    assert before_samples, "Meta before snapshot has no native Metrics summary sample"
    assert after_samples, "Meta after snapshot has no native Metrics summary sample"
    before = latest(before_samples)
    aggregate = metrics_samples_after(after_samples, fresh_latest_timestamp)
    pending_reason = None
    if aggregate is None:
        pending_reason = "META_AFTER_SAMPLE_PRECEDES_FRESH_D_MOUNTS"
    else:
        if aggregate["dram_total"] != TOTAL_BYTES:
            pending_reason = "META_AGGREGATE_TOTAL_MISMATCH"
        elif aggregate["unmount_success"] - before["unmount_success"] < 2:
            pending_reason = "META_UNMOUNT_SUCCESS_DELTA_LT_2"
        elif aggregate["unmount_fail"] - before["unmount_fail"] != 0:
            pending_reason = "META_UNMOUNT_FAIL_DELTA_NONZERO"
        elif aggregate["mount_success"] - before["mount_success"] < 2:
            pending_reason = "META_MOUNT_SUCCESS_DELTA_LT_2"
    return {
        "pass": pending_reason is None,
        "status": "PASS_FRESH_NATIVE_METRICS" if pending_reason is None else "PENDING_OR_FAILED_STRICT",
        "pending_reason": pending_reason,
        "delta": delta,
        "fresh": fresh,
        "fresh_latest_timestamp": fresh_latest_timestamp,
        "last_before": before,
        "aggregate": aggregate,
    }


def verify_stop(node, lib, run):
    stopped = raw_payload(run / ("stop-%d.json" % node))
    assert stopped["container_id"] == lib.OLD_D[node]["container_id"]
    assert stopped["resource_shutdown_complete"] is True
    assert stopped["provider_closed0"] is True
    assert stopped["all_NPUs_empty"] is True
    assert stopped["supervisor"]["phase"] == "STOPPED"
    assert stopped["supervisor"]["provider_exit_code"] == 0
    assert stopped["provider"]["phase"] == "CLOSE_COMPLETE"
    assert stopped["profile_before"] == stopped["profile_after"]
    streams = {}
    for stream in ("stdout", "stderr"):
        before = stopped["retained_model_logs_before"][stream]
        after = stopped["retained_model_logs_after"][stream]
        assert before["bytes"] > 0
        assert after["prefix_match"] is True
        assert after["prefix_bytes"] == before["bytes"]
        assert after["prefix_sha256"] == before["sha256"]
        assert after["after_full"]["bytes"] >= before["bytes"]
        streams[stream] = {
            "prefix_match": True,
            "before_bytes": before["bytes"],
            "before_sha256": before["sha256"],
            "after_full_bytes": after["after_full"]["bytes"],
            "after_full_sha256": after["after_full"]["sha256"],
            "shutdown_addition_bytes": after["shutdown_addition_bytes"],
        }
    return {
        "container_id": stopped["container_id"],
        "resource_shutdown_complete": True,
        "provider_closed0": True,
        "all_NPUs_empty": True,
        "supervisor_provider_exit_code": 0,
        "model_log_prefix": streams,
        "clean_model_shutdown": stopped["clean_model_shutdown"],
        "model_shutdown_diagnostic": stopped["model_shutdown_diagnostic"],
    }


def verify_pool(node, lib, receipt, service_ids, run):
    pool = raw_payload(run / ("candidate-pool-%d.json" % node))
    assert pool["ready"] is True
    assert pool["supervisor_status"]["phase"] == "POOL_READY_WAITING_MODEL_SIGNAL"
    assert pool["supervisor_status"]["model_started"] is False
    provider = pool["provider_status"]
    assert provider["phase"] == "READY"
    ready = [event for event in provider["events"] if event["event"] == "READY"]
    setup = [event for event in provider["events"] if event["event"] == "SETUP_COMPLETE"]
    init = [event for event in provider["events"] if event["event"] == "INIT_COMPLETE"]
    assert len(ready) == len(setup) == len(init) == 1
    assert setup[0].get("rc") == 0 and init[0].get("rc") == 0
    service_id = ready[0]["local_service_id"]
    assert type(service_id) is int and service_id >= 0
    assert service_id not in P_MOUNT_BYTES, service_id
    assert service_id not in OLD_D_SERVICE_IDS.values(), service_id
    service_ids.append(service_id)
    assert provider["configuration"]["dram_size"] == "160GB"
    assert provider["configuration"]["max_dram_size"] == "160GB"
    assert provider["configuration"]["protocol"] == "device_sdma"
    assert provider["configuration"]["init_bm"] is True
    receipt_row = receipt["pool"][str(node)]
    assert receipt_row["local_service_id"] == service_id
    assert receipt_row["container_id"] == pool["container_id"]
    assert receipt_row["provider_SETUP_COMPLETE_rc0"] is True
    assert receipt_row["provider_INIT_COMPLETE_rc0"] is True
    return {
        "container_id": pool["container_id"],
        "name": pool["name"],
        "local_service_id": service_id,
        "dram_GiB": 160,
        "protocol": "device_sdma",
        "init_bm": True,
        "provider_SETUP_COMPLETE_rc0": True,
        "provider_INIT_COMPLETE_rc0": True,
        "provider_READY_events": 1,
        "remote_root": lib.STAGES["candidate"]["run_root"],
        "run_id": lib.STAGES["candidate"]["run_id"],
    }


def verify_preflight(node, lib, run):
    raw = read_json(run / ("candidate-preflight-%d.json" % node))
    assert raw.get("rc") == 0, (node, raw.get("rc"), raw.get("stderr"))
    outer = json.loads(raw["stdout"])
    assert outer["query_rc"] == 0
    inner = json.loads(outer["stdout"].split("GLM53_L17_PREFLIGHT=", 1)[1].strip().splitlines()[0])
    assert inner["pass"] is True and inner["variant"] == "candidate"
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
    return inner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transition_run_id", nargs="?", default=DEFAULT_RID)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    assert pathlib.Path(args.transition_run_id).name == args.transition_run_id
    run = ROOT / "runs" / args.transition_run_id
    out = ROOT / args.out
    assert not out.exists(), "refusing to overwrite %s" % args.out
    lib = load_library()
    assert sha256_file(ROOT / STAGE_RECEIPT) == STAGE_RECEIPT_SHA
    assert sha256_file(ROOT / STAGE_REVIEW) == STAGE_REVIEW_SHA
    assert sha256_file(ROOT / P_CAPACITY) == P_CAPACITY_SHA
    receipt = read_json(run / "receipt.json")
    assert receipt.get("pass") is True
    assert receipt.get("phase") == "POOLS_READY_AWAITING_INDEPENDENT_CAPACITY_AUDIT"
    assert receipt.get("selected_P_variant") == "candidate"
    assert receipt.get("selected_D_variant") == "candidate"
    assert receipt.get("selected_variant") == "candidate"
    assert receipt.get("selected_variant_scope") == "D_ONLY_NOT_P"
    assert receipt.get("models_started") is False
    assert receipt.get("model_start_allowed") is False
    assert receipt.get("provider_SDK_setup_init_sequences") == 2
    assert receipt.get("SDK_init_calls") == 2
    assert receipt.get("preflight_SDK_init_calls") == 0
    assert receipt.get("stage_receipt_sha256") == STAGE_RECEIPT_SHA
    assert receipt.get("stage_review_sha256") == STAGE_REVIEW_SHA
    assert receipt.get("candidate_stage_sha256") == lib.STAGES["candidate"]["stage_sha256"]
    assert receipt.get("rollback_stage_sha256") == lib.STAGES["rollback"]["stage_sha256"]
    capacity_p = read_json(ROOT / P_CAPACITY)
    assert capacity_p.get("pass") is True
    assert lib.p_selection(capacity_p) == "candidate"
    assert "P_ONLY" in capacity_p.get("classification", "")
    stage_receipt = read_json(ROOT / STAGE_RECEIPT)
    assert stage_receipt.get("pass") is True
    stage_review = read_json(ROOT / STAGE_REVIEW)
    assert stage_review.get("review_status") == "APPROVED_FOR_STAGE_ONLY"

    # Old D: owned resource closure only; no per-rank native unmount claim.
    old_D_close = {str(node): verify_stop(node, lib, run) for node in (36, 41)}
    old_ids = {node: capacity_p["providers"][str(node)]["local_service_id"] for node in (36, 41)}
    assert old_ids == OLD_D_SERVICE_IDS, old_ids
    old_unmount = {
        str(service_id): {
            "native_unmount_line": None,
            "native_unmount_rank": None,
            "reason": "Meta native Metrics summary only logs aggregate unmount_req/success/fail counters; no per-rank Unmount lines were observed",
            "scope": "owned-D-provider-close-inference only",
        }
        for service_id in old_ids.values()
    }

    # Actual D service IDs come only from the two READY events.
    providers = {}
    service_ids = []
    for node in (36, 41):
        providers[str(node)] = verify_pool(node, lib, receipt, service_ids, run)
    assert len(set(service_ids)) == 2, service_ids

    # Meta snapshots: same owned node34 container, prefix retained.
    before = raw_payload(run / "meta-before.json")
    after = raw_payload(run / "meta-after.json")
    assert before["container_id"] == after["container_id"] == lib.META_CID
    assert before.get("meta_node") == after.get("meta_node") == 34
    assert before.get("meta_labels") == after.get("meta_labels")
    assert before["meta_labels"]["project_id"] == "glm53"
    assert before["meta_labels"]["run_id"] == lib.OLD_D_RID
    before_lines = before["critical_lines"]
    after_lines = after["critical_lines"]

    # Fresh D mounts plus the strictly fresh native aggregate/counter sample.
    meta_audit = audit_meta_capacity(before_lines, after_lines, service_ids)
    delta = meta_audit["delta"]
    fresh = meta_audit["fresh"]
    fresh_latest_timestamp = meta_audit["fresh_latest_timestamp"]
    last_before = meta_audit["last_before"]
    aggregate = meta_audit["aggregate"]
    pending_reason = meta_audit["pending_reason"]
    capacity_ok = meta_audit["pass"]
    metrics_scope = {
        "aggregate_scope": "Meta native aggregate dram_used=used/total; total is the configured 568 GiB pool, not per-rank active replay",
        "counter_scope": "cumulative native mount/unmount counters; delta from last before sample to the latest fresh after sample",
        "freshness_scope": "after sample timestamp >= latest fresh D Mount timestamp on the same node34 Meta log clock",
        "per_rank_scope": "per-rank unmount is not logged; null is retained instead of a fabricated record",
    }

    # Retained P reference lines (actual observed lines, not an active-table replay).
    retained_p = {}
    for service_id, expected in P_MOUNT_BYTES.items():
        matches = []
        for line in before_lines + after_lines:
            mount = parse_mount_line(line)
            if mount is not None and mount["rank"] == service_id and mount["bytes"] == expected:
                matches.append(mount)
        assert matches, "retained P mount reference missing for rank %d" % service_id
        retained_p[str(service_id)] = {
            "bytes": matches[-1]["bytes"],
            "native_line": matches[-1]["native_line"],
            "scope": "observed retained-P native Mount reference; not an active-capacity replay",
        }

    # Retained P identity/context and actual CIDs.
    for node in (34, 35):
        before_health = health_payload(run / ("health-%d.json" % node))
        after_health = health_payload(run / ("retained-P-health-%d.json" % node))
        assert before_health["healthy_idle"] is True and after_health["healthy_idle"] is True
        assert before_health["container_id"] == after_health["container_id"] == capacity_p["providers"][str(node)]["container_id"]
        assert before_health["NPU_context_PIDs"] == after_health["NPU_context_PIDs"]
        assert len(after_health["NPU_context_PIDs"]) == 18

    # Candidate prepare and D-only preflight (TP2 attention, DP16, EP32).
    preflight = {}
    for node in (36, 41):
        prepared = raw_payload(run / ("candidate-prepare-%d.json" % node))
        assert prepared["empty_NPU_cards"] == 8
        assert prepared["model_weights_files"] == 193 and prepared["model_weights_bytes"] == 764240829290
        assert prepared["MemAvailable"] > 1100 * 1024 ** 3
        assert prepared["private_optional_extension_mounted"] is False
        stage_config = read_json(ROOT / lib.STAGES["candidate"]["bundle"] / ("config/node-%d.json" % node))
        parallel = stage_config["parallel_intent"]
        assert parallel["launch_tp"] == 32 and parallel["attention_tp"] == 2
        assert parallel["attention_dp"] == 16 and parallel["expert_parallel"] == 32
        preflight[str(node)] = verify_preflight(node, lib, run)

    result = {
        "project_id": "glm53",
        "pass": capacity_ok,
        "capacity_status": "PASS_FRESH_NATIVE_METRICS" if capacity_ok else "PENDING_OR_FAILED_STRICT",
        "capacity_pending_reason": pending_reason,
        "classification": "D_ONLY_REMOUNT_568GiB_RETAINED_P_META",
        "selected_P_variant": "candidate",
        "selected_D_variant": "candidate",
        "selected_variant": "candidate",
        "selected_variant_scope": "D_ONLY_NOT_P",
        "transition_run_id": args.transition_run_id,
        "transition_receipt_sha256": sha256_file(run / "receipt.json"),
        "expected_MC_bytes": TOTAL_BYTES,
        "actual_MC_bytes": None if aggregate is None else aggregate["dram_total"],
        "native_aggregate_capacity": None if aggregate is None else {
            "timestamp": aggregate["timestamp"],
            "dram_used": aggregate["dram_used"],
            "dram_total": aggregate["dram_total"],
            "mount_success": aggregate["mount_success"],
            "mount_fail": aggregate["mount_fail"],
            "unmount_success": aggregate["unmount_success"],
            "unmount_fail": aggregate["unmount_fail"],
            "source_line_sha256": aggregate["source_line_sha256"],
            "last_before_sample": {
                "timestamp": last_before["timestamp"],
                "dram_total": last_before["dram_total"],
                "mount_success": last_before["mount_success"],
                "mount_fail": last_before["mount_fail"],
                "unmount_success": last_before["unmount_success"],
                "unmount_fail": last_before["unmount_fail"],
                "source_line_sha256": last_before["source_line_sha256"],
            },
            "deltas": {
                "mount_success": aggregate["mount_success"] - last_before["mount_success"],
                "mount_fail": aggregate["mount_fail"] - last_before["mount_fail"],
                "unmount_success": aggregate["unmount_success"] - last_before["unmount_success"],
                "unmount_fail": aggregate["unmount_fail"] - last_before["unmount_fail"],
            },
            "scope": metrics_scope,
        },
        "native_metrics_scope": metrics_scope,
        "fresh_D_native_mounts": {
            str(item["rank"]): {
                "bytes": item["bytes"],
                "timestamp": item["timestamp"],
                "native_line": item["native_line"],
                "scope": "Meta after-minus-before delta; one actual 160 GiB Mount per READY service ID",
            }
            for item in fresh
        },
        "retained_P_native_mount_reference": retained_p,
        "old_D_owned_close": old_D_close,
        "old_D_per_rank_native_unmount": old_unmount,
        "old_D_close_scope": "owned old-D provider CLOSE_COMPLETE rc0 + supervisor provider_exit_code0/STOPPED + 8 empty NPUs; no per-rank native unmount line exists",
        "retained_P_actual_container_ids": {
            str(node): capacity_p["providers"][str(node)]["container_id"] for node in (34, 35)
        },
        "retained_P_native_capacity_GiB": {str(node): capacity_p["providers"][str(node)]["dram_GiB"] for node in (34, 35)},
        "providers": providers,
        "meta_container_id": lib.META_CID,
        "Meta_retained": True,
        "P_identities_and_contexts_retained": True,
        "new_D_identities_and_contexts_created": True,
        "native_preflight": preflight,
        "provider_SDK_setup_init_sequences": 2,
        "preflight_SDK_init_calls": 0,
        "source_preflight": {"attention_tp": 2, "dp": 16, "ep": 32, "tp": 32},
        "capacity_floor_tokens": CAPACITY_FLOOR_TOKENS,
        "old_D_reported_KV_capacity": OLD_D_CAPACITY_TOKENS,
        "model_reported_KV_capacity": None,
        "model_capacity_check": "PENDING_MODEL_START",
        "model_started": False,
        "model_qualified": False,
        "no_static_service_id_expectation": True,
        "no_fabricated_rank_unmount": True,
        "no_historical_active_capacity_replay": True,
        "source_evidence_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted(run.glob("*.json"))
        },
        "model_requests": 0,
        "SDK_init_calls_in_preflight": 0,
        "formal_performance_qualified": False,
    }
    result["source_evidence_sha256"][P_CAPACITY] = P_CAPACITY_SHA
    result["source_evidence_sha256"][STAGE_RECEIPT] = STAGE_RECEIPT_SHA
    result["source_evidence_sha256"][STAGE_REVIEW] = STAGE_REVIEW_SHA
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pass": capacity_ok,
        "capacity_status": result["capacity_status"],
        "pending_reason": pending_reason,
        "out": args.out,
        "out_sha256": sha256_file(out),
        "expected_MC_bytes": TOTAL_BYTES,
        "actual_MC_bytes": None if aggregate is None else aggregate["dram_total"],
        "service_ids": sorted(service_ids),
        "fresh_D_mount_ids": sorted(item["rank"] for item in fresh),
        "unmount_success_delta": None if aggregate is None else aggregate["unmount_success"] - last_before["unmount_success"],
        "model_started": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
