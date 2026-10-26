"""DRAFT_PENDING_PRIMARY_REVIEW: local-only audit of the new selected D pool.

Reads only retained transition artifacts and the accepted P capacity evidence;
makes zero SSH/HTTP/NPU/SDK/model calls. Run it only after the controlled D
transition reached POOLS_READY_AWAITING_INDEPENDENT_CAPACITY_AUDIT and a primary
review accepted the transition receipt.

It derives the two new provider service IDs from the actual READY events, proves
the two fresh 160 GiB Meta mounts, the old D unmounts, the retained P 82/83
identity/context and the 568 GiB total. It records the model-reported KV capacity
as PENDING_MODEL_START; that floor is checked by the later observe/functional gate.
"""

import argparse
import ast
import collections
import hashlib
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB = "scripts/glm53_C32_L17_D_transition_library_v1.py"
LIB_SHA = "95309e2866324a3109e06560e33bdc0b8851e0d2bf99db937da2f05581d23b73"
DEFAULT_RID = "20261019T-L17-D-transition-and-pool-v1"
DEFAULT_OUT = "results/sglang-C32-L17-selected-D-pool-capacity-v1.json"
P_CAPACITY = "results/sglang-C32-L12C-selected-P-pool-capacity-v1.json"
P_CAPACITY_SHA = "19355aab78afeb145fd9f27a91954dabe14696f96d09be90da10a36ff66f7d50"
STAGE_RECEIPT = "runs/20261019T-L17-D-native-eager-stage-v3/receipt.json"
STAGE_REVIEW = "results/C32-L17-D-native-eager-stage-actual-review-v3.json"
CAPACITY_FLOOR_TOKENS = 161280
OLD_D_CAPACITY_TOKENS = 181632
FULL_CAPACITY_BYTES = 568 * 1024 ** 3
D_MOUNT_BYTES = 160 * 1024 ** 3
P_MOUNT_BYTES = {82: 136 * 1024 ** 3, 83: 112 * 1024 ** 3}
MOUNT_RE = re.compile(r"Mount bm on loc\{rank=(\d+),media=DRAM\}, capacity:(\d+)")
UNMOUNT_RE = re.compile(r"(?:Unmount|UnMount)", re.I)
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_library():
    path = ROOT / LIB
    assert sha256_file(path) == LIB_SHA, "plan library SHA mismatch"
    spec = importlib.util.spec_from_file_location("glm53_l17_d_audit_library", path)
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
    assert match, "critical Mount/Unmount line without timestamp: %r" % line[:200]
    return match.group(1)


def active_capacity(lines):
    """Replay Mount/Unmount in timestamp order; values are bytes per rank."""
    events = []
    for line in lines:
        mount = MOUNT_RE.search(line)
        if mount:
            events.append((line_timestamp(line), "mount", int(mount.group(1)), int(mount.group(2)), line))
            continue
        if UNMOUNT_RE.search(line):
            rank_match = re.search(r"rank=(\d+)", line)
            assert rank_match, "Unmount line without rank: %r" % line[:200]
            events.append((line_timestamp(line), "unmount", int(rank_match.group(1)), None, line))
    events.sort(key=lambda item: item[0])
    active = {}
    for _, kind, rank, capacity, line in events:
        if kind == "mount":
            active[rank] = capacity
        else:
            active.pop(rank, None)
    return active, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transition_run_id", nargs="?", default=DEFAULT_RID)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    run = ROOT / "runs" / args.transition_run_id
    assert pathlib.Path(args.transition_run_id).name == args.transition_run_id
    out = ROOT / args.out
    assert not out.exists(), "refusing to overwrite %s" % args.out
    lib = load_library()
    receipt = read_json(run / "receipt.json")
    assert receipt.get("pass") is True
    assert receipt.get("phase") == "POOLS_READY_AWAITING_INDEPENDENT_CAPACITY_AUDIT"
    assert receipt.get("selected_variant") == "candidate"
    assert receipt.get("models_started") is False
    assert receipt.get("model_start_allowed") is False
    assert sha256_file(ROOT / STAGE_RECEIPT) == receipt["stage_receipt_sha256"]
    assert sha256_file(ROOT / P_CAPACITY) == P_CAPACITY_SHA
    capacity_p = read_json(ROOT / P_CAPACITY)
    assert capacity_p.get("pass") is True and capacity_p.get("selected_variant") == "candidate"
    stage = read_json(ROOT / STAGE_RECEIPT)
    assert stage.get("pass") is True
    stage_review = read_json(ROOT / STAGE_REVIEW)
    assert stage_review.get("review_status") == "APPROVED_FOR_STAGE_ONLY"

    # Old D shutdown: resource closure is separate from a non-clean model exit.
    old_unmounts = {}
    for node in (36, 41):
        stopped = raw_payload(run / ("stop-%d.json" % node))
        assert stopped["resource_shutdown_complete"] is True
        assert stopped["provider_closed0"] is True
        assert stopped["all_NPUs_empty"] is True
        assert stopped["profile_before"] == stopped["profile_after"]
        for stream in ("stdout", "stderr"):
            assert stopped["retained_model_logs_before"][stream]["bytes"] > 0
            assert stopped["retained_model_logs_after"][stream]["bytes"] >= stopped["retained_model_logs_before"][stream]["bytes"]
        assert stopped["retained_model_and_diagnostic_files_in_place"] == lib.OLD_D_ROOT
        old_unmounts[str(node)] = {"clean_model_shutdown": stopped["clean_model_shutdown"], "model_shutdown_diagnostic": stopped["model_shutdown_diagnostic"]}

    # Meta snapshots: same owned container, complete critical lines, append-only.
    before = raw_payload(run / "meta-before.json")
    after = raw_payload(run / "meta-after.json")
    assert before["container_id"] == after["container_id"] == lib.META_CID
    before_lines = before["critical_lines"]
    after_lines = after["critical_lines"]
    before_counter = collections.Counter(before_lines)
    after_counter = collections.Counter(after_lines)
    removed = before_counter - after_counter
    assert not removed, "Meta critical snapshot lost %d previously observed lines" % sum(removed.values())
    delta_counter = after_counter - before_counter
    delta = [line for line in after_lines if delta_counter[line] > 0]

    # Actual service IDs come only from the two READY events.
    providers = {}
    service_ids = []
    for node in (36, 41):
        pool = raw_payload(run / ("candidate-pool-%d.json" % node))
        assert pool["ready"] is True
        assert pool["supervisor_status"]["phase"] == "POOL_READY_WAITING_MODEL_SIGNAL"
        assert pool["supervisor_status"]["model_started"] is False
        ready = [event for event in pool["provider_status"]["events"] if event["event"] == "READY"]
        assert len(ready) == 1
        service_id = ready[0]["local_service_id"]
        assert type(service_id) is int and service_id >= 0
        service_ids.append(service_id)
        providers[str(node)] = {
            "container_id": pool["container_id"],
            "name": pool["name"],
            "local_service_id": service_id,
            "dram_GiB": 160,
            "remote_root": lib.STAGES["candidate"]["run_root"],
            "run_id": lib.STAGES["candidate"]["run_id"],
        }
        assert pool["provider_status"]["configuration"]["dram_size"] == "160GB"
        assert pool["provider_status"]["configuration"]["max_dram_size"] == "160GB"
        assert pool["provider_status"]["configuration"]["protocol"] == "device_sdma"
    assert len(set(service_ids)) == 2, service_ids

    # Fresh 160 GiB contributions in the delta, exactly one line per actual ID.
    fresh = []
    for line in delta:
        mount = MOUNT_RE.search(line)
        if mount and int(mount.group(2)) == D_MOUNT_BYTES:
            fresh.append({"rank": int(mount.group(1)), "bytes": int(mount.group(2)), "native_line": line})
    assert sorted(item["rank"] for item in fresh) == sorted(service_ids), (fresh, service_ids)

    # Old D unmount proof in the delta (old IDs 2 and 3 from the accepted P cap file).
    old_ids = {node: capacity_p["providers"][str(node)]["local_service_id"] for node in (36, 41)}
    assert old_ids == {36: 2, 41: 3}
    unmounts = {}
    for node, old_id in old_ids.items():
        matches = [line for line in delta if UNMOUNT_RE.search(line) and re.search(r"rank=%d\b" % old_id, line)]
        assert matches, "old D service ID %d has no Unmount proof" % old_id
        unmounts[str(old_id)] = matches[-1]

    # Active capacity rebuild: P retention, old D gone, two new 160 GiB mounts, 568 GiB.
    before_active, _ = active_capacity(before_lines)
    after_active, _ = active_capacity(after_lines)
    for service_id, expected in P_MOUNT_BYTES.items():
        assert before_active.get(service_id) == expected, (service_id, before_active.get(service_id))
        assert after_active.get(service_id) == expected, (service_id, after_active.get(service_id))
    for old_id in old_ids.values():
        assert before_active.get(old_id) == D_MOUNT_BYTES
        if old_id not in service_ids:
            assert old_id not in after_active
    for service_id in service_ids:
        assert after_active.get(service_id) == D_MOUNT_BYTES
    assert set(after_active) == {82, 83, *service_ids}, after_active
    assert sum(after_active.values()) == FULL_CAPACITY_BYTES
    P_after_lines = [line for line in delta if re.search(r"rank=(?:82|83)\b", line) and UNMOUNT_RE.search(line)]
    assert not P_after_lines, "retained P rank was unmounted"
    retained_p_lines = {}
    for service_id, expected in P_MOUNT_BYTES.items():
        matches = []
        for line in after_lines:
            mount = MOUNT_RE.search(line)
            if mount and int(mount.group(1)) == service_id and int(mount.group(2)) == expected:
                matches.append(line)
        assert matches, "retained P mount line missing for rank %d" % service_id
        retained_p_lines[str(service_id)] = {"bytes": expected, "native_line": matches[-1]}

    # Retained P identity/context unchanged before vs after the transition.
    for node in (34, 35):
        before_health = health_payload(run / ("health-%d.json" % node))
        after_health = health_payload(run / ("retained-P-health-%d.json" % node))
        assert before_health["healthy_idle"] is True and after_health["healthy_idle"] is True
        assert before_health["container_id"] == after_health["container_id"] == capacity_p["providers"][str(node)]["container_id"]
        assert after_health["container_id"] == capacity_p["providers"][str(node)]["container_id"]
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
        raw = read_json(run / ("candidate-preflight-%d.json" % node))
        assert raw.get("rc") == 0, (node, raw.get("rc"), raw.get("stderr"))
        outer = json.loads(raw["stdout"])
        assert outer["query_rc"] == 0
        inner = json.loads(outer["stdout"].split("GLM53_L17_PREFLIGHT=", 1)[1].strip().splitlines()[0])
        assert inner["pass"] is True and inner["variant"] == "candidate"
        assert inner["tp"] == 32 and inner["dp"] == 16 and inner["ep"] == 32 and inner["attention_tp"] == 2
        assert inner["MTP_steps"] == 5 and inner["MTP_draft_tokens"] == 6
        assert inner["kv_cache_dtype"] == "bfloat16"
        assert inner["no_copy"] == "true" and inner["candidate_audit_marker"] is True
        assert inner["SDK_init_calls"] == 0 and inner["engine_created"] is False
        assert inner["NPU_compute_calls"] == 0
        preflight[str(node)] = inner

    result = {
        "project_id": "glm53",
        "pass": True,
        "classification": "D_ONLY_REMOUNT_568GiB_RETAINED_P_META",
        "selected_variant": "candidate",
        "transition_run_id": args.transition_run_id,
        "transition_receipt_sha256": sha256_file(run / "receipt.json"),
        "actual_MC_bytes": FULL_CAPACITY_BYTES,
        "fresh_D_native_mounts": {str(item["rank"]): {"bytes": item["bytes"], "native_line": item["native_line"]} for item in fresh},
        "old_D_unmount_proof": unmounts,
        "old_D_shutdown": old_unmounts,
        "retained_P_native_mounts": retained_p_lines,
        "active_capacity_by_rank": {str(rank): bytes_ for rank, bytes_ in sorted(after_active.items())},
        "providers": providers,
        "meta_container_id": lib.META_CID,
        "Meta_retained": True,
        "P_identities_and_contexts_retained": True,
        "D_identities_and_contexts_retained": True,
        "native_preflight": preflight,
        "source_preflight": {"attention_tp": 2, "dp": 16, "ep": 32, "tp": 32},
        "capacity_floor_tokens": CAPACITY_FLOOR_TOKENS,
        "old_D_reported_KV_capacity": OLD_D_CAPACITY_TOKENS,
        "model_reported_KV_capacity": None,
        "model_capacity_check": "PENDING_MODEL_START",
        "model_started": False,
        "model_qualified": False,
        "no_static_service_id_expectation": True,
        "source_evidence_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted(run.glob("*.json"))
        },
        "model_requests": 0,
        "SDK_init_calls": 0,
        "formal_performance_qualified": False,
    }
    result["source_evidence_sha256"][P_CAPACITY] = P_CAPACITY_SHA
    result["source_evidence_sha256"][STAGE_RECEIPT] = sha256_file(ROOT / STAGE_RECEIPT)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass": True, "out": args.out, "out_sha256": sha256_file(out), "actual_MC_bytes": FULL_CAPACITY_BYTES, "service_ids": sorted(service_ids), "model_started": False}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
