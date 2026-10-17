#!/usr/bin/env python3
"""Summarize glm5-3-vllm/evidence/20261017-inventory raw evidence into summary.json.

Offline, stdlib only; raw files read-only. Reads the SPECS matches plus every node*-hccl-*/ run
directory (auto-discovered by glob, so a later run is pulled in on the next invocation) and writes
only summary.json. `sources` lists SHA256 of exactly the files read. Communication runs are recorded
as they are: a failed run stays failed and no performance number is derived from any run.
"""
import glob, hashlib, json, re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INV = REPO / "glm5-3-vllm" / "evidence" / "20261017-inventory"
OUT = INV / "summary.json"
SPECS = dict([
    ("node32.inventory", "node32-inventory-*/stdout.json"), ("node32.assets", "node32-assets-*/stdout.json"), ("node32.detail", "node32-detail-*/stdout.json"), ("node32.k8s_pods", "node32-k8s-pods.json"), ("node32.reserved_devices", "node32-reserved-devices.json"),
    ("node33.inventory", "node33-inventory-*/stdout.json"), ("node33.assets", "node33-assets-*/stdout.json"), ("node33.detail", "node33-detail-*/stdout.json"), ("node33.k8s_pods", "node33-k8s-pods.json"), ("node33.image_metadata_v2", "node33-image-metadata-v2-*/stdout.json"),
    ("node33.source", "node33-source-*/stdout.json"), ("node33.npu_smoke", "node33-npu-smoke-*/stdout.json"), ("node33.sdma_source", "node33-sdma-source-*/stdout.json"), ("node33.reservation_create", "node33-reservation-create.json"), ("node33.reservation_ready", "node33-reservation-ready.json"),
    ("node33.reservation_cleanup", "node33-reservation-cleanup.json"), ("node33.reservation_events", "reservation_events.json"), ("node33.node_capacity", "node-capacity.json"), ("node32.reservation_create_authorized", "node32-reservation-create-authorized.json"), ("node33.reservation_create_authorized", "node33-reservation-create-authorized.json"),
    ("workload_release", "workload-release.json"), ("node32.reservation_create_v2", "node32-reservation-create-authorized-v2.json"), ("node33.reservation_create_v2", "node33-reservation-create-authorized-v2.json"), ("node32.reservation_ready_v2", "node32-reservation-ready-authorized-v2.json"), ("node33.reservation_ready_v2", "node33-reservation-ready-authorized-v2.json")])
NODES = ("node32", "node33")
SPECS.update({
    'capacity.ready': 'node33-capacity-readiness-*/metrics.json',
    'capacity.functional': 'node33-capacity-functional-*/metrics.json',
    'capacity.stopped': 'node33-capacity-stopped-*/metrics.json',
    'memcache.api': 'node32-memcache-api-*/metrics.json',
    'memcache.trace32': 'node32-memcache-sdma-close-profile-*/memcache-trace-analysis.json',
    'memcache.trace33': 'node33-memcache-sdma-close-profile-*/memcache-trace-analysis.json',
    'memcache.lifecycle': 'node32-memcache-init-close-only-*/metrics.json',
    'memcache.lifecycle_result': 'node32-memcache-init-close-only-*/result.json',
    'resources.final': 'final-task-resources.json',
    'idle32': 'node32-final-device-idle-*/metrics.json',
    'idle33': 'node33-final-device-idle-*/metrics.json',
})
MARKER = re.compile(r"GLM53_RESULT_JSON_BEGIN\s*(\{.*?\})\s*GLM53_RESULT_JSON_END", re.S)
RAW, REL, SOURCES, ERRORS = {}, {}, [], []


def rel(path):
    return str(path.relative_to(REPO))
def load(key):
    hits = sorted(glob.glob(str(INV / SPECS[key])))
    if len(hits) != 1:
        ERRORS.append("%s: %d matches" % (key, len(hits)))
        return
    data = Path(hits[0]).read_bytes()
    RAW[key], REL[key] = json.loads(data), rel(Path(hits[0]))
    SOURCES.append({"path": REL[key], "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
def load_abs(path):
    data = path.read_bytes()
    SOURCES.append({"path": rel(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    return data
def ev(key, field=None):
    return REL[key] if not field else "%s#%s" % (REL[key], field)
def line_field(text, name):
    return next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith(name)), None)
def cmd_stdout(inv, name):
    return next((c["stdout"] for c in inv["commands"] if c["name"] == name), "")
def kv_lines(text):
    return {" ".join(k.split()): " ".join(v.split()) for k, v in (ln.split(":", 1) for ln in text.splitlines() if ":" in ln)}
def parse_netdev(t):
    out = {}
    for ln in t.splitlines():
        if ":" in ln:
            name, rest = ln.split(":", 1)
            f = rest.split()
            if len(f) >= 16:
                out[name.strip()] = (int(f[0]), int(f[8]))
    return out
def parse_diskstats(t):
    return {f[2]: (int(f[5]), int(f[9])) for f in (ln.split() for ln in t.splitlines()) if len(f) >= 14}
def window_rates(d):  # 5s background counters; diskstats sectors assumed 512B; instantaneous only
    w0, w1 = d["quiet_window_before"], d["quiet_window_after"]
    dt = round(w1["monotonic"] - w0["monotonic"], 3)
    def diff(b, a, i, scale=1.0):
        return round((a[i] - b[i]) * scale / dt / 1e6, 3)
    nb, na = parse_netdev(w0["netdev"]), parse_netdev(w1["netdev"])
    db, da = parse_diskstats(w0["diskstats"]), parse_diskstats(w1["diskstats"])
    nics = {n: {"rx_MBps": diff(nb[n], na[n], 0), "tx_MBps": diff(nb[n], na[n], 1)} for n in sorted(set(nb) & set(na)) if n.startswith(("enp", "bond4"))}
    disks = {n: {"read_MBps": diff(db[n], da[n], 0, 512), "write_MBps": diff(db[n], da[n], 1, 512)} for n in sorted(set(db) & set(da)) if n.startswith("nvme") or n == "sda"}
    return {"duration_s": dt, "utc_window": [w0["utc"], w1["utc"]], "nic_MBps": nics, "disk_MBps_512B_sector": disks}
def topo_counts(text):
    rows = [ln.split()[1:] for ln in text.splitlines() if ln.strip().startswith("Phy-ID") and not ln.split()[1].startswith("Phy-ID")]
    cells = [c for row in rows for c in row]
    return {"rows": len(rows), "SIO_links": cells.count("SIO"), "HCCS_SW_links": cells.count("HCCS_SW")}
def ascend_holders(obj):
    pods, holders = json.loads(obj["stdout"]), []  # outer wrapper -> inner stdout JSON string
    for p in pods:
        for c in p.get("containers", []):
            res = c.get("resources") or {}
            req = {k: v for k, v in (res.get("requests") or {}).items() if "Ascend" in k}
            lim = {k: v for k, v in (res.get("limits") or {}).items() if "Ascend" in k}
            if req or lim:
                holders.append({"pod": p["name"], "namespace": p.get("namespace"), "phase": p.get("phase"), "container": c["name"], "requests": req, "limits": lim})
    return {"pods_total": len(pods), "ascend_die_requests": sum(int(v) for h in holders for v in h["requests"].values()), "holders": holders}
def node_facts(inv_key, detail_key):
    inv, detail = RAW[inv_key], RAW[detail_key]
    cmds = {" ".join(c["argv"][:4]): c["stdout"] for c in detail["commands"]}
    hccs = kv_lines(cmds.get("npu-smi info -t hccs", ""))
    image = json.loads(cmd_stdout(inv, "docker_image_identity"))
    lscpu, free = cmd_stdout(inv, "lscpu"), cmd_stdout(inv, "free").splitlines()[1].split()
    cpu = {k: line_field(lscpu, v) for k, v in (("arch", "Architecture"), ("model", "BIOS Model name"), ("cpus", "CPU(s)"), ("threads_per_core", "Thread(s) per core"), ("sockets", "Socket(s)"), ("numa_nodes", "NUMA node(s)"))}
    return {"hostname": cmd_stdout(inv, "hostname").strip(), "cpu": {**cpu, "evidence": ev(inv_key, "command:lscpu")},
        "memory": {"total": free[1], "used": free[2], "free": free[3], "available": free[6], "evidence": ev(inv_key, "command:free")}, "docker_image": {"id": image["Id"], "tags": image["RepoTags"], "created": image["Created"], "size_bytes": image["Size"], "evidence": ev(inv_key, "command:docker_image_identity")},
        "nic_numa_speed": {"entries": cmd_stdout(inv, "sysfs_net_numa_speed").strip().splitlines(), "evidence": ev(inv_key, "command:sysfs_net_numa_speed")}, "pci_numa": {"entries": len([ln for ln in cmd_stdout(inv, "sysfs_pci_numa").splitlines() if ln.strip()]), "evidence": ev(inv_key, "command:sysfs_pci_numa")},
        "inventory_warnings": inv.get("warnings", []), "topology": topo_counts(cmds.get("npu-smi info -t topo", "")), "spod_info": kv_lines(cmds.get("npu-smi info -t spod-info", "")),
        "hccs": {"health": hccs.get("hccs health status"), "retry_count": hccs.get("hccs retry count"), "error_count": hccs.get("hccs error count"), "link_speed": hccs.get("hccs link speed")},
        "quiet_window": {**window_rates(detail), "known_transfer_process_matches": detail["known_transfer_process_matches"], "evidence": ev(detail_key),
                         "scope_note": detail["note"] + " 512-byte sector assumed; instantaneous diagnostic only, isolation not guaranteed."},
        "dev_nodes": {"total": len(detail["dev_nodes"]), "davinci_dies": len([n for n in detail["dev_nodes"] if n.rsplit("/dev/davinci", 1)[-1].isdigit()])}}
def assets_facts(a, key):
    return {"weights_path": a["weights_path"], "canonical_alias": a["canonical_alias"], "file_count": a["file_count"], "total_bytes": a["total_bytes"],
        "receipt_bytes": a["receipt"]["bytes"], "manifest_matches_receipt": a["manifest_matches_receipt"], "scope": a["scope"], "missing": a["missing"],
        "receipt_all_payload_sha256_verified": a["receipt"]["all_file_payload_sha256_verified"], "size_mismatch": a["size_mismatch"],
        "index_shards_missing_from_receipts": a["index_shards_missing_from_receipts"], "index_tensor_count": a["index_tensor_count"], "evidence": ev(key),
        "quant_string_counts": a["quant_string_counts"], "quant_description_sha256": a["quant_description_sha256"]}


def split_stdout(text):  # stdout carries an ATB/mki_log prefix line; v2+ also wraps the report in markers
    m = MARKER.search(text)
    if m:
        return json.loads(m.group(1)), True
    return json.loads(text[text.find("\n{") + 1:text.rfind("}") + 1]), False
def run_env(argv):
    return {argv[i + 1].split("=", 1)[0]: argv[i + 1].split("=", 1)[1] for i, a in enumerate(argv) if a == "-e" and i + 1 < len(argv) and "=" in argv[i + 1]}
def env_gaps(env):  # mechanical caller-env checks; the authoritative cause is each rank's recorded error tail
    gaps = [] if "HCCL_INTRA_PCIE_ENABLE" in env else ["HCCL_INTRA_PCIE_ENABLE not set in caller env"]
    ct = env.get("HCCL_CONNECT_TIMEOUT")
    if ct and ct.isdigit() and int(ct) < 120:
        gaps.append("HCCL_CONNECT_TIMEOUT=%s below required >=120" % ct)
    return gaps
def rank_records(report):
    ranks = []
    for r in report["rank_results"]:
        err = r.get("error")
        ranks.append({"rank": r["rank"], "logical_device": r["logical_device"], "device_name": r["device_name"],
            "cases_completed": len(r.get("cases") or []), "timed_messages": sum(len(c.get("ns") or []) for c in (r.get("cases") or [])),
            "error_chars": len(err) if err else 0, "error_tail_1200": err[-1200:] if err else None,
            "error_tail_truncated": bool(err) and len(err) > 1200, "error_last_line": err.strip().splitlines()[-1][:200] if err else None})
    return ranks
def comm_run(d):
    res, ctx = json.loads(load_abs(d / "result.json")), json.loads(load_abs(d / "context.json"))
    text, log_bytes = load_abs(d / "stdout.json").decode("utf-8"), load_abs(d / "sdma-log-evidence.txt") if (d / "sdma-log-evidence.txt").is_file() else b""
    report, marked = split_stdout(text)
    env, ranks = run_env(res["argv"]), rank_records(report)
    log = log_bytes.decode("utf-8").strip()
    analysis = json.loads(load_abs(d / "sdma-analysis.json")) if (d / "sdma-analysis.json").is_file() else None
    return {"run": d.name, "node": ctx["node"], "label": ctx["label"], "npu_devices": ctx["npu_devices"], "git_sha": ctx["git_sha"], "run_started_host_utc": res["started_host_utc"],
        "run_duration_s": round(res["duration_s"], 3), "lease_read_at_collection_utc": ctx["client_started_utc"], "resource_lease": ctx["resource_lease"], "script_sha256": res["script_sha256"],
        "image_id": res["image_id"], "timed_out": res["timed_out"], "rc": res["returncode"], "ok": report["ok"], "marker_delimited_stdout": marked, "stdout_prefix_line": text.splitlines()[0],
        "env": {k: v for k, v in sorted(env.items()) if k.startswith(("HCCL_", "ASCEND_"))}, "env_gap_flags": env_gaps(env), "sdma_runtime_path_verified": report["sdma_runtime_path_verified"], "rank_message_counts": [r["timed_messages"] for r in ranks], "ranks": ranks,
        "sdma_log_evidence": {"file_bytes": len(log_bytes), "excerpt": log.splitlines()[0][:300] if log else None}, "sdma_analysis": analysis,
        "evidence": [rel(d / n) for n in ("result.json", "stdout.json", "context.json", "sdma-log-evidence.txt", "sdma-analysis.json") if (d / n).is_file()]}


for key in SPECS:
    load(key)
RUN_DIRS = [d for d in sorted(INV.glob("node*-hccl-*/")) if all((d / f).is_file() for f in ("result.json", "stdout.json", "context.json"))]
RUN_NOTES = ["skipped incomplete run dir: %s" % rel(d) for d in sorted(INV.glob("node*-hccl-*/")) if d not in RUN_DIRS]
RUNS = sorted((comm_run(d) for d in RUN_DIRS), key=lambda r: r["run_started_host_utc"])
if ERRORS:
    OUT.write_text(json.dumps({"read_errors": ERRORS, "sources": SOURCES}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit("read errors: %s" % ERRORS)

ann = json.loads(RAW["node32.reserved_devices"]["stdout"])["annotations"]
cfg = json.loads(ann["ascend.kubectl.kubernetes.io/ascend-910-configuration"])
meta, src, sd, s = RAW["node33.image_metadata_v2"], RAW["node33.source"], RAW["node33.sdma_source"], RAW["node33.npu_smoke"]
create, ready, cleanup = RAW["node33.reservation_create"], RAW["node33.reservation_ready"], RAW["node33.reservation_cleanup"]
ca_old, ca = {n: RAW["%s.reservation_create_authorized" % n] for n in NODES}, {n: RAW["%s.reservation_create_v2" % n] for n in NODES}
rq, release = {n: RAW["%s.reservation_ready_v2" % n] for n in NODES}, RAW["workload_release"]
events = json.loads(RAW["node33.reservation_events"]["stdout"])["items"]
cap = {n["name"]: {"capacity_Ascend910": n["capacity"].get("huawei.com/Ascend910"), "allocatable_Ascend910": n["allocatable"].get("huawei.com/Ascend910"), "ready": next((c["status"] for c in n["conditions"] if c["type"] == "Ready"), None)} for n in json.loads(RAW["node33.node_capacity"]["stdout"])}
ok_runs, gaps, run_contexts = [r for r in RUNS if r["ok"]], sorted({g for r in RUNS for g in r["env_gap_flags"]}), [r["evidence"][2] for r in RUNS]  # run_contexts: each run's context.json, for lease cross-reference
runtime_sdma_runs = [r for r in RUNS if r["sdma_analysis"] and r["sdma_analysis"].get("runtime_sdma_link_evidence")]  # separate plog-analysis artifact, not the probe flag

CONFLICTS = [
    ("inventory_warnings", "open", "npu_smi_usage rc=215 and ps_fields stdout truncated at 262144 bytes on both nodes.", ["node32.inventory", "node33.inventory"]),
    ("mooncake_package_name", "open", "pip name mooncake-transfer-engine shows installed=false, but NPU distribution mooncake-transfer-engine-npu 0.3.11.post1 is present; must not be read as absent.", ["node33.image_metadata_v2", "node33.source"]),
    ("receipt_scope", "open", "196 files / 764240832702 bytes are receipt identity plus current metadata; no independent payload rehash.", ["node32.assets", "node33.assets"]),
    ("node33_no_free_dies", "historical_resolved", "Initial snapshot only (17:37Z pod lists): node33 16/16 dies held by sglang-pd4p2d-reserve-33-20261014 and node32 2/16 by containerworkload-694f3t4n; in that state the first node33 16-die request failed scheduling (Insufficient huawei.com/Ascend910) and was deleted. Released 2026-10-16T17:57Z under user authorization; history, not current occupancy.", ["node33.reservation_events", "node33.node_capacity", "node33.k8s_pods", "node32.k8s_pods", "workload_release"]),
    ("lease_create_timestamp_ordering", "clock_domains", "Kubernetes, target hosts, and Mac wall clocks differ by minutes. Cross-domain creationTimestamp/client receipt ordering is not a race certificate; compare UID/node/resource identities and use same-host clocks for durations.", ["node33.reservation_create_v2"] + run_contexts),
    ("vllm_version_string", "open", "site-packages vllm version '0.30.0+empty' vs git SHA ced6857...; mapping not resolved by this summary.", ["node33.image_metadata_v2", "node33.source"]),
    ("quiet_window_limit", "open", "Five-second background snapshot only; supports no isolation guarantee for later runs.", ["node32.detail", "node33.detail"])]
POLICY = [
    ("sdma_only", "user directive (session), not raw evidence", "All data transfers must use SDMA; record control/bootstrap TCP separately.", []),
    ("no_fabricated_numbers", "user directive", "Report only verified scope: one node's tiny-window full-model capacity/functionality is established; no production-capacity, precision acceptance, throughput or 1P1D feasibility follows. Failed probes stay failed even when payloads pass.", []),
    ("memcache_priority", "latest user constraint relayed by the main conversation", "Pause new Mooncake work; prioritize installed MemCache/MemFabric and explicit device_sdma. A mixed MultiConnector example is not proof of pure MemCache PD.", []),
    ("allocation_ownership", "user authorization + workload-release.json + reservation v2 credentials",
     "User authorized killing all workloads on nodes 32/33; release executed (node32 testportal Deployment scaled to 0, node33 old reserve pod deleted) and this task's 16-die reservations are Ready with a 7200s deadline. Old occupancy is history, not a pending question.",
     ["workload_release", "node32.reservation_create_v2", "node33.reservation_create_v2", "node32.reservation_ready_v2", "node33.reservation_ready_v2"])]

layers = {
    "static_presence": {"done": True, "status": "present_static_only", "evidence": [ev("node33.sdma_source"), ev("node33.image_metadata_v2"), ev("node33.source")], "note": "ascend_transport.so and mooncake files present; pinned vllm/vllm-ascend git SHAs clean. File presence only, no routing proof."},
    "device_check": {"done": True, "status": "partial_pass", "evidence": [ev("node33.npu_smoke"), ev("node32.detail"), ev("node33.detail")], "note": "Single visible die correctness pass (model_loaded false); HCCS health OK and 16-die topology on both nodes. No multi-die collective exercised."},
    "communication_correctness": {"done": bool(ok_runs), "status": ("established" if len(ok_runs) == len(RUNS) else "established_for_passing_runs_only") if ok_runs else "not_established", "evidence": [r["evidence"][0] for r in RUNS], "note": "%d of %d discovered 2-rank HCCL runs returned ok=false; the remaining %d completed ok=true with all recorded payload checks passing. Failures kept as recorded." % (len(RUNS) - len(ok_runs), len(RUNS), len(ok_runs))},
    "runtime_sdma_evidence": {"done": False, "status": "plog_link_evidence_present" if runtime_sdma_runs else "unverified", "evidence": [r["evidence"][1] for r in RUNS] + [p for r in runtime_sdma_runs for p in r["evidence"][3:]], "note": "The probe field sdma_runtime_path_verified is false in all %d runs (kept as recorded, not flipped to true). %d run(s) carry a separate sdma-analysis.json whose plog analysis reports runtime_sdma_link_evidence=true; it is bounded to retained-log evidence with no device-profiler packet accounting." % (len(RUNS), len(runtime_sdma_runs))},
    "single_model_functionality": {'done': True, 'status': 'ready_one_request_then_stopped',
        'evidence': [ev('capacity.ready'), ev('capacity.functional'), ev('capacity.stopped')],
        'note': 'Full TP16 model loaded on node33, HTTP200 for 12 prompt/32 generated tokens, then stopped exit0. Tiny-window functionality, not accuracy/performance/PD acceptance.'},
    "memcache_sdma_data_path": {'done': True, 'status': 'payload_and_sdma_verified_lifecycle_failed',
        'evidence': [ev('memcache.api'), ev('memcache.trace32'), ev('memcache.trace33'), ev('memcache.lifecycle'), ev('memcache.lifecycle_result')],
        'note': 'Installed MemCache/MemFabric1.2.0; synthetic pool put/get hashes match, 8 SDMA_SQE tasks across two local traces. SDK close returns0 but processes exit133; init/close-only with zero torch allocation also reproduces. No dependency replacement or model PD launched.'},
    "complete_pd": {"done": False, "status": "not_started", "evidence": [ev('capacity.functional'), ev('memcache.lifecycle_result')], "note": "Single model functionality and synthetic MemCache transport are separate from P/D model execution. C8 KV production/consumption and exclusion of D-side full prefill remain unverified."}}

reservations = {n: {"pod": ca[n]["manifest"]["metadata"]["name"], "create_rc": ca[n]["rc"], "create_stdout": ca[n]["stdout"].strip(), "ready_rc": rq[n]["rc"], "activeDeadlineSeconds": ca[n]["manifest"]["spec"]["activeDeadlineSeconds"],
                    "command": ca[n]["manifest"]["spec"]["containers"][0]["command"], "requests_Ascend910": ca[n]["manifest"]["spec"]["containers"][0]["resources"]["requests"]["huawei.com/Ascend910"], "preemptionPolicy": ca[n]["manifest"]["spec"].get("preemptionPolicy"),
                    "ready_stdout": rq[n]["stdout"].strip(), "ready_stderr": rq[n]["stderr"].strip(), "evidence": [ev("%s.reservation_create_v2" % n), ev("%s.reservation_ready_v2" % n)]} for n in NODES}

summary = {
    "schema": "glm53-inventory-summary/3", "generated_utc": datetime.now(timezone.utc).isoformat(), "inventory_dir": str(INV.relative_to(REPO)),
    "scope_note": "Derived offline from read-only raw evidence; paths relative to repo root; no independent payload rehash; per-run communication records keep failures as recorded and derive no performance numbers; verification layers are separated below.",
    "read_errors": ERRORS, "verification_layers": layers,
    'memcache': {'memcache_build': RAW['memcache.api']['memcache_hybrid_version'],
        'memfabric_build': RAW['memcache.api']['memfabric_hybrid_version'],
        'trace_analysis': {n: RAW['memcache.trace'+n] for n in ('32', '33')},
        'init_close_only': RAW['memcache.lifecycle'],
        'init_close_only_returncode': RAW['memcache.lifecycle_result']['returncode'],
        'lingqu_network_version': None, 'lingqu_version_requirement': '>=1.5 per installed guide, not independently verified'},
    'final_scope': {'resources': RAW['resources.final'],
        'evidence': [ev('resources.final'), ev('idle32'), ev('idle33')],
        'devices_idle': {n: RAW['idle'+n]['npu_smi_stdout'].count('No running processes found in NPU') == 8 for n in ('32', '33')},
        'task_service_containers': {n: RAW['idle'+n]['task_containers'] for n in ('32', '33')},
        'note': 'All experiment/model/metadata service containers stopped; finite Kubernetes sleep holders retained only. Snapshot, not a perpetual lease guarantee.'},
    "nodes": {n: node_facts("%s.inventory" % n, "%s.detail" % n) for n in NODES},
    "weights_assets": {n: assets_facts(RAW["%s.assets" % n], "%s.assets" % n) for n in NODES},
    "k8s_ascend_holders_initial_snapshot": {"as_of_utc": "2026-10-16T17:37Z (file mtime)", "state": "initial snapshot / history; superseded by workload-release.json (17:57Z release); not current occupancy", **{n: {**ascend_holders(RAW["%s.k8s_pods" % n]), "evidence": ev("%s.k8s_pods" % n)} for n in NODES}},
    "reserved_devices_node32": {"pod": cfg["pod_name"], "host_ip": cfg["host_ip"], "super_pod_id": cfg["super_pod_id"], "devices": cfg["devices"], "ascend_real": ann["huawei.com/AscendReal"], "evidence": ev("node32.reserved_devices")},
    "node33_image_and_source": {"python": meta["python"], "packages": meta["packages"], "distributions": {d["name"]: d["version"] for d in src["distribution_names"]}, "git": {k: {"sha": v["sha"]["stdout"].strip(), "clean": not (v["status"]["stdout"] + v["diff"]["stdout"]).strip()} for k, v in src["git"].items()},
        "mooncake_npu_distribution": next((d["version"] for d in src["distribution_names"] if d["name"] == "mooncake-transfer-engine-npu"), None), "evidence": [ev("node33.image_metadata_v2"), ev("node33.source")]},
    "node33_npu_smoke": {"device_count": s["device_count"], "device_name": s["device_name"], "memory_free_total_bytes": s["memory_free_total_bytes"], "memory_free_GiB": [round(b / 2**30, 2) for b in s["memory_free_total_bytes"]],
        "correctness": s["correctness"], "model_loaded": s["model_loaded"], "performance_claim": s["performance_claim"], "limit": "Single visible die only; no communication path and no model performance verified.", "evidence": ev("node33.npu_smoke")},
    "node33_sdma_static": {"ascend_transport_so_present": any(Path(f).name == "ascend_transport.so" for f in sd["mooncake_files"]), "mooncake_files_count": len(sd["mooncake_files"]), "source_matches_count": len(sd["source_matches"]), "evidence": ev("node33.sdma_source"),
        "limit": "Static file/source presence only; does not prove runtime SDMA routing."},
    "resource_lifecycle": {
        "node_capacity": cap, "node_capacity_evidence": ev("node33.node_capacity"),
        "initial_snapshot_state": "History: node33 all 16 dies held by the old reservation and node32 2 dies by the testportal deployment; see k8s_ascend_holders_initial_snapshot.",
        "initial_failures": {
            "first_request": {"pod": create["manifest"]["metadata"]["name"], "requested_Ascend910": create["manifest"]["spec"]["containers"][0]["resources"]["requests"]["huawei.com/Ascend910"],
                "scheduler_event": events[0]["message"] if events else None, "ready_rc": ready["rc"], "ready_stderr": ready["stderr"].strip(), "deleted": cleanup["stdout"].strip(),
                "evidence": [ev("node33.reservation_create"), ev("node33.reservation_events"), ev("node33.reservation_ready"), ev("node33.reservation_cleanup")]},
            "authorized_create_rejected": {n: {"pod": ca_old[n]["manifest"]["metadata"]["name"], "rc": ca_old[n]["rc"], "stderr": ca_old[n]["stderr"].strip(),
                "preemptionPolicy": ca_old[n]["manifest"]["spec"].get("preemptionPolicy"), "evidence": ev("%s.reservation_create_authorized" % n)} for n in NODES}},
        "latest_resource_actions": {
            "authorization": release["authorization"],
            "release": {"started_utc": release["started_client_utc"], "completed_utc": release["completed_client_utc"], "before": release["before"], "evidence": ev("workload_release"),
                "actions": [{"argv": " ".join(a["argv"]), "rc": a["rc"], "stdout": a["stdout"].strip(), "stderr": a["stderr"].strip()} for a in release["actions"]],
                "node32_deployment_replicas_after": release["node32_deployment_replicas_after"], "status": release["status"]},
            "reservations": reservations,
            "current_state": "Both reservations Ready (ready rc=0), 16 Ascend910 dies each, activeDeadlineSeconds 7200; the node33 lease read back in the v3/timing run contexts shows 16 assigned device ids and active_deadline_seconds 7200; no model in the holder.",
            "evidence": [ev("workload_release")] + [ev("%s.reservation_create_v2" % n) for n in NODES] + [ev("%s.reservation_ready_v2" % n) for n in NODES]}},
    "communication_runs": RUNS, "communication_run_dir_notes": RUN_NOTES,
    "communication_run_status": {"basis": "raw evidence in this summary; no performance number derived", "derived_gap_causes": gaps,
        "note": "v1 lacked the PCIE pairing env and v2 used HCCL_CONNECT_TIMEOUT=30 below the required minimum; both failed with 0 messages. v3 (PCIE=1, CONNECT_TIMEOUT=120) and the timing run completed rc=0 with ok=true, 180 timed messages per rank and all recorded payload checks passing.",
        "auto_inclusion": "Any new node*-hccl-*/ directory is discovered by this script's glob and included on the next run."},
    "missing_or_conflicts": [{"id": i, "state": st, "detail": d, "evidence": [REL.get(k, k) for k in ks]} for i, st, d, ks in CONFLICTS],
    "policy_constraints": [{"id": i, "basis": b, "text": t, **({"evidence": [ev(k) for k in ks]} if ks else {})} for i, b, t, ks in POLICY],
    "sources": SOURCES}
OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("summary v3: HCCL runs=%d ok=%d sources=%d errors=%d; model functional, MemCache data path verified, lifecycle failed" % (len(RUNS), len(ok_runs), len(SOURCES), len(ERRORS)))
