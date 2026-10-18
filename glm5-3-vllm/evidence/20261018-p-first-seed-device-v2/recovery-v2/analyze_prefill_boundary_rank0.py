import csv, hashlib, json, os
from collections import Counter, defaultdict

BASE = "/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/P-boundary-profile/rank0-small"
OUT_DIR = "/Users/wio/work/Inference-Foundry-execution-plan/glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2"

profile_result_path = os.path.join(BASE, "profile-result.json")
profiler_info_path = os.path.join(BASE, "profiler_info_0.json")
export_report_path = os.path.join(BASE, "export-report.json")
kernel_csv_path = os.path.join(BASE, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv")
out_path = os.path.join(OUT_DIR, "prefill-boundary-rank0-summary-v1.json")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def merged_union_dur(intervals):
    # intervals: list of (start_us, end_us) already clipped to request window
    if not intervals:
        return 0.0
    iv = sorted(intervals)
    total = 0.0
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    total += ce - cs
    return total

def parse_shapes_field(raw):
    """CSV field for Input Shapes arrives as a quoted string; strip surrounding quotes
    and return the list of per-arg shapes (each a str, comma-joined dims)."""
    if raw is None:
        return []
    s = raw.strip()
    s = s.strip('"')
    if not s:
        return []
    return [a for a in s.split(";")]

def first_extent(shape):
    # shape like "1024,6144" -> 1024
    parts = shape.split(",")
    if parts and parts[0].strip().lstrip("-").isdigit():
        return int(parts[0].strip())
    return None

profile_result = json.load(open(profile_result_path))
profiler_info = json.load(open(profiler_info_path))
export_report = json.load(open(export_report_path))

rows = profile_result["rows"]
assert profile_result.get("completed") is True, "profile not completed"
rid_set = [r["request_id"] for r in rows]
assert len(rid_set) == len(set(rid_set)), "duplicate request ids"

# ---- 1. CSV hash verification against export-report
hash_checks = {}
for ef in export_report.get("exported_files", []):
    p = ef.get("path"); ba = ef.get("bytes"); sh = ef.get("sha256")
    if not p or not sh:
        continue
    full = os.path.join(BASE, p)
    if os.path.exists(full):
        actual = sha256_file(full)
        size = os.path.getsize(full)
        ok = (actual == sh and size == ba)
        hash_checks[p] = {
            "verified": ok, "sha256_match": actual == sh,
            "bytes_match": size == ba,
            "expected_sha256": sh, "actual_sha256": actual,
            "expected_bytes": ba, "actual_bytes": size,
        }
    else:
        hash_checks[p] = {"verified": False, "note": "file not present in rank0-small"}

if hash_checks.get("ASCEND_PROFILER_OUTPUT/kernel_details.csv", {}).get("verified") is not True:
    raise SystemExit("kernel_details.csv hash verification FAILED; abort")

# ---- 2. read kernel_details.csv
with open(kernel_csv_path, "r", newline="") as f:
    rd = csv.reader(f)
    header = next(rd)
    idx = {h: i for i, h in enumerate(header)}
    for col in ("Name", "Type", "Start Time(us)", "Duration(us)", "Input Shapes", "Input Data Types"):
        assert col in idx, col
    kern = []
    for r in rd:
        try:
            start_us = float(r[idx["Start Time(us)"]].strip())
            dur_us = float(r[idx["Duration(us)"]].strip())
        except Exception:
            continue  # malformed/empty tail
        kern.append({
            "name": r[idx["Name"]],
            "type": r[idx["Type"]],
            "start_us": start_us,
            "dur_us": dur_us,
            "in_shape_raw": r[idx["Input Shapes"]] if idx["Input Shapes"] < len(r) else "",
            "in_dtypes_raw": r[idx["Input Data Types"]] if idx["Input Data Types"] < len(r) else "",
        })

# ---- 3. profiler offset check vs each request offset
end_info = profiler_info.get("end_info", {})
prof_wall_end_ns = int(end_info.get("collectionTimeEnd"))
prof_mono_end_ns = int(end_info.get("MonotonicTimeEnd"))
prof_offset_ns = prof_wall_end_ns - prof_mono_end_ns

offset_checks = []
for r in rows:
    req_offset_ns = int(r["sent_wall_ns"]) - int(r["sent_monotonic_ns"])
    diff = prof_offset_ns - req_offset_ns
    within_tol = abs(diff) <= 1_000_000  # 1 ms
    offset_checks.append({
        "case_id": r["case_id"], "N": r["N"], "phase": r["phase"],
        "request_id": r["request_id"],
        "req_offset_ns_sent_wall_minus_sent_mono": req_offset_ns,
        "prof_offset_ns_collectTimeEnd_minus_MonoTimeEnd": prof_offset_ns,
        "offset_difference_ns": diff,
        "within_1ms_tol": within_tol,
    })

# ---- 4. assign each kernel row to a request window by host wall start time
win_lo = {r["request_id"]: int(r["sent_wall_ns"]) / 1000.0 for r in rows}
win_hi = {r["request_id"]: int(r["received_wall_ns"]) / 1000.0 for r in rows}

kern_by_req = defaultdict(list)
unassigned = []
for k in kern:
    st = k["start_us"]
    matched = None
    for r in rows:
        if win_lo[r["request_id"]] <= st <= win_hi[r["request_id"]]:
            matched = r["request_id"]
            break
    if matched is not None:
        kern_by_req[matched].append(k)
    else:
        unassigned.append(k)

# ---- per-request report
def input_shape_dist(ks):
    """For the given rows, aggregate Input Shapes values -> count and duration median,
    plus parsed first-extent distribution and dtype distribution."""
    shape_counter = Counter()
    shape_durs = defaultdict(list)
    first_extent_counter = Counter()
    dtype_counter = Counter()
    dtype_durs = defaultdict(list)
    for k in ks:
        shapes = parse_shapes_field(k["in_shape_raw"])
        dtypes = [a for a in (k["in_dtypes_raw"].strip().strip('"').split(";")) if a]
        key = "|".join(shapes) if shapes else "<none>"
        shape_counter[key] += 1
        shape_durs[key].append(k["dur_us"])
        dt_key = "|".join(dtypes) if dtypes else "<none>"
        dtype_counter[dt_key] += 1
        dtype_durs[dt_key].append(k["dur_us"])
        if shapes:
            fe = first_extent(shapes[0])
            if fe is not None:
                first_extent_counter[fe] += 1
    return {
        "shape_values": [
            {"shape": s, "count": c, "duration_us_median": _med(shape_durs[s])}
            for s, c in shape_counter.most_common()
        ],
        "first_input_extent_counts": {str(k): v for k, v in first_extent_counter.items()},
        "dtype_values": [
            {"dtypes": d, "count": c, "duration_us_median": _med(dtype_durs[d])}
            for d, c in dtype_counter.most_common()
        ],
        "row_count": len(ks),
        "duration_us_median_overall": _med([k["dur_us"] for k in ks]) if ks else None,
    }

def _med(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

KEY_TYPES = ["MlaPrologV3", "GroupedMatmulSwigluQuant", "GroupedMatmul",
             "MoeInitRoutingV3", "QuantBatchMatmulV3"]

per_request = []
for i, r in enumerate(rows):
    rid = r["request_id"]
    ks = kern_by_req.get(rid, [])
    lo = win_lo[rid]; hi = win_hi[rid]
    api_wall_s = (int(r["received_wall_ns"]) - int(r["sent_wall_ns"])) / 1e9
    starts = [k["start_us"] for k in ks]
    clipped = [(max(lo, k["start_us"]), min(hi, k["start_us"] + k["dur_us"])) for k in ks]
    union_dur = merged_union_dur(clipped)
    sum_dur = sum(k["dur_us"] for k in ks)
    by_type = defaultdict(lambda: {"count": 0, "dur_us_sum": 0.0})
    for k in ks:
        by_type[k["type"]]["count"] += 1
        by_type[k["type"]]["dur_us_sum"] += k["dur_us"]
    top_types = [{"type": t, "count": d["count"], "dur_us_sum": d["dur_us_sum"]}
                 for t, d in sorted(by_type.items(), key=lambda kv: (-kv[1]["count"], kv[0]))][:12]
    key_dist = {}
    for t in KEY_TYPES:
        tk = [k for k in ks if k["type"] == t]
        if tk:
            key_dist[t] = input_shape_dist(tk)
        else:
            key_dist[t] = {"row_count": 0}
    per_request.append({
        "request_index": i,
        "case_id": r["case_id"], "N": r["N"], "phase": r["phase"],
        "request_id": rid,
        "sent_wall_ns": r["sent_wall_ns"], "received_wall_ns": r["received_wall_ns"],
        "sent_monotonic_ns": r["sent_monotonic_ns"], "received_monotonic_ns": r["received_monotonic_ns"],
        "complete_answer_pass": r.get("complete_answer_pass"),
        "api_wall_duration_s": api_wall_s,
        "csv_row_count": len(ks),
        "min_event_start_us": min(starts) if starts else None,
        "max_event_end_us": max([s + k["dur_us"] for s, k in zip(starts, ks)]) if ks else None,
        "clipped_interval_union_dur_us": union_dur,
        "sum_dur_us_already_shelf": sum_dur,
        "top_kernel_types": top_types,
        "key_type_shape_dtype_distributions": key_dist,
    })

# ---- main Mla input extent (1024 vs 256) and real-257 MTP entries across ALL rows
mla_probe = {"rows_total": 0}
fe_counter = Counter()
mtp_shapes = Counter()  # shapes containing a 257 element among MlaPrologV3 rows
for k in kern:
    if k["type"] != "MlaPrologV3":
        continue
    mla_probe["rows_total"] += 1
    shapes = parse_shapes_field(k["in_shape_raw"])
    if shapes:
        fe = first_extent(shapes[0])
        if fe is not None:
            fe_counter[fe] += 1
        for sh in shapes:
            dims = [d for d in sh.split(",") if d.strip().lstrip("-").isdigit()]
            if any(int(d) == 257 for d in dims):
                mtp_shapes[sh] += 1
mla_probe["first_input_extent_counts"] = {str(k): v for k, v in fe_counter.items()}
mla_probe["shapes_containing_257"] = [{"shape": s, "count": c} for s, c in mtp_shapes.most_common()]

summary = {
    "schema": "prefill-boundary-rank0-summary-v1",
    "generated_by": os.path.basename(__file__),
    "rank_id": profiler_info.get("rank_id"),
    "source_identities": {
        "profile_result": {"path": profile_result_path,
                           "sha256": sha256_file(profile_result_path)},
        "profiler_info_0": {"path": profiler_info_path,
                            "sha256": sha256_file(profiler_info_path)},
        "export_report": {"path": export_report_path,
                          "sha256": sha256_file(export_report_path)},
    },
    "profiler_clock": {
        "collectionTimeEnd": prof_wall_end_ns,
        "MonotonicTimeEnd": prof_mono_end_ns,
        "offset_collectionTimeEnd_minus_MonotonicTimeEnd_ns": prof_offset_ns,
        "start_info": profiler_info.get("start_info"),
        "torch_npu_version": profiler_info.get("torch_npu_version"),
        "cann_version": profiler_info.get("cann_version"),
    },
    "csv_hash_verification": hash_checks,
    "offset_check_per_request": offset_checks,
    "request_windows_assign_rule": "assigned by CSV Start Time(us) (host wall epoch us) within "
                                    "sent_wall_ns/1000 .. received_wall_ns/1000; monotonic not used for assignment",
    "kernel_rows_total": len(kern),
    "kernel_rows_assigned_to_request": sum(len(v) for v in kern_by_req.values()),
    "kernel_rows_unassigned": len(unassigned),
    "delta_row_type_counts": {"note": "CSV row counts are profiler event tallies within each request "
                            "wall window, not full executed layer/kernel totals; varying counts do not "
                            "by themselves imply model layer skipping or loss."},
    "mla_extent_and_mtp_observed": mla_probe,
    "per_request": per_request,
    "union_semantics_note": "clipped_interval_union_dur_us is the merged coverage of these profiler CSV "
                            "intervals clipped to the request wall window; summed durations may overlap. "
                            "Union is not hardware utilization and not a full wall-time decomposition.",
    "limitations": [
        "Only rank0 NPU profiler kernel_details.csv events are analyzed, not the 482MB trace_view.json.",
        "CSV row counts are profiler-event tallies within request windows, not authoritative executed "
        "layer/kernel totals; differing counts across requests do not by themselves indicate model skipping or loss.",
        "clipped_interval_union_dur_us only measures coverage of the profiled CSV intervals; summed "
        "durations may overlap, so union is not hardware utilization or a complete wall-time decomposition.",
        "Shapes/dtypes are reported as observed in the profiler; no causal or correctness verdict is made.",
        "Time bases: CSV Start Time(us) is host wall epoch microseconds; request/timeline wall offsets "
        "were checked for consistency (see offset_check_per_request) and reported as-is.",
        "No benchmark performance gains or model correctness claims are made in this report.",
    ],
}

with open(out_path, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=False)
print("WROTE", out_path)
print("kernel rows:", len(kern), "assigned:", sum(len(v) for v in kern_by_req.values()),
      "unassigned:", len(unassigned))
print("JSON valid, request entries:", len(per_request))