#!/usr/bin/env python3
"""One-pass indexed Gen66 vLLM rank-parameterized replay fence and marker audit."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


FLOW_CAT = "async_npu"
FLOW_NAME = "torch_to_npu"
ANNOTATION_SUFFIXES = (".start", ".end")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dec(value):
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def ds(value):
    return None if value is None else format(value, "f")


def event_end(event):
    start = dec(event.get("ts"))
    duration = dec(event.get("dur", 0))
    return None if start is None or duration is None else start + duration


def interval_contains(outer, inner):
    lo, hi = dec(outer.get("ts")), event_end(outer)
    a, b = dec(inner.get("ts")), event_end(inner)
    return None not in (lo, hi, a, b) and lo <= a and b <= hi


def interval_overlaps(a, b):
    a0, a1 = dec(a.get("ts")), event_end(a)
    b0, b1 = dec(b.get("ts")), event_end(b)
    return None not in (a0, a1, b0, b1) and a0 < b1 and b0 < a1


class FenceIndex:
    """Index only valid non-overlapping fence interiors; retain all overlaps."""
    def __init__(self, fences):
        self.fences = sorted(fences, key=lambda x: (x["start"], x["end"], x["invocation"]))
        self.starts = [f["start"] for f in self.fences]
        self.prefix_max_end = []
        max_end = None
        self.valid = True
        self.overlaps = []
        for fence in self.fences:
            max_end = fence["end"] if max_end is None else max(max_end, fence["end"])
            self.prefix_max_end.append(max_end)
        for previous, current in zip(self.fences, self.fences[1:]):
            if previous["end"] > current["start"]:
                self.valid = False
                self.overlaps.append({"previous": previous["invocation"], "current": current["invocation"],
                                      "previous_end": ds(previous["end"]), "current_start": ds(current["start"])})

    def _overlapping(self, start, end):
        i = bisect_right(self.starts, end) - 1
        out = []
        while i >= 0 and self.prefix_max_end[i] > start:
            fence = self.fences[i]
            if start < fence["end"] and fence["start"] < end:
                out.append(fence)
            i -= 1
        out.reverse()
        return out

    def owners(self, event):
        if not self.valid:
            return []
        start, end = dec(event.get("ts")), event_end(event)
        if start is None or end is None:
            return []
        return [f for f in self._overlapping(start, end) if f["start"] <= start and end <= f["end"]]

    def crossing(self, event):
        if not self.valid:
            return []
        start, end = dec(event.get("ts")), event_end(event)
        if start is None or end is None:
            return []
        return [f for f in self._overlapping(start, end) if not (f["start"] <= start and end <= f["end"])]


def _native_key(event):
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    model, task, stream = args.get("Model Id"), args.get("Task Id"), event.get("tid")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (model, task)):
        return None
    try:
        return int(model), int(stream), int(task)
    except (ValueError, TypeError):
        return None


def _event_signature(event):
    return str(event.get("pid")), str(event.get("tid")), dec(event.get("ts"))


def _process_metadata(event):
    if event.get("ph") != "M" or event.get("name") not in {"process_name", "process_labels"}:
        return None
    args = event.get("args") if isinstance(event.get("args"), dict) else {}
    name = event.get("name")
    return str(event.get("pid")), name, str(args.get("name" if name == "process_name" else "labels", ""))


def load_indexes(trace_gz, temp_dir):
    """Parse exactly once; retain only rows needed by explicit identity joins."""
    raw_digest = hashlib.sha256()
    raw_bytes = 0
    with tempfile.NamedTemporaryFile(prefix="gen50-trace-", suffix=".json", dir=temp_dir) as raw:
        with gzip.open(trace_gz, "rb") as gz:
            while True:
                chunk = gz.read(8 << 20)
                if not chunk:
                    break
                raw_digest.update(chunk)
                raw_bytes += len(chunk)
                raw.write(chunk)
        raw.flush()
        raw.seek(0)
        doc = json.load(raw, parse_float=Decimal)
    if isinstance(doc, list):
        events = doc
    elif isinstance(doc, dict):
        events = next((doc[k] for k in ("traceEvents", "trace_events", "events", "Events", "activities") if isinstance(doc.get(k), list)), None)
    else:
        events = None
    if not isinstance(events, list):
        raise ValueError("trace root has no event array")
    metadata = defaultdict(dict)
    scopes = defaultdict(list)
    cpu_records = defaultdict(list)
    flow_pairs = defaultdict(lambda: {"s": [], "f": []})
    native_at_endpoint = defaultdict(list)
    native_tasks = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        meta = _process_metadata(event)
        if meta:
            pid, key, value = meta
            metadata[pid][key] = value
        ph, name = event.get("ph"), str(event.get("name", ""))
        pid, tid = str(event.get("pid")), str(event.get("tid"))
        if ph == "X" and (name.endswith(".start") or name.endswith(".end")):
            scopes[(pid, name)].append((index, event))
        if ph == "X" and name == "record_event":
            cpu_records[(pid, tid)].append((index, event))
        if ph in {"s", "f"} and event.get("cat") == FLOW_CAT and name == FLOW_NAME and event.get("id") is not None:
            flow_pairs[(str(event.get("cat")), name, str(event["id"]))][ph].append((index, event))
        args = event.get("args")
        if ph == "X" and isinstance(args, dict) and "Task Type" in args:
            native_tasks.append((index, event))
            native_at_endpoint[_event_signature(event)].append((index, event))
    return events, metadata, scopes, cpu_records, flow_pairs, native_at_endpoint, native_tasks, raw_digest.hexdigest(), raw_bytes


def _native_for_outer(row, events_by_end, hardware_pid):
    matches = []
    # Callback identity is exact: hardware PID is supplied by process metadata,
    # while trace PID/TID/ts must exactly match the unique flow finish.
    # The caller supplies the finish event to this function via `_finish`.
    finish = row["_finish"]
    for index, event in events_by_end.get(_event_signature(finish), []):
        args = event.get("args", {})
        if str(event.get("pid")) != str(hardware_pid):
            continue
        model_id, task_id = args.get("Model Id"), args.get("Task Id")
        task_type = args.get("Task Type")
        if isinstance(model_id, bool) or not isinstance(model_id, int):
            continue
        if task_id != row.get("task_id") or args.get("Physic Stream Id") != row.get("acl_stream_id"):
            continue
        if str(event.get("tid")) != str(row.get("acl_stream_id")):
            continue
        if not isinstance(task_type, str) or not task_type.strip():
            continue
        matches.append((index, event))
    return matches


def _resolve_outer(row, scopes, cpu_records, flow_pairs, native_at_endpoint, hardware_pid, device_label):
    problems = []
    invocation, side, pid = row.get("invocation"), row.get("side"), str(row.get("pid"))
    scope_name = f"{invocation}.{side}"
    scope_rows = scopes.get((pid, scope_name), [])
    if len(scope_rows) != 1:
        return None, [f"record_scope_count_{len(scope_rows)}"]
    scope_index, scope = scope_rows[0]
    records = [(i, e) for i, e in cpu_records.get((pid, str(scope.get("tid"))), []) if interval_contains(scope, e)]
    if len(records) != 1:
        return None, [f"record_event_count_in_scope_{len(records)}"]
    record_index, record = records[0]
    starts = []
    for (cat, name, flow_id), endpoints in flow_pairs.items():
        if cat != FLOW_CAT or name != FLOW_NAME:
            continue
        for i, event in endpoints["s"]:
            if str(event.get("pid")) == pid and str(event.get("tid")) == str(scope.get("tid")) and interval_contains(record, event):
                starts.append((flow_id, i, event, endpoints))
    if len(starts) != 1:
        return None, [f"async_npu_start_count_in_record_{len(starts)}"]
    flow_id, start_index, start, endpoints = starts[0]
    if len(endpoints["s"]) != 1 or len(endpoints["f"]) != 1 or endpoints["s"][0][0] != start_index:
        return None, [f"async_npu_flow_endpoint_count_s{len(endpoints['s'])}_f{len(endpoints['f'])}"]
    finish_index, finish = endpoints["f"][0]
    tmp = dict(row)
    tmp["_finish"] = finish
    native = _native_for_outer(tmp, native_at_endpoint, hardware_pid)
    if len(native) != 1:
        return None, [f"native_endpoint_task_count_{len(native)}"]
    native_index, native_event = native[0]
    args = native_event.get("args", {})
    checks = {
        "task_id": args.get("Task Id") == row.get("task_id"),
        "physical_stream": args.get("Physic Stream Id") == row.get("acl_stream_id"),
        "native_tid_stream": str(native_event.get("tid")) == str(row.get("acl_stream_id")),
        "model_id_present": isinstance(args.get("Model Id"), int) and not isinstance(args.get("Model Id"), bool),
        "task_type_present": isinstance(args.get("Task Type"), str) and bool(args.get("Task Type").strip()),
    }
    if not all(checks.values()):
        problems.extend(f"callback_native_mismatch_{k}" for k, ok in checks.items() if not ok)
    if row.get("event") != "vllm_replay_outer_event" or row.get("capture_status") != 0 or row.get("model_ri_nonnull") is not False:
        problems.append("callback_capture_status_or_classification_invalid")
    if any(row.get(k) != 0 for k in ("stream_rc", "task_rc", "capture_rc", "callback_rc")):
        problems.append("callback_return_code_nonzero")
    if problems:
        return None, problems
    result = {k: row.get(k) for k in ("rank_id", "pid", "graph_python_id", "graph_key", "invocation", "side", "replay_serial", "method", "work_key", "request_id", "acl_stream_id", "task_id", "capture_status", "callback_rc")}
    result.update({"cpu_scope_event_index": scope_index, "cpu_scope_name": scope_name,
                   "cpu_record_event_index": record_index, "cpu_record_interval_us": [ds(dec(record["ts"])), ds(event_end(record))],
                   "async_flow_id": flow_id, "async_start_event_index": start_index,
                   "async_finish_event_index": finish_index, "async_finish_pid": finish.get("pid"),
                   "async_finish_tid": finish.get("tid"), "async_finish_ts_us": ds(dec(finish.get("ts"))),
                   "native_event_index": native_index, "native_pid": native_event.get("pid"),
                   "native_tid": native_event.get("tid"), "native_model_id": args.get("Model Id"),
                   "native_stream_id": args.get("Physic Stream Id"), "native_task_id": args.get("Task Id"),
                   "native_task_type": args.get("Task Type"), "native_ts_us": ds(dec(native_event.get("ts"))),
                   "native_dur_us": ds(dec(native_event.get("dur", 0))), "native_device": device_label})
    return {"row": result, "native_event": native_event}, []


def select_rank_evidence(rank, identity, outer_rows, expected_fences=31):
    """Fail closed on absent rank or rows with mismatched explicit rank labels."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a nonnegative integer")
    rank_entries = [x for x in identity.get("ranks", []) if x.get("rank_id") == rank]
    if len(rank_entries) != 1:
        raise ValueError(f"expected exactly one identity entry for rank {rank}, found {len(rank_entries)}")
    selected_outer = [x for x in outer_rows if x.get("rank_id") == rank]
    expected_rows = 2 * expected_fences
    if len(selected_outer) != expected_rows:
        raise ValueError(f"rank {rank} expected {expected_rows} explicit outer rows, found {len(selected_outer)}")
    if any(x.get("rank_id") != rank for x in selected_outer):
        raise ValueError("outer rank filter mismatch")
    return rank_entries[0], selected_outer


def validate_rank_binding(rank_entry, outer_rows, expected_fences=31):
    """Bind explicit replay rows to this capture's single PID and graph object."""
    markers = rank_entry.get("exact_event_record_joins", [])
    pids = {str(x.get("pid")) for x in markers}
    graph_ids = {x.get("graph_python_id") for x in markers}
    if len(pids) != 1 or len(graph_ids) != 1 or None in graph_ids:
        raise ValueError("rank marker rows do not identify one PID and graph object")
    if {str(x.get("pid")) for x in outer_rows} != pids:
        raise ValueError("selected outer rows PID does not match selected rank marker PID")
    if {x.get("graph_python_id") for x in outer_rows} != graph_ids:
        raise ValueError("selected outer rows graph object does not match selected rank markers")
    pairs = {(x.get("invocation"), x.get("side")) for x in outer_rows}
    invocations = {x[0] for x in pairs}
    expected_rows = 2 * expected_fences
    if len(invocations) != expected_fences or len(pairs) != expected_rows or any(side not in {"start", "end"} for _, side in pairs):
        raise ValueError(f"selected rank does not contain {expected_fences} unique explicit start/end invocations")
    if any(sum(1 for row in outer_rows if row.get("invocation") == inv and row.get("side") == side) != 1
           for inv in invocations for side in ("start", "end")):
        raise ValueError("selected rank has missing or duplicate invocation side")


def outer_pair_type_approved(start_row, end_row, accepted_types):
    """Only explicit source-reviewed CANN task-type names can define fences."""
    accepted = set(accepted_types or ())
    if not accepted:
        return False
    return start_row.get("native_task_type") in accepted and end_row.get("native_task_type") in accepted


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def select_rank_state_input(state, rank, expected_pid):
    """Select the exact rankN SDK input and require one rank-specific source."""
    kind = f"rank{rank}"
    candidates = [x for x in state.get("inputs", []) if x.get("kind") == kind]
    if len(candidates) != 1:
        raise ValueError(f"SDK state must contain exactly one {kind} input, found {len(candidates)}")
    entry = candidates[0]
    source = entry.get("source")
    copy = entry.get("copy")
    if not isinstance(source, str) or not isinstance(copy, str):
        raise ValueError(f"{kind} input lacks source/copy profile paths")
    pid_pattern = re.compile(r"^bms-fae_(\d+)_\d+_ascend_pt$")
    pids = {m.group(1) for m in (pid_pattern.match(Path(source).name), pid_pattern.match(Path(copy).name)) if m}
    if pids != {str(expected_pid)}:
        raise ValueError(f"{kind} source/copy profile directory PID does not match marker PID {expected_pid}")
    if Path(entry.get("manifest_path", "")).name != f"{kind}-raw-copy.json":
        raise ValueError(f"{kind} input does not point to its exact raw-copy manifest")
    if not entry.get("manifest_sha256"):
        raise ValueError(f"{kind} input lacks raw-copy manifest SHA")
    result = entry.get("result")
    if not isinstance(result, dict):
        raise ValueError(f"{kind} input has no result object")
    trace_path = result.get("trace_path")
    if not isinstance(trace_path, str) or not trace_path.startswith(copy.rstrip("/") + "/"):
        raise ValueError(f"{kind} trace path is outside its selected copied profile directory")
    return entry, result


def validate_rank_manifest_binding(rank, expected_pid, state_entry, manifest, manifest_sha, manifest_bytes, copy_receipt):
    """Prove state source/copy, profiler_info_N entry, manifest digest, and download receipt agree."""
    kind = f"rank{rank}"
    source, copy_path = state_entry["source"], state_entry["copy"]
    if manifest.get("kind") != kind:
        raise ValueError(f"raw-copy manifest kind does not equal {kind}")
    if manifest.get("original_path") != source or manifest.get("copy_path") != copy_path:
        raise ValueError(f"raw-copy manifest source/copy paths mismatch for {kind}")
    if state_entry.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"SDK state raw-copy manifest SHA mismatch for {kind}")
    profile_file = f"profiler_info_{rank}.json"
    matches = [x for x in manifest.get("files", []) if x.get("relative_path") == profile_file]
    if len(matches) != 1 or not matches[0].get("sha256"):
        raise ValueError(f"raw-copy manifest must contain one hashed {profile_file}")
    receipt_key = f"{kind}-raw-copy.json"
    receipt = copy_receipt.get(receipt_key)
    if (not isinstance(receipt, dict) or receipt.get("sha256") != manifest_sha
            or receipt.get("bytes") != manifest_bytes):
        raise ValueError(f"download receipt does not match {receipt_key} SHA")
    return {"manifest_sha256": manifest_sha, "manifest_receipt_key": receipt_key,
            "profiler_info_file": profile_file, "profiler_info_sha256": matches[0]["sha256"],
            "profile_pid": str(expected_pid)}


def validate_startup_device_binding(rank, marker_rows, device_map):
    """Require an explicit startup rank/card-group/device crosswalk; never derive rank+offset."""
    startup_path = device_map.get("startup_config_path")
    expected_startup_sha = device_map.get("startup_config_sha256")
    if not isinstance(startup_path, str) or not expected_startup_sha or not Path(startup_path).is_file():
        raise ValueError("device map must name an available startup config and SHA256")
    actual_startup_sha = sha(startup_path)
    if actual_startup_sha != expected_startup_sha:
        raise ValueError("startup config SHA256 does not match device map provenance")
    assignments = device_map.get("rank_assignments")
    if not isinstance(assignments, list) or len(assignments) != 4:
        raise ValueError("startup mapping must explicitly declare four rank assignments")
    by_rank = {}
    for row in assignments:
        rank_id, group, device = row.get("rank_id"), row.get("card_group_id"), row.get("device_id")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (rank_id, group, device)):
            raise ValueError("startup mapping rank/card_group/device fields must be exact integers")
        if rank_id in by_rank:
            raise ValueError("startup mapping has duplicate rank assignment")
        by_rank[rank_id] = row
    if set(by_rank) != {0, 1, 2, 3}:
        raise ValueError("startup mapping ranks must be exactly 0..3")
    groups = [row["card_group_id"] for row in by_rank.values()]
    if set(groups) != {0, 1, 2, 3} or len(set(groups)) != 4:
        raise ValueError("startup mapping card groups must be an explicit permutation of 0..3")
    devices = [row["device_id"] for row in by_rank.values()]
    if len(set(devices)) != 4 or any(x < 0 for x in devices):
        raise ValueError("startup mapping device IDs must be four distinct nonnegative IDs")
    capture_devices = {int(row["capture_stream_device_id"]) for row in marker_rows}
    if len(capture_devices) != 1:
        raise ValueError("selected capture markers do not identify one capture_stream_info device_id")
    selected = by_rank[rank]
    capture_device = next(iter(capture_devices))
    if selected["device_id"] != capture_device:
        raise ValueError("startup rank/card-group device does not match capture_stream_info device_id")
    return {"startup_config_path": startup_path, "startup_config_sha256": actual_startup_sha,
            "rank_id": rank, "card_group_id": selected["card_group_id"],
            "device_id": selected["device_id"], "capture_stream_device_id": capture_device,
            "trace_device_label": f"NPU {capture_device}"}


def audit(trace_gz, trace_receipt, state_path, outer_path, identity_path, output_path, rank,
          receipt_key=None, raw_copy_manifest_path=None, copy_receipt_path=None, compact_output_path=None,
          startup_device_map_path=None, expected_fences=31, accepted_outer_task_types=()):
    trace_gz, trace_receipt = Path(trace_gz), Path(trace_receipt)
    digest = sha(trace_gz)
    receipt_doc = json.loads(trace_receipt.read_text())
    key = receipt_key or trace_gz.name
    if receipt_key is None and "sha256" in receipt_doc and "bytes" in receipt_doc:
        receipt_doc = {trace_gz.name: receipt_doc}
    if key not in receipt_doc and receipt_key is None and "serving-trace.json.gz" in receipt_doc:
        key = "serving-trace.json.gz"
    if key not in receipt_doc and receipt_key is None:
        candidates = [k for k, v in receipt_doc.items()
                      if isinstance(v, dict) and (k.endswith(".gz") or "trace" in k.lower())
                      and "sha256" in v and "bytes" in v]
        if len(candidates) == 1:
            key = candidates[0]
    if key not in receipt_doc or not isinstance(receipt_doc[key], dict):
        raise ValueError(f"download receipt does not identify one compressed trace entry ({key})")
    receipt = receipt_doc[key]
    expected_sha = receipt.get("sha256") or receipt.get("serving_trace_sha256")
    expected_gzip_bytes = receipt.get("bytes")
    if expected_gzip_bytes is None or not expected_sha:
        raise ValueError("download receipt lacks exact compressed trace SHA or byte count")
    if trace_gz.stat().st_size != expected_gzip_bytes:
        raise ValueError(f"compressed trace byte count mismatch: {trace_gz.stat().st_size} != {expected_gzip_bytes}")
    if expected_sha and digest != expected_sha:
        raise ValueError(f"compressed trace SHA mismatch: {digest} != {expected_sha}")
    if receipt.get("kind") not in (None, f"rank{rank}"):
        raise ValueError(f"download receipt kind does not match rank{rank}")
    state = json.loads(Path(state_path).read_text())
    outer_rows = json.loads(Path(outer_path).read_text())
    identity = json.loads(Path(identity_path).read_text())
    rank_entry, rank_outer = select_rank_evidence(rank, identity, outer_rows, expected_fences)
    marker_rows = rank_entry["exact_event_record_joins"]
    if len(marker_rows) != 128:
        raise ValueError(f"rank {rank} identity audit does not contain 128 exact marker keys")
    validate_rank_binding(rank_entry, rank_outer, expected_fences)
    if any(row.get("rank_id") != rank for row in marker_rows):
        raise ValueError(f"rank {rank} identity rows contain a mismatched rank label")
    marker_pids = {str(row.get("pid")) for row in marker_rows}
    if len(marker_pids) != 1:
        raise ValueError(f"rank {rank} marker rows do not identify one PID")
    selected_state_input, input_result = select_rank_state_input(state, rank, next(iter(marker_pids)))
    if raw_copy_manifest_path is None or copy_receipt_path is None:
        raise ValueError("rank audit requires downloaded raw-copy manifest and its download receipt")
    raw_copy_manifest_path = Path(raw_copy_manifest_path)
    copy_receipt_path = Path(copy_receipt_path)
    copy_manifest = json.loads(raw_copy_manifest_path.read_text())
    copy_receipt_doc = json.loads(copy_receipt_path.read_text())
    state_receipt = copy_receipt_doc.get("state.json")
    if (not isinstance(state_receipt, dict)
            or state_receipt.get("sha256") != sha(state_path)
            or state_receipt.get("bytes") != Path(state_path).stat().st_size):
        raise ValueError("download receipt does not match SDK state.json bytes/SHA")
    source_binding = validate_rank_manifest_binding(
        rank, next(iter(marker_pids)), selected_state_input, copy_manifest,
        sha(raw_copy_manifest_path), raw_copy_manifest_path.stat().st_size, copy_receipt_doc)
    expected_trace = input_result.get("trace_sha256") or state.get("trace_sha256") or state.get("output_trace_sha256")
    if not expected_trace or input_result.get("trace_bytes") is None or input_result.get("event_count") is None:
        raise ValueError(f"SDK state rank{rank} input lacks exact uncompressed trace SHA, byte count, or event count")
    expected_markers = {}
    expected_devices = {int(x["capture_stream_device_id"]) for x in marker_rows}
    if len(expected_devices) != 1:
        raise ValueError(f"rank {rank} marker capture rows disagree on device IDs: {sorted(expected_devices)}")
    device_id = next(iter(expected_devices))
    device_label = f"NPU {device_id}"
    if startup_device_map_path is None:
        raise ValueError("an explicit startup rank/card-group/device map is required")
    startup_device_map_path = Path(startup_device_map_path)
    startup_device_map = json.loads(startup_device_map_path.read_text())
    startup_binding = validate_startup_device_binding(rank, marker_rows, startup_device_map)
    if startup_binding["trace_device_label"] != device_label:
        raise ValueError("startup/capture device assignment does not match trace label")
    for row in marker_rows:
        key = (int(row["model_id"]), int(row["model_stream_id"]), int(row["task_id"]))
        if key in expected_markers:
            raise ValueError(f"duplicate raw marker key {key}")
        expected_markers[key] = row

    events, metadata, scopes, cpu_records, flows, native_at_endpoint, all_native, raw_digest, raw_bytes = load_indexes(trace_gz, Path(output_path).parent)
    if expected_trace and raw_digest != expected_trace:
        raise ValueError(f"uncompressed trace SHA mismatch: {raw_digest} != {expected_trace}")
    if input_result.get("trace_bytes") is not None and raw_bytes != input_result["trace_bytes"]:
        raise ValueError(f"uncompressed trace byte count mismatch: {raw_bytes} != {input_result['trace_bytes']}")
    if input_result.get("event_count") is not None and len(events) != input_result["event_count"]:
        raise ValueError(f"trace event count mismatch: {len(events)} != {input_result['event_count']}")
    hardware = [pid for pid, m in metadata.items() if m.get("process_name") == "Ascend Hardware" and m.get("process_labels") == device_label]
    if len(hardware) != 1:
        raise ValueError(f"expected one Ascend Hardware/{device_label} process, found {hardware}")
    hardware_pid = hardware[0]
    failures = []
    resolved = []
    for row in rank_outer:
        result, errors = _resolve_outer(row, scopes, cpu_records, flows, native_at_endpoint, hardware_pid, device_label)
        if errors:
            failures.append({"rank_id": rank, "invocation": row.get("invocation"), "side": row.get("side"), "reason": errors})
        else:
            resolved.append(result)

    paired = defaultdict(dict)
    for item in resolved:
        row = item["row"]
        key = (row["rank_id"], row["graph_python_id"], row["invocation"])
        if row["side"] in paired[key]:
            failures.append({"invocation": row["invocation"], "reason": "duplicate_side"})
        paired[key][row["side"]] = item
    fences = []
    candidate_fences = []
    unapproved_outer_pairs = []
    for key, pair in paired.items():
        if set(pair) != {"start", "end"}:
            failures.append({"invocation": key[2], "reason": "missing_start_or_end", "sides": sorted(pair)})
            continue
        start, end = pair["start"]["native_event"], pair["end"]["native_event"]
        srow, erow = pair["start"]["row"], pair["end"]["row"]
        same = (str(start.get("pid")) == str(end.get("pid")) == str(hardware_pid)
                and str(start.get("tid")) == str(end.get("tid"))
                and srow.get("acl_stream_id") == erow.get("acl_stream_id")
                and srow.get("graph_python_id") == erow.get("graph_python_id"))
        lo, hi = event_end(start), dec(end.get("ts"))
        if not same or lo is None or hi is None or hi <= lo:
            failures.append({"invocation": key[2], "reason": "invalid_fence_identity_or_duration",
                             "start_interval": [ds(dec(start.get("ts"))), ds(event_end(start))],
                             "end_interval": [ds(dec(end.get("ts"))), ds(event_end(end))]})
            continue
        fence = {"rank_id": key[0], "graph_python_id": key[1], "invocation": key[2],
                       "work_key": srow.get("work_key"), "request_id": srow.get("request_id"),
                       "method": srow.get("method"), "native_pid": hardware_pid,
                       "device": device_label, "physical_stream_id": srow.get("acl_stream_id"),
                       "start_event_index": srow["native_event_index"], "end_event_index": erow["native_event_index"],
                       "start_task_id": srow["native_task_id"], "end_task_id": erow["native_task_id"],
                       "start_task_type": srow.get("native_task_type"), "end_task_type": erow.get("native_task_type"),
                       "start_event_ts_us": ds(dec(start.get("ts"))), "start_event_dur_us": ds(dec(start.get("dur", 0))),
                       "end_event_ts_us": ds(dec(end.get("ts"))), "end_event_dur_us": ds(dec(end.get("dur", 0))),
                       "interior_start_us": ds(lo), "interior_end_us": ds(hi), "start": lo, "end": hi}
        candidate_fences.append(fence)
        if outer_pair_type_approved(srow, erow, accepted_outer_task_types):
            fences.append(fence)
        else:
            unapproved_outer_pairs.append({"invocation": key[2], "start_task_type": srow.get("native_task_type"),
                                           "end_task_type": erow.get("native_task_type"),
                                           "reason": "native Task Type is observed but not in explicit accepted_outer_task_types"})
    fences.sort(key=lambda x: (x["start"], x["invocation"]))
    candidate_fence_index = FenceIndex(candidate_fences)
    fence_index = FenceIndex(fences)
    if not fence_index.valid:
        failures.append({"reason": "overlapping_same_device_stream_fences", "overlaps": fence_index.overlaps})

    graph_events = []
    unknown_native = []
    marker_key_events = defaultdict(list)
    hw_tasks = [(i, e) for i, e in all_native if str(e.get("pid")) == hardware_pid and str(e.get("tid")) != "0"]
    for i, event in hw_tasks:
        key = _native_key(event)
        args = event.get("args", {})
        if key in expected_markers:
            marker_key_events[key].append((i, event))
        owners = fence_index.owners(event)
        crosses = fence_index.crossing(event)
        row = {"event_index": i, "native_pid": event.get("pid"), "device": device_label, "tid": event.get("tid"),
               "model_id": args.get("Model Id"), "stream_id": args.get("Physic Stream Id"), "task_id": args.get("Task Id"),
               "subtask_id": args.get("Subtask Id"), "task_type": args.get("Task Type"), "name": event.get("name"),
               "connection_id": args.get("Connection Id"), "batch_id": args.get("Batch Id"),
               "ts_us": ds(dec(event.get("ts"))), "dur_us": ds(dec(event.get("dur", 0))),
               "owner_invocations": [x["invocation"] for x in owners], "crossing_invocations": [x["invocation"] for x in crosses],
               "raw_graph_marker_key": list(key) if key in expected_markers else None}
        if key in expected_markers:
            graph_events.append(row)
        elif owners or crosses:
            unknown_native.append(row)

    per_replay = []
    fence_coverage = defaultdict(list)
    for fence in fences:
        selected = {}
        errors = []
        for key, marker in expected_markers.items():
            candidates = []
            for index, event in marker_key_events.get(key, []):
                owner = fence_index.owners(event)
                crossings = fence_index.crossing(event)
                if any(x["invocation"] == fence["invocation"] for x in owner):
                    candidates.append((index, event))
                elif any(x["invocation"] == fence["invocation"] for x in crossings):
                    errors.append({"key": list(key), "marker_id": marker["marker_id"], "reason": "marker_event_crosses_fence", "event_index": index})
            if len(candidates) != 1:
                errors.append({"key": list(key), "marker_id": marker["marker_id"], "reason": "marker_key_count_in_fence", "count": len(candidates), "event_indices": [x[0] for x in candidates]})
            else:
                selected[key] = candidates[0]
        layer_rows = []
        if not errors:
            by_layer = defaultdict(dict)
            for key, marker in expected_markers.items():
                by_layer[marker["layer_idx"]][marker["marker_side"]] = (key, marker, selected[key])
            for layer in range(64):
                pair = by_layer[layer]
                if set(pair) != {"pre", "post"}:
                    errors.append({"layer_idx": layer, "reason": "missing_pre_or_post_marker"})
                    continue
                pre_key, pre_marker, (pre_i, pre_event) = pair["pre"]
                post_key, post_marker, (post_i, post_event) = pair["post"]
                pre_ts, pre_end = dec(pre_event.get("ts")), event_end(pre_event)
                post_ts, post_end = dec(post_event.get("ts")), event_end(post_event)
                if None in (pre_ts, pre_end, post_ts, post_end) or pre_ts > post_ts or pre_end > post_ts:
                    errors.append({"layer_idx": layer, "reason": "invalid_or_overlapping_pre_post_intervals", "pre_event_index": pre_i, "post_event_index": post_i})
                    continue
                layer_rows.append({"layer_idx": layer, "marker_identity": {"pre_marker_id": pre_marker["marker_id"], "post_marker_id": post_marker["marker_id"]},
                                   "native_key": {"pre": list(pre_key), "post": list(post_key)},
                                   "pre_event_index": pre_i, "post_event_index": post_i,
                                   "pre_ts_us": ds(pre_ts), "pre_dur_us": ds(pre_end - pre_ts),
                                   "post_ts_us": ds(post_ts), "post_dur_us": ds(post_end - post_ts),
                                   "interior_endpoint_elapsed_us": ds(post_ts - pre_end),
                                   "boundary_inclusive_elapsed_us": ds(post_end - pre_ts),
                                   "scope": "same-device native EVENT_RECORD endpoint elapsed; not exclusive compute or full multi-stream critical path"})
        fence_coverage[fence["invocation"]] = ["ok" if not errors else "unknown", errors]
        per_replay.append({k: v for k, v in fence.items() if k not in {"start", "end"}} | {
            "marker_keys_expected": len(expected_markers), "marker_keys_closed": len(selected) if not errors else len(selected),
            "layer_rows": layer_rows, "unknowns": errors})

    raw_marker_types = Counter(str(m["raw_node"].get("args", {}).get("Task Type")) for m in marker_rows)
    serving_marker_types = Counter(str(e.get("args", {}).get("Task Type"))
                                   for rows in marker_key_events.values() for _, e in rows)
    result = {
        "schema_version": 1,
        "status": "completed_marker_boundary_audit_with_nonmarker_rows_retained" if not failures and len(fences) == expected_fences and fence_index.valid and all(len(x["layer_rows"]) == 64 and not x["unknowns"] for x in per_replay) else "completed_with_unknowns",
        "scope": f"Gen66 vLLM rank {rank} serving diagnostic only; explicit flow-linked native outer fences and exact captured marker keys; no performance baseline claim",
        "rank_id": rank,
        "inputs": {"trace_gzip": {"path": str(trace_gz), "bytes": trace_gz.stat().st_size, "sha256": digest},
                   "trace_receipt": {"path": str(trace_receipt), "sha256": sha(trace_receipt), "compressed_sha256": digest,
                                     "uncompressed_sha256": raw_digest, "uncompressed_bytes": raw_bytes},
                   "outer_rows": {"path": str(outer_path), "sha256": sha(outer_path)},
                   "identity_audit": {"path": str(identity_path), "sha256": sha(identity_path)},
                   "raw_copy_manifest": {"path": str(raw_copy_manifest_path), "sha256": source_binding["manifest_sha256"],
                                         "download_receipt_path": str(copy_receipt_path),
                                         "download_receipt_sha256": sha(copy_receipt_path),
                                         "profiler_info_file": source_binding["profiler_info_file"],
                                         "profiler_info_sha256": source_binding["profiler_info_sha256"]},
                   "startup_device_map": {"path": str(startup_device_map_path), "sha256": sha(startup_device_map_path),
                                          "startup_binding": startup_binding},
                   "sdk_state": {"path": str(state_path), "sha256": sha(state_path),
                                 "selected_kind": selected_state_input.get("kind"),
                                 "selected_source": selected_state_input.get("source"),
                                 "selected_copy": selected_state_input.get("copy"),
                                 "selected_trace_path": input_result.get("trace_path")}},
        "trace_event_count": len(events), "hardware_process": {"pid": hardware_pid, "name": "Ascend Hardware", "device": device_label},
        "marker_identity_keys": len(expected_markers), "outer_rows_selected_rank": len(rank_outer),
        "marker_task_type_domains": {"raw_graph_capture_nodes": dict(raw_marker_types),
                                      "serving_trace_exact_key_events": dict(serving_marker_types)},
        "outer_native_task_type_counts": dict(Counter(str(x["row"].get("native_task_type")) for x in resolved)),
        "accepted_outer_task_types": sorted(set(accepted_outer_task_types)),
        "candidate_fence_count": len(candidate_fences),
        "candidate_fence_nonoverlap_valid": candidate_fence_index.valid,
        "candidate_fence_overlaps": candidate_fence_index.overlaps,
        "outer_type_semantics_unapproved": unapproved_outer_pairs,
        "outer_resolution_counts": {"resolved": len(resolved), "unresolved": len(failures)},
        "outer_resolved": len(resolved), "outer_unresolved": failures,
        "fence_count": len(fences), "fence_nonoverlap_valid": fence_index.valid, "fence_overlaps": fence_index.overlaps,
        "native_marker_event_count": sum(len(v) for v in marker_key_events.values()),
        "unmapped_or_crossing_native_row_count": len(unknown_native),
        "graph_marker_events": graph_events, "unmapped_or_crossing_inside_fences": unknown_native,
        "per_replay": per_replay,
        "limits": ["No task ordinal/range, nearest-time, or unverified order is used.",
                   "Outer task type labels are observed at exact flow-finish/native callback identity and are not prefiltered by a CANN-version-specific name.",
                   "Replay labels come from explicit outer invocation scopes, exact flow IDs, callback task/stream IDs, and the exact native endpoint.",
                   "work_key/request_id are preserved as raw nullable metadata; they do not prove returned-token count.",
                   "Layer rows are only event-boundary elapsed between exact same-generation pre/post raw-graph EVENT_RECORD keys.",
                   "These values are not exclusive kernel time or an all-stream device critical path."]}
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if compact_output_path is not None:
        compact_fences = []
        for replay in per_replay:
            compact_fences.append({k: v for k, v in replay.items() if k not in {"layer_rows"}})
            compact_fences[-1]["layer_rows"] = replay["layer_rows"]
        compact = {
            "schema_version": 1,
            "rank_id": rank,
            "status": result["status"],
            "scope": "Exact outer-fence and same-generation marker boundary rows for map display; see full audit for all retained native rows.",
            "full_audit": {"path": str(output_path), "bytes": Path(output_path).stat().st_size, "sha256": sha(output_path)},
            "inputs": result["inputs"],
            "hardware_process": result["hardware_process"],
            "counts": {"outer_rows": len(rank_outer), "outer_resolved": len(resolved),
                       "fences": len(fences), "marker_identity_keys": len(expected_markers),
                       "marker_events": result["native_marker_event_count"],
                       "layer_rows": sum(len(x["layer_rows"]) for x in per_replay),
                       "unmapped_or_crossing_native_rows_retained_in_full_audit": len(unknown_native),
                       "outer_unresolved": len(failures),
                       "replay_unknown_count": sum(len(x["unknowns"]) for x in per_replay)},
            "marker_task_type_domains": result["marker_task_type_domains"],
            "outer_native_task_type_counts": result["outer_native_task_type_counts"],
            "fences": compact_fences,
            "limits": result["limits"] + ["Non-marker native rows remain fully preserved in the referenced full audit; this compact view does not replace them."]
        }
        Path(compact_output_path).write_text(json.dumps(compact, indent=2, sort_keys=True) + "\n")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--trace-receipt", type=Path, required=True)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--outer", type=Path, required=True)
    p.add_argument("--identity", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--receipt-key", help="exact file key in download receipt; defaults to trace basename")
    p.add_argument("--raw-copy-manifest", type=Path, required=True)
    p.add_argument("--copy-receipt", type=Path, required=True, help="download receipt containing rankN-raw-copy.json")
    p.add_argument("--compact-output", type=Path, required=True)
    p.add_argument("--startup-device-map", type=Path, required=True)
    p.add_argument("--expected-fences", type=int, default=31, help="exact explicit start/end invocation pairs per rank; Gen64 vLLM uses 31 from 62 rows")
    p.add_argument("--accepted-outer-task-type", action="append", default=[], help="exact Task Type label with source-reviewed boundary semantics; omitted means no candidate fence is accepted")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.compact_output.exists():
        raise FileExistsError(a.compact_output)
    result = audit(a.trace, a.trace_receipt, a.state, a.outer, a.identity, a.output, a.rank,
                   a.receipt_key, a.raw_copy_manifest, a.copy_receipt, a.compact_output,
                   a.startup_device_map, a.expected_fences, a.accepted_outer_task_type)
    print(json.dumps({k: result[k] for k in ("status", "rank_id", "trace_event_count", "outer_rows_selected_rank", "outer_resolved", "fence_count", "marker_identity_keys")}))


if __name__ == "__main__":
    main()
