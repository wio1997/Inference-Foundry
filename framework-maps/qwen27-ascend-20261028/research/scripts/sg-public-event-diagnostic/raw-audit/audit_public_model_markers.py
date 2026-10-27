#!/usr/bin/env python3
"""Fail-closed identity audit for public pre/post layer event markers.

This establishes marker -> exact capture receipt -> CANN physical/model stream
bridge -> unique raw EVENT_RECORD node. It does not assign intervening graph
tasks to layers or infer device time from event endpoints.
"""
from __future__ import annotations
import argparse, hashlib, json, struct
from collections import Counter, defaultdict
from pathlib import Path

FMT = "=HHIIIQHHHH" + "B" * 32
RECORD_SIZE = 64
SIDES = {"pre", "post"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def capture_rows(path: Path):
    raw = path.read_bytes()
    if len(raw) % RECORD_SIZE:
        raise ValueError(f"capture stream file has partial record: {path}")
    out = []
    for i in range(0, len(raw), RECORD_SIZE):
        f = struct.unpack_from(FMT, raw, i)
        out.append({"capture_status": f[6], "model_stream_id": f[7],
                    "original_stream_id": f[8], "model_id": f[9],
                    "device_id": f[10], "record_index": i // RECORD_SIZE})
    return out


def key_of(node):
    a = node.get("args", {})
    vals = (a.get("Model Id"), a.get("Stream Id"), a.get("Task Id"))
    if any(isinstance(x, bool) or not isinstance(x, int) for x in vals):
        return None
    return vals


def marker_identity(row):
    if row.get("event") != "layer_public_event_marker":
        raise ValueError("non-marker row passed to marker audit")
    required_ints = ("pid", "rank_id", "layer_idx", "graph_python_id",
                     "acl_stream_id", "task_id", "stream_rc", "task_rc", "capture_rc",
                     "capture_status", "callback_rc")
    for k in required_ints:
        if isinstance(row.get(k), bool) or not isinstance(row.get(k), int):
            raise ValueError(f"marker missing exact integer {k}")
    if "producer_pid" in row and row["pid"] != row["producer_pid"]:
        raise ValueError("marker PID differs from producer PID")
    if row.get("graph_identity_status") != "unique_backend_saved_graph_object":
        raise ValueError("marker graph object identity is not unique")
    if row.get("classification") != "capture_endpoint_candidate":
        raise ValueError("marker classification is not capture endpoint candidate")
    if row["capture_status"] != 1 or row.get("model_ri_nonnull") is not True:
        raise ValueError("callback is not a successful nonnull capture")
    if any(row[k] != 0 for k in ("stream_rc", "task_rc", "capture_rc", "callback_rc")):
        raise ValueError("marker callback return code failed")
    layer = row["layer_idx"]
    if not 0 <= layer < 64:
        raise ValueError("layer index outside 0..63")
    side = row.get("marker_side")
    if side not in SIDES:
        raise ValueError("marker side must be pre or post")
    marker_id = row.get("marker_id")
    if not isinstance(marker_id, str) or not marker_id:
        raise ValueError("missing marker_id")
    if not isinstance(row.get("capture_invocation_id"), str) or not row["capture_invocation_id"]:
        raise ValueError("missing capture_invocation_id")
    return row["pid"], row["rank_id"], row["graph_python_id"], layer, side, marker_id


def audit_rank(entry):
    rank = entry["rank_id"]
    paths = {k: Path(entry[k]) for k in ("endpoint_rows", "receipts", "raw_graph", "capture_stream_info")}
    data = {k: load(paths[k]) for k in ("endpoint_rows", "receipts", "raw_graph")}
    if isinstance(data["receipts"], dict):
        data["receipts"] = [data["receipts"]]
    if not isinstance(data["receipts"], list) or any(not isinstance(x, dict) for x in data["receipts"]):
        raise ValueError("receipt input must be one receipt object or an array of receipt objects")
    all_markers = [x for x in data["endpoint_rows"] if x.get("event") == "layer_public_event_marker"]
    if any(isinstance(x.get("rank_id"), bool) or not isinstance(x.get("rank_id"), int)
           or x["rank_id"] not in (0, 1, 2, 3) for x in all_markers):
        raise ValueError("every marker in endpoint input must carry exact TP4 rank_id")
    markers = [x for x in all_markers if x["rank_id"] == rank]
    if len(markers) != 128:
        raise ValueError(f"rank {rank}: expected exactly 128 marker rows, got {len(markers)}")
    counts = Counter(x.get("marker_side") for x in markers)
    if counts != Counter({"pre": 64, "post": 64}):
        raise ValueError(f"rank {rank}: expected 64 pre and 64 post markers, got {dict(counts)}")
    ids, layer_sides = set(), set()
    for row in markers:
        ident = marker_identity(row)
        pid, rr, gid, layer, side, mid = ident
        if rr != rank or pid != entry["pid"]:
            raise ValueError("marker rank/PID differs from manifest identity")
        if mid in ids:
            raise ValueError(f"duplicate marker_id {mid}")
        ids.add(mid)
        ls = (gid, layer, side)
        if ls in layer_sides:
            raise ValueError(f"duplicate layer side {ls}")
        layer_sides.add(ls)
    if layer_sides != {(gid, layer, side) for layer in range(64) for side in SIDES}:
        raise ValueError("each layer must have exactly one pre and one post marker")
    invocations = {x["capture_invocation_id"] for x in markers}
    if len(invocations) != 1:
        raise ValueError("markers do not share one capture invocation")
    summaries = [x for x in data["endpoint_rows"] if x.get("event") == "layer_public_event_summary"
                 and x.get("graph_python_id") == gid
                 and x.get("capture_invocation_id") in invocations]
    if len(summaries) != 1 or summaries[0].get("marker_rows") != 128:
        raise ValueError("expected one matching backend summary declaring 128 markers")
    graph_ids = {r["graph_python_id"] for r in markers}
    if len(graph_ids) != 1:
        raise ValueError("rank markers do not identify exactly one graph object")
    gid = next(iter(graph_ids))
    recs = [r for r in data["receipts"] if r.get("graph_python_id") == gid
            and r.get("pid") == entry["pid"] and r.get("rank_id") == rank
            and r.get("capture_end_status") == "success" and r.get("dump_status") == "success"
            and r.get("raw_exists") is True]
    if len(recs) != 1:
        raise ValueError(f"expected one successful same-PID/rank graph receipt, found {len(recs)}")
    rec = recs[0]
    capture_id = entry.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id or rec.get("capture_id") != capture_id:
        raise ValueError("receipt capture_id does not match explicit manifest capture identity")
    if entry.get("capture_stream_capture_id") != capture_id:
        raise ValueError("capture_stream_info provenance is not bound to this receipt capture_id")
    models = {str(x) for x in rec.get("graph_model_ids", [])}
    if len(models) != 1:
        raise ValueError("capture receipt must expose exactly one graph model ID")
    model_id = int(next(iter(models)))
    raw_nodes = data["raw_graph"]
    if not isinstance(raw_nodes, list):
        raise ValueError("raw graph must be a node list")
    caprows = capture_rows(paths["capture_stream_info"])
    joins, errors = [], []
    for row in markers:
        phys = row["acl_stream_id"]
        bridges = [b for b in caprows if b["original_stream_id"] == phys
                   and b["model_id"] == model_id]
        if len(bridges) != 1:
            errors.append({"marker_id": row["marker_id"], "reason": "physical-to-model bridge not unique",
                           "bridge_candidates": len(bridges)})
            continue
        b = bridges[0]
        k = (model_id, b["model_stream_id"], row["task_id"])
        matches = [(i, n) for i, n in enumerate(raw_nodes) if key_of(n) == k]
        event_nodes = [(i, n) for i, n in matches
                       if str(n.get("args", {}).get("Task Type", "")).upper() == "EVENT_RECORD"
                       and str(n.get("pid", "")).split()[0] == str(entry["pid"])]
        if len(matches) != 1 or len(event_nodes) != 1:
            errors.append({"marker_id": row["marker_id"], "reason": "raw graph key not a unique EVENT_RECORD",
                           "model_stream_task_key": list(k), "key_matches": len(matches),
                           "event_record_matches": len(event_nodes)})
            continue
        joins.append({"rank_id": rank, "pid": entry["pid"], "graph_python_id": gid,
                      "layer_idx": row["layer_idx"], "marker_side": row["marker_side"],
                      "marker_id": row["marker_id"], "acl_stream_id": phys,
                      "model_id": model_id, "model_stream_id": b["model_stream_id"],
                      "task_id": row["task_id"], "capture_stream_record_index": b["record_index"],
                      "capture_stream_record_status": b["capture_status"],
                      "capture_stream_device_id": b["device_id"],
                      "callback_capture_status": row["capture_status"],
                      "callback_return_codes": {k: row[k] for k in ("stream_rc", "task_rc", "capture_rc", "callback_rc")},
                      "raw_node_index": event_nodes[0][0], "raw_node": event_nodes[0][1],
                      "scope": "exact public event marker identity only; no neighboring task or elapsed-time attribution"})
    if errors:
        raise ValueError(json.dumps({"rank": rank, "unresolved": errors}, sort_keys=True))
    if len(joins) != 128 or len({j["marker_id"] for j in joins}) != 128:
        raise ValueError("rank must close 128 unique markers")
    return {"rank_id": rank, "pid": entry["pid"], "graph_python_id": gid,
            "marker_count": 128, "pre_count": 64, "post_count": 64, "exact_event_record_joins": joins}


def audit(manifest_path: Path):
    manifest = load(manifest_path)
    ranks = manifest.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != 4:
        raise ValueError("manifest must declare exactly four rank entries")
    if sorted(x.get("rank_id") for x in ranks) != [0, 1, 2, 3]:
        raise ValueError("manifest ranks must be exactly 0..3")
    out = []
    for e in ranks:
        expected_hashes = e.get("sha256", {})
        required_hashes = {"endpoint_rows", "receipts", "raw_graph", "capture_stream_info"}
        if set(expected_hashes) != required_hashes:
            raise ValueError(f"rank {e.get('rank_id')}: SHA256 receipts required for exactly {sorted(required_hashes)}")
        for field, expected in expected_hashes.items():
            p = Path(e[field])
            if sha(p) != expected:
                raise ValueError(f"SHA mismatch: {field} for rank {e['rank_id']}")
        out.append(audit_rank(e))
    total = sum(len(x["exact_event_record_joins"]) for x in out)
    if total != 512:
        raise ValueError(f"expected 512 exact marker joins, got {total}")
    return {"status": "verified_marker_identity_only", "rank_count": 4,
            "marker_count": total, "identity_scope": "marker→same PID/rank/graph receipt→unique capture stream bridge→unique raw EVENT_RECORD key",
            "prohibited_inferences": ["task ordinal or range to layer assignment", "intervening task membership", "device timing or exclusive layer duration", "performance claim"],
            "ranks": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = audit(a.manifest)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "rank_count", "marker_count")}))

if __name__ == "__main__":
    main()
