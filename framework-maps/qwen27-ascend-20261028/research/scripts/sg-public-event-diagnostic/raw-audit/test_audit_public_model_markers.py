import json
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

import audit_public_model_markers as m


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def valid_inputs(root, rank=0):
    pid, gid, inv = 1000 + rank, 9000 + rank, f"capture-{rank}"
    rows = []
    nodes = []
    for layer in range(64):
        for side in ("pre", "post"):
            task = layer * 2 + (side == "post")
            rows.append({"event": "layer_public_event_marker", "pid": pid,
                         "rank_id": rank, "layer_idx": layer, "graph_python_id": gid,
                         "graph_identity_status": "unique_backend_saved_graph_object",
                         "acl_stream_id": 65, "task_id": task, "stream_rc": 0, "task_rc": 0,
                         "capture_rc": 0, "capture_status": 1, "callback_rc": 0,
                         "model_ri_nonnull": True, "classification": "capture_endpoint_candidate",
                         "marker_side": side, "marker_id": f"{inv}:layer{layer}:{side}",
                         "capture_invocation_id": inv})
            nodes.append({"name": f"EVENT_RECORD_{task}", "pid": f"{pid} aclGraph", "ph": "X",
                          "args": {"Model Id": 44, "Stream Id": 12, "Task Id": task,
                                   "Task Type": "EVENT_RECORD"}})
    rows.append({"event": "layer_public_event_summary", "graph_python_id": gid,
                 "capture_invocation_id": inv, "marker_rows": 128})
    receipt = [{"graph_python_id": gid, "capture_id": f"capture-{rank}", "pid": pid, "rank_id": rank,
                "capture_end_status": "success", "dump_status": "success", "raw_exists": True,
                "graph_model_ids": ["44"]}]
    capture = struct.pack(m.FMT, 0, 0, 0, 0, 0, 0, 0, 12, 65, 44, 3, *([0] * 31))
    paths = {
        "endpoint_rows": write_json(root / f"end-{rank}.json", rows),
        "receipts": write_json(root / f"rec-{rank}.json", receipt),
        "raw_graph": write_json(root / f"raw-{rank}.json", nodes),
        "capture_stream_info": root / f"capture-{rank}.bin",
    }
    paths["capture_stream_info"].write_bytes(capture)
    return {"rank_id": rank, "pid": pid, "capture_id": f"capture-{rank}",
            "capture_stream_capture_id": f"capture-{rank}", **{k: str(v) for k, v in paths.items()}}


class AuditTests(unittest.TestCase):
    def test_complete_four_rank_identity_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ranks = [valid_inputs(root, i) for i in range(4)]
            result = m.audit_rank(ranks[0])
            self.assertEqual(result["exact_event_record_joins"].__len__(), 128)
            manifest = root / "manifest.json"
            for rank in ranks:
                rank["sha256"] = {k: hashlib.sha256(Path(rank[k]).read_bytes()).hexdigest()
                                  for k in ("endpoint_rows", "receipts", "raw_graph", "capture_stream_info")}
            write_json(manifest, {"ranks": ranks})
            result = m.audit(manifest)
            self.assertEqual(result["status"], "verified_marker_identity_only")
            self.assertEqual(result["marker_count"], 512)

    def test_rejects_missing_side_and_duplicate_marker(self):
        for mutate in (
            lambda rows: rows.pop(0),
            lambda rows: rows.__setitem__(1, dict(rows[0])),
        ):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as td:
                root = Path(td); entry = valid_inputs(root)
                rows = json.loads(Path(entry["endpoint_rows"]).read_text())
                mutate(rows); write_json(Path(entry["endpoint_rows"]), rows)
                with self.assertRaises(ValueError): m.audit_rank(entry)

    def test_rejects_callback_failure_bad_layer_duplicate_bridge_or_ambiguous_node(self):
        mutations = (
            ("callback", lambda rows: rows[0].update(capture_status=0)),
            ("layer", lambda rows: rows[0].update(layer_idx=64)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                root = Path(td); entry = valid_inputs(root)
                rows = json.loads(Path(entry["endpoint_rows"]).read_text()); mutate(rows)
                write_json(Path(entry["endpoint_rows"]), rows)
                with self.assertRaises(ValueError): m.audit_rank(entry)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); entry = valid_inputs(root)
            p = Path(entry["capture_stream_info"])
            p.write_bytes(p.read_bytes() * 2)
            with self.assertRaises(ValueError): m.audit_rank(entry)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); entry = valid_inputs(root)
            p = Path(entry["raw_graph"]); graph = json.loads(p.read_text())
            graph.append(dict(graph[0])); write_json(p, graph)
            with self.assertRaises(ValueError): m.audit_rank(entry)

    def test_rejects_receipt_pid_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); entry = valid_inputs(root)
            p = Path(entry["receipts"]); rec = json.loads(p.read_text()); rec[0]["pid"] += 1
            write_json(p, rec)
            with self.assertRaises(ValueError): m.audit_rank(entry)


if __name__ == "__main__":
    unittest.main()
