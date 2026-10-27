import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("audit_sg67_formal_by_rank.py")
spec = importlib.util.spec_from_file_location("sg67_auditor", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class SelectionTests(unittest.TestCase):
    def fixture(self, count):
        return [{"rank_id": 0, "invocation": f"inv{i}", "side": side}
                for i in range(count) for side in ("start", "end")]

    def test_accepts_exactly_32_explicit_pairs(self):
        identity = {"ranks": [{"rank_id": 0, "proof": "fixture"}]}
        entry, rows = mod.select_rank_evidence(0, identity, self.fixture(32), 32)
        self.assertEqual(entry["proof"], "fixture")
        self.assertEqual(len(rows), 64)
        mod.validate_rank_binding({"exact_event_record_joins": [
            {"pid": 7, "graph_python_id": "g"} for _ in range(128)]},
            [{**r, "pid": 7, "graph_python_id": "g"} for r in rows], 32)

    def test_rejects_31_and_35_pair_substitutions(self):
        identity = {"ranks": [{"rank_id": 0}]}
        for count in (31, 35):
            with self.subTest(count=count), self.assertRaises(ValueError):
                mod.select_rank_evidence(0, identity, self.fixture(count), 32)

    def test_list_download_receipt_is_exact_and_duplicate_safe(self):
        doc = {"files": [{"relative_path": "rank0-trace.json.gz", "bytes": 12, "sha256": "abc"}]}
        self.assertEqual(mod.normalize_download_files(doc)["rank0-trace.json.gz"],
                         {"bytes": 12, "sha256": "abc", "kind": None})
        with self.assertRaises(ValueError):
            mod.normalize_download_files({"files": [doc["files"][0], doc["files"][0]]})


if __name__ == "__main__":
    unittest.main()
