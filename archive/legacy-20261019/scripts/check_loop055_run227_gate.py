#!/usr/bin/env python3
"""Gate Run227 four-layer serving substitution on all-rank B shadow parity."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for rank in range(8):
    for layer in range(43):
        a = json.loads((out / f"A_rank{rank}_layer{layer}.json").read_text())
        b = json.loads((out / f"B_rank{rank}_layer{layer}.json").read_text())
        assert a["status"] == "captured" and b["status"] == "replayed"
        assert a["shape"] == b["shape"] == [11, 4096]
        if layer < 3:
            assert a["context_input_ids_ptr"] == b["context_input_ids_ptr"]
        c = b["comparisons"]
        assert c[0]["graph_exact"] and c[0]["graph_max_abs"] == 0
        assert c[1]["graph_max_abs"] <= min(0.03125, max(0.015625, 2 * c[1]["self_max_abs"]))
        rows.append({"rank": rank, "layer": layer, "routed_graph_max_abs": c[1]["graph_max_abs"], "routed_self_max_abs": c[1]["self_max_abs"]})
for tag in "AB":
    for rank in range(8):
        stage = json.loads((out / f"{tag}_rank{rank}_stage.json").read_text())
        assert stage["status"] == "pass" and stage["layers_seen"] == list(range(43))
result = {"gate": "pass", "rows": rows}
(out.parent / "gate_check.json").write_text(json.dumps(result, indent=2) + "\n")
(out.parent / "gate.txt").write_text("pass\n")
print(json.dumps({"gate": "pass", "comparisons": len(rows)}))
