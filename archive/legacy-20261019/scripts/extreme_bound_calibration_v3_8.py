#!/usr/bin/env python3
"""Source-gated V3.8 Bound update: dependency skeleton, not a latency floor."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prior_path = "evidence/20260927_loop077_bound/run396/bound_calibration_v3_7.json"
    dag_path = "evidence/20260927_loop078_bound/run400/dependency_ledger.json"
    review_path = "evidence/20260927_loop078_bound/run400/astra_review.md"
    model = load(prior_path)
    dag = load(dag_path)
    review = (ROOT / review_path).read_text()
    assert dag["status"] == "source_pinned_partial_two_cycle_dag"
    assert len(dag["sources"]) == 5
    assert len(dag["nodes"]) == 19 and len(dag["edges"]) == 22
    assert len(dag["conditional_edges"]) == len(dag["unproven_hazards"]) == 1
    assert "Verdict: ACCEPT as the revised source-pinned partial dependency inventory" in review
    assert all(node["duration_floor_ms"] is None for node in dag["nodes"])
    sched = model["bound_ladder"]["scheduling_execution"]
    sched["run400_two_cycle_source_dependency_inventory"] = {
        "nodes": len(dag["nodes"]),
        "typed_main_edges": len(dag["edges"]),
        "parking_conditional_edges": len(dag["conditional_edges"]),
        "unproven_storage_hazards": dag["unproven_hazards"],
        "graph_sync_condition": dag["conditional_context"]["graph_pre_replay_sync"],
        "source": f"{dag_path}; independent Astra review {review_path}",
        "scope": "Partial source-order/data ledger; mixed Host enqueue/device completion edges cannot feed a finish-to-start longest-path solver; no numerical Bound promotion.",
    }
    sched["missing"].append("settle count-copy read completion before next num_sampled overwrite, parking branch, actual graph replay/schedule mode and typed Host/device endpoints")
    model["next_measurement"]["priority"] = "close read-before-overwrite and true side-stream/Graph/HCCL completion joins while measuring compulsory Target/Draft per-token route"
    model["next_measurement"]["specific_gate"] = "Run401 sparse same-device copy_done(c) vs overwrite_pre(c+1), Host mirror consumer timestamps and A/A; Run399 exact W4A8 route IDs with row/position/group-list parity; no finite scheduling/product ceiling until complete resource DAG and formal E2E validation"
    model["input_paths"].extend([prior_path, dag_path, review_path])
    assert model["bound_ladder"]["product_e2e"]["finite_tps_upper_bound"] is None
    assert all(model["bound_ladder"][key]["latency_floor_s"] is None
               for key in ("algorithm_resource", "hardware_resource", "scheduling_execution"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "ok", "source_dag_nodes": len(dag["nodes"]),
                      "product_finite_upper": None}))


if __name__ == "__main__":
    main()
