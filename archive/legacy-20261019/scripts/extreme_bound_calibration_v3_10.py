#!/usr/bin/env python3
"""Add conditional current dense-transport cut, never compulsory network floor."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIOR = "evidence/20260927_loop078_bound/run406/bound_calibration_v3_9.json"
LEDGER = "evidence/20260927_loop077_bound/run378/ledger.json"
REVIEW = "evidence/20260927_loop078_bound/run414/astra_comm_resource_review.md"


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    m = load(PRIOR)
    tasks = load(LEDGER)["ordered_tasks"]
    review = (ROOT / REVIEW).read_text()
    assert len(tasks) == 265 and "306,659,328" in review
    cuts = []
    for k in (1, 2, 3, 4):
        out = inc = 0
        for op in tasks:
            c = op["reported_chunk_bytes"]
            if op["kind"] == "hcom_allGather":
                x, y = k*c, (8-k)*c
            elif op["kind"] == "hcom_reduceScatter":
                x, y = (8-k)*c, k*c
            elif op["kind"] == "hcom_alltoall":
                x = y = k*(8-k)*c
            else:
                raise ValueError(op["kind"])
            out += x
            inc += y
        cuts.append(dict(cut_rank_count=k, outgoing_B=out, incoming_B=inc, two_way_B=out+inc))
    assert cuts[-1] == dict(cut_rank_count=4, outgoing_B=153329664, incoming_B=153329664, two_way_B=306659328)
    h = m["bound_ladder"]["hardware_resource"]
    h["run414_current_dense_transport_cut_conditional_only"] = {
        "cuts": cuts,
        "assumptions": "Fixed current ownership and all265 API outputs, opaque fresh dense payload, no cross-operation reuse/recomputation/zero elision/compression/placement change; ideal within-side reduction/broadcast; counts logical cut crossings, not physical multi-hop wire bytes.",
        "bound_status": "conditional current transport scenario; NOT model-compulsory network bytes or finite latency floor",
        "source": [LEDGER, REVIEW],
    }
    h["missing"].append("derive legal architecture-dependent semantic information cuts and actual physical link/cut capacity; Run246 transit export is empty")
    m["bound_ladder"]["algorithm_resource"]["missing"].append("determine which hidden, expert, reduced contribution and verifier information must cross each cut under placement/recompute tradeoffs")
    m["next_measurement"]["priority"] = "complete row identity and actual resource numerator while closing count-copy Scheduling hazard; one real DSA A2A transport export capability gate before any network latency floor"
    m["next_measurement"]["specific_gate"] = "Run401 local copy-read/overwrite measurement; Run407 source+branch row-map certificate; single actual DSA A2A native peer/bytes/path capture capability on restored service"
    m["input_paths"].extend([PRIOR, LEDGER, REVIEW])
    assert m["bound_ladder"]["product_e2e"]["finite_tps_upper_bound"] is None
    assert all(m["bound_ladder"][key]["latency_floor_s"] is None for key in ("algorithm_resource", "hardware_resource", "scheduling_execution"))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(m, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"status": "ok", "conditional_4x4_cut_B": cuts[-1]["two_way_B"], "finite_product_ceiling": None}))


if __name__ == "__main__":
    main()
