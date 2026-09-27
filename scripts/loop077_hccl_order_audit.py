#!/usr/bin/env python3
"""All-rank exact HCCL task order and conditional API byte mapping."""
import argparse
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZE = {"BFP16": 2, "FP32": 4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    reference = None
    hashes = []
    for rank in range(8):
        folder = Path(sorted(glob.glob(str(ROOT / f"evidence/20260926_loop060_resource/run246/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT")))[-1])
        trace = json.loads((folder / "trace_view.json").read_text())
        scopes = sorted((x for x in trace if x.get("cat") == "cpu_op" and x.get("name") == "extreme::target"), key=lambda x: x["ts"])
        hccl = sorted((x for x in trace if x.get("name", "").startswith("hcom_")), key=lambda x: float(x["ts"]))
        assert len(scopes) == 2
        for cycle, scope in enumerate(scopes):
            lo = float(scope["ts"])
            hi = lo + float(scope["dur"])
            seq = [(x["name"].split("__")[0], int(x["args"]["count"]), x["args"]["data_type"])
                   for x in hccl if lo <= float(x["ts"]) < hi]
            assert len(seq) == 265
            if reference is None:
                reference = seq
            assert seq == reference, (rank, cycle)
            hashes.append({"rank": rank, "cycle": cycle,
                           "sha256": hashlib.sha256(json.dumps(seq).encode()).hexdigest()})
    ordered = []
    by_kind = {}
    for i, (kind, count, dtype) in enumerate(reference):
        chunk = count * SIZE[dtype]
        input_bytes = chunk if kind == "hcom_allGather" else 8 * chunk
        logical_output_bytes = 8 * chunk if kind == "hcom_allGather" else input_bytes if kind == "hcom_alltoall" else chunk
        ordered.append({"ordinal": i, "kind": kind, "elements": count,
                        "dtype": dtype, "reported_chunk_bytes": chunk,
                        "conditional_API_input_bytes": input_bytes,
                        "conditional_API_output_bytes": logical_output_bytes})
        entry = by_kind.setdefault(kind, {"tasks": 0, "reported_chunk_bytes": 0, "conditional_API_input_bytes": 0})
        entry["tasks"] += 1
        entry["reported_chunk_bytes"] += chunk
        entry["conditional_API_input_bytes"] += input_bytes
    out = {"status": "all_rank_ordered_target_hccl_task_inventory",
           "source": "Run246 original FULL Graph 8 ranks x2 cycles trace hcom args; Run339 source-pinned HCCL API size convention is conditional mapping prior",
           "ordered_tasks": ordered, "all_rank_cycle_sha256": hashes,
           "summary": {"rank_cycles_identical": len(hashes), "tasks_per_rank_cycle": len(ordered),
                       "reported_chunk_bytes_sum": sum(x["reported_chunk_bytes"] for x in ordered),
                       "conditional_API_input_bytes_sum": sum(x["conditional_API_input_bytes"] for x in ordered),
                       "by_kind": by_kind},
           "limits": ["This is a current implementation task order, not mathematical necessity of all 265 collectives under arbitrary execution architecture.",
                      "The count-to-API mapping follows Run339 matched prefill HCCL conventions but requires direct decode source/ABI confirmation before compulsory byte promotion.",
                      "API tensor sizes are not physical link bytes or latency. Collective algorithm, topology, network aggregation and compute overlap remain unknown.",
                      "Synthetic communication-only replay of the exact order can calibrate isolated attainable chain service, but cannot replace the product resource-constrained DAG."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
