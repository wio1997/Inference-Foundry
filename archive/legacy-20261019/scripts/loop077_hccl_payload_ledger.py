#!/usr/bin/env python3
"""Reported HCCL task count/dtype signature in original FULL Graph Target cycles."""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BYTES = {"BFP16": 2, "FP32": 4}
EXPECTED = {
    ("hcom_allGather", 49152, "BFP16"): 90,
    ("hcom_reduceScatter", 49152, "BFP16"): 87,
    ("hcom_alltoall", 49152, "BFP16"): 43,
    ("hcom_allGather", 3072, "FP32"): 43,
    ("hcom_allGather", 196608, "BFP16"): 1,
    ("hcom_allGather", 1551360, "BFP16"): 1,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    windows = []
    for rank in range(8):
        folder = Path(sorted(glob.glob(str(ROOT / f"evidence/20260926_loop060_resource/run246/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT")))[-1])
        trace = json.loads((folder / "trace_view.json").read_text())
        scopes = sorted((x for x in trace if x.get("cat") == "cpu_op" and x.get("name") == "extreme::target"), key=lambda x: x["ts"])
        hccl = [x for x in trace if x.get("name", "").startswith("hcom_")]
        assert len(scopes) == 2
        for cycle, scope in enumerate(scopes):
            lo = float(scope["ts"])
            hi = lo + float(scope["dur"])
            matched = [x for x in hccl if lo <= float(x["ts"]) < hi]
            sigs = Counter((x["name"].split("__")[0], int(x["args"]["count"]), x["args"]["data_type"])
                           for x in matched)
            assert sigs == Counter(EXPECTED), (rank, cycle, sigs)
            assert all(x["args"]["rank_size"] == 8 for x in matched)
            windows.append({"rank": rank, "cycle": cycle, "task_count": len(matched),
                            "reported_count_dtype_bytes_sum": sum(n * count * BYTES[dtype]
                                for (_, count, dtype), n in sigs.items())})
    signatures = [{"kind": kind, "count_elements": count, "dtype": dtype,
                   "tasks_per_rank_target_cycle": n,
                   "reported_count_dtype_bytes_per_task": count * BYTES[dtype]}
                  for (kind, count, dtype), n in EXPECTED.items()]
    total = windows[0]["reported_count_dtype_bytes_sum"]
    assert len(windows) == 16 and all(x["task_count"] == 265 and x["reported_count_dtype_bytes_sum"] == total for x in windows)
    out = {"status": "original_full_graph_target_hccl_reported_task_payload_ledger",
           "source": "Run246 latest 8-rank x2 Target scope CANN trace_view.json hcom task args",
           "signatures": signatures, "windows": windows,
           "summary": {"rank_cycles": len(windows), "tasks_per_rank_cycle": 265,
                       "reported_count_dtype_bytes_sum_per_rank_cycle": total},
           "limits": ["The count field meaning differs by collective ABI and is not yet proven to be input bytes, output bytes or physical link bytes.",
                      "Task payload products are an inventory, not additive link traffic or latency. A task may wait for later rank submission.",
                      "Graph/profiler capture can affect HCCL timings; logical compulsory collectives require source/API and dependency audit, and alternative TP schedule may change their number.",
                      "Run339 isolated service cases used residual-prefill payloads and do not give this Target decode chain service capacity."]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"]))


if __name__ == "__main__":
    main()
