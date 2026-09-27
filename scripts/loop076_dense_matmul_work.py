#!/usr/bin/env python3
"""Account fixed-shape non-GMM GEMM-equivalent work in saved FULL Graph cycles."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "evidence/20260926_loop060_resource/run246/profile"
FAMILIES = {
    "quant_matmul": "aclnnQuantMatmulWeightNz_",
    "plain_matmul": "aclnnMatmul_",
    "transpose_batch_matmul": "aclnnTransposeBatchMatMul_",
}
EXPECTED = {"quant_matmul": 236, "plain_matmul": 151, "transpose_batch_matmul": 43}


def dims(text):
    first = text.strip().strip('"').split(";")[0]
    values = [int(value) for value in first.split(",")]
    assert values and all(value > 0 for value in values), text
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    windows = []
    source_paths = []
    for rank in range(8):
        folders = sorted(PROFILE.glob(f"rank{rank}_*/ASCEND_PROFILER_OUTPUT"))
        assert len(folders) == 5
        folder = folders[-1]
        trace = json.loads((folder / "trace_view.json").read_text())
        scopes = sorted((row for row in trace if row.get("cat") == "cpu_op" and row.get("name") == "extreme::target"), key=lambda x: float(x["ts"]))
        assert len(scopes) == 2
        csv_path = folder / "kernel_details.csv"
        kernels = list(csv.DictReader(csv_path.open(newline="")))
        source_paths.append({"rank": rank, "path": str(csv_path.relative_to(ROOT)), "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()})
        for cycle, scope in enumerate(scopes):
            start = float(scope["ts"])
            end = start + float(scope["dur"])
            chosen = [row for row in kernels if start <= float(row["Start Time(us)"].strip()) < end]
            totals = {name: {"count": 0, "standard_Gflop": 0.0, "shape_groups": {}} for name in FAMILIES}
            for row in chosen:
                family = next((name for name, prefix in FAMILIES.items() if row["Name"].startswith(prefix)), None)
                if family is None:
                    continue
                x = dims(row["Input Shapes"])
                y = dims(row["Output Shapes"])
                assert len(x) >= 2 and len(y) >= 2
                k, n = x[-1], y[-1]
                m = math.prod(y[:-1])
                work = 2 * m * k * n / 1e9
                key = f"M{m}_K{k}_N{n}"
                slot = totals[family]
                slot["count"] += 1
                slot["standard_Gflop"] += work
                shape = slot["shape_groups"].setdefault(key, {"count": 0, "standard_Gflop": 0.0})
                shape["count"] += 1
                shape["standard_Gflop"] += work
            assert all(totals[name]["count"] == expected for name, expected in EXPECTED.items()), (rank, cycle, totals)
            windows.append({"rank": rank, "cycle": cycle, "families": totals,
                            "sum_standard_Gflop": sum(totals[name]["standard_Gflop"] for name in FAMILIES)})
    med = {name: {"count": EXPECTED[name],
                  "standard_Gflop_median": statistics.median(w["families"][name]["standard_Gflop"] for w in windows),
                  "standard_Gflop_range": [min(w["families"][name]["standard_Gflop"] for w in windows),
                                             max(w["families"][name]["standard_Gflop"] for w in windows)]}
           for name in FAMILIES}
    out = {
        "status": "fixed_shape_nonGMM_GEMM_equivalent_work_census",
        "source": "Run246 latest captured two Target scopes per all8 rank; original FULL Graph Level1 profile",
        "sources": source_paths,
        "windows": windows,
        "summary": {"valid_rank_cycles": len(windows), "families": med,
                    "sum_standard_Gflop_median": statistics.median(w["sum_standard_Gflop"] for w in windows)},
        "limits": [
            "These are standard 2MNK GEMM-equivalent arithmetic counts inferred from actual Input/Output shapes, not strict mathematical operation lower bounds; nonlinear and custom kernels are excluded.",
            "Run246 is profiler-perturbed and separate from Run367 route capture, but these three dense shape-family counts are fixed across all16 latest windows.",
            "GMM route-dependent work and Compressor custom-kernel arithmetic remain separate ledger entries; DSpark, prefill, attention, KV, communication and Algorithmic efficiency are still incomplete.",
            "No timing sum or Product TPS upper bound follows from these arithmetic counts.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "summary": out["summary"]}))


if __name__ == "__main__":
    main()
