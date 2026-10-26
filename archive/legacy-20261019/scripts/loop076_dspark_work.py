#!/usr/bin/env python3
"""Partial DSpark proposer arithmetic census from saved original FULL Graph traces."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from loop076_dense_matmul_work import dims, FAMILIES

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "evidence/20260926_loop060_resource/run246/profile"
TAIL_INPUT_SHAPES = {
    "12,256;129280,256",   # seven serial Markov-bias projections
    "84,4096;16160,4096", # prior Draft body projection
    "11,16384;4,16384",  # small Draft projection
}



def num(row, key):
    try:
        return float(row[key].strip())
    except (KeyError, ValueError, AttributeError):
        return 0.0


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
        scopes = sorted((row for row in trace if row.get("cat") == "cpu_op" and row.get("name") == "extreme::proposer"), key=lambda x: float(x["ts"]))
        assert len(scopes) == 2
        csv_path = folder / "kernel_details.csv"
        kernels = list(csv.DictReader(csv_path.open(newline="")))
        source_paths.append({"rank": rank, "path": str(csv_path.relative_to(ROOT)), "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()})
        for cycle, scope in enumerate(scopes):
            start = float(scope["ts"])
            end = start + float(scope["dur"])
            chosen = [row for row in kernels if start <= num(row, "Start Time(us)") < end]
            # Host proposer return may precede queued Draft device tasks.
            # The 6ms window is a search range, not attribution by itself.
            tail = [row for row in kernels if end <= num(row, "Start Time(us)") < end + 6000
                    and row["Name"].startswith("aclnnMatmul_")
                    and row["Input Shapes"].strip().strip(chr(34)) in TAIL_INPUT_SHAPES]
            assert all(row["Stream ID"] == "47" for row in tail)
            tail_provenance = [
                {"model_id": row["Model ID"], "task_id": row["Task ID"],
                 "stream_id": row["Stream ID"], "input_shapes": row["Input Shapes"],
                 "start_us_after_host_exit": num(row, "Start Time(us)") - end,
                 "duration_us": num(row, "Duration(us)")}
                for row in tail
            ]
            chosen += tail
            families = {name: {"count": 0, "standard_Gflop": 0.0, "shape_groups": {}} for name in FAMILIES}
            gmm = {"gmm1": {"count": 0, "AIC_read_GB": 0.0}, "gmm2": {"count": 0, "AIC_read_GB": 0.0}}
            for row in chosen:
                name = row["Name"]
                fam = next((key for key, prefix in FAMILIES.items() if name.startswith(prefix)), None)
                if fam is not None:
                    x, y = dims(row["Input Shapes"]), dims(row["Output Shapes"])
                    assert len(x) >= 2 and len(y) >= 2
                    m, k, n = math.prod(y[:-1]), x[-1], y[-1]
                    work = 2 * m * k * n / 1e9
                    family = families[fam]
                    family["count"] += 1
                    family["standard_Gflop"] += work
                    key = f"M{m}_K{k}_N{n}"
                    group = family["shape_groups"].setdefault(key, {"count": 0, "standard_Gflop": 0.0})
                    group["count"] += 1
                    group["standard_Gflop"] += work
                elif name.startswith("aclnnGroupedMatmulSwigluQuantWeightNzV2_"):
                    gmm["gmm1"]["count"] += 1
                    gmm["gmm1"]["AIC_read_GB"] += num(row, "aic_read_main_memory_datas(KB)") * 1024 / 1e9
                elif name.startswith("aclnnGroupedMatmulWeightNz_"):
                    gmm["gmm2"]["count"] += 1
                    gmm["gmm2"]["AIC_read_GB"] += num(row, "aic_read_main_memory_datas(KB)") * 1024 / 1e9
            assert families["quant_matmul"]["count"] == 18
            assert families["transpose_batch_matmul"]["count"] == 3
            assert families["plain_matmul"]["count"] == 16
            assert families["plain_matmul"]["shape_groups"]["M12_K256_N129280"]["count"] == 7
            assert families["plain_matmul"]["shape_groups"]["M84_K4096_N16160"]["count"] == 1
            assert gmm["gmm1"]["count"] == gmm["gmm2"]["count"] == 3
            windows.append({"rank": rank, "cycle": cycle,
                            "proposer_host_scope_ms_profiled": float(scope["dur"]) / 1000,
                            "tail_tasks": tail_provenance,
                            "families": families, "gmm": gmm,
                            "dense_standard_Gflop_partial": sum(x["standard_Gflop"] for x in families.values())})
    values = [row["dense_standard_Gflop_partial"] for row in windows]
    out = {"status": "DSpark_proposer_partial_GEMM_work_census_corrected_device_tail",
           "source": "Run246 latest all8 two proposer scopes per rank, original FULL Graph Level1 capture",
           "sources": source_paths, "windows": windows,
           "summary": {"rank_cycles": len(windows), "family_count_ranges_per_rank_cycle": {name: [min(row["families"][name]["count"] for row in windows), max(row["families"][name]["count"] for row in windows)] for name in FAMILIES} | {"gmm1": [3, 3], "gmm2": [3, 3]},
                       "dense_standard_Gflop_partial_range": [min(values), max(values)],
                       "dense_standard_Gflop_partial_median": statistics.median(values)},
           "limits": [
               "GMM routed counts and math for three DSpark layers are missing; 504 padded input rows must not be treated as actual routed tokens.",
               "Host-scope-only census falsely varied 7-16: selected Draft device tasks, including seven serial Markov-bias projections and M84_K4096_N16160, start up to 5.03ms and finish up to 5.06ms after Host scope exit. Exact-shape-filtered 6ms search completes sixteen plain MatMul tasks; it is not a generic task-owner proof. In this selected trace, next Target Host scope starts at least 6.76ms after proposer Host exit. This is a device-task attribution correction, not a scheduling saving.",
               "Tail Model ID=4294967295 and stream47 do not prove Graph capture; the seven Markov-bias projections have serial token feedback via bias/add/argmax.",
               "Dense 2MNK is executed-shape standard arithmetic, not a strict mathematical operation lower bound. Attention, nonlinear, quantization, KV/state and other custom kernels remain outside this subtotal.",
               "Profiled Host scope/task times are perturbed and not used as attainable service; Run246 is a diagnostic cohort, not formal Run99.",
               "No useful accepted tokens per same cycle, compulsory HBM bytes, or Product TPS bound follows.",
           ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"status": out["status"], "summary": out["summary"]}))


if __name__ == "__main__":
    main()
