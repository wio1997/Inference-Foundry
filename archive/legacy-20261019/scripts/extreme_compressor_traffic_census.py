#!/usr/bin/env python3
"""Exact-shape Compressor arithmetic/footprint screen on product graph counters.

Tensor footprint is not compulsory HBM traffic. Reported AIC+AIV task
counters are not automatically unique bytes or removable work.
"""
import argparse
import csv
import glob
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/"
              "_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/"
              "custom_transformer_impl/ascendc/compressor/arch32/"
              "compressor_block_cube_perf.h")
DTYPE_BYTES = {"DT_BF16": 2, "FLOAT": 4, "INT32": 4}
EXPECTED_CALLS = {256: 21, 512: 20, 1024: 21}

def parse_shape(text):
    return [tuple(int(v) for v in part.split(",")) if part else ()
            for part in text.strip('"').split(";")]

def numel(shape):
    out = 1
    for d in shape:
        out *= d
    return out

def get_float(row, key):
    return float(row.get(key, "0").strip() or 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not SOURCE.exists():
        raise SystemExit("missing installed Compressor source")
    src = SOURCE.read_text()
    anchors = {}
    for token in ("CopyXGmToL1(info, xL1Tensor",
                  "CopyWeightGmToL1(wL1Tensor",
                  "Fixpipe(kvMm1ResGm",
                  "Fixpipe(scoreMm1ResGm"):
        lines = [i for i, line in enumerate(src.splitlines(), 1) if token in line]
        if len(lines) != 1:
            raise SystemExit("source anchor missing or ambiguous: " + token)
        anchors[token] = lines[0]
    records = defaultdict(list)
    invalid = []
    for rank in range(8):
        folders = sorted(glob.glob(str(args.profile_dir /
                                      ("rank%d_*/ASCEND_PROFILER_OUTPUT" % rank))))
        if len(folders) != 5:
            invalid.append({"rank": rank, "capture_count": len(folders)})
            continue
        folder = Path(folders[-1])
        trace = json.loads((folder / "trace_view.json").read_text())
        scopes = sorted((e for e in trace if e.get("name") == "extreme::target"
                         and e.get("cat") == "cpu_op"),
                        key=lambda e: float(e["ts"]))
        if len(scopes) != 2:
            invalid.append({"rank": rank, "scope_count": len(scopes)})
            continue
        with (folder / "kernel_details.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        for cycle, scope in enumerate(scopes):
            start = float(scope["ts"])
            stop = start + float(scope["dur"])
            matched = [r for r in rows if r["Name"] == "Compressor"
                       and start <= get_float(r, "Start Time(us)") < stop]
            by_n = defaultdict(list)
            for row in matched:
                shapes = parse_shape(row["Input Shapes"])
                dtypes = row["Input Data Types"].split(";")
                if len(shapes) < 3 or len(dtypes) < 3:
                    invalid.append({"rank": rank, "cycle": cycle,
                                    "reason": "bad_input_abi"})
                    continue
                if shapes[0] != (96, 4096) or shapes[1] != shapes[2] or \
                        dtypes[:3] != ["DT_BF16"] * 3:
                    invalid.append({"rank": rank, "cycle": cycle,
                                    "reason": "unexpected_shape_dtype",
                                    "shapes": shapes[:3], "dtypes": dtypes[:3]})
                    continue
                n, k = shapes[1]
                if k != 4096 or n not in EXPECTED_CALLS:
                    invalid.append({"rank": rank, "cycle": cycle,
                                    "reason": "unexpected_weight"})
                    continue
                by_n[n].append(row)
            if {n: len(v) for n, v in by_n.items()} != EXPECTED_CALLS:
                invalid.append({"rank": rank, "cycle": cycle,
                                "reason": "call_counts",
                                "actual": {n: len(v) for n, v in by_n.items()}})
                continue
            for n, group in by_n.items():
                read_aic = sum(get_float(r, "aic_read_main_memory_datas(KB)")
                               for r in group) * 1024
                read_aiv = sum(get_float(r, "aiv_read_main_memory_datas(KB)")
                               for r in group) * 1024
                write_aic = sum(get_float(r, "aic_write_main_memory_datas(KB)")
                                for r in group) * 1024
                write_aiv = sum(get_float(r, "aiv_write_main_memory_datas(KB)")
                                for r in group) * 1024
                records[n].append({
                    "rank": rank, "cycle": cycle, "count": len(group),
                    "reported_read_bytes": read_aic + read_aiv,
                    "reported_write_bytes": write_aic + write_aiv,
                    "aic_read_bytes": read_aic, "aiv_read_bytes": read_aiv,
                    "aic_write_bytes": write_aic, "aiv_write_bytes": write_aiv,
                    "task_sum_ms": sum(get_float(r, "Duration(us)")
                                       for r in group) / 1000,
                })
    status = ("valid" if not invalid and all(len(records[n]) == 16
                                             for n in EXPECTED_CALLS)
              else "invalid")
    summary = {}
    for n, count in EXPECTED_CALLS.items():
        m, k = 96, 4096
        weight_per_call = 2 * n * k * 2
        x_per_call = m * k * 2
        rows = records[n]
        med = lambda key: statistics.median(r[key] for r in rows) if rows else None
        summary[str(n)] = {
            "calls_per_rank_cycle": count,
            "shape": {"x": [m, k], "wkv": [n, k], "wgate": [n, k]},
            "unique_two_weight_tensor_footprint_bytes_per_cycle":
                count * weight_per_call,
            "x_tensor_footprint_bytes_per_call": x_per_call,
            "nominal_two_projection_flops_per_cycle": count * 4 * m * k * n,
            "reported_aic_read_bytes_per_cycle_median": med("aic_read_bytes"),
            "reported_aiv_read_bytes_per_cycle_median": med("aiv_read_bytes"),
            "reported_read_bytes_per_cycle_median": med("reported_read_bytes"),
            "reported_write_bytes_per_cycle_median": med("reported_write_bytes"),
            "profiled_task_sum_ms_median": med("task_sum_ms"),
            "unexplained_read_after_unique_weight_footprint_bytes":
                (med("reported_read_bytes") - count * weight_per_call)
                if rows else None,
            "windows": rows,
        }
    out = {
        "status": status, "windows_per_shape": 16,
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_anchors": anchors, "summary": summary, "invalid": invalid,
        "limitations": [
            "Unique tensor footprints are not compulsory HBM reads; L2, tiling, "
            "repeat visits, aliases, and cache state matter.",
            "Unexplained read minus weight footprint includes X, state, "
            "metadata, workspace and possible repeat reads; not waste.",
            "Nominal matmul FLOPs omit normalization, cache and rope arithmetic.",
            "Profiler task durations and bytes are not exposed E2E savings."
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "source_sha256": out["source_sha256"],
        "summary": {n: {k: v for k, v in row.items() if k != "windows"}
                    for n, row in summary.items()},
        "invalid": invalid,
    }, indent=2))
    if status != "valid":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
