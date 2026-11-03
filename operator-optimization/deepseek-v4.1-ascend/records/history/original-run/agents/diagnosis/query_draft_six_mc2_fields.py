#!/usr/bin/env python3
"""Select six already identified MC2 rows from verified frozen CSVs."""
import argparse, csv, hashlib, json
from decimal import Decimal
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--trace-root", required=True)
args = ap.parse_args()
spec = json.loads(FROZEN_INPUT)
fields = [
    "Name", "Type", "Model ID", "Stream ID", "Task ID",
    "Start Time(us)", "Duration(us)", "Wait Time(us)", "Block Num", "Mix Block Num",
    "Input Shapes", "Input Data Types", "Input Formats",
    "Output Shapes", "Output Data Types", "Output Formats",
    "aicore_time(us)", "aic_total_cycles", "aic_mac_time(us)", "aic_mac_ratio",
    "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
    "aiv_time(us)", "aiv_total_cycles", "aiv_vec_ratio", "aiv_scalar_ratio",
    "aiv_mte2_ratio", "aiv_mte3_ratio", "cube_utilization(%)",
]
results = []
for rank, targets in spec["ranks"].items():
    p = Path(args.trace_root) / rank / "ASCEND_PROFILER_OUTPUT/kernel_details.csv"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    if h.hexdigest() != spec["csv_sha256_by_rank"][rank]:
        raise RuntimeError("frozen input hash mismatch")
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        if not set(fields).issubset(reader.fieldnames or ()):
            raise RuntimeError("missing field")
        matches = {t["mc2_task_id"]: [] for t in targets}
        for row in reader:
            if row["Type"] != "DispatchFFNCombine":
                continue
            for target in targets:
                if row["Stream ID"] != target["main_stream"] or row["Task ID"] != target["mc2_task_id"]:
                    continue
                start = int(Decimal(row["Start Time(us)"].strip()) * 1000)
                dur = int(Decimal(row["Duration(us)"].strip()) * 1000)
                if (start, start + dur) == (target["mc2_start_ns"], target["mc2_end_ns"]):
                    matches[target["mc2_task_id"]].append(row)
    for target in targets:
        rows = matches[target["mc2_task_id"]]
        if len(rows) != 1:
            raise RuntimeError("expected exactly one task match: " + rank)
        row = rows[0]
        if Decimal(row["aic_mac_ratio"]) * Decimal(row["aic_total_cycles"]) != Decimal(target["expected_aic_num"]):
            raise RuntimeError("AIC numerator differs")
        if Decimal(row["aic_total_cycles"]) != Decimal(target["expected_aic_den"]):
            raise RuntimeError("AIC denominator differs")
        results.append({"rank": rank, "local_op_id": target["local_op_id"],
                        "fields": {k: row[k] for k in fields}})
print(json.dumps({"scope": "Six same-rank selected historical draft MC2 CSV tasks, no NPU",
                  "csv_sha256_by_rank": spec["csv_sha256_by_rank"], "samples": results,
                  "limitations": ["Exported counters are not physical FLOPs, idle time, memory bandwidth or internal MC2 phases.",
                                  "Samples are not matched workload or proven identical cross-rank collectives.",
                                  "Mixed pipeline overlap, counter definitions and exporter version need independent verification."]}, indent=2))
