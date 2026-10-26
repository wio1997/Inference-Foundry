"""Summarize old cold TP0 kernel shape, task duration, and active-core time."""
import glob
import json
from pathlib import Path

import pandas as pd

root = Path("/data/wio/Inference_Foundry")
files = glob.glob(str(root / "evidence/20260920_diagnostic/app_profile_cold2/raw/PROF_*/mindstudio_profiler_output/op_summary*.csv"))
if len(files) != 1:
    raise RuntimeError(files)
columns = ["Task ID", "OP Type", "Task Start Time(us)", "Task Duration(us)", "Task Wait Time(us)",
           "Input Shapes", "Input Data Types", "Block Num", "aiv_time(us)", "aicore_time(us)"]
d = pd.read_csv(files[0], usecols=columns, low_memory=False)
d = d.dropna(subset=["Task ID", "OP Type", "Task Start Time(us)", "Task Duration(us)"])
wanted = {"ScatterNdUpdateSk", "Compressor", "QuantBatchMatmulV3", "GroupedMatmulSwigluQuantV2"}
x = d[d["OP Type"].isin(wanted)].copy()
out = {"source": files[0], "rows": len(x), "operators": {}}
for name, group in x.groupby("OP Type"):
    op = {"count": len(group), "total_task_ms": float(group["Task Duration(us)"].sum() / 1000),
          "total_aiv_ms": float(group["aiv_time(us)"].sum() / 1000),
          "p50_task_us": float(group["Task Duration(us)"].median()),
          "p90_task_us": float(group["Task Duration(us)"].quantile(.9)),
          "shapes": []}
    for (shape, dtype), frame in group.groupby(["Input Shapes", "Input Data Types"], dropna=False):
        op["shapes"].append({"shape": str(shape), "dtype": str(dtype), "count": len(frame),
                             "total_task_ms": float(frame["Task Duration(us)"].sum() / 1000),
                             "p50_task_us": float(frame["Task Duration(us)"].median()),
                             "p90_task_us": float(frame["Task Duration(us)"].quantile(.9)),
                             "p50_aiv_us": float(frame["aiv_time(us)"].median()),
                             "p50_block_num": float(frame["Block Num"].median())})
    op["shapes"].sort(key=lambda item: item["total_task_ms"], reverse=True)
    out["operators"][name] = op
target = root / "evidence/20260920_loop015_cold_kernels/shape_summary.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(out, indent=2) + "\n")
for name, op in out["operators"].items():
    print(name, op["count"], round(op["total_task_ms"], 1), "ms", op["shapes"][:3])
