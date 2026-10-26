"""Summarize no-profiler prepare-stage timings from the measured c12 request window."""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("/data/wio/Inference_Foundry/evidence/20260920_loop014_prepare_trace")
bench = json.loads((root / "run1/decode_c12.json").read_text())
requests = bench["requests"]
lo = min(r["start"] for r in requests) * 1e9
hi = max(r["end"] for r in requests) * 1e9
columns = ["start", "states", "inputs", "mamba", "attn", "preprocess", "end",
           "num_reqs", "num_tokens", "with_prefill"]
names = [
    ("state_update", "start", "states"),
    ("input_assembly", "states", "inputs"),
    ("dispatch_mamba", "inputs", "mamba"),
    ("compress_attention", "mamba", "attn"),
    ("preprocess", "attn", "preprocess"),
    ("cos_final", "preprocess", "end"),
    ("total_prepare", "start", "end"),
]
out = {"bench_summary": bench["summary"], "monotonic_window_ns": [lo, hi],
       "filter": "start within measured request window and with_prefill=0", "ranks": {}}
for path in sorted(glob.glob(str(root / "raw/rank*_pid*.csv"))):
    rank = int(Path(path).name.split("_")[0][4:])
    d = pd.read_csv(path, names=columns)
    all_window = d[(d.start >= lo) & (d.start < hi)]
    x = all_window[all_window.with_prefill == 0]
    result = {"all_window_steps": len(all_window), "decode_steps": len(x), "stages": {}}
    for name, left, right in names:
        values = (x[right] - x[left]).to_numpy(dtype=np.float64) / 1e6
        if len(values):
            result["stages"][name] = {
                "sum_ms": float(values.sum()),
                "mean_ms": float(values.mean()),
                "median_ms": float(np.median(values)),
                "p90_ms": float(np.percentile(values, 90)),
                "max_ms": float(values.max()),
            }
    out["ranks"][str(rank)] = result
(root / "run1/stage_summary.json").write_text(json.dumps(out, indent=2) + "\n")
for rank, value in out["ranks"].items():
    print(rank, value["decode_steps"], {k: round(v["median_ms"], 3) for k, v in value["stages"].items()})
