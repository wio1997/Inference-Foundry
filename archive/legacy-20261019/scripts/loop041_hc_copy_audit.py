#!/usr/bin/env python3
"""Offline, bounded HC/copy/cache kernel census for Run107 valid target scopes."""
import glob
import json
import statistics
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
prior = json.loads((root / "evidence/20260924_loop038_cycle/run107/target_window.json").read_text())
excluded = {(x["rank"], x["cycle"]) for x in prior["excluded_windows"]}
family = {
    "hc_pre": lambda n: n == "HcPre",
    "hc_post": lambda n: n == "HcPost",
    "rms_norm": lambda n: n == "RmsNorm",
    "triton_rms": lambda n: n == "triton_rms_kernel",
    "rms_dynamic_quant": lambda n: n == "RmsNormDynamicQuant",
    "compressor": lambda n: n == "Compressor",
    "cache_scatter_sk": lambda n: n.startswith("aclnnScatterNdUpdateSk_"),
    "copy_cast": lambda n: n.startswith("aclnnInplaceCopy_Cast"),
    "copy_transpose": lambda n: n.startswith("aclnnInplaceCopy_Transpose"),
    "copy_tensor_move": lambda n: n.startswith("aclnnInplaceCopy_TensorMove"),
    "memcpy_async": lambda n: n == "MEMCPY_ASYNC",
}
rows = []
for rank in range(8):
    paths = glob.glob(str(root / f"evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    assert len(paths) == 1, (rank, paths)
    events = json.loads(Path(paths[0]).read_text())
    scopes = sorted((e for e in events if e.get("cat") == "cpu_op" and e.get("name") == "extreme::target"), key=lambda e: float(e["ts"]))
    assert len(scopes) == 2, (rank, len(scopes))
    for cycle, scope in zip((64, 65), scopes):
        if (rank, cycle) in excluded:
            continue
        start = float(scope["ts"])
        end = start + float(scope["dur"])
        active = [e for e in events if start <= float(e.get("ts", -1)) < end]
        devices = [e for e in active if e.get("args", {}).get("Task Type", "").startswith(("KERNEL_", "SDMA_"))]
        row = {"rank": rank, "cycle": cycle, "device_kernel_count": len(devices)}
        for key, pred in family.items():
            selected = [e for e in devices if pred(e.get("name", ""))]
            row[key] = {"count": len(selected), "kernel_sum_ms": sum(float(e["dur"]) for e in selected) / 1000}
        row["cpu_clone_count"] = sum(e.get("cat") == "cpu_op" and e.get("name") == "aten::clone" for e in active)
        row["cpu_copy_count"] = sum(e.get("cat") == "cpu_op" and e.get("name") in ("aten::copy_", "aclnnInplaceCopy") for e in active)
        rows.append(row)
assert len(rows) == 15
assert all(r["device_kernel_count"] >= 2600 for r in rows)
summary = {}
for key in family:
    counts = [r[key]["count"] for r in rows]
    sums = [r[key]["kernel_sum_ms"] for r in rows]
    summary[key] = {"count_median": statistics.median(counts), "count_range": [min(counts), max(counts)],
                    "kernel_sum_ms_median": statistics.median(sums), "kernel_sum_ms_range": [min(sums), max(sums)]}
for key in ("cpu_clone_count", "cpu_copy_count"):
    vals = [r[key] for r in rows]
    summary[key] = {"median": statistics.median(vals), "range": [min(vals), max(vals)]}
output = {"run": "run134", "source": "Run107 cycle-synchronized trace, 15 valid target rank-cycle scopes",
          "caveat": "Kernel duration sums can overlap and are not removable wall time; profiler/forced sync distort duration. CPU aten::clone counts during graph replay do not count graph-build clones. Device copy-kernel names do not identify the originating source clone.",
          "source_findings": [
              "deepseek_v4.py decoder forward clones hidden_states twice per layer around two hc_pre calls, then separate RMSNorm and HC post; graph replay can elide source-level clones.",
              "hc_pre_inv_rms custom op takes only x and epsilon and returns y; it cannot replace HC pre with its hc_fn/scale/base inputs and three outputs.",
              "device_op.py stores cache with npu_scatter_nd_update_sk; attention/sfa_v1.py also has scatter update paths. Kernel count alone does not prove redundant writes."
          ], "summary": summary, "windows": rows}
path = root / "evidence/20260925_loop041_hc_copy/run134/audit.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(summary, indent=2))
