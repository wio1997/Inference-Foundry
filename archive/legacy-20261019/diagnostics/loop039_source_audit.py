#!/usr/bin/env python3
"""Verify the Loop039 MoE grouped-matmul source and trace mapping."""
import csv
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
MODEL = Path("/data/yxy/DeepSeek-V4-Flash-0731-w4a8/config.json")
SOURCE = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend")
CONFIG = json.loads(MODEL.read_text())
assert CONFIG["num_hidden_layers"] == 43
device_op = (SOURCE / "device/device_op.py").read_text()
moe_mlp = (SOURCE / "ops/fused_moe/moe_mlp.py").read_text()
assert "grouped_matmul_swiglu_quant_weight_nz(" in device_op
assert "npu_grouped_matmul_gmm2(" in device_op
assert "DeviceOperator.npu_grouped_matmul_swiglu_quant(" in moe_mlp
assert "DeviceOperator.npu_grouped_matmul_gmm2(" in moe_mlp
profile = ROOT / "evidence/20260924_loop038_cycle/run107/profile"
ranks = []
for rank in range(8):
    paths = list(profile.glob(
        f"rank{rank}_*ascend_pt/ASCEND_PROFILER_OUTPUT"))
    assert len(paths) == 1
    p = paths[0]
    events = json.loads((p / "trace_view.json").read_text())
    scopes = sorted((float(e["ts"]), float(e["ts"]) + float(e["dur"]))
                    for e in events if e.get("ph") == "X"
                    and e.get("name") == "extreme::target")
    assert len(scopes) == 2
    with (p / "kernel_details.csv").open(newline="") as handle:
        kernels = list(csv.DictReader(handle))
    start, end = scopes[0]
    chosen = [row for row in kernels
              if float(row["Start Time(us)"].strip()) < end
              and float(row["Start Time(us)"].strip()) +
                  float(row["Duration(us)"]) > start]
    gmm1 = [row for row in chosen
            if "GroupedMatmulSwigluQuantWeightNzV2" in row["Name"]]
    gmm2 = [row for row in chosen
            if "GroupedMatmulWeightNz" in row["Name"]]
    assert len(gmm1) == len(gmm2) == 43, (rank, len(gmm1), len(gmm2))
    ranks.append({
        "rank": rank,
        "gmm1_count": len(gmm1),
        "gmm1_sum_ms": sum(float(row["Duration(us)"])
                           for row in gmm1) / 1000,
        "gmm2_count": len(gmm2),
        "gmm2_sum_ms": sum(float(row["Duration(us)"])
                           for row in gmm2) / 1000,
    })
out = {
    "run": "run108",
    "model_hidden_layers": CONFIG["num_hidden_layers"],
    "model_hidden_size": CONFIG["hidden_size"],
    "model_moe_intermediate_size": CONFIG["moe_intermediate_size"],
    "model_routed_experts": CONFIG["n_routed_experts"],
    "model_experts_per_token": CONFIG["num_experts_per_tok"],
    "source_gmm1": str(SOURCE / "device/device_op.py"),
    "source_gmm2": str(SOURCE / "device/device_op.py"),
    "source_caller": str(SOURCE / "ops/fused_moe/moe_mlp.py"),
    "inference": "The one GMM1 and one GMM2 kernel per configured layer "
                 "match the 43-layer source path; exact runtime tensor "
                 "shapes and alternative operator parity remain unmeasured.",
    "ranks": ranks,
}
path = ROOT / "evidence/20260924_loop039_gmm/run108/source_audit.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
