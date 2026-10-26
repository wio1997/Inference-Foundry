#!/usr/bin/env python3
"""Correct Loop039 source branch mapping for the frozen W4A8 checkpoint."""
import glob
import json
import struct
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
MODEL = Path("/data/yxy/DeepSeek-V4-Flash-0731-w4a8")
SOURCE = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend")
q = json.loads((MODEL / "quant_model_description.json").read_text())
assert q["group_size"] == 0
assert q["layers.0.ffn.experts.0.w1.weight"] == "W4A8_DYNAMIC"
w4 = (SOURCE / "quantization/methods/w4a8.py").read_text()
stage = (SOURCE / "ops/fused_moe/moe_stage_params.py").read_text()
mlp = (SOURCE / "ops/fused_moe/moe_mlp.py").read_text()
assert "self.is_per_channel_weight = self.group_size == 0" in w4
assert "return self.quant_type == QuantType.W4A8 and self.is_per_channel_weight" in stage
assert "use_w4a8_per_channel_gmm_swiglu" in mlp
assert "torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2(" in mlp
assert "DeviceOperator.npu_grouped_matmul_gmm2(" in mlp
weight_shapes = {}
for path in sorted(MODEL.glob("*.safetensors")):
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    for name in ("w1", "w2", "w3"):
        key = f"layers.0.ffn.experts.0.{name}.weight"
        if key in header:
            weight_shapes[name] = header[key]["shape"]
    if len(weight_shapes) == 3:
        break
assert weight_shapes == {
    "w1": [1024, 4096], "w2": [2048, 2048],
    "w3": [1024, 4096],
}
out = {
    "run": "run109",
    "corrects": "Run108 source_gmm1 claim; its 43+43 kernel counts remain valid",
    "quant_group_size": q["group_size"],
    "per_channel_weight": True,
    "checkpoint_layer0_expert0_packed_i8_shapes": weight_shapes,
    "gmm1_active_branch": "moe_mlp.py per-channel W4A8 custom "
                          "torch.ops._C_ascend.grouped_matmul_swiglu_quant_v2",
    "gmm2_active_branch": "moe_mlp.py DeviceOperator.npu_grouped_matmul_gmm2 "
                          "to torch_npu.npu_grouped_matmul",
    "limits": "Checkpoint weights and branch predicates are verified. "
              "Live routed token counts, operator input shapes, graph "
              "addresses, and a faster parity-preserving replacement "
              "are not yet established.",
}
path = ROOT / "evidence/20260924_loop039_gmm/run109/active_branch.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
