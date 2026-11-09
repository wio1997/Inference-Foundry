import os

from deepinfer.npu_ops.patch_imports import patch_imports

patches = {
    "deepinfer.npu_ops.deep_select": {"deep_select.interface": "topk"},
    "deepinfer.npu_ops.vit_attn": {"flash_attn.cute.interface": "_flash_attn_fwd"},
}
if os.getenv("DEEPINFER_ASCEND_910C_REFERENCE", "0") == "1":
    patches["deepinfer.npu_ops.compat910c"] = {
        "deepinfer.model_executor.layers.tilelang_ops.norm": "norm",
    }
patch_imports(patches)
