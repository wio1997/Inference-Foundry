from deepinfer.npu_ops.patch_imports import patch_imports

patch_imports(
    {
        "deepinfer.npu_ops.deep_select": {
            "deep_select.interface": "topk",
        },
        "deepinfer.npu_ops.vit_attn": {
            "flash_attn.cute.interface": "_flash_attn_fwd",
        },
    }
)
