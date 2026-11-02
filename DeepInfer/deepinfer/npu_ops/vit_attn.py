import math
from typing import Any

import torch
import torch_npu


# TODO(NPU): To be removed when `_flash_attn_fwd` for ViT is ready
def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: list[int] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """ViT 用的 varlen attention：非 causal，每段内做全量注意力。

    cu_seqlens_q: host list（InputMetadataVL.vit_attn_cu_seqlens，每段累计末端）。
    """
    result = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        q.shape[-2],
        "TND",
        scale=1.0 / math.sqrt(q.shape[-1]),
        actual_seq_qlen=cu_seqlens_q,
        actual_seq_kvlen=cu_seqlens_q,
        sparse_mode=0,
    )[0]
    return (result, None)
