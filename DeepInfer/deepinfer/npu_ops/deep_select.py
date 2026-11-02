from typing import Any

import torch
from deep_select.interface import topk as _topk


def _torch_topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    indices_type: torch.dtype = torch.int64,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    data, index = torch.topk(input, topk, dim=-1, sorted=sorted)
    if index.dtype is not indices_type:
        index = index.to(indices_type)
    return data, index


# TODO(NPU): To be removed when `topk` for sampler is ready
def topk(*args: Any, **kwargs: Any) -> tuple[torch.Tensor | None, torch.Tensor]:
    if kwargs.get("return_value", True):
        # Sampler is not supported yet, fallback to torch.
        return _torch_topk(*args, **kwargs)
    # Indexer use real topk kernel.
    return _topk(*args, **kwargs)
