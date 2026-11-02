"""RMSNorm parameter containers for HF loading.

`convert_rmsnorm` transfers hidden_size/eps/use_weight/get_weight()/out_scale
into the runtime RMSNorm. HF supplies the effective norm weights.
"""

from __future__ import annotations

import torch
from torch import nn


class DropoutAddRMSNorm(nn.Module):
    """Keep the native parameter path and runtime norm interface, without forward."""

    def __init__(
        self,
        hidden_size: int,
        *,
        eps: float = 1e-5,
        is_exp_norm: bool = False,
        out_scale: float = 1.0,
        use_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.is_exp_norm = is_exp_norm
        self.out_scale = out_scale
        self.use_weight = use_weight
        if use_weight:
            init = torch.zeros if is_exp_norm else torch.ones
            self.weight = nn.Parameter(init(hidden_size, dtype=dtype))

    def get_weight(self) -> torch.Tensor | None:
        if not self.use_weight:
            return None
        return self.weight.exp() if self.is_exp_norm else self.weight
