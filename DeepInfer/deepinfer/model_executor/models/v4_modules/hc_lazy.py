"""HC lazy post-application bundle, shared by HC forward, YoCo compaction and the decoder layer.

Dependency-free leaf on purpose: the operator layer (`yoco`) needs to type, inspect and
rebuild this bundle without importing the HC forward path and its fused kernel stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HCLazyPostData:
    """Backbone and mixing coefficients produced by `_hc_pre`, consumed by the paired `hc_post`.

    All fields are per token and carry a batch dim of 1.
    """

    residual: torch.Tensor | None  # [1, n, hc_mult, hidden_size]
    post_layer_mix: torch.Tensor | None  # [1, n, hc_mult, 1]
    comb_res_mix: torch.Tensor | None  # [1, n, hc_mult, hc_mult]
    # [1, n, hc_mult, 1]; shifted HC only: applied by the next sub-layer or at the model end.
    pre_layer_mix: torch.Tensor | None

    def clear_post(self) -> None:
        """Release consumed post data, keeping the pre mix needed by the next shifted-HC sub-layer."""
        self.residual = None
        self.post_layer_mix = None
        self.comb_res_mix = None

    def clear(self) -> None:
        self.clear_post()
        self.pre_layer_mix = None
