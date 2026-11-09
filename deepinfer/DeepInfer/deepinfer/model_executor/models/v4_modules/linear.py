"""HF linear parameter containers with FP8 layout metadata."""

from __future__ import annotations

from torch import nn


class ReplicatedLinear(nn.Linear):
    """Complete linear weights, optionally prepared as FP8 on meta."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        use_fp8_gemm: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.use_fp8_gemm = use_fp8_gemm


class ParallelRoutedLinear(nn.Linear):
    """Expert-major grouped GEMM metadata: (n_local_experts * h_in, h_out).

    HF materialization allocates packed FP4 directly. Dummy initialization uses
    this 2D representation before converting to the runtime layout.
    """

    def __init__(self, h_in: int, h_out: int, n_local_experts: int, *, use_fp8_gemm: bool = False) -> None:
        super().__init__(h_out, h_in * n_local_experts, bias=False)
        self.use_fp8_gemm = use_fp8_gemm
        self.n_local_experts = n_local_experts
        self.weight.n_experts = n_local_experts
