"""CUDA outlier detection and PyTorch reference."""

import torch

from deepinfer.model_executor.layers.tilelang_ops import ops

MAX_ABS = 1e11


def detect_outlier_and_abort(data: torch.Tensor, id: int = 0) -> None:
    """Queue a CUDA assertion for NaN, infinity, or values outside (-MAX_ABS, MAX_ABS).

    The check runs asynchronously. A failed assertion surfaces on a subsequent
    CUDA operation or synchronization and leaves the CUDA context unusable
    until the process exits. Failures print the id, fp32 value, and element
    offset in the flattened tensor.

    Args:
        data: Any shape, fp16/bf16/fp32 CUDA tensor. Noncontiguous inputs are
            copied before checking; comparisons use fp32.
        id: Runtime int64 identifier included in failure diagnostics.
    """
    if data.numel() == 0:
        return
    flat_data = data.contiguous().view(-1)
    ops.detect_outlier_and_abort_tl(data.dtype, MAX_ABS)(flat_data, id)
