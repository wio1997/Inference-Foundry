"""DeepSeek Extreme fixed-product runtime."""

from .fixed_decode import (
    AcceptanceOutput,
    FixedDecodeConfig,
    FixedDecodeRuntime,
    FixedDecodeState,
    TargetOutput,
)
from .greedy_accept import greedy_accept
from .target_adapter import FixedTargetAdapter, FixedTargetBinding

__all__ = [
    "AcceptanceOutput",
    "FixedDecodeConfig",
    "FixedDecodeRuntime",
    "FixedDecodeState",
    "TargetOutput",
    "greedy_accept",
    "FixedTargetAdapter",
    "FixedTargetBinding",
]
