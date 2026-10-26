"""Direct target-model adapter for the fixed DeepSeek decode runtime.

The adapter knows nothing about vLLM scheduler or request types. Its backend
callable is installed once after weights, TP groups, attention metadata, and
KV tensors have been bound. Every cycle thereafter consumes runtime-owned
fixed buffers and returns only tensors needed by verification and DSpark.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .fixed_decode import FixedDecodeConfig, FixedDecodeState, TargetOutput


@dataclass(frozen=True)
class FixedTargetBinding:
    """One-time binding of real target operators to the fixed runtime ABI."""

    forward: Callable[
        [torch.Tensor, torch.Tensor],
        torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
    ]
    compute_logits: Callable[[torch.Tensor], torch.Tensor]
    state_fingerprint: Callable[[], dict[str, torch.Tensor]]


class FixedTargetAdapter:
    """Execute target verification directly from fixed input buffers.

    The binding enters the already-frozen Ascend forward context and exposes
    target KV tensors to attention operators. Those objects are setup-time
    state and are not rebuilt per cycle.
    """

    def __init__(
        self,
        config: FixedDecodeConfig,
        binding: FixedTargetBinding,
    ) -> None:
        self.config = config
        self.binding = binding
        self._input_ptr: int | None = None
        self._position_ptr: int | None = None

    def execute(self, state: FixedDecodeState) -> TargetOutput:
        cfg = self.config
        if state.target_input_ids.numel() != cfg.target_token_count:
            raise ValueError("target input token count changed")
        if state.target_positions.numel() != cfg.target_token_count:
            raise ValueError("target position count changed")

        input_ptr = state.target_input_ids.data_ptr()
        position_ptr = state.target_positions.data_ptr()
        if self._input_ptr is None:
            self._input_ptr = input_ptr
            self._position_ptr = position_ptr
        elif input_ptr != self._input_ptr or position_ptr != self._position_ptr:
            raise RuntimeError("fixed target input buffer address changed")

        model_output = self.binding.forward(
            state.target_input_ids,
            state.target_positions,
        )
        if isinstance(model_output, tuple):
            hidden_states, aux_hidden_states = model_output
        else:
            hidden_states = model_output
            aux_hidden_states = []

        if hidden_states.shape[0] != cfg.target_token_count:
            raise ValueError("target hidden-state token count changed")
        sample_hidden = hidden_states[state.target_logits_indices]
        logits = self.binding.compute_logits(sample_hidden)
        return TargetOutput(
            logits=logits,
            hidden_states=hidden_states,
            aux_hidden_states=tuple(aux_hidden_states),
        )

    def state_fingerprint(self) -> dict[str, torch.Tensor]:
        """Return backend-selected KV/cache fingerprints without host reads."""

        return self.binding.state_fingerprint()
