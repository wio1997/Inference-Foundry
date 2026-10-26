"""Request-free greedy verification for the fixed DSpark7 decode path."""

from __future__ import annotations

import torch

from .fixed_decode import AcceptanceOutput, FixedDecodeConfig, FixedDecodeState, TargetOutput
from .greedy_accept import greedy_accept


class FixedGreedyAcceptance:
    """Turn target logits into accepted tokens without sampler/request objects."""

    def __init__(self, config: FixedDecodeConfig) -> None:
        self.config = config

    def execute(
        self,
        state: FixedDecodeState,
        target: TargetOutput,
    ) -> AcceptanceOutput:
        cfg = self.config
        expected = cfg.target_token_count
        if target.logits.shape[0] != expected:
            raise ValueError("target logits token count changed")

        predicted = target.logits.argmax(dim=-1).view(
            cfg.batch_size,
            cfg.target_tokens_per_request,
        )
        accepted, counts = greedy_accept(
            state.draft_tokens,
            predicted[:, : cfg.speculative_tokens],
            predicted[:, cfg.speculative_tokens],
            output=state.accepted_tokens,
        )
        return AcceptanceOutput(accepted, counts)
