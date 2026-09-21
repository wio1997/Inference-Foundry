"""Greedy DSpark verification for the frozen temperature=0 product path."""

from __future__ import annotations

import torch


def greedy_accept(
    draft_tokens: torch.Tensor,
    target_argmax: torch.Tensor,
    bonus_tokens: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept the common prefix and emit recovery/bonus tokens.

    Args:
        draft_tokens: `[B, K]` target-vocabulary DSpark tokens.
        target_argmax: `[B, K]` target predictions for those draft positions.
        bonus_tokens: `[B]` target prediction after an entirely accepted block.
        output: optional preallocated `[B, K + 1]` integer tensor.

    Returns:
        Fixed-width sampled tokens padded with `-1`, and `[B]` int32 counts.
    """

    if draft_tokens.shape != target_argmax.shape or draft_tokens.ndim != 2:
        raise ValueError("draft_tokens and target_argmax must have equal [B,K] shape")
    batch, width = draft_tokens.shape
    if tuple(bonus_tokens.shape) != (batch,):
        raise ValueError("bonus_tokens must have shape [B]")
    if output is None:
        output = torch.empty(
            (batch, width + 1),
            dtype=draft_tokens.dtype,
            device=draft_tokens.device,
        )
    if tuple(output.shape) != (batch, width + 1):
        raise ValueError("output shape mismatch")

    output.fill_(-1)
    leading_match = torch.cumprod(
        draft_tokens.eq(target_argmax).to(torch.int32), dim=1
    ).to(torch.bool)
    accepted_drafts = leading_match.sum(dim=1).to(torch.int64)
    positions = torch.arange(width, device=draft_tokens.device).unsqueeze(0)
    copy_mask = positions < accepted_drafts.unsqueeze(1)
    output[:, :width].copy_(torch.where(copy_mask, draft_tokens, -1))

    all_accepted = accepted_drafts.eq(width)
    recovery_index = accepted_drafts.clamp_max(width - 1)
    recovery = torch.gather(target_argmax, 1, recovery_index.unsqueeze(1)).squeeze(1)
    final_token = torch.where(all_accepted, bonus_tokens, recovery).to(output.dtype)
    output.scatter_(1, accepted_drafts.unsqueeze(1), final_token.unsqueeze(1))
    return output, (accepted_drafts + 1).to(torch.int32)
