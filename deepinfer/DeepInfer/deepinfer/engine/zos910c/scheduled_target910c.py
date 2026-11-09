"""Bridge the native 4×6 verifier packet to original Scheduler output tensors."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DecodeSamplerInput:
    step: int
    accepted: torch.Tensor
    tokens: torch.Tensor
    drafts: torch.Tensor
    confidence: torch.Tensor
    num_accepted: torch.Tensor


def pack_decode_sample(step, logical_q, *, proposals, targets, accepted_counts, next_drafts, confidence):
    """Inputs are CPU packets from the native target, draft and greedy verifier."""
    if step.metadata.is_prompt or step.metadata.num_slots != 4:
        raise ValueError("native full target packet requires four-slot decode")
    for value, shape, dtype in (
            (proposals, (4, 5), torch.int32), (targets, (4, 6), torch.int32),
            (accepted_counts, (4,), torch.int32), (next_drafts, (4, 5), torch.int32),
            (confidence, (4, 5), torch.float32)):
        if value.device.type != "cpu" or value.shape != shape or value.dtype != dtype:
            raise ValueError("invalid native verifier packet")
    rows = step.metadata.rows
    slots = torch.tensor([r.slot for r in rows], dtype=torch.int64)
    accepted = accepted_counts.index_select(0, slots)
    flat, output = [], []
    for row, q, n in zip(rows, logical_q, accepted.tolist(), strict=True):
        if not 0 <= n < q <= 6:
            raise ValueError("native acceptance exceeds refreshed logical Q")
        prefix = proposals[row.slot, :n].tolist()
        if prefix != targets[row.slot, :n].tolist():
            raise ValueError("native accepted drafts differ from the held target")
        flat.extend(prefix)
        flat.extend([-1] * (q - n))
        output.append(targets[row.slot, n].item())
    return DecodeSamplerInput(step.number, torch.tensor(flat, dtype=torch.int32),
                              torch.tensor(output, dtype=torch.int32),
                              next_drafts.index_select(0, slots), confidence.index_select(0, slots),
                              accepted)


def physical_commits(step, committed):
    """Return physical vectors from original C++ pushes, with empty inactive slots."""
    if step.metadata.is_prompt:
        raise ValueError("prompt commit does not update a target graph ticket")
    slots = step.metadata.num_slots
    counts, emitted = [0] * slots, [()] * slots
    for row, (seq_id, tokens, count, _) in zip(step.metadata.rows, committed, strict=True):
        if seq_id != row.seq_id or len(tokens) != count or not 0 <= count <= 6:
            raise ValueError("original Scheduler commit differs from the held slot identity")
        counts[row.slot] = count
        emitted[row.slot] = tokens
    return tuple(counts), tuple(emitted)
