# SPDX-License-Identifier: Apache-2.0
"""Validate one scheduler sequence against an existing GLM request owner.

This is a metadata-only integration primitive, not a model registry, scheduler,
cache allocator, or claim of end-to-end support. The caller must coordinate
validation failures across participating ranks before entering collectives.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScheduledTextSpan:
    request_id: str
    seq_id: int
    token_start: int
    token_stop: int
    is_prompt: bool

    @property
    def token_count(self) -> int:
        return self.token_stop - self.token_start


def scheduled_text_span(metadata, *, request_id: str, seq_id: int,
                        tokens_seen: int, max_context_tokens: int) -> ScheduledTextSpan:
    """Map a real SequenceMetadata span onto a live private request position.

    DeepInfer's input preparation uses [seqlen_k - seqlen_q, seqlen_k).
    This bounded primitive accepts one text sequence, ordinary one-token decode,
    and an already established request/sequence identity. It performs no token
    extraction and no state mutation; callers remain responsible for actual
    scheduler lifecycle, batching, distributed agreement and cache ownership.
    """
    if not isinstance(request_id, str) or not request_id:
        raise ValueError('nonempty request identity required')
    if type(seq_id) is not int or seq_id < 0:
        raise ValueError('nonnegative sequence identity required')
    if type(tokens_seen) is not int or tokens_seen < 0:
        raise ValueError('nonnegative committed token position required')
    if type(max_context_tokens) is not int or max_context_tokens < 1:
        raise ValueError('positive context bound required')
    if metadata.request_id != request_id or metadata.seq_id != seq_id:
        raise ValueError('scheduler/request identity mismatch')
    if metadata.is_placeholder:
        raise ValueError('placeholder sequence is not executable')
    if metadata.multi_modal_data is not None:
        raise ValueError('multimodal scheduler input is not supported')
    if type(metadata.is_prompt) is not bool:
        raise ValueError('explicit prompt/decode mode required')
    q, k, full_len = metadata.seqlen_q, metadata.seqlen_k, metadata.full_len
    if any(type(x) is not int for x in (q, k, full_len)):
        raise ValueError('scheduler lengths must be integers')
    if q < 1 or k < q or k > full_len or k > max_context_tokens:
        raise ValueError('scheduler lengths exceed the valid token/context span')
    if not metadata.is_prompt and q != 1:
        raise ValueError('only single-token decode is supported')
    start = k - q
    if start != tokens_seen:
        raise ValueError('scheduler position differs from committed request state')
    return ScheduledTextSpan(request_id, seq_id, start, k, metadata.is_prompt)
