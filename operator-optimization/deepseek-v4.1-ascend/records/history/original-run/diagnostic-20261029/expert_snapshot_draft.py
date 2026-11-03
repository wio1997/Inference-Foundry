"""CPU-only contract for a future MC2 expert-count diagnostic.

This is deliberately NOT an instrumentation backend. It never imports torch,
opens data files, interacts with a device, or produces sampled observations.
No manifest or boolean can waive the missing device/lifetime proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


class SamplingUnavailable(RuntimeError):
    """Sampling must not proceed with the currently unverified adapter."""


@dataclass(frozen=True)
class CallIdentity:
    """Fields that the future real adapter must bind to one executed call."""

    process_id: int
    device_index: int
    global_rank: int
    mc2_group_token: str  # nonsecret anonymized token, not communicator name
    mc2_rank: int
    mc2_world_size: int
    model_instance: str  # main model vs draft, verified rather than inferred
    layer_index: int
    call_sequence: int
    execution_mode: str  # eager, capture, replay, or unknown
    graph_token: str | None
    replay_sequence: int | None
    resolved_max_output_size: int | None
    source_sha256: str
    binary_sha256: str | None


def request_device_snapshot(_identity: CallIdentity) -> None:
    """Fail closed until a reviewed device adapter supplies actual proofs.

    A false is_current_stream_capturing result by itself does not prove eager
    execution or the absence of graph replay; do not relax this refusal.
    """
    raise SamplingUnavailable(
        "No NPU adapter is implemented: actual call identity, execution stream, "
        "buffer reuse, graph/replay identity, and completion are unverified."
    )


def calculate_effective_rows(counts: Sequence[int], capacity: int) -> dict:
    """Calculate the source-conditional clipping rule on caller-supplied values.

    This function does not certify where values came from or that a binary
    implements the frozen source. Callers must prove those separately.
    """
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if not counts or any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("counts must be a nonempty sequence of nonnegative integers")
    used = 0
    effective = []
    for count in counts:
        clipped = min(count, max(capacity - used, 0))
        effective.append(clipped)
        used += clipped
    assert used == min(sum(counts), capacity)
    return {"effective_per_expert": effective, "effective_total": used,
            "before_capacity": sum(counts), "truncated_total": sum(counts) - used}
