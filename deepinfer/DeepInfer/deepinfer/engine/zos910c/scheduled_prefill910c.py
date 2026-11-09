"""Project original Scheduler prompt steps into the native fixed slot model."""

from dataclasses import dataclass


def initialize_prompt_cursor(sequence, *, position, chunk_size, prefix_points):
    """Called for a fresh Sequence before add_sequence, after native restore."""
    end = sequence.get_prompt_len()
    if not 0 <= position < end or sequence.num_cached_tokens != 0 or chunk_size < 1:
        raise ValueError("invalid fresh prompt cursor")

    # Match the native chunk ceiling and cold prefix capture boundaries.
    # Scheduler still decides the batch and is free to cut an earlier chunk.
    checkpoints = []
    cursor = position
    for boundary in (*(p for p in prefix_points if position == 0 and 0 < p < end), end):
        if boundary <= cursor:
            raise ValueError("prefix points must be sorted and distinct")
        checkpoints.extend(range(cursor + chunk_size, boundary, chunk_size))
        if boundary < end:
            checkpoints.append(boundary)
        cursor = boundary

    sequence.chunk_checkpoints = checkpoints
    sequence.num_cached_tokens = position
    sequence.cached_hit_tokens = position


@dataclass(frozen=True)
class PromptChunk:
    seq_id: int
    slot: int
    start: int
    end: int
    tokens: tuple[int, ...]
    is_last: bool


def prompt_chunks(step, sequences, positions, *, chunk_size):
    """Use the original Q/K ranges in logical output order."""
    if not step.is_prompt:
        raise ValueError("a decode step cannot run as native prefill")
    chunks = []
    for row in step.rows:
        sequence = sequences[row.seq_id]
        end = row.scheduled_k
        start = end - row.q
        length = sequence.get_prompt_len()
        if (positions[row.slot] != start or sequence.num_cached_tokens != start
                or not 0 <= start < end <= length or row.q > chunk_size):
            raise ValueError("native cursor or chunk capacity differs from Scheduler")
        tokens = tuple(sequence.data.get_token_ids()[start:end])
        if len(tokens) != row.q:
            raise ValueError("Scheduler prompt range exceeds its token sequence")
        chunks.append(PromptChunk(row.seq_id, row.slot, start, end, tokens, end == length))
    return tuple(chunks)
