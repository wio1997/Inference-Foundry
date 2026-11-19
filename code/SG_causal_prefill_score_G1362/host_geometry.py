"""Plan causal K widths for source-verified, contiguous eager Prefill rows."""

from dataclasses import dataclass

from causal_score import ScoreTile


@dataclass(frozen=True)
class RequestRows:
    row_start: int
    prefix_length: int
    row_count: int

    def score_tiles(self, *, compress_ratio, heads, key_count, budget_bytes):
        """Use the native request width to preserve query-row tiling."""
        if compress_ratio not in (1, 2) or min(heads, key_count, budget_bytes) <= 0:
            return None
        if key_count != (self.prefix_length + self.row_count) // compress_ratio:
            return None
        rows_per_tile = max(1, budget_bytes // (heads * key_count * 2))
        tiles = []
        for start in range(0, self.row_count, rows_per_tile):
            end = min(start + rows_per_tile, self.row_count)
            # Positions are [prefix_length, prefix_length + row_count).
            width = (self.prefix_length + end) // compress_ratio
            tiles.append(ScoreTile(start, end, width))
        return tuple(tiles)


def contiguous_request_rows(prefix_lengths, extend_lengths, sequence_lengths, *, row_count):
    """Validate CPU metadata supplied by the caller's ordinary eager position path.

    The caller must verify row provenance and exclude transformed position paths.
    Return None for a mismatched geometry so the native scorer remains available.
    """
    if any(lengths is None for lengths in (prefix_lengths, extend_lengths, sequence_lengths)):
        return None
    if not len(prefix_lengths) == len(extend_lengths) == len(sequence_lengths):
        return None
    result = []
    offset = 0
    for prefix, extend, sequence in zip(prefix_lengths, extend_lengths, sequence_lengths):
        if min(prefix, extend) < 0 or prefix + extend != sequence:
            return None
        result.append(RequestRows(offset, prefix, extend))
        offset += extend
    if offset != row_count:
        return None
    return tuple(result)
