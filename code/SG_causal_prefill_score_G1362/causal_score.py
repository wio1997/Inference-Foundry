"""Test-only causal suffix trimming for the native 128 MiB Prefill scorer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreTile:
    start: int
    end: int
    key_width: int


def score_tiles(lens, *, heads, key_count, budget_bytes):
    """Plan from authoritative host lengths in the order of the native request rows."""
    if not (heads > 0 and key_count > 0 and budget_bytes > 0):
        raise ValueError('Positive head count, key count and budget are required')
    if not all(0 <= length <= key_count for length in lens):
        raise ValueError('Visibility length is outside the request key table')
    rows_per_tile = max(1, budget_bytes // (heads * key_count * 2))
    tiles = []
    for start in range(0, len(lens), rows_per_tile):
        end = min(start + rows_per_tile, len(lens))
        tiles.append(ScoreTile(start, end, max(lens[start:end])))
    return tuple(tiles)


def causal_chunks(indexer, q, weights, index_k, request, tiles, chunk_type, *, output_dtype):
    """Keep the native masking and full top-k width while trimming GEMM columns."""
    for tile in tiles:
        rows = slice(tile.start, tile.end)
        tok, lens = request.tok[rows], request.lens[rows]
        width = tile.key_width
        if width == 0:
            # The native scorer always returns FP32. Every column is masked here.
            scores = weights.new_full((tile.end - tile.start, request.lc), float('-inf'),
                                      dtype=output_dtype)
        else:
            scores = indexer.scores(q[tok], index_k[:width], weights[tok])
            scores = scores.masked_fill(request.columns[None, :width] >= lens[:, None], float('-inf'))
            if width < request.lc:
                full = scores.new_full((tile.end - tile.start, request.lc), float('-inf'))
                full[:, :width].copy_(scores)
                scores = full
        yield chunk_type(rows=rows, tok=tok, lens=lens, scores=scores)
