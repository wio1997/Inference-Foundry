from __future__ import annotations
def scores(self, q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """q [t, H, d], k [n, d], weights [t, H] -> [t, n], summed over all heads."""
    s = torch.einsum('bhd,nd->bhn', q, k)
    s = (s.relu() * weights.unsqueeze(-1)).sum(dim=1)
    return s.float()

class RequestScores(msgspec.Struct, frozen=True):
    lc: int
    k: int
    columns: torch.Tensor
    tok: torch.Tensor
    lens: torch.Tensor
    slots: torch.Tensor

class ChunkScores(msgspec.Struct, frozen=True):
    rows: slice
    tok: torch.Tensor
    lens: torch.Tensor
    scores: torch.Tensor

def _score_chunks(*, indexer: DeepseekV41Indexer, q: torch.Tensor, weights: torch.Tensor, index_k: torch.Tensor, request: RequestScores) -> Iterator[ChunkScores]:
    lc, j = (request.lc, request.columns)
    rows_per_chunk = max(1, _TORCH_SCORE_BUDGET_BYTES // (q.shape[1] * index_k.shape[0] * 2))
    for start in range(0, request.tok.numel(), rows_per_chunk):
        rows = slice(start, start + rows_per_chunk)
        tok_c, lens_c = (request.tok[rows], request.lens[rows])
        s = indexer.scores(q[tok_c], index_k, weights[tok_c])[:, :lc]
        s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)
        yield ChunkScores(rows=rows, tok=tok_c, lens=lens_c, scores=s)
