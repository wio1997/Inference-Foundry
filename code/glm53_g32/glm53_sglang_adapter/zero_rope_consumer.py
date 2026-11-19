"""Candidate CANN9.1 NoPE consumer bridge. Initialize before capture; unverified on NPU."""
import torch
import torch_npu


class ZeroRopeSparseFA:
    """Preserve logits: [q,0]@[k,0]^T == q@k^T, with unchanged scale.

    One immutable zero KV tensor may be shared by all equal-geometry layers on
    one worker. Never write it through pool writers or transfer/register it as
    model cache. Query and index scratch belong to one non-overlapping call lane.
    """
    def __init__(self, key_template, *, max_query_tokens, query_heads=8,
                 rope_dim=64, sparse_width=2112):
        assert key_template.ndim == 4 and key_template.shape[-2:] == (1, 512)
        assert key_template.dtype == torch.bfloat16 and key_template.device.type == 'npu'
        assert rope_dim in [0, 64] and sparse_width in [2048, 2112]
        assert max_query_tokens > 0
        self.query_zeros = key_template.new_zeros((max_query_tokens, query_heads, rope_dim))
        self.key_zeros = key_template.new_zeros((*key_template.shape[:-1], rope_dim))
        self.index_scratch = torch.full((max_query_tokens, 1, sparse_width), -1,
                                        dtype=torch.int32, device=key_template.device)
        self.rope_dim = rope_dim
        self.sparse_width = sparse_width

    def __call__(self, *, query, key, value, sparse_indices, scale_value,
                 query_rope=None, key_rope=None, **kwargs):
        assert query_rope is None or query_rope.shape[-1] == 0, 'Real RoPE must never enter the NoPE bridge'
        assert key_rope is None or key_rope.shape[-1] == 0
        assert query.shape[1:] == self.query_zeros.shape[1:-1] + (512,)
        assert query.dtype == key.dtype == value.dtype == self.query_zeros.dtype
        assert key.shape == value.shape and key.shape[1:-1] == self.key_zeros.shape[1:-1]
        assert key.shape[0] <= self.key_zeros.shape[0]
        assert query.shape[0] <= self.query_zeros.shape[0]
        assert kwargs.get('layout_query') == 'TND' and kwargs.get('layout_kv') == 'PA_BSND'
        assert kwargs.get('attention_mode') == 2 and kwargs.get('sparse_block_size') == 1
        if sparse_indices.ndim == 2:
            sparse_indices = sparse_indices.unsqueeze(1)
        assert sparse_indices.dtype == torch.int32 and sparse_indices.shape[1] == 1
        assert sparse_indices.shape[0] == query.shape[0]
        assert sparse_indices.shape[2] <= self.sparse_width, 'Never truncate valid KPool tail indices'
        indices = self.index_scratch[:query.shape[0]]
        indices.fill_(-1)
        indices[:, :, :sparse_indices.shape[2]].copy_(sparse_indices)
        return torch_npu.npu_sparse_flash_attention(
            query=query, key=key, value=value, sparse_indices=indices,
            query_rope=self.query_zeros[:query.shape[0]],
            key_rope=self.key_zeros[:key.shape[0]], scale_value=scale_value, **kwargs)

    def allocation_metadata(self):
        return {'rope_dim': self.rope_dim, 'sparse_width': self.sparse_width,
                'query_zero_bytes': self.query_zeros.numel() * self.query_zeros.element_size(),
                'key_zero_bytes': self.key_zeros.numel() * self.key_zeros.element_size(),
                'index_scratch_bytes': self.index_scratch.numel() * self.index_scratch.element_size(),
                'extra_key_bytes_per_physical_token_slot': self.rope_dim * 2,
                'extra_dot_dimensions_over_512': self.rope_dim / 512,
                'scope': 'Arithmetic/work upper-accounting only, not measured latency or bandwidth'}
