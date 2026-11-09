# SPDX-License-Identifier: Apache-2.0
"""Resident W8A8 index projections for native 910C paged DSA.

Source layers own K projection/norm; later index sources reuse these source
keys. The indexer sees the unrotated compressor latent. Query projection
consumes q_a after its normalization. All index heads remain replicated.
"""
import os
import torch

from deepinfer.npu_ops.dsa910c import quantize_query, select
from .w8a8_math910c import rms_norm_mixed, rope_frequencies, device_rope_frequencies, rotate_positions


class AscendW8A8Indexer(torch.nn.Module):
    def __init__(self, reader, device, *, layer_id):
        super().__init__()
        self.config = reader.config['text_config']
        c = self.config
        if layer_id not in c['index_source_layer_ids']:
            raise ValueError('only index-source layers own query projections')
        self.fp4_index = os.environ.get('DEEPINFER_910C_OFFICIAL_FP4_INDEX','0') == '1'
        if self.fp4_index:
            from deepinfer.npu_ops.fp4_index910c import prepare_runtime
            prepare_runtime(device)
        self.layer_id = layer_id
        self.ratio = c['compress_ratios'][layer_id]
        self.owns_key = layer_id in c['kv_source_layer_ids']
        prefix = f'layers.{layer_id}.attn.indexer'
        self.query_projection = reader.load_linear(prefix+'.wq_b', device)
        self.weights_projection = reader.load_linear(prefix+'.weights_proj', device)
        self.register_buffer('frequencies', device_rope_frequencies(c,device,compressed=True))
        if self.owns_key:
            self.key_projection = reader.load_linear(prefix+'.wk', device)
            self.register_buffer('key_norm', reader.tensor(prefix+'.k_norm.weight')[0].to(device))

    def query_values(self, normalized_q_a, hidden, positions, *, projected_query=None):
        c = self.config
        if projected_query is None:
            projected = self.query_projection(normalized_q_a)
        else:
            if (projected_query.shape != (normalized_q_a.shape[0],c['index_n_heads']*c['index_head_dim'])
                    or projected_query.dtype != normalized_q_a.dtype or projected_query.device != normalized_q_a.device):
                raise ValueError('preprojected index query must preserve original shape, dtype and device')
            projected = projected_query
        projected = projected.unflatten(-1, (c['index_n_heads'], c['index_head_dim']))
        query = rotate_positions(projected, positions, self.frequencies)
        weights = (self.weights_projection(hidden).float()
                   * (c['index_head_dim']*c['index_n_heads'])**-.5)
        weights = weights.bfloat16() if self.fp4_index else weights.half()
        return query, weights

    def query(self, normalized_q_a, hidden, positions, *, projected_query=None):
        query, weights = self.query_values(normalized_q_a, hidden, positions, projected_query=projected_query)
        if self.fp4_index:
            from deepinfer.npu_ops.fp4_index910c import quantize_packed
            codes, scales = quantize_packed(query)
        else:
            codes, scales = quantize_query(query)
        return codes, scales, weights

    def key_values(self, unrotated_latent, source_positions):
        if not self.owns_key:
            raise ValueError('consumer indexers must use source-owned keys')
        projected = self.key_projection(unrotated_latent)
        normalized = rms_norm_mixed(projected, self.key_norm, self.config['rms_norm_eps'])[0]
        return rotate_positions(normalized, source_positions, self.frequencies)

    def key(self, unrotated_latent, source_positions):
        import torch_npu
        key = self.key_values(unrotated_latent, source_positions)
        if self.fp4_index:
            from deepinfer.npu_ops.fp4_index910c import quantize_packed
            codes, scale = quantize_packed(key)
            return codes[:, None], scale
        codes, scale = torch_npu.npu_dynamic_quant(key, dst_type=torch.int8)
        # Key scales are rounded AFTER quantization; query scales before it.
        return codes[:, None], scale.half()[:, None]

    def select(self, normalized_q_a, hidden, batch, key_cache, scale_cache,
               block_table, *, candidates=None, projected_query=None):
        c = self.config
        codes, scales, weights = self.query(normalized_q_a, hidden, batch.positions(), projected_query=projected_query)
        source = self.layer_id == c['candidate_source_layer_id']
        consumer = self.layer_id > c['candidate_source_layer_id']
        if consumer and candidates is None:
            raise ValueError('candidate consumer requires current-forward source block indices')
        if not consumer and candidates is not None:
            raise ValueError('only candidate consumers accept source block indices')
        backend = select
        if self.fp4_index:
            from deepinfer.npu_ops.fp4_index910c import select as fp4_select
            backend = fp4_select
        return backend(codes, key_cache, weights, scales, scale_cache, batch, block_table,
                      ratio=self.ratio, candidate_mode=1 if source else 2 if consumer else 3,
                      candidates=candidates, topk=c['index_topk'],
                      candidate_topk_blocks=c['candidate_topk_blocks'],
                      candidate_block_size=c['candidate_block_size'])
