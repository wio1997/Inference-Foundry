# SPDX-License-Identifier: Apache-2.0
"""Single-request rank-local GLM caches and real installed metadata builders.

Each cache layer receives independent storage. Main MLA and compressed indexer
share logical block IDs, while each KDA state and FP32 tail has its own block 1.
Block 0 stays reserved. No reference scheduler/block manager owns these buffers.
"""
from math import ceil
from typing import get_type_hints

import torch

from .reference_backend import ForwardInputs, LayerContract


class ReferenceCacheRuntime:
    def __init__(self, loaded, *, max_context_tokens):
        from vllm.model_executor.layers.attention.mla_attention import MLAAttention
        from vllm.model_executor.layers.mamba.abstract import MambaBase
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
        from vllm_ascend.models.glm5next.kv_cache import Glm5NextIndexerCache, Glm5NextTailCache
        self.loaded, self.config, self.device = loaded, loaded.vllm_config, loaded.device
        self.max_context_tokens = max_context_tokens
        self.layers = {name: layer for name, layer in
                       self.config.compilation_config.static_forward_context.items()
                       if hasattr(layer, 'kv_cache')}
        self.specs, self.builders, self.roles, self.contracts = {}, {}, {}, {}
        for name, layer in self.layers.items():
            if isinstance(layer, MambaBase):
                spec, role = layer.get_kv_cache_spec(self.config), 'kda'
            elif isinstance(layer, Glm5NextTailCache):
                spec, role = layer.get_kv_cache_spec(self.config), 'tail'
            elif isinstance(layer, Glm5NextIndexerCache):
                spec, role = layer.get_kv_cache_spec(self.config), 'indexer'
            elif isinstance(layer, MLAAttention):
                # Matches the sparse-MLA branch of NPUModelRunner.get_kv_cache_spec.
                h = self.config.model_config.hf_text_config
                spec = AscendMLAAttentionSpec(
                    block_size=self.config.cache_config.block_size,
                    num_kv_heads=1, head_size=h.kv_lora_rank + h.qk_rope_head_dim,
                    dtype=self.config.model_config.dtype,
                    cache_dtype_str=self.config.cache_config.cache_dtype,
                    model_version=getattr(layer, 'model_version', None),
                    indexes_kv_by_block_stride=bool(layer.indexes_kv_by_block_stride))
                role = 'mla'
            else:
                raise TypeError('unsupported registered cache layer: ' + name)
            if spec is None:
                raise ValueError('registered GLM cache layer returned no spec: ' + name)
            backend = layer.get_attn_backend()
            builder = backend.get_builder_cls()(spec, [name], self.config, self.device)
            metadata_type = get_type_hints(builder.build)['return']
            if not isinstance(metadata_type, type):
                raise TypeError('installed builder must declare a concrete metadata type: ' + name)
            self.specs[name], self.roles[name], self.builders[name] = spec, role, builder
            self.contracts[name] = LayerContract(
                metadata_type,
                lambda value, span, cache, n=name: self.validate_metadata(n, value, span, cache),
                lambda cache, n=name: self.validate_cache(n, cache))
        if set(self.roles.values()) != {'kda', 'mla', 'indexer', 'tail'}:
            raise ValueError('complete mixed GLM requires KDA, MLA, compressed indexer and FP32 tail')
        counts = {role: list(self.roles.values()).count(role) for role in set(self.roles.values())}
        if (not counts['mla'] == counts['indexer'] == counts['tail']
                or counts['kda'] + counts['mla'] != loaded.model.config.num_hidden_layers):
            raise ValueError('registered cache roles do not cover every complete GLM layer')

    def num_blocks(self, name):
        return (2 if self.roles[name] in ('kda', 'tail') else
                1 + ceil(self.max_context_tokens / self.specs[name].block_size))

    def allocate(self, request_id):
        from vllm_ascend.models.glm5next.cache_views import view_glm5_next_cache
        caches = {}
        for name, spec in self.specs.items():
            layer, blocks = self.layers[name], self.num_blocks(name)
            if self.roles[name] == 'kda':
                caches[name] = tuple(torch.zeros((blocks, *shape), dtype=dtype, device=self.device)
                                     for shape, dtype in zip(layer.get_state_shape(), layer.get_state_dtype(), strict=True))
            else:
                raw = torch.zeros(blocks * spec.page_size_bytes, dtype=torch.uint8, device=self.device)
                views = view_glm5_next_cache(
                    name, spec, raw, attn_backend=layer.get_attn_backend(),
                    kernel_block_size=128, num_blocks=blocks,
                    get_kv_cache_dims=lambda _, s: (s.head_size, 0))
                if views is None:
                    raise ValueError('pinned GLM cache view rejected spec: ' + name)
                caches[name] = views
        return caches

    def release(self, cache):
        # Device work and outstanding offload stream uses complete before owner
        # drops its tensors. This is bounded request teardown, not a pool/cache.
        self.loaded.offloader.join_after_forward()
        torch.npu.synchronize(self.device)

    def validate_cache(self, name, cache):
        spec, role, blocks = self.specs[name], self.roles[name], self.num_blocks(name)
        if not isinstance(cache, (list, tuple)) or not cache:
            raise TypeError('rank-local cache views must be a nonempty sequence: ' + name)
        if role == 'kda':
            expected = list(zip(self.layers[name].get_state_shape(), self.layers[name].get_state_dtype(), strict=True))
            if len(cache) != len(expected) or any(tuple(t.shape) != (blocks, *s) or t.dtype != d
                                                  for t, (s, d) in zip(cache, expected)):
                raise ValueError('KDA rank-local state shape/dtype mismatch: ' + name)
        elif role == 'tail':
            if len(cache) != 1 or cache[0].dtype != torch.float32 or tuple(cache[0].shape) != (blocks, 2, spec.block_size, spec.head_size):
                raise ValueError('tail must retain raw K/gate FP32 ring: ' + name)
        elif any(t.dtype != spec.dtype for t in cache):
            raise ValueError('MLA/indexer dtype mismatch: ' + name)

    def validate_metadata(self, name, value, span, cache):
        role = self.roles[name]
        count = span.token_stop - span.token_start
        if role == 'kda':
            if value.num_actual_tokens != count:
                raise ValueError('GDN token count differs from scheduler')
            mode = 'non_spec_prefill_metadata' if value.num_prefills else 'non_spec_decode_metadata'
            if getattr(value, mode, None) is None:
                raise ValueError('Ascend GDN builder omitted actual conv/KDA descriptors')
        elif role == 'indexer':
            if (value.num_actual_tokens != count or value.cum_query_lens is None
                    or value.raw_seq_lens is None
                    or (self.builders[name].use_dsa_cp and value.query_metadata is None)):
                raise ValueError('compressed indexer metadata is incomplete')
        elif role == 'tail':
            if value.block_size != cache[0].shape[2]:
                raise ValueError('tail ring capacity differs from metadata')
        elif value.num_actual_tokens != count:
            raise ValueError('MLA metadata token count differs from scheduler')

    def build_metadata(self, span, cache):
        from vllm_ascend.attention.attention_v1 import AscendAttentionState
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
        from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
        count = span.token_stop - span.token_start
        query_cpu = torch.tensor([0, count], dtype=torch.int32)
        seq_cpu = torch.tensor([span.token_stop], dtype=torch.int32)
        computed_cpu = torch.tensor([span.token_start], dtype=torch.int32)
        positions = torch.arange(span.token_start, span.token_stop, dtype=torch.int64, device=self.device)
        state = (AscendAttentionState.DecodeOnly if not span.is_prompt else
                 AscendAttentionState.PrefillNoCache if span.token_start == 0 else
                 AscendAttentionState.ChunkedPrefill)
        metadata, slots = {}, {}
        common_ratio_to_sas_metadata = {}
        for name, builder in self.builders.items():
            role, spec = self.roles[name], self.specs[name]
            table = torch.arange(1, self.num_blocks(name), dtype=torch.int32, device=self.device).unsqueeze(0)
            if role in ('mla', 'indexer'):
                # Natural cache views split each logical allocation page into
                # 128-token SFA pages (128/compress_ratio for indexer rows).
                # The common block table must use the same physical page IDs.
                if spec.block_size % 128:
                    raise ValueError('GLM SFA allocation block is not a multiple of128')
                split = spec.block_size // 128
                subblocks = torch.arange(split, dtype=torch.int32, device=self.device)
                table = (table.unsqueeze(-1) * split + subblocks).reshape(1, -1)
            mapping = (spec.block_size + positions.remainder(spec.block_size)
                       if role == 'tail' else self.config.cache_config.block_size + positions)
            common = AscendCommonAttentionMetadata(
                query_start_loc=query_cpu.to(self.device), query_start_loc_cpu=query_cpu,
                seq_lens=seq_cpu.to(self.device), _seq_lens_cpu=seq_cpu,
                seq_lens_cpu=seq_cpu, seq_lens_cpu_upper_bound=seq_cpu,
                num_computed_tokens_cpu=computed_cpu,
                num_reqs=1, num_actual_tokens=count, max_query_len=count,
                max_seq_len=span.token_stop, block_table_tensor=table,
                slot_mapping=mapping, causal=True,
                is_prefilling=torch.tensor([span.is_prompt], dtype=torch.bool),
                num_input_tokens=count, actual_seq_lengths_q=[count],
                positions=positions, attn_state=state, decode_token_per_req=1,
                graph_pad_size=-1)
            extra = ({'common_ratio_to_sas_metadata': common_ratio_to_sas_metadata,
                      'num_actual_reqs': 1, 'full_graph_mode': False}
                     if isinstance(builder, AscendDSAMetadataBuilder) else {})
            metadata[name] = builder.build(0, common, **extra)
            if role == 'kda':
                # The stable FLA AscendC API has per-request state slots,
                # not paged/APC state scheduling. Its caller needs a vector.
                for phase in ('non_spec_prefill_metadata', 'non_spec_decode_metadata'):
                    value = getattr(metadata[name], phase, None)
                    if value is None:
                        continue
                    conv = value.causal_conv1d
                    indices = conv.cache_indices
                    if indices.ndim == 2 and indices.shape[1] == 1:
                        conv.cache_indices = indices[:, 0].contiguous()
                    elif indices.ndim != 1:
                        raise ValueError('stable GLM conv requires one state slot per request')
            slots[name] = mapping
        return ForwardInputs(metadata, slots)
