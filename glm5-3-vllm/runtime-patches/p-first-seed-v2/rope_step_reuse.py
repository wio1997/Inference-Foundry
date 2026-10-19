"""Share a completed SFA RoPE result with its same-step cache-only Indexer.

The pinned native cache-only callback does not advance positions. Its original
Indexer builder still copies into the original per-step owned graph buffers.
No memo survives this callback, no target/capture path is selected.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import time


def install(ns, root=None, deps=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '0'
    if deps is None:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        from vllm_ascend.attention import indexer, sfa_v1
        from vllm_ascend.spec_decode import llm_base_proposer as proposer
        for module, sha in [(indexer, '1f345bafb7c57ee02b66294ba72314e052f3c9b16f1faf5a8fb2bddde33c6eea'),
                            (sfa_v1, '9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc'),
                            (proposer, '458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31')]:
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == sha
        deps = dict(indexer=indexer, SFA=sfa_v1.AscendSFAMetadataBuilder,
                    Proposer=proposer.AscendSpecDecodeBaseProposer,
                    ctx=get_forward_context, ctx_available=is_forward_context_available)
    torch, Runner = ns['torch'], ns['NPUModelRunner']
    indexer, Proposer = deps['indexer'], deps['Proposer']
    root = Path('/out/rope-step-reuse-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    active = {'frame': None}
    native = Proposer._build_cache_only_group_next_step_attn_metadata
    records = 0

    def eligible(r):
        p = r.parallel_config
        ids = list(r.input_batch.req_ids)
        return bool(r.is_kv_consumer and not r.is_kv_producer and r.use_async_scheduling
                    and r.num_spec_tokens == 5 and r.model_config.use_mla
                    and not r.model_config.is_hybrid and not getattr(r, 'lora_config', None)
                    and not r.dynamic_eplb and not ns['enable_sp']()
                    and p.tensor_parallel_size == 16 and p.data_parallel_size == 1
                    and p.pipeline_parallel_size == 1 and p.decode_context_parallel_size == 1
                    and p.prefill_context_parallel_size == 1 and ids
                    and all(i in r.requests and r.requests[i].sampling_params is not None
                            and r.requests[i].sampling_params.max_tokens > 1 for i in ids))

    @functools.wraps(native)
    def callback(self, cad, draft_index, num_input_tokens, primary_group, primary_metadata, cache_only_groups):
        frame = active['frame']
        if frame is None:
            return native(self, cad, draft_index, num_input_tokens, primary_group, primary_metadata, cache_only_groups)
        capturing = deps['ctx_available']() and deps['ctx']().capturing
        pos = cad.positions
        source_pos = primary_metadata.positions
        pair = (primary_metadata.cos, primary_metadata.sin)
        selected = bool(frame['requested'] and not capturing and self.method == 'mtp'
                        and 1 <= draft_index < 5 and len(cache_only_groups) == 1
                        and type(primary_group.get_metadata_builder()) is deps['SFA']
                        and type(cache_only_groups[0].get_metadata_builder()) is indexer.AscendSFAIndexerMetadataBuilder
                        and pos.dtype == source_pos.dtype == torch.int64
                        and pos.device.type == source_pos.device.type == 'npu'
                        and pos.data_ptr() == source_pos.data_ptr()
                        and tuple(source_pos.shape) == (num_input_tokens,)
                        and tuple(source_pos.stride()) == (1,)
                        and all(t is not None and t.dtype == torch.bfloat16
                                and t.device.type == 'npu' and tuple(t.shape) == (num_input_tokens, 1, 1, 64)
                                for t in pair))
        saved = indexer.get_cos_and_sin_mla
        row = dict(draft_index=draft_index, N=num_input_tokens, selected=selected,
                   reused_lookups=0, bitwise_comparisons=0, original_indexer_copy_retained=True)

        def lookup(positions, use_cache=False):
            valid = (selected and not use_cache and positions.dtype == torch.int64
                     and positions.device == source_pos.device and positions.data_ptr() == source_pos.data_ptr()
                     and tuple(positions.shape) == tuple(source_pos.shape)
                     and tuple(positions.stride()) == tuple(source_pos.stride()))
            if not valid:
                return saved(positions, use_cache=use_cache)
            assert row['reused_lookups'] == 0, 'Pinned one-getter Indexer contract changed'
            row['reused_lookups'] += 1
            if frame['observe']:
                reference = saved(positions, use_cache=False)
                assert all(torch.equal(a, b) for a, b in zip(pair, reference)), 'Shared RoPE differs from original getter'
                row['bitwise_comparisons'] += 1
            return pair

        try:
            if selected:
                indexer.get_cos_and_sin_mla = lookup
            result = native(self, cad, draft_index, num_input_tokens, primary_group, primary_metadata, cache_only_groups)
            if selected:
                assert row['reused_lookups'] == 1
                if frame['observe']:
                    target = result[cache_only_groups[0].layer_names[0]]
                    assert all(torch.equal(a, b) for a, b in zip(pair, (target.cos, target.sin)))
                    row['owned_copy_bitwise'] = True
                    row['source_ptrs'] = [t.data_ptr() for t in pair]
                    row['indexer_ptrs'] = [target.cos.data_ptr(), target.sin.data_ptr()]
                    assert all(a != b for a, b in zip(row['source_ptrs'], row['indexer_ptrs']))
                    row['independent_indexer_storage'] = True
            return result
        finally:
            indexer.get_cos_and_sin_mla = saved
            if frame['observe']:
                frame['calls'].append(row)
    Proposer._build_cache_only_group_next_step_attn_metadata = callback

    original_sample = Runner.sample_tokens
    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        nonlocal records
        if not eligible(self):
            return original_sample(self, *args, **kwargs)
        frame = dict(requested=(root/'enable').exists(), observe=not (root/'disabled').exists(), calls=[])
        previous = active['frame']; active['frame'] = frame
        try:
            return original_sample(self, *args, **kwargs)
        finally:
            active['frame'] = previous
            if frame['observe'] and frame['calls'] and records < 5000:
                records += 1
                frame.update(rank=self.tp_rank, pid=os.getpid(), monotonic_ns=time.monotonic_ns())
                with (root/('rank-%d-%d.jsonl' % (self.tp_rank, os.getpid()))).open('a') as f:
                    f.write(json.dumps(frame)+'\n')
    Runner.sample_tokens = sample
