"""Private native K-producer graph alongside original MTP eager Prolog.

The main-stream handshake protects both source readiness and reuse of owned
graph outputs. Original Indexer cache writes, Q/top-k and KV fences stay native.
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
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.attention import sfa_v1, indexer
        for module, sha in [(sfa_v1, '9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc'),
                            (indexer, '1f345bafb7c57ee02b66294ba72314e052f3c9b16f1faf5a8fb2bddde33c6eea')]:
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == sha
        deps = dict(SFA=sfa_v1.AscendSFAImpl, Indexer=indexer.AscendSFAIndexerBackend,
                    quant=sfa_v1.AscendW8A8DynamicLinearMethod,
                    prolog_type=sfa_v1.PreprocessType.PROLOG_V3,
                    ctx=get_forward_context, available=is_forward_context_available,
                    is_draft=lambda: _EXTRA_CTX.is_draft_model,
                    capturing=lambda: _EXTRA_CTX.capturing)
    torch, Runner, Mode = ns['torch'], ns['NPUModelRunner'], ns['CUDAGraphMode']
    SFA, Indexer = deps['SFA'], deps['Indexer']
    root = Path('/out/indexer-k-graph-overlap-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    state = dict(frame=None, stream=None, entries={}, pending={}, records=0)
    native_sample = Runner.sample_tokens
    native_forward = SFA.forward
    native_prolog = SFA._sfa_preprocess_prolog_v3
    native_k = Indexer.forward_k

    def runner_eligible(r):
        p = r.parallel_config
        ids = list(r.input_batch.req_ids)
        return bool(r.is_kv_consumer and not r.is_kv_producer and r.use_async_scheduling
                    and r.num_spec_tokens == 5 and r.model_config.use_mla
                    and not r.model_config.is_hybrid and not getattr(r, 'lora_config', None)
                    and not r.dynamic_eplb and not ns['enable_sp']()
                    and p.tensor_parallel_size == 16 and p.data_parallel_size == 1
                    and p.pipeline_parallel_size == p.decode_context_parallel_size == p.prefill_context_parallel_size == 1
                    and ids and all(i in r.requests and r.requests[i].sampling_params is not None
                                    and r.requests[i].sampling_params.max_tokens > 1 for i in ids))

    @functools.wraps(native_sample)
    def sample(self, *args, **kwargs):
        if not runner_eligible(self):
            return native_sample(self, *args, **kwargs)
        frame = dict(requested=(root/'enable').exists(), observe=not (root/'disabled').exists(),
                     rank=self.tp_rank, profile_copies=(root/'profile-copy-labels').exists(), calls=[])
        previous = state['frame']
        state['frame'] = frame
        try:
            return native_sample(self, *args, **kwargs)
        finally:
            state['frame'] = previous
            assert not state['pending'], 'A K producer escaped its SFA consumer scope'
            if frame['observe'] and frame['calls'] and state['records'] < 5000:
                state['records'] += 1
                frame.update(pid=os.getpid(), monotonic_ns=time.monotonic_ns(), graph_entries=len(state['entries']))
                with (root/('rank-%d-%d.jsonl' % (self.tp_rank, os.getpid()))).open('a') as f:
                    f.write(json.dumps(frame)+'\n')
    Runner.sample_tokens = sample

    def tensor_signature(t):
        if t is None:
            return None
        return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device))

    def guard(self, hidden, metadata):
        if (type(self) is not SFA or type(self.indexer) is not Indexer
                or not deps['available']() or not deps['is_draft']() or deps['capturing']()):
            return False
        ctx, idx = deps['ctx'](), self.indexer
        return bool(not ctx.capturing and ctx.cudagraph_runtime_mode != Mode.FULL
                    and self.preprocess_type == deps['prolog_type']
                    and self._quant_type is deps['quant'] and self.enable_sparse_sfa_c8
                    and not self.enable_sparse_sfa_turboquant and self.runtime_has_indexer
                    and self.has_indexer and self.local_num_heads == 4 and self.kv_lora_rank == 512
                    and self.qk_rope_head_dim == 64 and idx.head_dim == 128
                    and not idx._pcp_active and not idx._dsa_cp_active and not idx.enable_sparse_li_c4
                    and hidden.dtype == torch.bfloat16 and hidden.device.type == 'npu'
                    and hidden.ndim == 2 and hidden.shape[0] in (6, 12, 18, 24)
                    and hidden.shape[1] == 6144 and hidden.is_contiguous()
                    and idx.wk_weights_proj.weight.dtype == torch.bfloat16
                    and all(t is not None and t.device == hidden.device
                            and t.dtype in (torch.bfloat16, torch.float32) and t.is_contiguous()
                            and t.numel() == hidden.shape[0]*64 and t.shape[-1] == 64
                            for t in (metadata.cos, metadata.sin)))

    def copy_inputs(inputs, sources, phase):
        frame = state['frame']
        if frame is not None and frame['profile_copies']:
            for name, dst, src in zip(('hidden', 'cos', 'sin'), inputs, sources):
                with torch.profiler.record_function('glm53.indexer.Kgraph.input.%s.%s.N%d' %
                                                     (name, phase, sources[0].shape[0])):
                    dst.copy_(src, non_blocking=True)
        else:
            for dst, src in zip(inputs, sources):
                dst.copy_(src, non_blocking=True)

    def prepare_entry(idx, hidden, cos, sin, main):
        signature = tuple(tensor_signature(t) for t in
                          [idx.wk_weights_proj.weight, idx.k_norm.weight, idx.k_norm.bias,
                           idx.q_hadamard if idx.enable_sparse_li_c8 else None])
        key = (id(idx), tuple(hidden.shape), str(hidden.dtype), tuple(cos.shape), str(cos.dtype),
               tuple(sin.shape), str(sin.dtype), idx.enable_sparse_li_c8, signature)
        entry = state['entries'].get(key)
        cold = entry is None
        aux = state['stream']
        if aux is None:
            aux = state['stream'] = torch.npu.Stream()
        if entry is None:
            if len(state['entries']) >= 16:
                return None, cold
            inputs = tuple(torch.empty_like(t) for t in (hidden, cos, sin))
            aux.wait_stream(main)
            with torch.npu.stream(aux):
                copy_inputs(inputs, (hidden, cos, sin), 'cold')
                for _ in range(3):
                    native_k(idx, *inputs)
            aux.synchronize()
            graph, pool = torch.npu.NPUGraph(), torch.npu.graph_pool_handle()
            with torch.npu.graph(graph, pool=pool, stream=aux):
                outputs = native_k(idx, *inputs)
            entry = dict(inputs=inputs, outputs=outputs, graph=graph, pool=pool,
                         source_signature=signature, owner=idx)
            state['entries'][key] = entry
        # Prior main-stream consumers finish before owned graph buffers are
        # reused. This also orders current hidden/metadata writers before copy.
        aux.wait_stream(main)
        with torch.npu.stream(aux):
            copy_inputs(entry['inputs'], (hidden, cos, sin), 'replay')
            entry['graph'].replay()
        return entry, cold

    @functools.wraps(native_prolog)
    def prolog(self, hidden_states, kv_cache, cos, sin, slot_mapping, *, attn_metadata=None):
        frame = state['frame']
        if frame is None:
            return native_prolog(self, hidden_states, kv_cache, cos, sin, slot_mapping, attn_metadata=attn_metadata)
        metadata = self._get_indexer_attn_metadata() if self.runtime_has_indexer else None
        eligible = metadata is not None and guard(self, hidden_states, metadata)
        row = dict(N=int(hidden_states.shape[0]), eligible=bool(eligible), selected=False,
                   native_cache_write_retained=True, native_main_prolog_retained=True)
        if eligible and frame['requested']:
            idx, main = self.indexer, torch.npu.current_stream()
            assert id(idx) not in state['pending']
            try:
                entry, cold = prepare_entry(idx, hidden_states, metadata.cos, metadata.sin, main)
            except BaseException:
                if state['stream'] is not None:
                    main.wait_stream(state['stream'])
                raise
            if entry is not None:
                row.update(selected=True, cold_graph=cold, added_input_copies=3,
                           Indexer_C8=idx.enable_sparse_li_c8)
                state['pending'][id(idx)] = dict(frame=frame, main=main, entry=entry, row=row,
                                                hidden=hidden_states, cos=metadata.cos, sin=metadata.sin)
        if frame['observe']:
            frame['calls'].append(row)
        return native_prolog(self, hidden_states, kv_cache, cos, sin, slot_mapping, attn_metadata=attn_metadata)
    SFA._sfa_preprocess_prolog_v3 = prolog

    @functools.wraps(native_k)
    def forward_k(self, hidden_states, cos, sin):
        pending = state['pending'].get(id(self))
        if pending is None:
            return native_k(self, hidden_states, cos, sin)
        assert pending['frame'] is state['frame']
        assert hidden_states is pending['hidden'] and cos is pending['cos'] and sin is pending['sin']
        assert torch.npu.current_stream().npu_stream == pending['main'].npu_stream
        pending['main'].wait_stream(state['stream'])
        outputs = pending['entry']['outputs']
        if pending['frame']['observe']:
            reference = native_k(self, hidden_states, cos, sin)
            assert all((a is None and b is None) or
                       (a is not None and b is not None and torch.equal(a, b))
                       for a, b in zip(reference, outputs)), 'Captured K differs from current native'
            pending['row'].update(current_native_bitwise=True,
                                  owned_input_ptrs=[t.data_ptr() for t in pending['entry']['inputs']],
                                  independent_inputs=all(a.data_ptr() != b.data_ptr() for a,b in
                                                         zip(pending['entry']['inputs'], (hidden_states,cos,sin))))
            assert pending['row']['independent_inputs']
        pending['row']['consumed_once'] = True
        state['pending'].pop(id(self))
        return outputs
    Indexer.forward_k = forward_k

    @functools.wraps(native_forward)
    def forward(self, *args, **kwargs):
        try:
            result = native_forward(self, *args, **kwargs)
        except BaseException:
            pending = state['pending'].pop(id(self.indexer), None) if self.runtime_has_indexer else None
            if pending is not None:
                pending['main'].wait_stream(state['stream'])
            raise
        pending = state['pending'].pop(id(self.indexer), None) if self.runtime_has_indexer else None
        if pending is not None:
            pending['main'].wait_stream(state['stream'])
            raise RuntimeError('Prepared K did not reach the original Indexer consumer')
        return result
    SFA.forward = forward
    return state
