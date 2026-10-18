"""Trim only graph-excluded native P attention; preserve the captured graph shape."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time


def install(ns, root=None, sfa_cls=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '1'
    if sfa_cls is None:
        from vllm_ascend.attention import sfa_v1
        assert hashlib.sha256(Path(sfa_v1.__file__).read_bytes()).hexdigest() == '9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc'
        sfa_cls = sfa_v1.AscendSFAImpl
    Runner, Mode = ns['NPUModelRunner'], ns['CUDAGraphMode']
    root = Path('/out/prefill-attention-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    state = {'frame': None, 'records': 0}
    build_original = Runner._build_attention_metadata
    signature = inspect.signature(build_original)

    @functools.wraps(build_original)
    def build(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        ids = list(self.input_batch.req_ids)
        batch, n = int(values['num_reqs']), int(values['num_tokens'])
        padded = int(values['num_tokens_padded'] or n)
        real = bool(ids and len(ids) == batch and all(rid in self.requests for rid in ids))
        scheduled = [int(x) for x in values['num_scheduled_tokens_np'][:batch]] if real else []
        parallel = self.parallel_config
        eligible = bool(
            real and self.is_kv_producer and not self.is_kv_consumer
            and parallel.tensor_parallel_size == 16 and parallel.data_parallel_size == 1
            and parallel.pipeline_parallel_size == 1 and parallel.decode_context_parallel_size == 1
            and parallel.prefill_context_parallel_size == 1 and self.num_spec_tokens == 1
            and not ns['enable_sp']() and not self.model_config.is_hybrid
            and self.model_config.use_mla and not self.sparse_kv_offload_enabled
            and values['cudagraph_runtime_mode'] == Mode.PIECEWISE
            and not values['for_cudagraph_capture'] and not values['use_spec_decode']
            and values['ubatch_slices'] is None and values['cascade_attn_prefix_lens'] is None
            and (values['num_reqs_padded'] is None or values['num_reqs_padded'] == batch)
            and values['batch_descriptor'] is not None and values['batch_descriptor'].num_tokens == padded
            and padded > n and all(x > self.uniform_decode_query_len for x in scheduled)
            and all(int(self.input_batch.num_computed_tokens_cpu[i]) < int(self.input_batch.num_prompt_tokens[i]) for i in range(batch))
            and all(self.requests[rid].sampling_params is not None
                    and self.requests[rid].sampling_params.max_tokens == 1
                    and not getattr(self.requests[rid], 'mm_features', []) for rid in ids))
        requested = (root / 'enable').exists()
        selected = bool(eligible and requested)
        state['frame'] = {'request_ids': ids, 'tp_rank': self.tp_rank,
                          'real_tokens': n, 'graph_tokens': padded,
                          'native_graph_mode': str(values['cudagraph_runtime_mode']),
                          'selected': selected, 'eligible': eligible, 'requested': requested,
                          'observe': not (root / 'disabled').exists(),
                          'computed_before': [int(self.input_batch.num_computed_tokens_cpu[i]) for i in range(batch)] if real else [],
                          'scheduled': scheduled, 'sfa_forward': 0, 'tail_clear': 0,
                          'prolog_rows': {}, 'o_proj_rows': {}} if real else None
        if selected:
            # This argument controls only native attention metadata and views.
            # The caller retains its original BatchDescriptor/forward shape.
            bound.arguments['num_tokens_padded'] = n
        return build_original(*bound.args, **bound.kwargs)

    Runner._build_attention_metadata = build

    execute_original = Runner.execute_model

    @functools.wraps(execute_original)
    def execute(self, *args, **kwargs):
        state['frame'] = None
        try:
            return execute_original(self, *args, **kwargs)
        finally:
            frame = state['frame']
            if frame is not None and frame['observe'] and state['records'] < 2000:
                state['records'] += 1
                frame.update(pid=os.getpid(), monotonic_ns=time.monotonic_ns(),
                             kind='P_attention_native_work',
                             scope='Native Python call shapes/counters; not GPU duration')
                with (root / ('rank-%d-%d.jsonl' % (self.tp_rank, os.getpid()))).open('a') as out:
                    out.write(json.dumps(frame, sort_keys=True) + '\n')
            state['frame'] = None

    Runner.execute_model = execute
    forward_original = sfa_cls.forward

    @functools.wraps(forward_original)
    def forward(self, layer_name, hidden_states, kv_cache, attn_metadata, output=None):
        frame = state['frame']
        if frame is not None and frame['observe']:
            frame['sfa_forward'] += 1
        result = forward_original(self, layer_name, hidden_states, kv_cache, attn_metadata, output)
        if frame is not None and frame['selected']:
            assert output is not None and result is output
            assert attn_metadata.num_input_tokens == attn_metadata.num_actual_tokens == frame['real_tokens']
            assert hidden_states.shape[0] == output.shape[0] == frame['graph_tokens']
            # Keep padded graph rows defined. This uses the existing zero_ fill
            # primitive; its launch/work is included in candidate measurements.
            output[frame['real_tokens']:].zero_()
            if frame['observe']:
                frame['tail_clear'] += 1
        return result

    sfa_cls.forward = forward
    prolog_original = sfa_cls._sfa_preprocess_prolog_v3

    @functools.wraps(prolog_original)
    def prolog(self, hidden_states, *args, **kwargs):
        frame = state['frame']
        if frame is not None and frame['observe']:
            key = str(int(hidden_states.shape[0]))
            frame['prolog_rows'][key] = frame['prolog_rows'].get(key, 0) + 1
        return prolog_original(self, hidden_states, *args, **kwargs)

    sfa_cls._sfa_preprocess_prolog_v3 = prolog
    o_proj_original = sfa_cls._finalize_o_proj

    @functools.wraps(o_proj_original)
    def o_proj(self, attn_output, *args, **kwargs):
        frame = state['frame']
        if frame is not None and frame['observe']:
            key = str(int(attn_output.shape[0]))
            frame['o_proj_rows'][key] = frame['o_proj_rows'].get(key, 0) + 1
        return o_proj_original(self, attn_output, *args, **kwargs)

    sfa_cls._finalize_o_proj = o_proj
