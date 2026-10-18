"""Own contiguous copies of the native read-only RoPE tables; keep native lookup.

Only the source tables used by the unchanged get_cos_and_sin_mla are selected.
The persistent MLA output buffers, graph pointers, arithmetic and KV stay native.
"""
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def install(ns, root=None, deps=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '0'
    if deps is None:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        from vllm_ascend.ops import rotary_embedding as rotary
        from vllm_ascend.attention import sfa_v1
        assert hashlib.sha256(Path(rotary.__file__).read_bytes()).hexdigest() == '403e1d0c0cac58a82898fa2596b923b60f19392d23560f4609967e4dfc8a411a'
        assert hashlib.sha256(Path(sfa_v1.__file__).read_bytes()).hexdigest() == '9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc'
        deps = {'rotary': rotary, 'ctx': get_forward_context, 'ctx_available': is_forward_context_available}
    rotary, torch, Runner = deps['rotary'], ns['torch'], ns['NPUModelRunner']
    root = Path('/out/rope-cache-layout-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    original = rotary.get_cos_and_sin_mla
    active = {'frame': None}
    cache = {}
    records = 0

    def describe(tensor):
        if tensor is None:
            return None
        return {'shape': list(tensor.shape), 'stride': list(tensor.stride()),
                'dtype': str(tensor.dtype), 'device': str(tensor.device),
                'data_ptr': tensor.data_ptr(), 'contiguous': tensor.is_contiguous()}

    def eligible_runner(runner):
        p = runner.parallel_config
        return bool(runner.is_kv_consumer and not runner.is_kv_producer
                    and runner.use_async_scheduling and runner.num_spec_tokens == 5
                    and p.tensor_parallel_size == 16 and p.data_parallel_size == 1
                    and p.pipeline_parallel_size == 1 and p.decode_context_parallel_size == 1
                    and p.prefill_context_parallel_size == 1 and not ns['enable_sp']()
                    and runner.model_config.use_mla and not runner.model_config.is_hybrid
                    and not getattr(runner, 'lora_config', None) and not runner.dynamic_eplb)

    native_record = rotary._record_cos_and_sin_cache_interleaved

    @functools.wraps(native_record)
    def record_cache(owner, cos_sin_cache):
        native_record(owner, cos_sin_cache)
        if cache:
            return
        pair = (rotary._cos_cache, rotary._sin_cache)
        if not all(t is not None and t.device.type == 'npu' and t.dtype == torch.bfloat16
                   and tuple(t.shape) == (1048576, 64) and tuple(t.stride()) == (128, 1)
                   for t in pair):
            return
        buffers = getattr(owner, '_buffers', {})
        if not (any(t is pair[0] for t in buffers.values()) and any(t is pair[1] for t in buffers.values())):
            return
        names = ['_glm53_rope_cos_contiguous', '_glm53_rope_sin_contiguous']
        assert all(n not in buffers for n in names)
        result = tuple(t.contiguous() for t in pair)
        assert all(torch.equal(a, b) for a, b in zip(pair, result)), 'Read-only RoPE cache copy changed bits'
        for key, tensor in zip(names, result):
            owner.register_buffer(key, tensor, persistent=False)
        cache.update(native=pair, native_ptrs=tuple(t.data_ptr() for t in pair), cached=result,
                     cached_ptrs=tuple(t.data_ptr() for t in result), owner=type(owner).__name__,
                     owner_module=owner, bitwise_equal=True,
                     bytes=sum(t.numel() * t.element_size() for t in result))

    # Register derived buffers while the native owner is being constructed,
    # before torch.compile guards or graph capture inspect module buffers.
    rotary._record_cos_and_sin_cache_interleaved = record_cache

    def cached_pair(runner, pair):
        if not cache:
            return None, 'cache-not-prepared-before-compilation'
        if not all(a is b and a.data_ptr() == ptr for a, b, ptr in zip(pair, cache['native'], cache['native_ptrs'])):
            return None, 'native-cache-replaced'
        if not all(t.data_ptr() == ptr for t, ptr in zip(cache['cached'], cache['cached_ptrs'])):
            return None, 'derived-cache-address-changed'
        owner = cache['owner_module']
        if not all(owner._buffers.get(name) is tensor for name, tensor in zip(['_glm53_rope_cos_contiguous', '_glm53_rope_sin_contiguous'], cache['cached'])):
            return None, 'derived-cache-owner-buffer-replaced'
        return cache['cached'], None

    @functools.wraps(original)
    def lookup(positions, use_cache=False):
        frame = active['frame']
        if frame is None:
            return original(positions, use_cache=use_cache)
        runner = frame['runner']
        ids = list(runner.input_batch.req_ids)
        real = bool(ids and all(rid in runner.requests and runner.requests[rid].sampling_params is not None
                               and runner.requests[rid].sampling_params.max_tokens > 1 for rid in ids))
        capturing = deps['ctx_available']() and deps['ctx']().capturing
        pair = (rotary._cos_cache, rotary._sin_cache)
        persistent = (rotary._cos_mla, rotary._sin_mla)
        persistent_ptrs = tuple(t.data_ptr() if t is not None else None for t in persistent)
        selected, reason, replacement = False, None, None
        if frame['requested'] and real and not capturing:
            replacement, reason = cached_pair(runner, pair)
            selected = replacement is not None
        row = None
        if frame['observe'] and real:
            frame['ids'] = sorted(set(frame['ids']) | set(ids))
            if 'inventory' not in frame:
                frame['inventory'] = {'native': [describe(t) for t in pair],
                    'cached': [describe(t) for t in cache['cached']] if cache else None,
                    'owner': cache.get('owner'), 'cache_bytes': cache.get('bytes', 0),
                    'full_cache_bitwise_equal': cache.get('bitwise_equal', False)}
            row = {'N': positions.shape[0], 'use_cache': bool(use_cache),
                   'requested': frame['requested'], 'selected': selected, 'fallback': reason}
        try:
            if selected:
                rotary._cos_cache, rotary._sin_cache = replacement
            output = original(positions, use_cache=use_cache)
            if row is not None:
                row['output_shapes'] = [list(t.shape) for t in output]
            return output
        finally:
            rotary._cos_cache, rotary._sin_cache = pair
            assert rotary._cos_mla is persistent[0] and rotary._sin_mla is persistent[1]
            assert tuple(t.data_ptr() if t is not None else None for t in persistent) == persistent_ptrs
            if row is not None:
                row['persistent_output_addresses_unchanged'] = True
                frame['calls'].append(row)

    aliases = []
    for name, module in list(sys.modules.items()):
        if name.startswith('vllm_ascend') and getattr(module, 'get_cos_and_sin_mla', None) is original:
            module.get_cos_and_sin_mla = lookup
            aliases.append(name)
    rotary.get_cos_and_sin_mla = lookup

    def wrap(method, phase):
        @functools.wraps(method)
        def run(self, *args, **kwargs):
            nonlocal records
            if not eligible_runner(self):
                return method(self, *args, **kwargs)
            frame = {'runner': self, 'phase': phase, 'requested': (root / 'enable').exists(),
                     'observe': not (root / 'disabled').exists(), 'calls': [], 'ids': []}
            previous = active['frame']
            active['frame'] = frame
            try:
                return method(self, *args, **kwargs)
            finally:
                active['frame'] = previous
                if frame['observe'] and frame['calls'] and records < 4000:
                    records += 1
                    frame.pop('runner')
                    frame.update(rank=self.tp_rank, pid=os.getpid(), monotonic_ns=time.monotonic_ns(),
                                 aliases=aliases)
                    with (root / ('rank-%d-%d.jsonl' % (self.tp_rank, os.getpid()))).open('a') as f:
                        f.write(json.dumps(frame, sort_keys=True) + '\n')
        return run

    Runner.execute_model = wrap(Runner.execute_model, 'execute_model')
    Runner.sample_tokens = wrap(Runner.sample_tokens, 'sample_tokens')
