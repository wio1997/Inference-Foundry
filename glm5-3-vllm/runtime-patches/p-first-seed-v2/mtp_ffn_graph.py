"""Bounded graph of the native MTP-only MoE, with attention/KV left eager.

No operator implementation or whole-draft graph flag is changed. Each graph
owns its input and output buffers and private graph pool. Copy and replay run
on the original current stream, after the original post-attention RMSNorm.
"""
from contextvars import ContextVar
import functools
import hashlib
import json
import os
from pathlib import Path
import time
import types


def install(ns, root=None, deps=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '0'
    if deps is None:
        from vllm.forward_context import get_forward_context
        from vllm.model_executor.models import deepseek_v2
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.ops.fused_moe import fused_moe, routed_experts, shared_experts
        for module, expected in [
            (deepseek_v2, 'c658788dc880c5e57400f3bc59537386f24605e157d3cad5b6d5bc8f31bf2328'),
            (fused_moe, '2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4'),
            (routed_experts, 'f9db74836ae210a541432b923118696c7a15be9a10d5776834e1d3b64ce11d77'),
            (shared_experts, '471db26e9e9e4ba70c2ccd43502ad39bff6699bbd4ca191319a9b66fc20458f9'),
        ]:
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
        deps = {'MoE': deepseek_v2.DeepseekV2MoE, 'ctx': get_forward_context,
                'extra': _EXTRA_CTX}
    Runner, MoE, torch, Mode = ns['NPUModelRunner'], deps['MoE'], ns['torch'], ns['CUDAGraphMode']
    root = Path('/out/mtp-ffn-graph-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    active = ContextVar('glm53_mtp_ffn_graph', default=None)
    modules, cache, warm = {}, {}, {}
    records = 0
    sample_original, moe_original = Runner.sample_tokens, MoE.forward

    @functools.wraps(sample_original)
    def sample(self, *args, **kwargs):
        nonlocal records
        ids = list(self.input_batch.req_ids)
        parallel = self.parallel_config
        drafter = self.drafter
        eligible = bool(ids and all(rid in self.requests and self.requests[rid].sampling_params is not None
                                    and self.requests[rid].sampling_params.max_tokens > 1 for rid in ids)
                        and self.is_kv_consumer and not self.is_kv_producer and self.use_async_scheduling
                        and parallel.tensor_parallel_size == 16 and parallel.data_parallel_size == 1
                        and parallel.pipeline_parallel_size == 1 and parallel.decode_context_parallel_size == 1
                        and parallel.prefill_context_parallel_size == 1 and self.num_spec_tokens == 5
                        and not ns['enable_sp']() and self.model_config.use_mla and not self.model_config.is_hybrid
                        and not self.dynamic_eplb and not getattr(self, 'lora_config', None)
                        and getattr(drafter, 'method', None) == 'mtp'
                        and getattr(drafter, 'use_cuda_graph', None) is False)
        if eligible and not modules:
            found = []
            for name, module in drafter.model.named_modules():
                if isinstance(module, MoE) and name.endswith('.mtp_block.mlp'):
                    found.append((name, module))
            assert len(found) == 1, 'Expected exactly one existing GLM MTP MoE'
            for name, module in found:
                modules[id(module)] = name
                # Bind only the loaded drafter instance. The target's class
                # must retain its original forward for torch.compile tracing.
                module.forward = types.MethodType(moe, module)
        if not eligible:
            return sample_original(self, *args, **kwargs)
        frame = {'ids': ids, 'rank': self.tp_rank, 'calls': [],
                 'observe': not (root / 'disabled').exists(), 'requested': (root / 'enable').exists()}
        token = active.set(frame)
        try:
            return sample_original(self, *args, **kwargs)
        finally:
            active.reset(token)
            if frame['observe'] and records < 3000:
                records += 1
                frame.update(pid=os.getpid(), monotonic_ns=time.monotonic_ns(), kind='mtp_ffn_native_graph')
                with (root / ('rank-%d-%d.jsonl' % (self.tp_rank, os.getpid()))).open('a') as f:
                    f.write(json.dumps(frame, sort_keys=True) + '\n')

    @functools.wraps(moe_original)
    def moe(self, hidden_states, already_sequence_parallel=False):
        frame = active.get()
        if frame is None or id(self) not in modules:
            return moe_original(self, hidden_states, already_sequence_parallel=already_sequence_parallel)
        ctx, extra = deps['ctx'](), deps['extra']
        x = hidden_states
        comm = getattr(getattr(extra, 'moe_comm_type', None), 'name', None)
        eligible = bool(not already_sequence_parallel and not self.is_sequence_parallel
                        and not self.enable_eplb and not ctx.capturing and extra.is_draft_model
                        and not extra.in_profile_run and ctx.cudagraph_runtime_mode == Mode.NONE
                        and getattr(ctx, 'device_metadata_executor', None) is None
                        and comm == 'ALLGATHER' and x.device.type == 'npu'
                        and x.dtype == torch.bfloat16 and x.is_contiguous()
                        and x.ndim == 2 and 0 < x.shape[0] <= 24 and x.shape[1] == 6144)
        key = (id(self), tuple(x.shape), str(x.dtype), str(x.device), comm,
               getattr(extra, 'num_tokens', None), getattr(extra, 'num_actual_tokens', None),
               getattr(extra, 'num_accept_tokens', None))
        # Scalar context values are specialized by the native pure FFN body.
        # No attention metadata, positions, KV pointers or sampling is captured.
        eligible = eligible and all(value is None or type(value) is int for value in key[-3:])
        context_scalars = [value if value is None or type(value) is int else 'non-scalar:' + type(value).__name__
                           for value in key[-3:]]
        action = 'native'
        if eligible and frame['requested']:
            if key not in cache and len(cache) >= 16:
                action = 'native-cache-bound'
            elif key not in cache and warm.get(key, 0) < 2:
                warm[key] = warm.get(key, 0) + 1
                action = 'native-warmup'
            else:
                if key not in cache:
                    # Strong references retain both static input and output.
                    # A private graph pool avoids aliasing target/other entries.
                    static_input = torch.empty_like(x)
                    static_input.copy_(x)
                    graph = torch.npu.NPUGraph()
                    old_capturing = ctx.capturing
                    ctx.capturing = True
                    try:
                        with torch.npu.graph(graph):
                            output = moe_original(self, static_input, already_sequence_parallel=False)
                    finally:
                        ctx.capturing = old_capturing
                    cache[key] = (static_input, output, graph)
                    action = 'capture-and-replay'
                else:
                    action = 'copy-and-replay'
                static_input, output, graph = cache[key]
                static_input.copy_(x)
                graph.replay()
                if frame['observe']:
                    frame['calls'].append({'module': modules[id(self)], 'shape': list(x.shape),
                                           'comm': comm, 'eligible': True, 'action': action,
                                           'context_scalars': context_scalars, 'cache_entries': len(cache)})
                return output
        if frame['observe']:
            frame['calls'].append({'module': modules[id(self)], 'shape': list(x.shape),
                                   'comm': comm, 'eligible': bool(eligible), 'action': action,
                                   'context_scalars': context_scalars, 'cache_entries': len(cache)})
        return moe_original(self, x, already_sequence_parallel=already_sequence_parallel)

    Runner.sample_tokens = sample
