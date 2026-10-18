"""Exact narrow prefix for GMM1; original routing and GMM2 count semantics.

The build embeds the separately tested graph-family owner in _BASE_PREFIX_SOURCE.
The existing target-only selector fixes MTP to the original path in both arms.
"""
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import types


def representations(torch, native_dispatch, native_fused, native_gmm1, native_gmm2):
    tag = object()
    stats = dict(prefix=0, fused=0, restored=0)

    def saved(inp):
        item = getattr(inp.group_list, '_glm53_i32_counts', None)
        return item[1] if isinstance(item, tuple) and len(item) == 2 and item[0] is tag else None

    @functools.wraps(native_dispatch)
    def candidate(self, inp):
        out = native_dispatch(self, inp)  # Same operator, including count1.
        count = out.group_list
        if not (out.group_list_type == 1 and count.dtype == torch.int64
                and count.device.type == 'npu' and tuple(count.shape) == (256,)
                and count.is_contiguous()):
            return out
        n = inp.hidden_states.shape[0]
        assert 0 < n <= 24 and self.top_k == 8
        # Native nonnegative histogram sum <= N*topK*TP <= 3072 < 2**31.
        # This changes integer bookkeeping, never model/activation precision.
        assert n * self.top_k * 16 <= 3072
        prefix = count.to(torch.int32).cumsum(0, dtype=torch.int32).to(torch.int64)
        prefix._glm53_i32_counts = (tag, count)
        stats['prefix'] += 1
        return dataclasses.replace(out, group_list=prefix, group_list_type=0)

    @functools.wraps(native_fused)
    def fused(self, inp, *args, **kwargs):
        own = saved(inp) is not None
        if own:
            assert inp.group_list_type == 0 and inp.fusion and not inp.dynamic_eplb
            assert getattr(inp.activation, 'name', '') != 'SWIGLUOAI_UNINTERLEAVE'
        out = native_fused(self, inp, *args, **kwargs)
        stats['fused'] += int(own)
        return out

    @functools.wraps(native_gmm1)
    def gmm1(self, inp, *args, **kwargs):
        assert saved(inp) is None, 'Narrow prefix requires the original fused GMM1 path'
        return native_gmm1(self, inp, *args, **kwargs)

    @functools.wraps(native_gmm2)
    def gmm2(self, inp, *args, **kwargs):
        count = saved(inp)
        if count is None:
            return native_gmm2(self, inp, *args, **kwargs)
        assert inp.group_list_type == 0 and count.dtype == torch.int64 and tuple(count.shape) == (256,)
        original_count_input = dataclasses.replace(inp, group_list=count, group_list_type=1)
        out = native_gmm2(self, original_count_input, *args, **kwargs)
        stats['restored'] += 1
        return out

    return candidate, fused, gmm1, gmm2, stats


def install(ns, root=None, deps=None, factory=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '0'
    torch, Mode = ns['torch'], ns['CUDAGraphMode']
    root = Path('/out/moe-cumsum-i32-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    (root / 'target-only').touch()
    if deps is None:
        from vllm.forward_context import get_forward_context
        from vllm.distributed import get_tp_group
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.attention import sfa_v1
        from vllm_ascend.compilation import acl_graph
        from vllm_ascend.ops.fused_moe import token_dispatcher
        from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
        for module, expected in [
            (token_dispatcher, '1dcfd8181cb2d34c584d8336c3379dbeefba75c0bf664f924063701b93295cbb'),
            (w8a8_dynamic, 'a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910'),
            (acl_graph, '8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955'),
            (sfa_v1, '9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc'),
        ]:
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
        deps = dict(Graph=acl_graph.ACLGraphWrapper,
                    Dispatcher=token_dispatcher.TokenDispatcherWithAllGather,
                    Quant=w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod,
                    get_context=get_forward_context, is_draft=lambda: _EXTRA_CTX.is_draft_model,
                    SFA=sfa_v1.AscendSFABackend, updatable=acl_graph.use_updatable_graph,
                    W8A8=token_dispatcher.QuantType.W8A8, has_lora=token_dispatcher.has_lora,
                    rank=lambda: get_tp_group().rank_in_group)
    Q, Graph = deps['Quant'], deps['Graph']
    candidate, fused, gmm1, gmm2, stats = representations(
        torch, deps['Dispatcher'].token_dispatch, Q.apply_gmm1_act_quant, Q.apply_gmm1, Q.apply_gmm2)
    Q.apply_gmm1_act_quant, Q.apply_gmm1, Q.apply_gmm2 = fused, gmm1, gmm2
    deps = dict(deps, candidate=candidate)
    if factory is None:
        assert hashlib.sha256(_BASE_PREFIX_SOURCE.encode()).hexdigest() == '173c720dd79dd656ec9cd88b51eeb373752ca71567254f88bcb1d367fcfac90c'
        base = types.ModuleType('_glm53_i32_graph_owner')
        exec(compile(_BASE_PREFIX_SOURCE, '<glm53-i32-graph-owner>', 'exec'), base.__dict__)
        factory = base.install
    factory(ns, root=root, deps=deps)
    graph_call = Graph.__call__
    captured = set()

    @functools.wraps(graph_call)
    def verified_graph(self, *args, **kwargs):
        ctx = deps['get_context']()
        full = self.runtime_mode == Mode.FULL and ctx.cudagraph_runtime_mode == Mode.FULL
        before = dict(stats)
        out = graph_call(self, *args, **kwargs)
        delta = {k: stats[k] - before[k] for k in stats}
        if full and delta['prefix']:
            n = ctx.batch_descriptor.num_tokens
            assert ctx.capturing and n in [6, 12, 18, 24]
            assert delta == dict(prefix=75, fused=75, restored=75), delta
            key = id(self), n
            if key not in captured:
                captured.add(key)
                row = dict(rank=deps['rank'](), pid=os.getpid(), N=n, **delta,
                           native_routing_histogram_type=1, GMM1_prefix_dtype='torch.int64',
                           intermediate_dtype='torch.int32', native_GMM2_histogram_type=1,
                           conservative_route_bound=3072)
                with (root / ('gmm2-restore-rank-%d-%d.jsonl' % (row['rank'], row['pid']))).open('a') as f:
                    f.write(json.dumps(row, sort_keys=True) + '\n')
        return out

    Graph.__call__ = verified_graph
