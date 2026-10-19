"""Native routing prefix for GMM1, native utility count recovery at GMM2.

Only existing operators and exact int64 bookkeeping are used. The embedded graph
owner gives each family its own original warmup/capture context; MTP stays native.
"""
import dataclasses
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import types


def representations(torch, prefix_dispatch, native_fused, native_gmm1,
                    native_gmm2, convert_group_list):
    tag = object()
    stats = dict(prefix=0, fused=0, restored=0)

    def owned(inp):
        value = getattr(inp.group_list, '_glm53_recover_count', None)
        return isinstance(value, tuple) and len(value) == 1 and value[0] is tag

    @functools.wraps(prefix_dispatch)
    def candidate(self, inp):
        out = prefix_dispatch(self, inp)
        prefix = out.group_list
        assert out.group_list_type == 0 and prefix.dtype == torch.int64
        assert prefix.device.type == 'npu' and tuple(prefix.shape) == (256,)
        assert prefix.is_contiguous() and 0 < inp.hidden_states.shape[0] <= 24
        assert self.top_k == 8
        prefix._glm53_recover_count = (tag,)
        stats['prefix'] += 1
        return out

    @functools.wraps(native_fused)
    def fused(self, inp, *args, **kwargs):
        selected = owned(inp)
        if selected:
            assert inp.group_list_type == 0 and inp.fusion and not inp.dynamic_eplb
            assert getattr(inp.activation, 'name', '') != 'SWIGLUOAI_UNINTERLEAVE'
        result = native_fused(self, inp, *args, **kwargs)
        stats['fused'] += int(selected)
        return result

    @functools.wraps(native_gmm1)
    def gmm1(self, inp, *args, **kwargs):
        assert not owned(inp), 'Prefix recovery requires original fused GMM1'
        return native_gmm1(self, inp, *args, **kwargs)

    @functools.wraps(native_gmm2)
    def gmm2(self, inp, *args, **kwargs):
        if not owned(inp):
            return native_gmm2(self, inp, *args, **kwargs)
        assert inp.group_list_type == 0
        # Exact pinned framework utility: diff(prefix), cat(first, differences).
        # Defer recovery to the consumer rather than placing it before GMM1.
        count = convert_group_list(inp.group_list, 0, 1)
        assert count.dtype == torch.int64 and tuple(count.shape) == (256,)
        assert count.is_contiguous()
        count_input = dataclasses.replace(inp, group_list=count, group_list_type=1)
        result = native_gmm2(self, count_input, *args, **kwargs)
        stats['restored'] += 1
        return result

    return candidate, fused, gmm1, gmm2, stats


def install(ns, root=None, deps=None, factory=None):
    assert os.environ.get('GLM53_P_FIRST_DRAFT_EXPORT') == '0'
    torch, Mode = ns['torch'], ns['CUDAGraphMode']
    root = Path('/out/moe-prefix-gmm2-count-probe') if root is None else Path(root)
    root.mkdir(exist_ok=True)
    (root / 'target-only').touch()
    if deps is None:
        from vllm.forward_context import get_forward_context
        from vllm.distributed import get_tp_group
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX
        from vllm_ascend.attention import sfa_v1
        from vllm_ascend.compilation import acl_graph
        from vllm_ascend.ops.fused_moe import token_dispatcher, moe_utils
        from vllm_ascend.quantization.methods.w8a8 import w8a8_dynamic
        for module, expected in [
            (token_dispatcher, '1dcfd8181cb2d34c584d8336c3379dbeefba75c0bf664f924063701b93295cbb'),
            (moe_utils, '2c3dc13f2f7955f0e077df07adf8538929404a8f0af35a0ea9da42d42bcf5b5a'),
            (w8a8_dynamic, 'a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910'),
            (acl_graph, '8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955'),
            (sfa_v1, '9b94f67f30364d40caf4a7a67af0f0365d1ba0b5048939873ab0464e95b096dc'),
        ]:
            assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == expected
        original = token_dispatcher.TokenDispatcherWithAllGather.token_dispatch
        source = textwrap.dedent(inspect.getsource(original))
        assert source.count('expert_tokens_num_type=1,') == 1
        assert source.count('group_list_type = 1  # `count` mode') == 1
        source = source.replace('expert_tokens_num_type=1,', 'expert_tokens_num_type=0,')
        source = source.replace('group_list_type = 1  # `count` mode', 'group_list_type = 0  # prefix from native routing')
        local = {}
        exec(compile(source, '<glm53-prefix-count-routing>', 'exec'), original.__globals__, local)
        deps = dict(Graph=acl_graph.ACLGraphWrapper,
                    Dispatcher=token_dispatcher.TokenDispatcherWithAllGather,
                    prefix_dispatch=local['token_dispatch'],
                    Quant=w8a8_dynamic.AscendW8A8DynamicFusedMoEMethod,
                    convert_group_list=moe_utils.cumsum_group_list,
                    get_context=get_forward_context, is_draft=lambda: _EXTRA_CTX.is_draft_model,
                    SFA=sfa_v1.AscendSFABackend, updatable=acl_graph.use_updatable_graph,
                    W8A8=token_dispatcher.QuantType.W8A8, has_lora=token_dispatcher.has_lora,
                    rank=lambda: get_tp_group().rank_in_group)
    Q, Graph = deps['Quant'], deps['Graph']
    candidate, fused, gmm1, gmm2, stats = representations(
        torch, deps['prefix_dispatch'], Q.apply_gmm1_act_quant,
        Q.apply_gmm1, Q.apply_gmm2, deps['convert_group_list'])
    Q.apply_gmm1_act_quant, Q.apply_gmm1, Q.apply_gmm2 = fused, gmm1, gmm2
    if factory is None:
        assert hashlib.sha256(_BASE_PREFIX_SOURCE.encode()).hexdigest() == '173c720dd79dd656ec9cd88b51eeb373752ca71567254f88bcb1d367fcfac90c'
        base = types.ModuleType('_glm53_prefix_count_graph_owner')
        exec(compile(_BASE_PREFIX_SOURCE, '<glm53-prefix-count-owner>', 'exec'), base.__dict__)
        factory = base.install
    factory(ns, root=root, deps=dict(deps, candidate=candidate))
    graph_call = Graph.__call__
    captured = set()

    @functools.wraps(graph_call)
    def verified_graph(self, *args, **kwargs):
        ctx = deps['get_context']()
        full = self.runtime_mode == Mode.FULL and ctx.cudagraph_runtime_mode == Mode.FULL
        before = dict(stats)
        result = graph_call(self, *args, **kwargs)
        delta = {k: stats[k] - before[k] for k in stats}
        if full and delta['prefix']:
            n = ctx.batch_descriptor.num_tokens
            assert ctx.capturing and n in [6, 12, 18, 24]
            assert delta == dict(prefix=75, fused=75, restored=75), delta
            key = id(self), n
            if key not in captured:
                captured.add(key)
                row = dict(rank=deps['rank'](), pid=os.getpid(), N=n, **delta,
                           native_routing_prefix_type=0, GMM1_prefix_dtype='torch.int64',
                           recovered_GMM2_count_dtype='torch.int64',
                           native_GMM2_histogram_type=1,
                           recovery_utility='cumsum_group_list(0,1)', recovery_at='GMM2')
                with (root / ('gmm2-count-rank-%d-%d.jsonl' % (row['rank'], row['pid']))).open('a') as f:
                    f.write(json.dumps(row, sort_keys=True) + '\n')
        return result

    Graph.__call__ = verified_graph
