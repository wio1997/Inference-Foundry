"""Scoped C1/C2 native prefix-cache integration; no storage/layout flags faked.

The native unified-KV layout cannot represent low ratios. C1/C2 byte ownership
follows FULL pages; native SWA tail and request-state PD transfers stay intact.
Full model PD and prefix correctness still require runtime acceptance.
"""
from pathlib import Path
import ast,hashlib,inspect,json,os


def supported(model_config,pool):
    from lowratio_pool import LowRatioNPUTokenToKVPool
    import torch
    if not isinstance(pool,LowRatioNPUTokenToKVPool):return False
    assert getattr(model_config.hf_config,'model_type',None)=='deepseek_v41'
    assert not pool._unified_kv and pool.page_size==128
    assert pool.c4_kv_pool is None and pool.c128_kv_pool is None
    assert pool.request_window is None and pool.swa_req_ring_size is None
    assert pool.swa_kv_pool is not None and pool.full_to_swa_index_mapping is not None
    assert set(pool.compression_ratios)<=set([0,1,2])
    assert set(pool.sources_by_ratio)<=set([1,2]) and pool.sources_by_ratio
    assert pool.swa_kv_pool.dtype==torch.bfloat16
    for ratio,owners in pool.sources_by_ratio.items():
        kv=pool.kv_pools[ratio];index=pool.index_pools[ratio]
        assert kv.dtype==torch.bfloat16 and kv.get_bytes_per_token()==1024
        assert kv.page_size==128//ratio and index.get_bytes_per_token()==68
        assert len(kv.kv_buffer)==len(owners)
    pairs=[p for p in pool.compress_state_pools if p is not None]
    assert pairs and all(p.ratio==2 and p.request_scoped and p.ring_size==8 for p in pairs)
    ptrs,lens,items=pool.get_request_state_buf_infos()
    assert len(ptrs)==len(lens)==len(items)==len(pairs) and all(items)
    return True


def has_real_c128(ctx):
    from lowratio_allocator import LowRatioNPUAllocator
    a=ctx.params.token_to_kv_pool_allocator
    if not isinstance(a,LowRatioNPUAllocator):return True
    pool=a.get_kvcache()
    assert pool.c128_kv_pool is None and a.c128_attn_allocator is None
    assert set(pool.compression_ratios)<=set([0,1,2])
    return False


def install(root):
    import sglang.srt.mem_cache.kv_cache_builder as builder
    import sglang.srt.mem_cache.registry as registry
    if getattr(builder,'_lowratio_radix_g491',False):return
    pins=json.loads((root/'native_radix_sha256.json').read_text())
    receipt={'native_source_sha256':{},'changes':{},'full_PD_prefix_quality_performance_accepted':False}
    for module,name,helper,helper_name,kind in [
        (builder,'validate_decode_radix_cache',supported,'_ds41_lowratio_supported','layout'),
        (registry,'create_unified_radix_cache',has_real_c128,'_ds41_has_real_c128','c128')]:
        path=Path(module.__file__);expected=pins[path.name]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        receipt['native_source_sha256'][path.name]=expected
        fn=getattr(module,name);tree=ast.parse('from __future__ import annotations\n'+inspect.getsource(fn));changed=0
        for node in ast.walk(tree):
            if not isinstance(node,ast.If):continue
            text=ast.unparse(node.test)
            if kind=='layout' and text=="getattr(model_config, 'is_deepseek_v4_arch', False) and (not getattr(token_to_kv_pool, '_unified_kv', False))":
                node.test=ast.BoolOp(op=ast.And(),values=[node.test,ast.UnaryOp(op=ast.Not(),operand=ast.parse(helper_name+'(model_config, token_to_kv_pool)',mode='eval').body)]);changed+=1
            elif kind=='c128' and text=="hasattr(params.req_to_token_pool, 'req_to_c128_sidecar')":
                node.test=ast.BoolOp(op=ast.And(),values=[node.test,ast.parse(helper_name+'(ctx)',mode='eval').body]);changed+=1
        assert changed==1,(name,changed)
        tree=ast.fix_missing_locations(tree);source=ast.unparse(tree)+'\n'
        overlay=root/f'native_radix_{name}.pid{os.getpid()}.py';overlay.write_text(source)
        module.__dict__[helper_name]=helper;context=dict(module.__dict__)
        exec(compile(tree,str(overlay),'exec'),context);updated=context[name]
        assert not fn.__code__.co_freevars and not updated.__code__.co_freevars
        fn.__code__=updated.__code__;receipt['changes'][name]={'conditions_extended':changed,'source_sha256':hashlib.sha256(source.encode()).hexdigest()}
    builder._lowratio_radix_g491=True
    receipt['scope']='Exact lowratio BF16/page128 C1/C2 FULL-owned pages with native paged SWA and request-scoped C2 state. No fake unified flag, no prefix disable, no native tree/transport replacement. C128 component only omitted when no real C128 pool/allocator.'
    (root/f'radix_install.pid{os.getpid()}.json').write_text(json.dumps(receipt,indent=2))
