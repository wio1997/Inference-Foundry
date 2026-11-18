"""Allocate the complete native target/draft pool topology on each NPU.

No model weights, attention backend or network PD is exercised here.
"""
from pathlib import Path
import datetime, hashlib, json, os, traceback
import torch, torch_npu

rank = int(os.environ['LOCAL_RANK'])
os.sched_setaffinity(0, {168 + rank})
torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.npu.set_device(rank)
root = Path(__file__).resolve().parent
out = root / 'evidence'
out.mkdir(exist_ok=True)
record = dict(rank=rank, pid=os.getpid(), phase='initializing', cases=[],
              scope='G491 real NPU native cache insert/match/lock/evict and request-state contract, '
                    'local transfer byte geometry and index graphs; not weights, '
                    'attention, cross-host PD, full-model capacity or performance')

def save():
    p = out / f'rank{rank}.json'
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(record, indent=2))
    q.replace(p)

try:
    save()
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.runtime_context import create_context, use_context
    import importlib, sglang
    package_root = Path(sglang.__file__).resolve().parent.parent
    pins = json.loads((root / 'native_pool_sha256.json').read_text())
    for path, expected in pins.items():
        assert hashlib.sha256((package_root / path).read_bytes()).hexdigest() == expected, path
    record['native_source_sha256_verified'] = pins
    from lowratio_pool import LowRatioNPUTokenToKVPool
    from sglang.kernels.ops.quantization.fp4_fake_quant import fake_quant_fp4
    candidate = json.loads((root / 'candidate_d.json').read_text())
    model = Path('/models/ds41')
    config = json.loads((model / 'config.json').read_text())['text_config']
    argv = [str(model) if x == '/models/ds41' else x
            for x in candidate['argv'][1:]]
    # This component has no serving role and creates no transport endpoints.
    argv[argv.index('--disaggregation-mode') + 1] = 'null'
    argv.remove('--disaggregation-decode-enable-radix-cache')
    args = prepare_server_args(argv)
    context = create_context(args, role='model_runner')
    ratios = config['compress_ratios']
    sources = config['kv_source_layer_ids']
    target_layers = config['num_hidden_layers']
    record['checkpoint_topology'] = dict(ratios=ratios, sources=sources,
        target_layers=target_layers, config_sha256=hashlib.sha256(
            (model / 'config.json').read_bytes()).hexdigest())
    with use_context(context):
        kwargs = dict(max_num_reqs=16, num_req_slots=17, swa_size=8192,
            c4_size=0, c128_size=0, c4_state_pool_size=0,
            c128_state_pool_size=0, page_size=128, swa_page_size=128,
            dtype=torch.bfloat16, c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32, qk_nope_head_dim=448,
            qk_rope_head_dim=64, indexer_head_dim=128,
            device=f'npu:{rank}', enable_memory_saver=False,
            compression_ratios=ratios, sliding_window=128,
            online_mtp_max_draft_tokens=6, kv_source_layers=sources,
            full_size=8192)
        target = LowRatioNPUTokenToKVPool(
            **kwargs, layer_num=target_layers, start_layer=0,
            end_layer=target_layers, is_draft_worker=False)
        draft = LowRatioNPUTokenToKVPool(
            **kwargs, layer_num=len(ratios)-target_layers,
            start_layer=target_layers, end_layer=len(ratios),
            is_draft_worker=True)
        assert target.c4_kv_pool is None and target.c128_kv_pool is None
        assert draft.get_contiguous_buf_infos() == ([], [], [])
        assert target.get_c128_kv_buf_infos() == ([], [], [])
        ptrs, sizes, items = target.get_contiguous_buf_infos()
        expected = []
        for ratio, owners in target.sources_by_ratio.items():
            pool = target.kv_pools[ratio]
            indexer = target.index_pools[ratio]
            assert indexer.get_bytes_per_token() == 68
            assert not hasattr(indexer, 'index_k_buffer')
            assert pool.get_bytes_per_token() == 1024
            rows = 128 // ratio
            expected.extend([rows*1024]*len(owners) + [rows*68]*len(owners))
            view = pool.paged_buffer_info()
            assert view.page_slots == rows and view.page_bytes == rows*1024
            assert all(a.data_ptr() == b.data_ptr()
                       for a,b in zip(view.buffers, pool.kv_buffer))
        assert items == expected and len(items) == 2*len(sources)
        for layer in range(target_layers):
            ratio = ratios[layer]
            if ratio not in (1,2):
                continue
            owner = max(x for x in sources if x <= layer and ratios[x] == ratio)
            assert target.source_layer_of(layer) == owner
            assert target.get_key_buffer(layer).data_ptr() == target.get_key_buffer(owner).data_ptr()
        pairs = [p for p in target.compress_state_pools if p is not None]
        assert len(pairs) == len(target.sources_by_ratio[2])
        assert all(p.ratio == 2 and p.request_scoped and p.ring_size == 8
                   and p.kv_score_buffer.kv_score.dtype == torch.float32 for p in pairs)
        record['topology'] = dict(main_regions=len(ptrs), item_bytes=items,
            region_bytes=sizes, native_source_mapping=True,
            no_INT8_side_indexer_buffers=True, C2_request_ring_size=8,
            pair_state_buffers=len(pairs), draft_layers=len(ratios)-target_layers)
        save()
        from types import SimpleNamespace as NS
        from lowratio_allocator import LowRatioNPUAllocator
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_req_to_token_pool import DSV4NPUReqToTokenPool
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams,MatchPrefixParams,EvictParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        import sglang.srt.mem_cache.registry as registry
        import sglang.srt.mem_cache.kv_cache_builder as builder
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
        a=LowRatioNPUAllocator(8192,8192,128,torch.bfloat16,f'npu:{rank}',target,True)
        reqpool=DSV4NPUReqToTokenPool(16,8193,f'npu:{rank}',False)
        reqpool.register_dsv4_allocator(a)
        config=NS(is_deepseek_v4_arch=True,is_hybrid_swa_compress=False,hf_config=NS(model_type='deepseek_v41'))
        try:
            builder.validate_decode_radix_cache(model_config=config,token_to_kv_pool=target,is_hybrid_swa=True,enable_hierarchical_cache=False)
        except ValueError as exc:
            assert 'requires the unified KV layout' in str(exc)
            record['original_layout_guard_reproduced']=str(exc)
        else:raise AssertionError('Original native guard unexpectedly accepted')
        def cache_context():
            params=CacheInitParams(disable=False,req_to_token_pool=reqpool,token_to_kv_pool_allocator=a,page_size=128,sliding_window_size=128)
            return registry.TreeCacheBuildContext(params=params,server_args=args,is_hybrid_swa=True,is_hybrid_ssm=False,full_tokens_per_layer=1,enable_hierarchical_cache=False,disable_radix_cache=False,tp_worker=None,model_config=config,tp_size=1,tp_rank=0,tp_group=None)
        original=registry.create_unified_radix_cache(cache_context())
        assert ComponentType.C128 in original.components
        try:original.components[ComponentType.C128].floor_cache_len(512)
        except AttributeError as exc:
            assert 'page_size' in str(exc);record['original_absent_C128_failure_reproduced']=str(exc)
        else:raise AssertionError('Expected absent C128 allocator contract failure')
        from lowratio_radix_compat import install,supported
        install(root)
        builder.validate_decode_radix_cache(model_config=config,token_to_kv_pool=target,is_hybrid_swa=True,enable_hierarchical_cache=False)
        assert supported(config,target) and target._unified_kv is False
        try:
            builder.validate_decode_radix_cache(model_config=config,token_to_kv_pool=target,is_hybrid_swa=True,enable_hierarchical_cache=True)
        except ValueError:record['hierarchical_guard_preserved']=True
        else:raise AssertionError('Hierarchical guard removed')
        tree=registry.create_unified_radix_cache(cache_context())
        assert set(tree.components)=={ComponentType.FULL,ComponentType.SWA}
        record['native_tree_core']=type(tree.tree_core).__name__
        initial=(a.full_available_size(),a.swa_available_size())
        def allocate(n,slot):
            cpu0=torch.tensor([0],dtype=torch.int64);cpuN=torch.tensor([n],dtype=torch.int64)
            bundle=a.alloc_extend(cpu0.to(f'npu:{rank}'),cpu0,cpuN.to(f'npu:{rank}'),cpuN,torch.tensor([-1],device=f'npu:{rank}'),n,req_pool_indices=torch.tensor([slot],device=f'npu:{rank}'),req_to_token_pool=reqpool)
            assert bundle is not None and bundle.out_c128_loc.numel()==0
            return bundle.out_full_loc
        for n in [128,256,512,1024]:
            ids=list(range(n));loc=allocate(n,1)
            result=tree.insert(InsertParams(key=RadixKey(ids),value=loc.to(torch.int64)))
            hit=tree.match_prefix(MatchPrefixParams(key=RadixKey(ids)))
            got=tree.path_device_indices(hit.best_match_node)
            assert len(got)==n and torch.equal(got,loc.to(torch.int64))
            lock=tree.lock(hit.best_match_node)
            before=a.full_available_size();tree.evict_for_alloc(EvictParams(num_tokens=n))
            assert a.full_available_size()==before
            tree.unlock(lock);tree.evict_for_alloc(EvictParams(num_tokens=8192,swa_num_tokens=8192))
            torch.npu.synchronize()
            assert (a.full_available_size(),a.swa_available_size())==initial
            record['cases'].append(dict(kind='native_FULL_SWA_insert_match_lock_evict',tokens=n,no_allocation_leak=True,C1_C2_page_ownership_preserved=True))
            save()
        # Same FULL pages may be shared; C2 pending state remains private to a
        # request slot, cleared before receiving another request's odd tail.
        states=[x.kv_score_buffer.kv_score for x in pairs]
        for seq in [127,128,129,8191,8192,8193]:
            for state in states:
                state[8:16].fill_(rank+1.25);state[16:24].fill_(-99);state[24:32].fill_(7)
            indices1=target.request_state_transfer_indices(1,seq)
            indices2=target.request_state_transfer_indices(2,seq)
            assert indices1.tolist()==([1] if seq%2 else [])
            assert indices2.tolist()==([2] if seq%2 else [])
            target.clear_request_scoped_state(2)
            for state in states:
                assert torch.all(state[8:16]==rank+1.25).item() and torch.all(state[24:32]==7).item()
                if seq%2:state[16:24].copy_(state[8:16]);assert torch.equal(state[16:24],state[8:16])
                else:assert not torch.any(state[16:24]==-99).item()
            record['cases'].append(dict(kind='native_C2_request_state_clear_and_positional_copy',length=seq,source_slot=1,destination_slot=2,neighbor_unchanged=True,odd_pending_pair=bool(seq%2),scope='Local actual native raw state copy, not cross-host transport'))
            save()
        assert target._unified_kv is False and a.c128_attn_allocator is None
    record.update(phase='passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    save()
except BaseException as exc:
    record.update(phase='failed',error=repr(exc),traceback=traceback.format_exc())
    save()
    raise
