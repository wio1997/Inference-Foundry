"""Diagnostic allocation: actual native C2 ring8 on no-SP D, matching SP P.

No fake layout flags and no validator change. Shared resolver identity keeps
native pool allocation and native fixed-byte pricing on the same geometry.
Requires real model/PD/correctness gates; this is not the final aligned config.
"""
from pathlib import Path
import hashlib,json,os,types

_DONE=False
_ROOT=None
_SEEN=set()

def _record(kind,**fields):
    key=(kind,tuple((k,str(v)) for k,v in fields.items()))
    if key in _SEEN:return
    _SEEN.add(key)
    with (_ROOT/f'targetonly_ring8.pid{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind,pid=os.getpid(),**fields))+'\n')

def _ring(compress_ratio,num_draft_tokens=0):
    prior=_targetonly_ring8_prior(compress_ratio,num_draft_tokens)
    return 8 if compress_ratio==2 and num_draft_tokens==0 else prior

def install(root):
    global _DONE,_ROOT
    if _DONE:return
    import torch
    import sglang.srt.mem_cache.deepseek_v4_memory_pool as pools
    import sglang.srt.model_executor.pool_configurator as planner
    from sglang.srt.utils import is_npu
    assert is_npu(),'Diagnostic is scoped to the actual NPU platform'
    root=Path(root);cfg=json.loads((root/'candidate.json').read_text());argv=cfg['argv']
    assert argv[argv.index('--disaggregation-mode')+1]=='decode'
    assert '--speculative-algorithm' not in argv
    assert '--disaggregation-decode-enable-radix-cache' in argv
    pins=json.loads((root/'native_pool_sha256.json').read_text())
    assert hashlib.sha256(Path(pools.__file__).read_bytes()).hexdigest()==pins['sglang/srt/mem_cache/deepseek_v4_memory_pool.py']
    pricepin=json.loads((root/'KV_native_source_sha256.json').read_text())['pool_configurator.py']
    assert hashlib.sha256(Path(planner.__file__).read_bytes()).hexdigest()==pricepin
    resolver=pools.get_compress_state_ring_size
    assert resolver(2,0)==2 and resolver(2,6)==8
    assert planner.resolve_compress_state_ring_size is pools.resolve_compress_state_ring_size
    assert pools.resolve_compress_state_ring_size.__globals__['get_compress_state_ring_size'] is resolver
    prior=types.FunctionType(resolver.__code__,dict(resolver.__globals__),resolver.__name__,resolver.__defaults__,resolver.__closure__)
    prior.__kwdefaults__=resolver.__kwdefaults__
    assert not resolver.__code__.co_freevars and not _ring.__code__.co_freevars
    pools.__dict__['_targetonly_ring8_prior']=prior
    resolver.__code__=_ring.__code__
    assert resolver(2,0)==8 and resolver(2,6)==8
    _ROOT=root
    cls=pools.DeepSeekV4TokenToKVPool
    old_make=cls._make_pair_state_pool
    def make(self,enable_memory_saver):
        assert self.get_ring_size(2)==8
        value=old_make(self,enable_memory_saver)
        assert value.ratio==2 and value.request_scoped and value.ring_size==8
        state=value.kv_score_buffer.kv_score
        assert state.dtype==torch.float32 and state.ndim==2
        # Metadata only: no tensor value read, device synchronization, or math.
        _record('actual_pool',ring_size=value.ring_size,request_scoped=value.request_scoped,
                shape=list(state.shape),dtype=str(state.dtype),bytes=state.nbytes,
                row_bytes=state.shape[-1]*state.element_size(),
                wire_item_bytes=value.ring_size*state.shape[-1]*state.element_size())
        return value
    cls._make_pair_state_pool=make
    old_infos=cls.get_request_state_buf_infos
    def infos(self):
        ptrs,lengths,items=old_infos(self)
        pairs=[v for v in self.compress_state_pools if v is not None and v.request_scoped]
        assert len(ptrs)==len(lengths)==len(items)==len(pairs)
        shapes=[]
        for value,item in zip(pairs,items):
            state=value.kv_score_buffer.kv_score
            assert value.ratio==2 and value.ring_size==8
            assert item==8*state.shape[-1]*state.element_size()
            shapes.append(list(state.shape))
        _record('actual_wire',entries=len(items),item_bytes=items,buffer_bytes=lengths,shapes=shapes)
        return ptrs,lengths,items
    cls.get_request_state_buf_infos=infos
    old_price=planner.DSV4PoolConfigurator._get_c2_state_fixed_bytes
    def price(self,max_running_requests):
        if self.num_layers(2):assert self.ring_sizes[2]==8
        result=old_price(self,max_running_requests)
        _record('actual_pricing',max_running_requests=max_running_requests,
                req_slots=self._get_num_req_slots(max_running_requests),
                C2_layers=self.num_layers(2),ring_size=self.ring_sizes.get(2),fixed_bytes=result)
        return result
    planner.DSV4PoolConfigurator._get_c2_state_fixed_bytes=price
    _DONE=True
    _record('installed',native_pool_SHA256=pins['sglang/srt/mem_cache/deepseek_v4_memory_pool.py'],
            native_pricing_SHA256=pricepin,resolver_alias_identity_preserved=True,
            allocation_and_pricing_same_resolver=True,
            scope='Diagnostic actualC2ring8 allocationonnoSPD; no validator/layoutflags/maths changed. Fullmodel/PD/qualityunverified.')
