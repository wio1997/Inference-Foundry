"""FULL-derived C1/C2 locations without a nonexistent C128 side allocator.

Retains native NPU location bundles, FULL/SWA lifecycle and request-state clear.
Full service, prefix and DSPARK rollback must be verified by runtime evidence.
"""
from pathlib import Path
import hashlib,json
import torch
from sglang.srt.hardware_backend.npu.dsv4.dsv4_allocator import DSV4NPUTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc


class LowRatioNPUAllocator(DSV4NPUTokenToKVPoolAllocator):
    def __init__(self,size,size_swa,page_size,dtype,device,kvcache,need_sort):
        assert kvcache.c128_kv_pool is None
        assert all(ratio in (1,2) or pool is None or not pool.kv_buffer
                   for ratio,pool in kvcache.kv_pools.items()), 'Real C4/C128 layers must retain native allocator'
        assert page_size==128 and dtype==torch.bfloat16 and str(device).startswith('npu')
        # C1/C2 source pools derive their address from FULL slots (loc // ratio),
        # as in the complete native generic backend. Only FULL/SWA are paged.
        SWATokenToKVPoolAllocator.__init__(self,size,size_swa,page_size,dtype,device,kvcache,need_sort)
        self.c128_attn_allocator=None
        self.c128_page_refcount=None
        self._empty_loc=torch.empty((0,),dtype=torch.int64,device=device)
        self._cur_req_to_token_pool=None

    def _alloc_compressed_kv(self,out_full_loc,out_swa_loc,*args,**kwargs):
        return DSV4OutCacheLoc(out_full_loc=out_full_loc,out_swa_loc=out_swa_loc,
                              out_c4_loc=self._empty_loc,out_c128_loc=self._empty_loc)

    def c128_num_pages_needed(self,prefix_lens_cpu,seq_lens_cpu):
        return 0

    def ensure_c128_capacity(self,tree_cache,num_pages):
        assert num_pages==0, 'A C128 demand cannot be served by this low-ratio pool'
        return True

    def _has_c128_sidecar_capacity(self,prefix_lens_cpu,seq_lens_cpu):
        return True

    def retain_c128_pages(self,page_ids):
        if page_ids.numel():
            assert not bool(torch.any(page_ids!=0).item()), 'Unexpected C128 prefix payload'

    def release_c128_pages(self,page_ids):
        self.retain_c128_pages(page_ids)

    def replace_req_c128_prefix(self,req_pool_idx,page_ids,req_to_token_pool):
        self.retain_c128_pages(page_ids)
        # Sidecar contains no C128 pages for this model. No C1/C2 payload is
        # dropped: those pools share the native FULL-page ownership and PD bytes.
        req_to_token_pool.req_to_c128_sidecar[req_pool_idx,:page_ids.numel()].zero_()

    def available_size(self):
        return SWATokenToKVPoolAllocator.available_size(self)

    def resize(self,config):
        return SWATokenToKVPoolAllocator.resize(self,config)


def install(root):
    import sglang.srt.mem_cache.kv_cache_configurator as native
    import sglang
    package=Path(sglang.__file__).resolve().parent.parent
    for name,expected in json.loads((root/'native_allocator_sha256.json').read_text()).items():
        assert hashlib.sha256((package/name).read_bytes()).hexdigest()==expected,name
    from lowratio_pool import LowRatioNPUTokenToKVPool
    cls=native.KVCacheConfigurator
    prior=cls._build_token_to_kv_pool_allocator
    def create(self,*,sizes,token_to_kv_pool,is_dsv4_model,req_to_token_pool,token_to_kv_pool_allocator):
        if not isinstance(token_to_kv_pool,LowRatioNPUTokenToKVPool):
            return prior(self,sizes=sizes,token_to_kv_pool=token_to_kv_pool,is_dsv4_model=is_dsv4_model,req_to_token_pool=req_to_token_pool,token_to_kv_pool_allocator=token_to_kv_pool_allocator)
        assert is_dsv4_model and token_to_kv_pool_allocator is None
        assert getattr(self.model_config.hf_config,'model_type',None)=='deepseek_v41'
        from sglang.srt.runtime_context import get_disagg,get_schedule
        allocator=LowRatioNPUAllocator(sizes.full_max_total_num_tokens,sizes.swa_max_total_num_tokens,
            get_schedule().page_size,self.kv_cache_dtype,self.device,token_to_kv_pool,
            get_disagg().disaggregation_mode in ('decode','prefill'))
        assert hasattr(req_to_token_pool,'register_dsv4_allocator')
        req_to_token_pool.register_dsv4_allocator(allocator)
        return allocator
    cls._build_token_to_kv_pool_allocator=create
