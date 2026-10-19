"""Reuse only this C32 native host pool's verified existing SDK mapping.

The standard backend still owns registration bookkeeping, warmup, transfers and
cleanup. Ordinary pointers retain its original register_buffer implementation.
"""
import hashlib,json,os
from pathlib import Path

SOURCE_SHA='ceda6ee4a7515e2bbf1177aa18eea6949ee9ebf80f5bcb0041c0ff360f83a1c5'
START=0x280040000000
END=START+2*1024**4
REVISION='glm53-C32-native-MC-existing-mapping-v1'

def install(module):
    assert os.environ.get('GLM53_NODE') in ['34','35']
    assert os.environ.get('GLM53_C32_NATIVE_MC_MAPPING_REUSE')=='1'
    assert os.environ.get('GLM53_C32_NATIVE_HICACHE_MTP')=='1'
    assert os.environ.get('SGLANG_HICACHE_HOST_MEM_BACKEND')=='memfabric'
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SOURCE_SHA
    cls=module.NpuMemcacheStore
    assert not getattr(cls,'_glm53_native_mapping_reuse',False)
    original=cls.register_buffer
    def register(self,tensor):
        ptr=int(tensor.data_ptr());size=int(tensor.numel()*tensor.element_size())
        if ptr+size<=START or ptr>=END:
            return original(self,tensor)
        # Fail closed on a partial overlap, unknown pool or unproven mapping.
        assert START<=ptr<ptr+size<=END and self.store is not None
        host=self.mem_pool_host
        assert host.layout=='page_first_kv_split' and host.page_size==128
        assert host.kv_buffer is host.k_buffer and host.index_k_scale_buffer is None
        fields={'k_buffer':(634,79,128,1,512),
                'v_buffer':(634,79,128,1,64),
                'index_k_buffer':(634,78,128,1,128)}
        names=[name for name in fields if tensor is getattr(host,name)]
        assert len(names)==1 and tuple(tensor.shape)==fields[names[0]]
        assert tensor.device.type=='cpu' and str(tensor.dtype)=='torch.bfloat16'
        assert tensor.element_size()==2 and tensor.is_contiguous()
        from memfabric_hybrid import offload
        assert offload.get_dva(ptr)==ptr and offload.get_dva(ptr+size-1)==ptr+size-1
        print('GLM53_C32_NATIVE_MC_MAPPING_REUSED='+json.dumps({
            'project_id':'glm53','revision':REVISION,'pid':os.getpid(),
            'node':os.environ['GLM53_NODE'],'component':names[0],
            'shape':list(tensor.shape),'bytes':size,'existing_linear_mapping':True,
            'explicit_second_MC_registration':False}),flush=True)
        return None
    cls.register_buffer=register
    cls._glm53_original_register_buffer=original
    cls._glm53_native_mapping_reuse=True
    return {'revision':REVISION,'source_sha256':SOURCE_SHA}
