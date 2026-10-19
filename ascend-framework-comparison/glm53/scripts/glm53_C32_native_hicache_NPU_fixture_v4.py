"""Bounded native byte-equivalence fixture, 1 owned die, ~55MiB HBM/1GiB DRAM.

No model calls, SDK/driver installation, pool eviction, or performance claim.
Private offload entity and allocations end with this fixture process.
"""
import hashlib,json,os,sys,types
from pathlib import Path

def main():
    assert os.environ['GLM53_C32_NATIVE_HICACHE_MTP']=='1'
    extension=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so')
    assert not os.environ.get('MEMFABRIC_HYBRID_EXTEND_LIB_PATH')
    assert str(extension)=='/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so'
    assert hashlib.sha256(extension.read_bytes()).hexdigest()==os.environ['GLM53_NATIVE_EXTENSION_SHA256']
    assert os.environ['SGLANG_HICACHE_HOST_MEM_BACKEND']=='memfabric'
    import torch,torch_npu
    from importlib.metadata import version
    assert version('memfabric_hybrid')==version('memcache_hybrid')=='1.3.0'
    from memfabric_hybrid import offload
    print('GLM53_NATIVE_FIXTURE_VISIBILITY='+json.dumps({'device_nodes':sorted(str(p) for p in Path('/dev').glob('davinci[0-9]*')),'ASCEND_VISIBLE_DEVICES':os.environ.get('ASCEND_VISIBLE_DEVICES'),'ASCEND_RT_VISIBLE_DEVICES':os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),'device_count':torch.npu.device_count()}),flush=True)
    torch.npu.set_device(0)
    from sglang.srt.mem_cache.pool_host import mla,npu_memfabric
    assert hashlib.sha256(Path(mla.__file__).read_bytes()).hexdigest()=='4009474db3858a74ca89e314bb4ea3a7840f0ce884d7c422fcb4fe6131ef5041'
    assert hashlib.sha256(Path(npu_memfabric.__file__).read_bytes()).hexdigest()=='121ab59ea6f558ec41b69dcb5b859821f40e4425812e46cb73ed91055cc22d0a'
    assert offload.kv_exchange_copy and offload.get_dva
    torch.npu.set_device(0);device=torch.device('npu:0')
    assert torch.npu.device_count()==16
    assert torch.npu.mem_get_info()[0]>=1024**3
    config=offload.OffloadConfig();config.device_id=0
    config.reserve_size=1024**3;config.alloc_size=1024**3
    config.flags=offload.OFFLOAD_FLAG_GIANT_PAGE;config.scene=offload.Scene.LOCAL
    rc=offload.initialize(config);assert rc==0,('offload.initialize',rc)
    compatibility=types.ModuleType('glm53_native_MTP_compatibility')
    exec(compile(COMPATIBILITY_SOURCE,'frozen-native-MTP-compatibility','exec'),compatibility.__dict__)
    compatibility.install(mla)
    h=mla.MLATokenToKVPoolHost.__new__(mla.MLATokenToKVPoolHost);h.page_size=128
    h.k_buffer=offload.empty([5,79,128,1,512],dtype=torch.bfloat16)
    h.v_buffer=offload.empty([5,79,128,1,64],dtype=torch.bfloat16)
    h.index_k_buffer=offload.empty([5,78,128,1,128],dtype=torch.bfloat16)
    h.index_k_scale_buffer=None
    allocated=sum(t.numel()*t.element_size() for t in [h.k_buffer,h.v_buffer,h.index_k_buffer])
    assert allocated==71024640 and all(offload.get_dva(t.data_ptr())!=0 for t in [h.k_buffer,h.v_buffer,h.index_k_buffer])
    def pool(layers,index):
        return types.SimpleNamespace(k_buffer=torch.empty((layers,4,128,1,512),dtype=torch.bfloat16,device=device),v_buffer=torch.empty((layers,4,128,1,64),dtype=torch.bfloat16,device=device),index_k_buffer=torch.empty((layers,4,128,1,128),dtype=torch.bfloat16,device=device) if index else None,index_k_scale_buffer=None)
    target=pool(78,True);draft=pool(1,False);h.device_pool=target;h.mtp_draft_device_pools=(draft,)
    components=[('targetK',target.k_buffer,h.k_buffer[:,:78],11),('targetV',target.v_buffer,h.v_buffer[:,:78],29),('targetIndexer',target.index_k_buffer,h.index_k_buffer,47),('draftK',draft.k_buffer,h.k_buffer[:,78:79],83),('draftV',draft.v_buffer,h.v_buffer[:,78:79],101)]
    def pattern(shape,seed):
        size=1
        for d in shape:size*=d
        return ((torch.arange(size,dtype=torch.int64)+seed)%251).to(torch.bfloat16).reshape(shape)
    host_pages=[4,1,3];device_pages=[2,0,3]
    host_indices=torch.tensor([p*128+t for p in host_pages for t in range(128)],dtype=torch.int64)
    device_indices=torch.tensor([p*128+t for p in device_pages for t in range(128)],dtype=torch.int64)
    original_kernel=offload.kv_exchange_copy;launches=[]
    def kernel(meta,device_id):
        assert device_id.index==0,'All nativecopy operations must remain onphysical/logicaldie0'
        rc=original_kernel(meta,device_id);launches.append(rc);assert rc==0
        return rc
    offload.kv_exchange_copy=kernel
    for name,d,host,seed in components:host.copy_(pattern(host.shape,seed));d.fill_(-17)
    torch.npu.synchronize()
    for layer in range(78):
        h._transfer_ascendc_sparse_copy(target,host_indices,device_indices,mla.TransferDirection.H2D,layer_start=layer,layer_num=1,index_k_layer_start=layer,index_k_layer_num=1)
    h._transfer_ascendc_sparse_copy(draft,host_indices,device_indices,mla.TransferDirection.H2D,layer_num=1,index_k_layer_num=1)
    torch.npu.synchronize();checks=[]
    for name,d,host,seed in components:
        expected=torch.full(d.shape,-17,dtype=torch.bfloat16)
        for hp,dp in zip(host_pages,device_pages):expected[:,dp]=host[hp]
        actual=d.cpu();assert torch.equal(actual,expected),(name,'H2D mismatch')
        checks.append({'component':name,'direction':'H2D','whole_tensor_equal':True,'unselected_device_page_sentinel_preserved':True})
    for name,d,host,seed in components:d.copy_(pattern(d.shape,seed+113));host.fill_(-31)
    torch.npu.synchronize()
    h._transfer_ascendc_sparse_copy(target,host_indices,device_indices,mla.TransferDirection.D2H)
    torch.npu.synchronize()
    for name,d,host,seed in components:
        expected=torch.full(host.shape,-31,dtype=torch.bfloat16);source=pattern(d.shape,seed+113)
        for hp,dp in zip(host_pages,device_pages):expected[hp]=source[:,dp]
        assert torch.equal(host,expected),(name,'D2H mismatch')
        checks.append({'component':name,'direction':'D2H','whole_tensor_equal':True,'unselected_host_page_sentinel_preserved':True})
    before_draft={name:d.cpu() for name,d,host,seed in components if name.startswith('draft')}
    for name,d,host,seed in components:
        if not name.startswith('draft'):d.fill_(-43)
    h._transfer_ascendc_sparse_copy(target,host_indices,device_indices,mla.TransferDirection.H2D,layer_start=17,layer_num=1,index_k_layer_start=17,index_k_layer_num=1)
    torch.npu.synchronize()
    for name,d,host,seed in components:
        if name.startswith('draft'):expected=before_draft[name]
        else:
            expected=torch.full(d.shape,-43,dtype=torch.bfloat16)
            for hp,dp in zip(host_pages,device_pages):expected[17,dp]=host[hp,17]
        assert torch.equal(d.cpu(),expected),(name,'perlayer isolation mismatch')
    assert len(launches)==82 and set(launches)=={0}
    loaded=sorted({line.split(maxsplit=5)[5].strip() for line in Path('/proc/self/maps').read_text().splitlines() if 'libmf_hybm_accoffload.so' in line})
    assert loaded==[str(extension)]
    print('GLM53_NATIVE_HICACHE_FIXTURE_RESULT='+json.dumps({'project_id':'glm53','pass':True,'model_requests':0,'SDK_bm_init_calls':0,'SDK_offload_init_calls':1,'selected_device':0,'visible_devices_for_runtime_compatibility':16,'all_kernel_calls_device0':True,'native_kernel_launches':len(launches),'native_return_codes':sorted(set(launches)),'reserved_host_bytes':1024**3,'allocated_host_bytes':allocated,'fixture_device_tensor_bytes':sum(d.numel()*d.element_size() for _,d,_,_ in components),'page_size':128,'target_layers':78,'draft_layers':1,'target_indexer_rows':78,'BF16':True,'checks':checks,'single_layer_isolation_all_components':True,'private_process_exit_cleanup':True,'loaded_optional_extension_paths':loaded,'optional_extension_sha256':os.environ['GLM53_NATIVE_EXTENSION_SHA256'],'native_MLA_sha256':compatibility.SOURCE_SHA,'compatibility_sha256':hashlib.sha256(COMPATIBILITY_SOURCE.encode()).hexdigest(),'not_performance_evidence':True}))
if __name__=='__main__':main()
