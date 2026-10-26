"""Bounded same-SDK MC/Offload registered-mapping reuse and native-copy prerequisite.
No model HTTP, no package/core/driver change, no model restart. Every MC key is unique and removed.
"""
import hashlib,json,os,sys,types
from pathlib import Path
OFFLOAD_START=0x280040000000
OFFLOAD_END=OFFLOAD_START+2*1024**4
NAMESPACE='glm53-20261019T-L12C-combined-MC-native-fixture-v1'
CASE=Path(os.environ['GLM53_NATIVE_CASE_ROOT'])
def emit(phase,**fields):
 data={'project_id':'glm53','phase':phase,**fields};temp=CASE/'step.tmp';temp.write_text(json.dumps(data)+'\n');temp.replace(CASE/'step.json');print('GLM53_COMBINED_MRE_STEP='+json.dumps(data),flush=True)
def reserve_keys(store,keys,planned):
 assert keys and all(k.startswith(NAMESPACE+'-') and len(k)<128 for k in keys) and not set(keys)&set(planned)
 assert list(store.batch_is_exist(keys))==[0]*len(keys),'No overwrite of existing MC keys'
 planned.extend(keys);(CASE/'keys.json').write_text(json.dumps(planned)+'\n')
def codes(value,count):
 out=list(map(int,value));assert len(out)==count and out==[0]*count,out;return out
def setup_store(store,mc,result):
 cfg=mc.LocalConfig();cfg.meta_service_url='tcp://172.16.12.34:29780';cfg.config_store_url='tcp://172.16.12.34:29781';cfg.protocol='device_sdma';cfg.world_size=128;cfg.dram_size='0GB';cfg.max_dram_size='160GB';cfg.hbm_size=cfg.max_hbm_size='0GB';cfg.log_level='info'
 result['SDK_setup_rc']=store.setup(cfg);assert result['SDK_setup_rc']==0
 result['SDK_bm_init_calls']=1;result['SDK_init_rc']=store.init(0,init_bm=True);assert result['SDK_init_rc']==0
 result['SDK_bm_init_calls']=1;result['SDK_configuration']={'device_id':0,'protocol':'device_sdma','dram_size':'0GB','max_dram_size':'160GB','world_size':128,'init_bm':True};emit('MC_client_initialized')
def plain_control(store,torch,result,planned):
 tensor=torch.empty(1024**2,dtype=torch.uint8,pin_memory=True);tensor.copy_((torch.arange(tensor.numel(),dtype=torch.int64)%251).to(torch.uint8));expected=tensor.clone();ptr=tensor.data_ptr();size=tensor.numel()
 assert not (OFFLOAD_START<=ptr<OFFLOAD_END)
 result['ordinary_pinned_tensor_retained_until_close']=tensor;result['ordinary_pinned_register_rc']=store.register_buffer(ptr,size);assert result['ordinary_pinned_register_rc']==0
 keys=[NAMESPACE+'-ordinary-pinned-control'];reserve_keys(store,keys,planned)
 codes(store.batch_put_from(keys,[ptr],[size]),1);tensor.fill_(253);codes(store.batch_get_into(keys,[ptr],[size]),1);assert torch.equal(tensor,expected)
 result['ordinary_registered_CPU_control_pass']=True;result['ordinary_pinned_tensor_retained_until_close']=tensor;emit('ordinary_registered_control_pass')
def storage_roundtrip(store,offload,torch,host,phase,result,planned):
 arrays=[('K',host.k_buffer),('V',host.v_buffer),('indexK',host.index_k_buffer)];expected=[x.clone() for _,x in arrays];keys=[];ptrs=[];sizes=[];checks=[]
 for name,tensor in arrays:
  assert tensor.device.type=='cpu' and tensor.dtype==torch.bfloat16 and tensor.is_contiguous() and tensor.shape[0]==5
  for page in range(5):
   view=tensor[page];ptr=view.data_ptr();size=view.numel()*view.element_size()
   assert OFFLOAD_START<=ptr and ptr+size<=OFFLOAD_END
   assert offload.get_dva(ptr)==ptr and offload.get_dva(ptr+size-1)==ptr+size-1,'Existing native local mapping must be linear and cover the entire buffer'
   keys.append(NAMESPACE+'-'+phase+'-'+name+'-'+str(page));ptrs.append(ptr);sizes.append(size)
 assert len(keys)==15 and sum(sizes)==71024640
 reserve_keys(store,keys,planned);put=codes(store.batch_put_from(keys,ptrs,sizes),15)
 infos=store.batch_get_key_info(keys);assert len(infos)==15
 for info,size in zip(infos,sizes):
  assert info is not None and int(info.size())==size
  gvas=[int(x) for x in info.gva_list() if int(x)>0];assert gvas and all(x>=OFFLOAD_END for x in gvas),'MC global key storage must be disjoint from the native Offload address span'
 for _,tensor in arrays:tensor.fill_(-79)
 get=codes(store.batch_get_into(keys,ptrs,sizes),15)
 for (name,tensor),wanted in zip(arrays,expected):
  assert torch.equal(tensor,wanted),(phase,name,'MC byte roundtrip mismatch')
  checks.append({'component':name,'whole_tensor_equal':True,'all5pages_including_packed_draft':True})
 result['storage_roundtrips'].append({'phase':phase,'objects':15,'bytes':sum(sizes),'put_codes':put,'get_codes':get,'existing_offload_mapping_reused':True,'explicit_MC_register_for_native_buffers':0,'native_mapping_linear_and_bounded':True,'MC_object_GVAs_disjoint_from_Offload_span':True,'checks':checks});emit('native_mapped_storage_roundtrip_pass',roundtrip=phase)
def verify_one_core_after(scope,mf):
 import ctypes
 core=ctypes.CDLL(str(scope['CORE']),mode=ctypes.RTLD_GLOBAL);alias=ctypes.CDLL(str(scope['ALIAS']),mode=ctypes.RTLD_GLOBAL);assert core._handle==alias._handle==mf._lib_handles[2]._handle
 class Info(ctypes.Structure):
  _fields_=[('base',ctypes.c_void_p),('name',ctypes.c_char_p)]
 names=[];callback_type=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.POINTER(Info),ctypes.c_size_t,ctypes.c_void_p)
 @callback_type
 def visit(ptr,size,data):
  value=(ptr.contents.name or b'').decode()
  if Path(value).name in ['libmf_hybm_core.so','libmf_smem.so','libmf_acc_offload.so']:names.append(Path(value).name)
  return 0
 libc=ctypes.CDLL(None);libc.dl_iterate_phdr.argtypes=[callback_type,ctypes.c_void_p];libc.dl_iterate_phdr.restype=ctypes.c_int;assert libc.dl_iterate_phdr(visit,None)==0
 assert sorted(names)==sorted(['libmf_hybm_core.so','libmf_smem.so','libmf_acc_offload.so'])
 assert scope['CORE'].with_name('libmf_hybm_accoffload.so').stat().st_size==0
 assert hashlib.sha256(scope['CORE'].read_bytes()).hexdigest()=='53736834872ab58915869c60ada0a719473b9b3628c112ded4dc2de8272bd126'

def main():
 assert os.environ['SGLANG_HICACHE_HOST_MEM_BACKEND']=='memfabric' and os.environ['GLM53_C32_NATIVE_HICACHE_MTP']=='1'
 scope={};exec(compile(LOADER_BOOTSTRAP_SOURCE,'qualified-same-inode-loader','exec'),scope);mf,mc,torch,loader=scope['load_sdk']()
 import torch_npu
 torch.npu.set_device(0);assert torch.npu.device_count()==16 and torch.npu.is_available()
 from memfabric_hybrid import offload
 extension=Path(os.environ['GLM53_NATIVE_LOADER_DIR'])/'libmf_hybm_accoffload.so';assert hashlib.sha256(extension.read_bytes()).hexdigest()=='3e87649f4d2ea85936cd94b439afd3473973bd62abcc06f777c9b875cf1a9d46'
 result={'project_id':'glm53','pass':False,'model_requests':0,'SDK_bm_init_calls':0,'SDK_offload_init_calls':0,'loader':loader,'storage_roundtrips':[],'native_explicit_MC_register_calls':0,'native_kernel_launches':0,'physical_device':0,'Host_reserve_bytes':1024**3,'maximum_own_MC_objects_bytes':2*71024640+1024**2,'original_model_configuration_changed':False};store=None;offload_started=False;planned_keys=[];launches=[];h=None;components=[];host=None
 try:
        from sglang.srt.mem_cache.pool_host import mla,npu_memfabric
        assert hashlib.sha256(Path(mla.__file__).read_bytes()).hexdigest()=='4009474db3858a74ca89e314bb4ea3a7840f0ce884d7c422fcb4fe6131ef5041'
        assert hashlib.sha256(Path(npu_memfabric.__file__).read_bytes()).hexdigest()=='121ab59ea6f558ec41b69dcb5b859821f40e4425812e46cb73ed91055cc22d0a'
        assert offload.kv_exchange_copy and offload.get_dva
        torch.npu.set_device(0);device=torch.device('npu:0')
        assert torch.npu.device_count()==16
        assert torch.npu.mem_get_info()[0]>=2*1024**3
        config=offload.OffloadConfig();config.device_id=0
        config.reserve_size=1024**3;config.alloc_size=1024**3
        config.flags=offload.OFFLOAD_FLAG_GIANT_PAGE;config.scene=offload.Scene.LOCAL
        result['SDK_offload_init_calls']=1;rc=offload.initialize(config);result['SDK_offload_init_rc']=rc;assert rc==0,('offload.initialize',rc)
        offload_started=True;emit('offload_initialized')
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
        store=mc.DistributedObjectStore();setup_store(store,mc,result);plain_control(store,torch,result,planned_keys)
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
        storage_roundtrip(store,offload,torch,h,'before-native-copy',result,planned_keys)
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
        storage_roundtrip(store,offload,torch,h,'after-native-copy',result,planned_keys)
        loaded=sorted({line.split(maxsplit=5)[5].strip() for line in Path('/proc/self/maps').read_text().splitlines() if 'libmf_hybm_accoffload.so' in line})
        assert loaded==[str(extension)]
        verify_one_core_after(scope,mf);result['single_SDK_core_after_all_operations']=True;result['compatibility_source_sha256']=hashlib.sha256(COMPATIBILITY_SOURCE.encode()).hexdigest()
        result.update(pass_=True,native_kernel_launches=len(launches),native_return_codes=sorted(set(launches)),native_copy_checks=checks,single_layer_isolation_all_components=True,allocated_host_bytes=allocated,fixture_device_tensor_bytes=sum(d.numel()*d.element_size() for _,d,_,_ in components),loaded_optional_extension_paths=loaded,optional_extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),target_layers=78,draft_layers=1,target_indexer_rows=78,BF16=True,not_performance_evidence=True)
        result['pass']=result.pop('pass_');assert len(result['storage_roundtrips'])==2 and len(checks)==10
 except BaseException as error:
        result['error']=repr(error);raise
 finally:
        cleanup_ok=True;result['native_kernel_launches']=len(launches)
        try:torch.npu.synchronize()
        except BaseException as e:cleanup_ok=False;result['cleanup_sync_error']=repr(e)
        if store is not None:
            try:
                states=list(store.batch_is_exist(planned_keys)) if planned_keys else [];assert len(states)==len(planned_keys)
                result['key_cleanup']=[]
                for key,state in zip(planned_keys,states):
                    assert int(state) in [0,1]
                    if int(state)==1:
                        rc=store.remove(key);result['key_cleanup'].append({'key':key,'remove_rc':rc});assert rc==0
                if planned_keys:assert list(store.batch_is_exist(planned_keys))==[0]*len(planned_keys)
                result['all_own_keys_absent_after']=True
            except BaseException as e:cleanup_ok=False;result['key_cleanup_error']=repr(e)
            try:result['SDK_close_rc']=store.close();assert result['SDK_close_rc']==0
            except BaseException as e:cleanup_ok=False;result['SDK_close_error']=repr(e)
        result.pop('ordinary_pinned_tensor_retained_until_close',None)
        h=None;components=[];host=None
        if result['SDK_offload_init_calls']>0:
            try:
                rc=offload.uninitialize();assert rc is None;result['offload_uninitialize_completed_void_API']=True
            except BaseException as e:cleanup_ok=False;result['offload_uninitialize_error']=repr(e)
        result['pass']=result['pass'] and cleanup_ok
        print('GLM53_MC_OFFLOAD_COMBINED_MRE_RESULT='+json.dumps(result),flush=True)
 assert result['pass']
if __name__=='__main__':main()
