# SPDX-License-Identifier: Apache-2.0
"""Existing native Memcache plus strict MF1.2 SDMA, synthetic bytes only."""
import pathlib,json,hashlib,os,time,traceback,importlib.metadata,sys
R=pathlib.Path('/run-evidence');node=int(os.environ['FLASH_COMPONENT_NODE']);assert node in (14,15);role='Prefill' if node==14 else 'Decode'
out={'project_id':'glm53-flash','workstream':'A','node':node,'passed':False,'scope':'Singledevice synthetic nativecache restore + strictSDMA coexistence component only, not native16rank/model/Graph/MTP/cachehit80K/performance','generation_requests':0,'formal_requests':0,'stage':'start'};e=store=None;cache_initialized=False
def save():
 p=R/'worker-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2));t.replace(p)
def wait_json(name,seconds=150):
 deadline=time.monotonic()+seconds
 while time.monotonic()<deadline:
  if (R/'abort-component.json').is_file():raise RuntimeError('Peer component aborted')
  p=R/name
  if p.is_file():return json.loads(p.read_text())
  time.sleep(.5)
 raise TimeoutError(name)
def flags(values):return [int(x) for x in values]
try:
 out['allocator_env']=os.environ.get('PYTORCH_NPU_ALLOC_CONF');assert out['allocator_env']=='expandable_segments:True';save()
 import torch,torch_npu
 torch.set_num_threads(2);torch.npu.set_device(0) # Nativeframeworkdeviceowner exists BEFORE lazySDKbackendimport
 from vllm_ascend.distributed.kv_transfer.utils.flash_native_sdk_bindings import load_bindings
 hybrid,transfer,mmc=load_bindings();TransferEngine=transfer.TransferEngine
 out['native_binding_files']={module.__name__:{'path':module.__file__,'sha256':hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest()} for module in [hybrid,transfer,mmc]};out['original_public_SDK_packages_imported']=any(name in sys.modules for name in ['memfabric_hybrid','memcache_hybrid']);assert not out['original_public_SDK_packages_imported'];out['native_device_initialized_before_bindings']=torch.npu.current_device()==0;assert out['native_device_initialized_before_bindings'];save()
 assert hasattr(TransferEngine,'batch_register_memory');out['MF_batch_register_doc']=TransferEngine.batch_register_memory.__doc__;out['lifecycle_API_docs']={cls.__name__:{n:getattr(cls,n).__doc__ for n in ['close','shutdown','destroy','uninitialize','uninit'] if hasattr(cls,n)} for cls in [TransferEngine,mmc.DistributedObjectStore,mmc.MetaService]};save()
 required=['init','register_buffer','batch_is_exist','batch_get_key_info','batch_put_from_layers','batch_get_into_layers']
 out['native_API_present']={n:hasattr(mmc.DistributedObjectStore,n) for n in required};out['native_API_doc']={n:getattr(mmc.DistributedObjectStore,n).__doc__ for n in required if hasattr(mmc.DistributedObjectStore,n)};save();assert all(out['native_API_present'].values()),'Native whole-segment API missing: no silent packing/fallback'
 assert importlib.metadata.version('memcache-hybrid')==importlib.metadata.version('memfabric-hybrid')=='1.2.0'
 from vllm_ascend.distributed.kv_transfer.utils.vllm_flash_sdma_performance_engine import FlashPerformanceSdmaEngine
 from vllm_ascend.distributed.kv_transfer.utils.sdma_payload_hash import PayloadHasher
 assert not any(x=='mooncake' or x.startswith('mooncake.') for x in sys.modules),'Unexpected broken Mooncake loaded'
 # Honest component topology; do not spoof official DP2TP8/DP16TP1 qualification.
 e=FlashPerformanceSdmaEngine('172.16.12.'+str(node),role,0,{'scope':'singledevicecomponent','tp_size':1,'dp_size':1,'tp_rank':0,'dp_rank':0,'global_device_ordinal':0})
 out['protocol']=str(e.protocol);assert out['protocol']=='TransDataOpType.SDMA';out['session_id']=e.session_id;out['adapter_sha256']=e.source_sha256
 if node==14:(R/'PD-engine-ready.json').write_text(json.dumps({'project_id':'glm53-flash','node':14,'session_id':e.session_id,'protocol':out['protocol']}))
 out['stage']='strict_SDMA_initialized';save()
 store=mmc.DistributedObjectStore();out['cache_init_rc']=int(store.init(0,init_bm=(node==14)));save();assert out['cache_init_rc']==0;cache_initialized=True;out['cache_init_bm']=node==14
 expected=json.loads((R/'expected.json').read_text());sizes=expected['sizes'];key='glm53-flash-r157-native-bf16-f32-coexist'
 def register_groups(groups):
  all_tensors=[x for group in groups for x in group];all_ptrs=[int(x.data_ptr()) for x in all_tensors];all_sizes=sizes*len(groups)
  assert [x.numel()*x.element_size() for x in all_tensors]==all_sizes
  out['native_registration_vectors']={'ptrs':all_ptrs,'sizes':all_sizes,'SDK_batch_calls':1};save()
  e.register_memories(all_ptrs,all_sizes)
  for ptr,size in zip(all_ptrs,all_sizes):
   if node==14:
    rc=int(store.register_buffer(ptr,size));assert rc==0,('cache_register',rc)
  out['stage']='all_native_buffers_registered';save()
  return [all_ptrs[i*3:(i+1)*3] for i in range(len(groups))]
 def checksum(ptrs):return e.hasher.hash_ranges(ptrs,sizes)
 def check(value):
  assert value['range_sha256']==expected['range_sha256'] and value['sha256']==expected['sha256'] and value['bytes']==sum(sizes)
 def object_info():
  deadline=time.monotonic()+30
  while True:
   exists=flags(store.batch_is_exist([key]))
   if exists==[1]:break
   if time.monotonic()>=deadline:raise TimeoutError('Known key not present after one put')
   time.sleep(.2)
  info=store.batch_get_key_info([key]);assert len(info)==1 and info[0] is not None and int(info[0].size())==sum(sizes)
  return {'exists':exists,'size_bytes':int(info[0].size()),'native_media_type_list':flags(info[0].type_list()),'native_GVA_count':len(info[0].gva_list()),'tier_name':None}
 if node==14:
  cpu=[((torch.arange(257*128,dtype=torch.int32)%251)-125).to(torch.bfloat16).reshape(257,128),(((torch.arange(8*128*128,dtype=torch.int32)%257)-128).float()/1024).reshape(8,128,128),(((torch.arange(3*8*128,dtype=torch.int32)%61)-30).float()/256).reshape(3,8,128)]
  raw=[x.view(torch.uint8).numpy().tobytes() for x in cpu];assert [hashlib.sha256(x).hexdigest() for x in raw]==expected['range_sha256'] and hashlib.sha256(b''.join(raw)).hexdigest()==expected['sha256']
  original=[x.to('npu') for x in cpu];restored=[torch.zeros_like(x) for x in original];source,restored_ptrs=register_groups([original,restored]);e.synchronize_producer_stream();check(checksum(source))
  out['cache_put_rc']=flags(store.batch_put_from_layers([key],[source],[sizes],0));save();assert out['cache_put_rc']==[0]
  out['cache_object']=object_info();out['cache_first_get_rc']=flags(store.batch_get_into_layers([key],[restored_ptrs],[sizes],1));save();assert out['cache_first_get_rc']==[0];torch.npu.synchronize();out['restored_before_PD']=checksum(restored_ptrs);check(out['restored_before_PD'])
  out['stage']='cache_restored_bytes_ready';save();(R/'producer-ready.json').write_text(json.dumps({'project_id':'glm53-flash','node':14,'session_id':e.session_id,'protocol':out['protocol'],'source_ptrs':restored_ptrs,'sizes':sizes,'expected_sha256':expected['sha256'],'cache_object':out['cache_object']}))
  done=wait_json('consumer-done.json',240);assert done.get('consumer_stopped_idle8') and done.get('consumer_passed'),done
  # Intentional second get, not retry: ensure raw PD engine did not invalidate
  # the same-process native cache context or its registered source buffers.
  for x in original:x.zero_()
  out['cache_after_PD_get_rc']=flags(store.batch_get_into_layers([key],[source],[sizes],1));assert out['cache_after_PD_get_rc']==[0];torch.npu.synchronize();out['restored_after_PD']=checksum(source);check(out['restored_after_PD']);check(checksum(restored_ptrs))
 else:
  destination=[torch.zeros((257,128),dtype=torch.bfloat16,device='npu'),torch.zeros((8,128,128),dtype=torch.float32,device='npu'),torch.zeros((3,8,128),dtype=torch.float32,device='npu')];ptrs=register_groups([destination])[0]
  peer=wait_json('producer-peer.json');assert peer['project_id']=='glm53-flash' and peer['node']==14 and peer['protocol']=='TransDataOpType.SDMA' and peer['sizes']==sizes and peer['expected_sha256']==expected['sha256']
  # Only metadata uses Memcache on D. All cross-node payload is this explicit
  # SDMA SDK call; cache payload put/get remains local on P, matching disabled
  # consumer load/put in the eventual model profile.
  out['cache_object_metadata_only']=object_info()
  out['SDMA_rc']=e.batch_transfer_sync_read(peer['session_id'],ptrs,peer['source_ptrs'],sizes);assert out['SDMA_rc']==0;out['D_after_PD']=checksum(ptrs);check(out['D_after_PD']);assert e._stats['calls']==e._stats['successful_calls']==1 and e._stats['bytes']==sum(sizes) and e._stats['ranges']==3
 out['transport_stats']=dict(e._stats);out['passed']=True;out['stage']='completed';save()
except BaseException as exc:out.update(error=repr(exc),traceback=traceback.format_exc());save()
finally:
 # Actual SDK lifecycle APIs/docstrings captured from R138, no guessed names.
 # Strong tensor references remain alive through both SDK releases. P reaches
 # here only after D terminal or peer failure; D native read is synchronous.
 try:
  if e is not None:
   torch.npu.synchronize();out['cleanup_device_drained']=True
   if e.hash_server is not None:e.hash_server.shutdown();e.hash_server.server_close();out['own_hash_server_stopped']=True
  if cache_initialized:
   out['cache_close_rc']=int(store.close());assert out['cache_close_rc']==0
  if e is not None:e._engine.destroy();out['MF_destroy_returned']=True
 except BaseException as exc:out.update(passed=False,cleanup_error=repr(exc),cleanup_traceback=traceback.format_exc())
 out['public_SDK_packages_imported_at_terminal']=any(name in sys.modules for name in ['memfabric_hybrid','memcache_hybrid']);out['native_SDK_mapped_libraries_at_terminal']=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if 'libmf_' in line or '_pymf_' in line or '_pymmc' in line});assert not out['public_SDK_packages_imported_at_terminal'];save()
(R/'worker-result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out));sys.exit(0 if out['passed'] else 1)
