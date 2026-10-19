"""Project SDMA guard/audit, preserving native SGLang transfer ownership."""
import functools,hashlib,json,os,pathlib,threading,time
_local=threading.local();_lock=threading.RLock()
def audit(event,**fields):
 row={'project_id':'glm53','event':event,'pid':os.getpid(),'thread_id':threading.get_ident(),'monotonic_ns':time.monotonic_ns(),**fields}
 data=(json.dumps(row,sort_keys=True,default=str)+'\n').encode()
 root=pathlib.Path('/run/glm53/audit');root.mkdir(exist_ok=True)
 with _lock:
  fd=os.open(root/f'sdma-{os.getpid()}.jsonl',os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
  try:
   assert os.write(fd,data)==len(data)
  finally:os.close(fd)
 print('GLM53_SGLANG_SDMA='+data.decode().rstrip(),flush=True)
def prepare_sdk():
 import memfabric_hybrid as mf
 import memcache_hybrid as mc
 from importlib.metadata import version
 assert version('memfabric_hybrid')=='1.3.0' and version('memcache_hybrid')=='1.3.0'
 assert 'a085a7c905a51e8cd55c2ef16c74ffe95ded31bc' in pathlib.Path(mf.__file__).with_name('VERSION').read_text()
 assert '06495e8a381c65b7f3b2615687e6b93e5672e242' in pathlib.Path(mc.__file__).with_name('VERSION').read_text()
 assert os.environ['ASCEND_MF_TRANSFER_PROTOCOL']=='sdma' and os.environ['SGLANG_MOONCAKE_SEND_AUX_TCP']=='0'
 assert mf.TransferEngine.TransDataOpType.SDMA.value==1
 mf.set_log_level(0);mf.set_conf_store_tls(False,'')
 audit('sdk_guard',protocol='SDMA',protocol_value=1,sdk_paths=[mf.__file__,mc.__file__])
def initialized(engine):
 assert engine._get_transfer_protocol()=='sdma' and engine._resolve_trans_op_type('sdma').value==1
 assert engine.hostname in ['172.16.12.'+str(n) for n in (34,35,36,41)]
 assert engine.role in ('Prefill','Decode') and engine.store_url=='tcp://172.16.12.34:29590'
 assert engine.engine.get_rpc_port()>0
 audit('initialize_complete',rc=0,protocol='SDMA',protocol_value=1,role=engine.role,device_id=engine.npu_id,hostname=engine.hostname,session=engine.session_id,actual_rpc_port=engine.engine.get_rpc_port(),store_url=engine.store_url,loaded_libraries=sorted({line.split()[-1] for line in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(x in line for x in ('libmf_','libhcom','_pymf_transfer','libascend_hal','libmmc'))}))
def registered(engine,ptrs,lengths,rc):
 audit('register_complete',session=engine.session_id,allocations=len(ptrs),allocation_bytes=sum(lengths),rc=rc)
 if rc!=0:raise RuntimeError(f'GLM53 native SDMA memory registration failed rc={rc}')
def kv_registered(manager):
 args=manager.kv_args
 keys=['engine_rank','system_dp_rank','gpu_id','page_size','kv_cache_dtype_str','kv_buf_groups','hidden_kv_layers','draft_kv_layers','kv_item_lens','kv_data_lens','kv_layer_ids','aux_item_lens','aux_data_lens','state_types','state_item_lens','state_data_lens','state_layer_ids','kv_data_mem_kinds']
 metadata={key:getattr(args,key,None) for key in keys}
 assert args.kv_cache_dtype_str in ('bfloat16','torch.bfloat16')
 audit('model_state_registered',session=manager.engine.session_id,metadata=metadata)
def send_audited(function):
 @functools.wraps(function)
 def wrapped(self,*args,**kwargs):
  old=getattr(_local,'context',None);name=function.__name__;request=args[0] if args else None
  session=kwargs.get('mooncake_session_id',request if isinstance(request,str) else getattr(request,'mooncake_session_id',None))
  context={'kind':name,'session':session,'room':getattr(request,'room',None),'logical_call_id':f'{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}'}
  _local.context=context;audit('model_state_send_begin',**context)
  try:
   rc=function(self,*args,**kwargs);audit('model_state_send_complete',rc=rc,**context);return rc
  except BaseException as error:
   audit('model_state_send_error',error=repr(error),**context);raise
  finally:_local.context=old
 return wrapped
def transfer(engine,remote,buffers,peers,lengths,call):
 assert len(buffers)==len(peers)==len(lengths) and all(int(x)>0 for x in [*buffers,*peers,*lengths])
 descriptor_bytes=sum(int(x) for x in lengths);begin=time.monotonic_ns();context=getattr(_local,'context',None)
 fields={'local_session':engine.session_id,'remote_session':remote,'ranges':len(lengths),'descriptor_sha256':hashlib.sha256(json.dumps([list(buffers),list(peers),list(lengths)],default=int,separators=(',',':')).encode()).hexdigest(),'payload_bytes_requested':descriptor_bytes,'physical_link_bytes':None,'context':context}
 audit('write_submit',**fields)
 try:rc=call()
 except BaseException as error:
  audit('write_error',error=repr(error),**fields);raise
 audit('write_complete',rc=rc,payload_bytes_completed=descriptor_bytes if rc==0 else None,elapsed_ns=time.monotonic_ns()-begin,**fields)
 return rc
def reject_tcp(*args,**kwargs):
 audit('forbidden_tcp_model_payload');raise RuntimeError('GLM53 model state requires SDMA; TCP auxiliary payload forbidden')
