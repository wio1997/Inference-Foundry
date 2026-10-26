"""Original backend, owned NPU buffers, stdin control only; CONFIG supplied by host controller."""
import hashlib,importlib.metadata,json,os,pathlib,subprocess,sys,traceback
plan=CONFIG['plan'];node=CONFIG['node'];role='P' if node==44 else 'D';N=plan['buffer_bytes'];engine=None;buffer=None;registered=False;checks=[];stage='imports';success=False

def event(kind,**values):print('L039_EVENT='+json.dumps({'role':role,'kind':kind,'stage':stage,**values}),flush=True)
def digest(t):return hashlib.sha256(t.numpy().tobytes()).hexdigest()
def source(g):return ((torch.arange(N,dtype=torch.int64)*17+31*g+7)%256).to(torch.uint8)
try:
 os.environ['ASCEND_MF_STORE_URL']=plan['store_url'];os.environ['ASCEND_MF_TRANSFER_PROTOCOL']=plan['protocol']
 import torch,torch_npu
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.npu.set_device(0)
 stage='memfabric_import_after_context';import memfabric_hybrid
 from sglang.srt.disaggregation.ascend.transfer_engine import AscendTransferEngine
 from sglang.srt.disaggregation.utils import DisaggregationMode
 root=pathlib.Path('/sgl-workspace/sglang');head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==plan['source_commit'];files={}
 for name in ['python/sglang/srt/disaggregation/ascend/transfer_engine.py','python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py']:
  data=(root/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD:'+name],text=True).strip()==blob;files[name]={'sha256':hashlib.sha256(data).hexdigest(),'git_blob_sha1':blob,'matches_HEAD':True}
 stage='common_init';ret=memfabric_hybrid.initialize();assert ret==0,ret
 if role=='D':
  stage='create_owned_shared_store';ret=memfabric_hybrid.create_config_store(plan['store_url']);assert ret==0,ret
 stage='original_engine_init';engine=AscendTransferEngine(hostname=f'172.16.12.{node}',npu_id=0,disaggregation_mode=DisaggregationMode.PREFILL if role=='P' else DisaggregationMode.DECODE);assert engine._get_transfer_protocol()==plan['protocol']
 stage='register_owned_buffer';buffer=torch.empty(N,dtype=torch.uint8,device='npu:0');engine.register(buffer.data_ptr(),N);registered=True
 event('ready',node=node,session_id=engine.get_session_id(),buffer_address=buffer.data_ptr(),buffer_bytes=N,protocol=engine._get_transfer_protocol(),store_url=plan['store_url'],source_commit=head,source_files=files,memfabric_version=importlib.metadata.version('memfabric-hybrid'),NPU_name=torch.npu.get_device_name(0))
 for line in sys.stdin:
  command=json.loads(line);op=command['op'];stage=op
  if op=='finish':
   assert len(checks)==len(plan['cases']);stage='deregister';engine.deregister(buffer.data_ptr());registered=False;success=True;break
  assert op in ['prepare','send','verify'],op
  case=next(x for x in plan['cases'] if x['id']==command['case']);expected_index=len(checks);assert case==plan['cases'][expected_index]
  if op=='prepare':
   assert role=='D'
   buffer.fill_(case['sentinel']);torch.npu.synchronize();event('armed',case=case['id'],generation=case['generation'],sentinel=case['sentinel']);continue
  if op=='send':
   assert role=='P';peer=command['peer'];assert peer['buffer_bytes']==N and peer['node']==46 and peer['session_id'].startswith('172.16.12.46:')
   cpu=source(case['generation']);buffer.copy_(cpu);torch.npu.synchronize();assert torch.equal(buffer.cpu(),cpu),'Source H2D mismatch'
   segments=case['segments'];assert all(0<=s['source_offset']<N and 0<=s['destination_offset']<N and s['bytes']>0 and s['source_offset']+s['bytes']<=N and s['destination_offset']+s['bytes']<=N for s in segments)
   if case['method']=='transfer_sync':
    s=segments[0];assert len(segments)==1;ret=engine.transfer_sync(peer['session_id'],buffer.data_ptr()+s['source_offset'],peer['buffer_address']+s['destination_offset'],s['bytes'])
   else:ret=engine.batch_transfer_sync(peer['session_id'],[buffer.data_ptr()+s['source_offset'] for s in segments],[peer['buffer_address']+s['destination_offset'] for s in segments],[s['bytes'] for s in segments])
   row={'case':case['id'],'method':case['method'],'return_code':ret,'bytes':sum(s['bytes'] for s in segments),'source_sha256':digest(cpu)};event('sent',**row);assert ret==0,row;checks.append(row);continue
  assert role=='D'
  torch.npu.synchronize();actual=buffer.cpu();golden=torch.full((N,),case['sentinel'],dtype=torch.uint8);src=source(case['generation']);written=torch.zeros(N,dtype=torch.bool)
  for s in case['segments']:
   a=s['destination_offset'];b=s['bytes'];golden[a:a+b]=src[s['source_offset']:s['source_offset']+b];written[a:a+b]=True
  unequal=actual!=golden;count=int(unequal.sum());row={'case':case['id'],'generation':case['generation'],'checked_bytes':N,'transferred_bytes':int(written.sum()),'untouched_bytes':int((~written).sum()),'mismatched_bytes':count,'untouched_mismatched_bytes':int((unequal & ~written).sum()),'first_mismatch_offsets':torch.nonzero(unequal).flatten()[:16].tolist(),'actual_sha256':digest(actual),'golden_sha256':digest(golden),'pass':count==0};event('verified',**row);assert row['pass'],row;checks.append(row)
 assert success,'Control channel closed before finish'
except BaseException as e:event('error',error={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
finally:
 if registered:
  try:engine.deregister(buffer.data_ptr());registered=False
  except BaseException as e:event('cleanup_error',error={'type':type(e).__name__,'message':str(e)})
 event('final',status='PASS' if success and not registered else 'FAIL',registered=False if not registered else True,deregistered=not registered,cases=checks,model_loaded=False,PD_MTP_serving_executed=False,performance='NOT_MEASURED')
sys.exit(0 if success and not registered else 1)
