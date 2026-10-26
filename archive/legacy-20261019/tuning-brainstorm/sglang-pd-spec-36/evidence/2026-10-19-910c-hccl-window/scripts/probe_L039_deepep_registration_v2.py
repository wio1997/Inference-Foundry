#!/usr/bin/env python3
"""Remote16rank BF16 low-latency dispatch support diagnostic, never a serving baseline."""
import collections, datetime, hashlib, importlib.metadata, importlib.util, json, os, pathlib, sys, traceback
rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE']); assert world==16
out=pathlib.Path('/work')/f'rank-{rank:02}.json'
v={'rank':rank,'world':world,'status':'STARTING','stage':'before_torch','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'NPU_compute':False,'model_loaded':False,'PD_MTP_serving':False,'performance':'NOT_MEASURED','checks':[],'maps':{}}
def save():
 out.write_text(json.dumps(v,indent=2)+'\n')
def maps(stage):
 lines=pathlib.Path('/proc/self/maps').read_text().splitlines()
 v['maps'][stage]=sorted({s.split()[-1] for s in lines if any(k in s for k in ['optiling','opsproto','libcust_opapi','deep_ep_cpp'])})
 save()
def payload(r):
 # All BF16 values are exactly representable. First2 columns identify rank/token.
 t=torch.arange(128,dtype=torch.int32)[:,None]; c=torch.arange(6144,dtype=torch.int32)[None,:]
 a=((r*131+t*17+c*3)%127).float()/16
 a[:,0]=r; a[:,1]=torch.arange(128).float()
 return a.to(torch.bfloat16)
def routes(r):
 return (torch.arange(128)[:,None]+r*16+torch.arange(8)[None,:]*32)%256
try:
 v['environment_before_torch']={k:os.environ.get(k) for k in ['ASCEND_CUSTOM_OPP_PATH','LD_LIBRARY_PATH','ASCEND_OPP_PATH','ASCEND_GLOBAL_LOG_LEVEL','DEEP_USE_MODE','DEEP_NORMAL_MODE_USE_INT8_QUANT']}
 v['packages']={n:importlib.metadata.version(n) for n in ['torch','torch-npu','deep-ep']}
 assert v['packages']['deep-ep']=='1.0.0+00db6cd6.cann.9.1.0.b243'
 pkg=pathlib.Path(importlib.util.find_spec('deep_ep').origin).parent
 plan=json.loads(pathlib.Path('/work/plan.json').read_text())
 for name,digest in plan['package_source_hashes'].items():assert hashlib.sha256((pkg/name).read_bytes()).hexdigest()==digest,name
 if plan['early_custom_opp']:assert os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':')[0]==str(pkg/'vendors/hwcomputing')
 else:assert not os.environ.get('ASCEND_CUSTOM_OPP_PATH')
 maps('before_torch')
 import torch, torch_npu
 import torch.distributed as dist
 torch.npu.set_device(rank); v['device_name']=torch.npu.get_device_name(rank); assert v['device_name']=='Ascend910_9392'
 dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=90))
 a=torch.tensor([rank],dtype=torch.int32,device=f'npu:{rank}'); dist.all_reduce(a); torch.npu.synchronize()
 assert a.cpu().item()==120
 v['NPU_compute']=True; v['checks'].append({'name':'16rank HCCL all_reduce exact rank sum','pass':True}); maps('after_context_and_collective_before_DeepEP')
 v['stage']='import_DeepEP'; save()
 from deep_ep import Buffer
 v['custom_opp_after_DeepEP']=os.environ.get('ASCEND_CUSTOM_OPP_PATH'); maps('after_DeepEP_import')
 hint=Buffer.get_low_latency_rdma_size_hint(256,6144,16,256)
 v['buffer']={'num_nvl_bytes':0,'num_rdma_bytes':hint,'low_latency_mode':True,'num_qps_per_rank':16,'max_dispatch_tokens_per_rank':256}
 v['stage']='create_buffer'; save()
 b=Buffer(dist.group.WORLD,num_nvl_bytes=0,num_rdma_bytes=hint,low_latency_mode=True,num_qps_per_rank=16)
 maps('after_buffer')
 cpu=payload(rank); ids_cpu=routes(rank).long(); x=cpu.to(f'npu:{rank}'); ids=ids_cpu.to(f'npu:{rank}'); weights=torch.full((128,8),0.125,dtype=torch.float32,device=f'npu:{rank}')
 v['case']={'tokens_per_rank':128,'hidden':6144,'experts':256,'topk':8,'dtype':'bfloat16','use_fp8':False,'async_finish':False,'return_recv_hook':False}
 v['stage']='low_latency_dispatch'; save()
 recv,counts,handle,event,hook=b.low_latency_dispatch(x,ids,256,256,use_fp8=False,async_finish=False,return_recv_hook=False,topk_weights=weights)
 torch.npu.synchronize(); maps('after_dispatch')
 v['stage']='independent_golden'; save()
 assert isinstance(recv,torch.Tensor),type(recv)
 actual=recv.cpu().contiguous(); count_cpu=counts.cpu().reshape(-1).tolist()
 v['returned']={'recv_shape':list(actual.shape),'recv_dtype':str(actual.dtype),'counts_shape':list(counts.shape),'counts_dtype':str(counts.dtype),'counts':count_cpu}; save()
 assert len(count_cpu)==16 and actual.ndim in (2,3) and actual.shape[-1]==6144 and actual.dtype==torch.bfloat16, v['returned']
 if actual.ndim==3:assert actual.shape[0]==16,v['returned']
 else:assert actual.shape[0]>=sum(int(n) for n in count_cpu),v['returned']
 offset=0
 checked=0
 for local in range(16):
  expert=rank*16+local; expected=[]
  for source_rank in range(16):
   selected=(routes(source_rank)==expert).any(dim=1)
   rows=payload(source_rank)[selected]
   expected.extend(hashlib.sha256(row.view(torch.uint16).numpy().tobytes()).hexdigest() for row in rows)
  count=int(count_cpu[local]); assert count==len(expected),(local,count,len(expected))
  valid_rows=actual[local,:count] if actual.ndim==3 else actual[offset:offset+count]
  observed=[hashlib.sha256(row.view(torch.uint16).numpy().tobytes()).hexdigest() for row in valid_rows]
  offset+=count
  assert collections.Counter(observed)==collections.Counter(expected),('payload multiset mismatch',rank,local)
  checked+=count
 v['checks'].append({'name':'all16 local experts counts and every full BF16 payload row equal independent CPU golden multiset','pass':True,'checked_rows':checked,'checked_payload_bytes':checked*6144*2})
 v['stage']='complete'; v['status']='PASS'; maps('complete'); del b; dist.destroy_process_group(); v['process_group_destroyed']=True
except BaseException as e:
 v['status']='FAIL'; v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 try:maps('failure')
 except BaseException:pass
finally:
 # Release the diagnostic buffer while its distributed group is still alive.
 try:
  if 'b' in globals():del b
  if 'dist' in globals() and dist.is_initialized():dist.destroy_process_group()
  v['process_group_destroyed']=True
 except BaseException as e:
  v['cleanup_error']=str(e); v['status']='FAIL'
 save(); print('L039_RANK_RESULT='+json.dumps({'rank':rank,'status':v['status'],'stage':v['stage']}),flush=True)
sys.exit(0 if v['status']=='PASS' else 1)
