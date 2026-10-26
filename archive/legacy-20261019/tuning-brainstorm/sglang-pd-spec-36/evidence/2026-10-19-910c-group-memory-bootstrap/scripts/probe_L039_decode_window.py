#!/usr/bin/env python3
"""Remote16rank BF16 low-latency dispatch support diagnostic, never a serving baseline."""
import ast, collections, datetime, hashlib, importlib.metadata, importlib.util, json, os, pathlib, sys, traceback
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
 t=torch.arange(64,dtype=torch.int32)[:,None]; c=torch.arange(6144,dtype=torch.int32)[None,:]
 a=((r*131+t*17+c*3)%127).float()/16
 a[:,0]=r; a[:,1]=torch.arange(64).float()
 return a.to(torch.bfloat16)
def routes(r):
 return (torch.arange(64)[:,None]+r*16+torch.arange(8)[None,:]*32)%256
try:
 assert int(os.environ.get('SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK','128'))==128
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
 source=pathlib.Path(plan['parallel_state_path']).read_bytes();assert hashlib.sha256(source).hexdigest()==plan['parallel_state_sha256']
 fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='get_torch_distributed_pg_options');ns={'_is_npu':True,'os':os};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(plan['parallel_state_path']),'exec'),ns);make=ns['get_torch_distributed_pg_options']
 default_opt=make();tp_opt=make('tp');ep_opt=make('moe_ep');assert tp_opt is None and ep_opt.hccl_config['hccl_buffer_size']==plan['ep_hccl_buffer_MB'] and default_opt.hccl_config['hccl_buffer_size']==plan['ep_hccl_buffer_MB']
 v['effective_policy']={'global_HCCL_BUFFSIZE_MB':int(os.environ['HCCL_BUFFSIZE']),'DEEPEP_HCCL_BUFFSIZE_MB':int(os.environ['DEEPEP_HCCL_BUFFSIZE']),'default_options':dict(default_opt.hccl_config),'TP_explicit_options':None,'EP_options':dict(ep_opt.hccl_config),'source_sha256':plan['parallel_state_sha256']}
 dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=90),pg_options=default_opt)
 tp_group=dist.new_group(ranks=list(range(16)),backend='hccl',pg_options=tp_opt,timeout=datetime.timedelta(seconds=90));ep_group=dist.new_group(ranks=list(range(16)),backend='hccl',pg_options=ep_opt,timeout=datetime.timedelta(seconds=90))
 v['memory_snapshots']={}
 def sample(name):
  try:v['memory_snapshots'][name]={'device_free_total_bytes':list(torch.npu.mem_get_info(rank)),'torch_allocated_bytes':torch.npu.memory_allocated(rank),'torch_reserved_bytes':torch.npu.memory_reserved(rank)}
  except Exception as error:v['memory_snapshots'][name]={'metadata_error':str(error)}
  save()
 sample('before_synthetic_pressure');v['stage']='synthetic_HBM_pressure';save();pressure_blocks=[]
 for i in range(55):pressure_blocks.append(torch.empty((1024**3,),dtype=torch.uint8,device=f'npu:{rank}').zero_())
 torch.npu.synchronize();v['synthetic_pressure_bytes']=sum(t.numel()*t.element_size() for t in pressure_blocks);assert v['synthetic_pressure_bytes']==55*1024**3;sample('after_55GiB_pressure');v['stage']='TP_all_reduce';save()
 a=torch.tensor([rank],dtype=torch.int32,device=f'npu:{rank}'); dist.all_reduce(a,group=tp_group); torch.npu.synchronize()
 assert a.cpu().item()==120
 v['NPU_compute']=True; v['checks'].append({'name':'16rank HCCL all_reduce exact rank sum','pass':True}); maps('after_context_and_collective_before_DeepEP');sample('after_TP_allreduce')
 v['stage']='import_DeepEP'; save()
 from deep_ep import Buffer
 v['deep_ep_cpp_origin']=sys.modules['deep_ep_cpp'].__file__;v['deep_ep_cpp_sha256']=hashlib.sha256(pathlib.Path(v['deep_ep_cpp_origin']).read_bytes()).hexdigest();assert v['deep_ep_cpp_sha256']==plan['deep_ep_cpp_sha256']
 v['custom_opp_after_DeepEP']=os.environ.get('ASCEND_CUSTOM_OPP_PATH'); maps('after_DeepEP_import')
 hint=Buffer.get_low_latency_rdma_size_hint(128,6144,16,256)
 v['buffer']={'num_nvl_bytes':0,'num_rdma_bytes':hint,'low_latency_mode':True,'num_qps_per_rank':16,'max_dispatch_tokens_per_rank':128}
 v['stage']='create_buffer'; save()
 b=Buffer(ep_group,num_nvl_bytes=0,num_rdma_bytes=hint,low_latency_mode=True,num_qps_per_rank=16)
 maps('after_buffer');sample('after_EP_buffer')
 cpu=payload(rank); ids_cpu=routes(rank).long(); x=cpu.to(f'npu:{rank}'); ids=ids_cpu.to(f'npu:{rank}'); weights=torch.full((64,8),0.125,dtype=torch.float32,device=f'npu:{rank}')
 v['case']={'tokens_per_rank':64,'hidden':6144,'experts':256,'topk':8,'dtype':'bfloat16','use_fp8':False,'async_finish':True,'return_recv_hook':False}
 v['stage']='low_latency_dispatch'; save()
 recv,counts,handle,event,hook=b.low_latency_dispatch(x,ids,128,256,use_fp8=False,async_finish=True,return_recv_hook=False,topk_weights=weights)
 torch.npu.synchronize(); maps('after_dispatch');sample('after_EP_dispatch')
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
 v['stage']='complete'; v['status']='PASS'; maps('complete');pressure_blocks.clear();torch.npu.empty_cache();v['synthetic_pressure_released']=True; del b; dist.destroy_process_group(); v['process_group_destroyed']=True
except BaseException as e:
 v['status']='FAIL'; v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 try:maps('failure')
 except BaseException:pass
finally:
 # Release own synthetic pressure before process-group cleanup.
 try:
  if 'pressure_blocks' in globals():pressure_blocks.clear()
  if 'torch' in globals():torch.npu.empty_cache()
 except BaseException as error:v['pressure_cleanup_error']=str(error)
 # Release the diagnostic buffer while its distributed group is still alive.
 try:
  if 'b' in globals():del b
  if 'dist' in globals() and dist.is_initialized():dist.destroy_process_group()
  v['process_group_destroyed']=True
 except BaseException as e:
  v['cleanup_error']=str(e); v['status']='FAIL'
 save(); print('L039_RANK_RESULT='+json.dumps({'rank':rank,'status':v['status'],'stage':v['stage']}),flush=True)
sys.exit(0 if v['status']=='PASS' else 1)
