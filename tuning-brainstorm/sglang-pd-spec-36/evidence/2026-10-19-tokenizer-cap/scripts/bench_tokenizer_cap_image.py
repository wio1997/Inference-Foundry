#!/usr/bin/env python3
"""Clean per-case A/A and A/B for the actual CPU tokenizer actor; no inference."""
import argparse,asyncio,datetime,hashlib,importlib,importlib.metadata,json,os,pathlib,platform,random,subprocess,sys,time,traceback
ap=argparse.ArgumentParser();ap.add_argument('--source-root',required=True);ap.add_argument('--expected-hashes',required=True);ap.add_argument('--model-hashes',required=True);ap.add_argument('--stdout',action='store_true');ap.add_argument('--case-key',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--fixture-root',required=True);args=ap.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=pathlib.Path('/sgl-workspace/sglang');model=pathlib.Path('/model');snapshot=pathlib.Path(args.source_root);fixture_root=pathlib.Path(args.fixture_root);protocol_path=pathlib.Path(args.protocol);protocol=json.loads(protocol_path.read_text());expected=json.loads(args.expected_hashes);model_hashes=json.loads(args.model_hashes)
assert args.case_key in protocol['cases'];assert protocol['single_factor']=={'A':{'max_batch_size':32},'B':{'max_batch_size':'predeclared_case_cap'}};assert protocol['candidate_caps']=={'tokenizer-cpu-c2':2,'tokenizer-cpu-c3':3};assert protocol['fixed']['batch_wait_timeout_s']==0.002
nodes=sorted({str(p) for pattern in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for p in pathlib.Path('/dev').glob(pattern)});assert not nodes
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1' and os.environ.get('TOKENIZERS_PARALLELISM')=='false'
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==protocol['source_commit'];hashes={}
for f,h in expected.items():hashes[f]=sha(root/f);assert hashes[f]==h and sha(snapshot/f)==h
for f,h in model_hashes.items():assert sha(model/f)==h
manifest_path=fixture_root/protocol['input_manifest'];manifest=json.loads(manifest_path.read_text());by={f['label']:f for f in manifest['fixtures']};fixtures=[]
for label in protocol['cases'][args.case_key]:
 info=by[label];p=fixture_root/info['path'];assert p.resolve().is_relative_to(fixture_root.resolve()) and sha(p)==info['sha256'];data=json.loads(p.read_text());assert len(data['input_ids'])==info['input_tokens'];fixtures.append(data)
digest=lambda ids:hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
def discover_cgroup():
 membership=pathlib.Path('/proc/self/cgroup').read_text();mountinfo=pathlib.Path('/proc/self/mountinfo').read_text();groups={}
 for line in membership.splitlines():
  _,controllers,path=line.split(':',2)
  for controller in controllers.split(','):groups[controller]=path
 candidates=[]
 for line in mountinfo.splitlines():
  left,right=line.split(' - ',1);fields=left.split();fs,_,options=right.split()[:3];mount_root=pathlib.PurePosixPath(fields[3]);mount=pathlib.Path(fields[4])
  if fs=='cgroup2':version=2;group=groups.get('')
  elif fs=='cgroup' and 'cpu' in options.split(','):version=1;group=groups.get('cpu')
  else:continue
  if group is None:continue
  group_path=pathlib.PurePosixPath(group)
  if group_path.is_relative_to(mount_root):paths=[mount/str(group_path.relative_to(mount_root))]
  elif group=='/':paths=[mount]
  else:paths=[]
  for path in paths:
   if (path/'cpu.stat').is_file():candidates.append((version,path,group,str(mount_root)))
 assert len(candidates)==1,{'cgroup_candidates':[(v,str(p),g,r) for v,p,g,r in candidates],'membership':membership,'mountinfo':mountinfo}
 version,path,group,mount_root=candidates[0]
 if version==1:
  quota=int((path/'cpu.cfs_quota_us').read_text());period=int((path/'cpu.cfs_period_us').read_text());assert quota==period and quota>0
 else:
  quota,period=(path/'cpu.max').read_text().split();assert quota!='max' and int(quota)==int(period)
 return {'version':version,'cpu_path':str(path),'group':group,'mount_root':mount_root,'quota_us':int(quota),'period_us':int(period),'membership':membership}
cgroup=discover_cgroup()
def cgroup_stat():
 raw={line.split()[0]:int(line.split()[1]) for line in (pathlib.Path(cgroup['cpu_path'])/'cpu.stat').read_text().splitlines() if len(line.split())==2}
 assert {'nr_periods','nr_throttled'}<=raw.keys()
 if cgroup['version']==1:
  assert 'throttled_time' in raw;raw['throttled_usec']=raw['throttled_time']//1000
 else:assert 'throttled_usec' in raw
 return raw
def read_optional(path):
 p=pathlib.Path(path);return p.read_text().strip() if p.is_file() else None
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','case_key':args.case_key,'source_commit':head,'source_hashes':hashes,'model_hashes':model_hashes,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(protocol_path),'manifest_sha256':sha(manifest_path),'fixture_metadata':[{'label':f['label'],'tokens':f['input_tokens'],'ids_sha256':f['input_ids_sha256'],'rendered_utf8_bytes':len(f['rendered_text'].encode())} for f in fixtures],'environment':{'python':sys.version,'machine':platform.machine(),'affinity':sorted(os.sched_getaffinity(0)),'cgroup':cgroup,'memory_max':read_optional('/sys/fs/cgroup/memory.max'),'tokenizers_parallelism':False},'accelerator_device_nodes':nodes,'samples':[],'scope':'Actual AsyncDynamicbatchTokenizer and full SGLang tokenizer loader in explicit CPU package view. Per-case existing max_batch_size32 versus predeclared cap2/3 contrast only; same2ms window, all requests scheduled together, fixed input bytes/IDs. No HTTP, weights, NPU, MTP or serving performance. No physical-core pin or peak calibration.'}
stage='import'
try:
 sys.path.insert(0,str(root/'python'));import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 mod=importlib.import_module('sglang.srt.utils.hf_transformers.tokenizer');batch_mod=importlib.import_module('sglang.srt.managers.async_dynamic_batch_tokenizer');assert 'torch_npu' not in sys.modules
 stage='load';tok=mod.get_tokenizer(str(model),tokenizer_mode='auto',tokenizer_backend='huggingface',trust_remote_code=False,local_files_only=True)
 result['environment'].update({x:importlib.metadata.version(x) for x in ['transformers','tokenizers','jinja2']});result['environment']['torch']=torch.__version__;result['tokenizer_class']=type(tok).__module__+'.'+type(tok).__name__;result['batcher_file_sha256']=sha(pathlib.Path(batch_mod.__file__))
 prompts=[f['rendered_text'] for f in fixtures];kwargs={'add_special_tokens':False};reference=[dict(tok(p,**kwargs)) for p in prompts]
 for f,out in zip(fixtures,reference):assert out['input_ids']==f['input_ids'] and digest(out['input_ids'])==f['input_ids_sha256']
 stage='benchmark'
 async def main():
  actors={label:batch_mod.AsyncDynamicbatchTokenizer(tok,max_batch_size=size,batch_wait_timeout_s=0.002) for label,size in [('A0',32),('A1',32),('B',protocol['candidate_caps'][args.case_key])]};rng=random.Random(protocol['random_seed'])
  async def encode(actor):return await asyncio.gather(*(actor.encode(text,**kwargs) for text in prompts))
  def verify(values):
   assert [dict(x) for x in values]==reference
   return [digest(x['input_ids']) for x in values]
  try:
   for label,actor in actors.items():
    for _ in range(protocol['warmup_batches_per_actor_per_case']):verify(await encode(actor))
   result['warmup']={'actors':list(actors),'batches_per_actor':protocol['warmup_batches_per_actor_per_case'],'values_verified':True}
   for block in range(protocol['paired_blocks']):
    pairs=['AA','AB'];rng.shuffle(pairs)
    for pair in pairs:
     members=['A0','A1'] if pair=='AA' else ['A0','B'];rng.shuffle(members)
     for order,label in enumerate(members):
      await asyncio.sleep(protocol['inter_sample_gap_s']);before=cgroup_stat()
      # Tasks are queued together before the first yield; task creation is outside timing.
      tasks=[asyncio.create_task(actors[label].encode(text,**kwargs)) for text in prompts]
      wall_start=time.perf_counter_ns();cpu_start=time.process_time_ns();values=await asyncio.gather(*tasks);cpu_end=time.process_time_ns();wall_end=time.perf_counter_ns();after=cgroup_stat()
      ids=verify(values);delta={k:after[k]-v for k,v in before.items() if k in after and isinstance(v,int)}
      result['samples'].append({'block':block,'pair':pair,'order':order,'actor':label,'max_batch_size':actors[label].max_batch_size,'wall_ns':wall_end-wall_start,'process_cpu_ns':cpu_end-cpu_start,'batch_start_perf_ns':wall_start,'cgroup_before':before,'cgroup_after':after,'cgroup_delta':delta,'ids_sha256':ids,'values_match':True,'submission':'all queued before first yield'})
  finally:
   for actor in actors.values():
    if actor._batcher_task:
     actor._batcher_task.cancel()
     try:await actor._batcher_task
     except asyncio.CancelledError:pass
     actor._batcher_task=None
    actor._executor.shutdown(wait=True)
 asyncio.run(main());assert len(result['samples'])==protocol['paired_blocks']*4;assert 'torch_npu' not in sys.modules
 result.update({'probe_status':'PASS','correctness':'PASS','measurement_complete':True})
except ImportError as exc:result.update({'probe_status':'UNAVAILABLE','correctness':'NOT_EVALUATED','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
except Exception as exc:result.update({'probe_status':'ERROR','correctness':'FAIL','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
for f,h in model_hashes.items():assert sha(model/f)==h
print(json.dumps(result));sys.exit(1 if result['probe_status']=='ERROR' else 0)
