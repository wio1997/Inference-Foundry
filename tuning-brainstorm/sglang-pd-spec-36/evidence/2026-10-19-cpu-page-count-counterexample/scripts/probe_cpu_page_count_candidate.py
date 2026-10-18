#!/usr/bin/env python3
"""Differential metadata experiment; native imports stop without fake outputs."""
import argparse,builtins,contextlib,datetime,hashlib,importlib,importlib.util,json,os,pathlib,subprocess,sys,traceback
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes);p=json.loads(pathlib.Path(args.protocol).read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1';nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit']
for f,h in expected.items():assert sha(root/f)==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(pathlib.Path('/model')/f)==h
candidate=snapshot/p['candidate_path'];assert sha(candidate)==p['candidate_sha256']
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':expected,'model_hashes':models,'candidate_sha256':sha(candidate),'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(pathlib.Path(args.protocol)),'accelerator_device_nodes':nodes,'cases':[],'counterexamples':[],'semantic_checks':0,'native_executed':False,'native_success_proven':False,'production_applied':False,'candidate_accepted':False,'performance':'NOT_MEASURED','scope':'Complete baseline and isolated candidate modules on Torch CPU logical page metadata, kvcache=None/debugFalse. Tagged CPU device lengths count explicit item calls. Exact native module import raises before import/launch; no fake kernel/output, no hardware synchronization measurement.'};stage='imports'
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);baseline=importlib.import_module('sglang.srt.hardware_backend.npu.allocator_npu');spec=importlib.util.spec_from_file_location('if_cpu_count_candidate',candidate);cand=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cand;spec.loader.exec_module(cand);classes={'baseline':baseline.NPUPagedTokenToKVPoolAllocator,'candidate':cand.NPUPagedTokenToKVPoolAllocator};result['actual_modules']={'baseline':baseline.__file__,'candidate':str(candidate),'full_module_imports':True};result['environment']={'python':sys.version,'torch':torch.__version__}
 class NativeBoundary(RuntimeError):pass
 class UnexpectedScalar(RuntimeError):pass
 class DeviceLens(torch.Tensor):
  reads=0;forbid=False
  @staticmethod
  def wrap(x):return torch.Tensor._make_subclass(DeviceLens,x,False)
  def item(self,*a,**kw):
   DeviceLens.reads+=1
   if DeviceLens.forbid:raise UnexpectedScalar('Tagged logical-device item not allowed in this case')
   return torch.Tensor.item(self,*a,**kw)
 @contextlib.contextmanager
 def native_stop(events):
  original=builtins.__import__
  def guard(name,globals=None,locals=None,fromlist=(),level=0):
   if name=='sgl_kernel_npu.mem_cache.allocator':events.append(name);raise NativeBoundary('Stopped before native module import; not a successful kernel')
   return original(name,globals,locals,fromlist,level)
  builtins.__import__=guard
  try:yield
  finally:builtins.__import__=original
 def check(x,message):
  assert x,message
  result['semantic_checks']+=1
 def create(which,ps,pages):
  a=classes[which](pages*ps,ps,torch.float32,'cpu',None,True);a.debug_mode=False;return a
 def run(which,ps,prefix,seq,dtype,*,cpu_variant='valid',explicit=None,pool_pages=None,forbid=False,cpu_seq=None):
  need=sum((e+ps-1)//ps-(s+ps-1)//ps for s,e in zip(prefix,seq));a=create(which,ps,pool_pages if pool_pages is not None else need+32);last=[]
  for x in prefix:
   v=a.alloc(((x+ps-1)//ps)*ps);assert v is not None;last.append(int(v[x-1]) if x else -1)
  pt=torch.tensor(prefix,dtype=dtype);st=torch.tensor(seq,dtype=dtype);pc=pt.clone();sc=torch.tensor(cpu_seq,dtype=dtype) if cpu_seq is not None else st.clone()
  if cpu_variant=='None':pc=sc=None
  elif cpu_variant=='non-Tensor':pc=prefix[:];sc=seq[:]
  elif cpu_variant=='meta device':pc=torch.empty(pt.shape,dtype=dtype,device='meta');sc=torch.empty(st.shape,dtype=dtype,device='meta')
  elif cpu_variant=='float dtype':pc=pc.float();sc=sc.float()
  elif cpu_variant=='device/cpu dtype mismatch':pc=pc.to(torch.int32 if dtype==torch.int64 else torch.int64);sc=sc.to(pc.dtype)
  elif cpu_variant=='shape mismatch':pc=torch.cat((pc,pc[:1]));sc=torch.cat((sc,sc[:1]))
  elif cpu_variant=='ndim2':pc=pc.reshape(1,-1);sc=sc.reshape(1,-1)
  before=a.get_all_free_pages().clone();DeviceLens.reads=0;DeviceLens.forbid=forbid;events=[];out=None;outcome='RETURN'
  with native_stop(events):
   try:out=a.alloc_extend(DeviceLens.wrap(pt),pc,DeviceLens.wrap(st),sc,torch.tensor(last,dtype=torch.int64),sum(e-s for s,e in zip(prefix,seq)),num_new_pages=explicit)
   except NativeBoundary:outcome='NATIVE_IMPORT_BOUNDARY'
  after=a.get_all_free_pages().clone();reads=DeviceLens.reads;DeviceLens.forbid=False
  return {'outcome':outcome,'item_reads':reads,'native_import_attempts':len(events),'before':before,'after':after,'out':out,'need':need,'available':a.available_size()}
 def record(label,left,right,*,fresh=True):
  check(left['outcome']==right['outcome'],'baseline/candidate outcome agrees');check(torch.equal(left['before'],right['before']),'identical starting page state');check(torch.equal(left['after'],right['after']),'identical final page identity');check(left['available']==right['available'],'identical capacity')
  if left['outcome']=='NATIVE_IMPORT_BOUNDARY':check(left['out'] is right['out'] is None and left['native_import_attempts']==right['native_import_attempts']==1,'native boundary only, no fake output');check(torch.equal(left['before'],left['after']),'no page consumption before native import')
  elif left['out'] is None:check(right['out'] is None and left['native_import_attempts']==right['native_import_attempts']==0,'same capacity rejection')
  else:check(torch.equal(left['out'],right['out']) and left['out'].dtype==right['out'].dtype==torch.int32,'exact Torch loc/dtype differential');check(left['native_import_attempts']==right['native_import_attempts']==0,'Torch branch did not enter native')
  result['cases'].append({**label,'status':'PASS','outcome':left['outcome'],'new_pages':left['need'],'baseline_tagged_CPU_item_reads':left['item_reads'],'candidate_tagged_CPU_item_reads':right['item_reads'],'native_success_proven':False})
 stage='fresh-small-boundary'
 for ps in p['pages']:
  for dtype in [torch.int32,torch.int64]:
   for count in [0,1,199]:
    counts=[count//2,count-count//2];prefix=[0,0];seq=[x*ps for x in counts];label={'family':'fresh-small','page_size':ps,'dtype':str(dtype),'count_ref':count};result['in_progress']=label;l=run('baseline',ps,prefix,seq,dtype);r=run('candidate',ps,prefix,seq,dtype,forbid=True);check(l['item_reads']==1 and r['item_reads']==0,'fresh candidate avoids explicit device item');record(label,l,r);result.pop('in_progress')
 stage='fresh-threshold-and-long'
 for ps in p['pages']:
  for dtype in [torch.int32,torch.int64]:
   for count in [200,201]:
    prefix=[0,0];seq=[100*ps,(count-100)*ps];label={'family':'fresh-threshold','page_size':ps,'dtype':str(dtype),'count_ref':count};result['in_progress']=label;l=run('baseline',ps,prefix,seq,dtype);r=run('candidate',ps,prefix,seq,dtype,forbid=True);check(l['item_reads']==1 and r['item_reads']==0,'fresh threshold read difference');record(label,l,r);result.pop('in_progress')
   for bs in [2,3]:
    prefix=[0]*bs;seq=[32768]*bs;label={'family':'fresh-long-reference','page_size':ps,'dtype':str(dtype),'batch':bs,'raw_length':32768};result['in_progress']=label;l=run('baseline',ps,prefix,seq,dtype);r=run('candidate',ps,prefix,seq,dtype,forbid=True);check(l['item_reads']==1 and r['item_reads']==0,'fresh long read difference');record(label,l,r);result.pop('in_progress')
 stage='fresh-cached-prefix'
 for ps in p['pages']:
  for dtype in [torch.int32,torch.int64]:
   for bs in [2,3]:
    prefix=[ps-1,ps,ps+1][:bs];seq=[32768]*bs;label={'family':'fresh-cached-prefix','page_size':ps,'dtype':str(dtype),'batch':bs,'prefix':prefix,'raw_length':32768};result['in_progress']=label;l=run('baseline',ps,prefix,seq,dtype);r=run('candidate',ps,prefix,seq,dtype,forbid=True);check(l['item_reads']==1 and r['item_reads']==0,'fresh cached-prefix read difference');record(label,l,r);result.pop('in_progress')
 stage='explicit-count'
 for ps in p['pages']:
  for variant in ['valid','None','non-Tensor']:
   for count in [1,200]:
    prefix=[0,0];seq=[0,count*ps];label={'family':'explicit-count','page_size':ps,'CPU_variant':variant,'count_ref':count};result['in_progress']=label;l=run('baseline',ps,prefix,seq,torch.int64,cpu_variant=variant,explicit=count,forbid=True);r=run('candidate',ps,prefix,seq,torch.int64,cpu_variant=variant,explicit=count,forbid=True);check(l['item_reads']==r['item_reads']==0,'supplied count never reads logical-device scalar');record(label,l,r);result.pop('in_progress')
 stage='fallback'
 for ps in p['pages']:
  for variant in p['fallback_variants']:
   label={'family':'fallback','page_size':ps,'CPU_variant':variant};result['in_progress']=label;l=run('baseline',ps,[0,0],[0,ps],torch.int64,cpu_variant=variant);r=run('candidate',ps,[0,0],[0,ps],torch.int64,cpu_variant=variant);check(l['item_reads']==r['item_reads']==1,'fallback retains original scalar read');record(label,l,r);result.pop('in_progress')
 stage='capacity-rejection'
 for ps in p['pages']:
  for variant in ['valid','None']:
   label={'family':'capacity-rejection','page_size':ps,'CPU_variant':variant};l=run('baseline',ps,[0,0],[ps,ps],torch.int64,cpu_variant=variant,pool_pages=0);r=run('candidate',ps,[0,0],[ps,ps],torch.int64,cpu_variant=variant,pool_pages=0,forbid=variant=='valid');check(l['outcome']==r['outcome']=='RETURN' and l['out'] is r['out'] is None,'actual capacity rejection retained');record(label,l,r)
 stage='stale-mirror-counterexample'
 for ps in p['pages']:
  device_seq=[100*ps,100*ps];stale_cpu_seq=[100*ps,101*ps];l=run('baseline',ps,[0,0],device_seq,torch.int64,cpu_seq=stale_cpu_seq,pool_pages=512);r=run('candidate',ps,[0,0],device_seq,torch.int64,cpu_seq=stale_cpu_seq,pool_pages=512,forbid=True);check(l['outcome']==r['outcome']=='RETURN' and l['out'] is not None,'counterexample executes actual Torch path');check(torch.equal(l['out'],r['out']),'stale count can hide behind identical output locs');check(l['available']-r['available']==ps and not torch.equal(l['after'],r['after']),'one page overconsumed with stale eligible metadata');result['counterexamples'].append({'status':'EXPECTED_DIFFERENCE','page_size':ps,'device_new_pages':200,'CPU_new_pages':201,'locs_equal':True,'candidate_extra_consumed_slots':ps,'scope':'Metadata gate cannot establish CPU freshness. Not a current serving bug claim; invalidates unqualified default-method adoption.'})
 result.update(probe_status='PASS',correctness='PASS_WITH_EXPECTED_COUNTEREXAMPLES',candidate_accepted=False,unqualified_default_method_compatibility='FALSIFIED_BY_STALE_CPU_COUNTEREXAMPLE',native_executed=False)
except Exception as exc:
 result.update(probe_status='UNAVAILABLE' if stage=='imports' else 'FAIL',correctness='NOT_EVALUATED' if stage=='imports' else 'FAIL',failed_stage=stage,error=repr(exc),traceback=traceback.format_exc())
print(json.dumps(result,ensure_ascii=False));sys.exit(0 if result['probe_status'] in ['PASS','UNAVAILABLE'] else 1)
