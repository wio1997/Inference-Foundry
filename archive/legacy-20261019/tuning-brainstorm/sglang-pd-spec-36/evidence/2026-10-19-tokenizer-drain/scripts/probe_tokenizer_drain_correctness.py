#!/usr/bin/env python3
"""Untimed actual-loader differential test. Instrumentation is not performance."""
import argparse,ast,asyncio,datetime,hashlib,importlib,importlib.metadata,importlib.util,json,os,pathlib,platform,subprocess,sys,threading,time,traceback
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
root=pathlib.Path('/sgl-workspace/sglang');model=pathlib.Path('/model');snapshot=pathlib.Path(args.source_root);fixture_root=pathlib.Path(args.fixture_root);protocol_path=pathlib.Path(args.protocol);protocol=json.loads(protocol_path.read_text());expected=json.loads(args.expected_hashes);model_hashes=json.loads(args.model_hashes)
nodes=sorted({str(p) for pattern in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for p in pathlib.Path('/dev').glob(pattern)});assert not nodes
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1' and os.environ.get('TOKENIZERS_PARALLELISM')=='false'
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==protocol['source_commit'];hashes={}
for f,h in expected.items():hashes[f]=sha(root/f);assert hashes[f]==h and sha(snapshot/f)==h
for f,h in model_hashes.items():assert sha(model/f)==h
candidate_path=fixture_root/protocol['candidate'];assert sha(candidate_path)==protocol['candidate_sha256'];manifest_path=fixture_root/protocol['fixture_manifest'];manifest=json.loads(manifest_path.read_text());fixtures=[]
for info in manifest['fixtures']:
 p=fixture_root/info['path'];assert p.resolve().is_relative_to(fixture_root.resolve()) and sha(p)==info['sha256'];d=json.loads(p.read_text());assert len(d['input_ids'])==info['input_tokens'];fixtures.append(d)
helper_path=fixture_root/'scripts/bench_tokenizer_cap_image.py';tree=ast.parse(helper_path.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['discover_cgroup','cgroup_stat']];assert len(functions)==2;ns={'pathlib':pathlib};exec(compile(ast.Module(body=functions,type_ignores=[]),str(helper_path),'exec'),ns);ns['cgroup']=ns['discover_cgroup']();cgroup_stat=ns['cgroup_stat']
cases_path=fixture_root/'specs/L013-cases-v1.json';plan=json.loads(cases_path.read_text());digest=lambda ids:hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','case_key':args.case_key,'source_commit':head,'source_hashes':hashes,'model_hashes':model_hashes,'script_sha256':sha(pathlib.Path(__file__)),'candidate_sha256':sha(candidate_path),'protocol_sha256':sha(protocol_path),'cases_sha256':sha(cases_path),'manifest_sha256':sha(manifest_path),'cgroup_helper_sha256':sha(helper_path),'environment':{'python':sys.version,'machine':platform.machine(),'cgroup':ns['cgroup'],'affinity':sorted(os.sched_getaffinity(0))},'accelerator_device_nodes':nodes,'scenarios':[],'scope':'Untimed correctness with delegating instrumentation, actual unmodified loader/base and isolated candidate subclass. Queue grouping may differ; no CPU/serving/NPU speed claim.'}
stage='import'
try:
 sys.path.insert(0,str(root/'python'));import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 mod=importlib.import_module('sglang.srt.utils.hf_transformers.tokenizer');batch_mod=importlib.import_module('sglang.srt.managers.async_dynamic_batch_tokenizer');assert 'torch_npu' not in sys.modules
 stage='load';tok=mod.get_tokenizer(str(model),tokenizer_mode='auto',tokenizer_backend='huggingface',trust_remote_code=False,local_files_only=True);base=batch_mod.AsyncDynamicbatchTokenizer;assert sha(pathlib.Path(batch_mod.__file__))==protocol['base_batcher_sha256']
 spec=importlib.util.spec_from_file_location('task_drain_candidate',candidate_path);cand_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(cand_module);candidate=cand_module.make_candidate(base)
 assert cand_module.BASE_SOURCE_COMMIT==head and cand_module.BASE_BATCHER_SHA256==protocol['base_batcher_sha256']
 assert all(getattr(candidate,n) is getattr(base,n) for n in ['__init__','encode','_process_dynamic_batch','_ensure_initialized','__call__'])
 result['environment'].update({x:importlib.metadata.version(x) for x in ['transformers','tokenizers','jinja2']});result['environment']['torch']=torch.__version__;result['tokenizer_class']=type(tok).__module__+'.'+type(tok).__name__;result['base_batcher_sha256']=sha(pathlib.Path(batch_mod.__file__))
 texts=[f['rendered_text'] for f in fixtures];label_by_text={x:f['label'] for x,f in zip(texts,fixtures)};default_kw={'add_special_tokens':False};references={}
 def oracle(i,kw):
  key=(i,json.dumps(kw,sort_keys=True))
  if key not in references:references[key]=dict(tok(texts[i],**kw))
  return references[key]
 for i,f in enumerate(fixtures):assert oracle(i,default_kw)['input_ids']==f['input_ids'] and digest(f['input_ids'])==f['input_ids_sha256']
 class Recorder:
  def __init__(self):self.calls=[];self.fail_next=False;self.gate_next=False;self.entered=threading.Event();self.release=threading.Event()
  def __call__(self,prompts,**kwargs):
   items=[prompts] if isinstance(prompts,str) else prompts;self.calls.append({'labels':[label_by_text[x] for x in items],'form':'single' if isinstance(prompts,str) else 'batch','kwargs':kwargs})
   if self.gate_next:
    self.gate_next=False;self.entered.set()
    if not self.release.wait(2):raise TimeoutError('task gate not released')
   if self.fail_next:self.fail_next=False;raise ValueError('task-injected-tokenizer-failure')
   return tok(prompts,**kwargs)
 stage='correctness'
 async def main():
  for role,cls in [('baseline',base),('candidate',candidate)]:
   for case in plan['scenarios']:
    rec=Recorder();actor=cls(rec,**protocol['parameters']);row={'role':role,'name':case['name'],'mode':case['mode'],'indices':case['indices'],'groups':[],'enqueues':[],'operations':[]};result['in_progress']=row;tasks=[];process=actor._process_dynamic_batch;before=cgroup_stat()
    async def observed_process(prompts,kwargs_list,futures):
     group={'labels':[label_by_text[x] for x in prompts],'kwargs':kwargs_list,'future_cancelled_at_dispatch':[f.cancelled() for f in futures]};row['groups'].append(group)
     await process(prompts,kwargs_list,futures);group['future_states_after']=[{'done':f.done(),'cancelled':f.cancelled()} for f in futures]
    actor._process_dynamic_batch=observed_process
    indices=case['indices'];kws=[dict(default_kw) for _ in indices]
    if case['mode']=='mixed_kwargs':kws[1]['return_attention_mask']=False
    wanted=[oracle(i,kw) for i,kw in zip(indices,kws)]
    async def request(j,delay=0):
     if delay:await asyncio.sleep(delay)
     row['enqueues'].append({'request':j,'fixture':fixtures[indices[j]]['label'],'loop_time':asyncio.get_running_loop().time()})
     return await actor.encode(texts[indices[j]],**kws[j])
    def check(values,cancelled=None,error=False):
     for j,value in enumerate(values):
      if j==cancelled:assert isinstance(value,asyncio.CancelledError)
      elif error:assert isinstance(value,ValueError) and str(value)=='task-injected-tokenizer-failure'
      else:assert dict(value)==wanted[j]
     row['operations'].append({'complete_dict_match':not error,'expected_cancelled_request':cancelled,'expected_injected_error':error,'result_ids_sha256':[None if isinstance(v,BaseException) else digest(v['input_ids']) for v in values]})
    try:
     mode=case['mode'];rec.fail_next=mode=='error_recovery';rec.gate_next=mode=='cancel_in_worker'
     for j in range(len(indices)):tasks.append(asyncio.create_task(request(j,case.get('last_delay_s',0) if j==len(indices)-1 else 0)))
     cancelled=None
     if mode=='cancel_before_enqueue':cancelled=1;tasks[cancelled].cancel()
     elif mode=='cancel_enqueued':
      await asyncio.sleep(0);row['queue_size_before_cancel']=actor._queue.qsize();assert row['queue_size_before_cancel']==len(indices);cancelled=1;tasks[cancelled].cancel()
     elif mode=='cancel_in_worker':
      deadline=asyncio.get_running_loop().time()+1
      while not rec.entered.is_set():
       assert asyncio.get_running_loop().time()<deadline;await asyncio.sleep(.0001)
      cancelled=1;tasks[cancelled].cancel();await asyncio.sleep(0);rec.release.set()
     values=await asyncio.wait_for(asyncio.gather(*tasks,return_exceptions=True),5);check(values,cancelled,error=mode=='error_recovery')
     if mode in ['error_recovery','repeat']:
      repetitions=1 if mode=='error_recovery' else 2
      for _ in range(repetitions):tasks=[asyncio.create_task(request(j)) for j in range(len(indices))];values=await asyncio.wait_for(asyncio.gather(*tasks,return_exceptions=True),5);check(values)
     assert actor._queue.empty() and not actor._batcher_task.done() and all(t.done() for t in tasks)
     assert all(len(g['labels'])<=protocol['parameters']['max_batch_size'] for g in row['groups'])
     row.update({'status':'PASS','queue_empty':True,'worker_calls':rec.calls,'cgroup_before':before,'cgroup_after':cgroup_stat()});result['scenarios'].append(row);result.pop('in_progress')
    finally:
     rec.release.set()
     for task in tasks:
      if not task.done():task.cancel()
     if tasks:await asyncio.gather(*tasks,return_exceptions=True)
     if actor._batcher_task:
      actor._batcher_task.cancel()
      try:await actor._batcher_task
      except asyncio.CancelledError:pass
      actor._batcher_task=None
     actor._executor.shutdown(wait=True);assert all(not t.is_alive() for t in actor._executor._threads)
     row['cleanup']='batcher cancelled and worker threads joined'
 asyncio.run(main());assert len(result['scenarios'])==2*len(plan['scenarios']);assert 'torch_npu' not in sys.modules
 paired=[]
 for case in plan['scenarios']:
  rows={x['role']:x for x in result['scenarios'] if x['name']==case['name']};paired.append({'name':case['name'],'baseline_groups':[len(g['labels']) for g in rows['baseline']['groups']],'candidate_groups':[len(g['labels']) for g in rows['candidate']['groups']]})
 result.update({'probe_status':'PASS','correctness':'PASS','paired_group_comparison':paired,'scenario_count':len(result['scenarios']),'performance':'NOT_MEASURED'})
except ImportError as exc:result.update({'probe_status':'UNAVAILABLE','correctness':'NOT_EVALUATED','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
except Exception as exc:result.update({'probe_status':'ERROR','correctness':'FAIL','stage':stage,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
for f,h in model_hashes.items():assert sha(model/f)==h
print(json.dumps(result));sys.exit(1 if result['probe_status']=='ERROR' else 0)
