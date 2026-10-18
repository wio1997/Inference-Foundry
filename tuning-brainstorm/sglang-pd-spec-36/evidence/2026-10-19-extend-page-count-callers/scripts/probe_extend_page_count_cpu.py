#!/usr/bin/env python3
"""Actual caller source bodies and NPU allocator scalar-read boundary on CPU."""
import argparse,ast,copy,datetime,hashlib,importlib,inspect,json,logging,os,pathlib,subprocess,sys,traceback,types
from collections import defaultdict
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes);p=json.loads(pathlib.Path(args.protocol).read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1';nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit']
for f,h in expected.items():assert sha(root/f)==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(pathlib.Path('/model')/f)==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':expected,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(pathlib.Path(args.protocol)),'accelerator_device_nodes':nodes,'source_bodies':{},'actual_modules':[],'cases':[],'semantic_checks':0,'native_executed':False,'small_native_success_executed':False,'performance':'NOT_MEASURED','scope':'Unchanged caller function bodies and actual full NPU allocator on CPU. Tagged CPU tensors raise at explicit logical-device item before any native kernel. Eviction/context/req table are boundaries. No serving, device-sync duration or small native success claim; existing optional-count capability is not a new optimization.'};stage='imports'
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);npu=importlib.import_module('sglang.srt.hardware_backend.npu.allocator_npu');Alloc=npu.NPUPagedTokenToKVPoolAllocator;sizing=importlib.import_module('sglang.srt.mem_cache.allocation_sizing');common=importlib.import_module('sglang.srt.utils.common');page_mod=importlib.import_module('sglang.srt.mem_cache.allocator.page_interleave')
 for m in [npu,sizing,common,page_mod]:
  f=str(pathlib.Path(m.__file__).relative_to(root));assert sha(root/f)==expected[f];result['actual_modules'].append({'name':m.__name__,'path':f,'sha256':expected[f],'full_module_import':True})
 result['environment']={'python':sys.version,'torch':torch.__version__};future=ast.parse('from __future__ import annotations').body[0];module=types.ModuleType('if_page_count_callers');sys.modules[module.__name__]=module;env=module.__dict__;env.update(torch=torch,defaultdict=defaultdict,logger=logging.getLogger('if_page_count'),_is_npu=True,_is_hip=False,attention_backends=lambda:('ascend','ascend'),page_interleave_shard_size=page_mod.page_interleave_shard_size,is_deepseek_v4=lambda cfg:False,available_and_evictable_str=lambda cache:'CPU observation boundary',get_spec=lambda:types.SimpleNamespace(speculative_eagle_topk=1),get_alloc_reserve_per_decode=lambda:8,page_aligned_decode_alloc_lens=sizing.page_aligned_decode_alloc_lens)
 def load(f,names):
  t=ast.parse((root/f).read_text());selected=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name) for name in names]
  for n in selected:result['source_bodies'][f+':'+n.name]={'ast_sha256':hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),'unchanged':True}
  exec(compile(ast.Module(body=[future,*copy.deepcopy(selected)],type_ignores=[]),f,'exec'),env)
 base='python/sglang/srt/';load(base+'mem_cache/allocation.py',['get_last_loc_torch','get_last_loc','alloc_paged_token_slots_extend','alloc_for_spec_decode']);load(base+'disaggregation/decode.py',['alloc_for_decode_prealloc']);load(base+'speculative/eagle_utils.py',['eagle_prepare_for_decode']);load(base+'hardware_backend/npu/dsv4/dsv4_allocator.py',['alloc_paged_token_slots_extend_npu']);t=ast.parse((root/(base+'mem_cache/allocation.py')).read_text());mapping=next(n for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='ALLOC_EXTEND_FUNCS' for x in n.targets));env['_alloc_paged_token_slots_extend_npu']=env['alloc_paged_token_slots_extend_npu'];exec(compile(ast.Module(body=[mapping],type_ignores=[]),'actual-dispatch-map','exec'),env)
 def check(value,message):
  assert value,message
  result['semantic_checks']+=1
 class DeviceScalarRead(RuntimeError):pass
 class DeviceLens(torch.Tensor):
  reads=0
  @staticmethod
  def wrap(value):return torch.Tensor._make_subclass(DeviceLens,value,False)
  def item(self,*a,**kw):DeviceLens.reads+=1;raise DeviceScalarRead('Tagged logical-device scalar item; CPU observation only')
 def create(ps,pages=1024):
  a=Alloc(ps*pages,ps,torch.float32,'cpu',None,True);a.debug_mode=False;return a
 captures=[];evictions=[];assignments=[]
 env['evict_from_tree_cache']=lambda cache,n:evictions.append(n)
 env['assign_req_to_token_pool_func']=lambda *a,**kw:assignments.append('unexpected after observation stop')
 def observe(a):
  original=a.alloc_extend;sig=inspect.signature(original)
  def call(*args,**kwargs):
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments;captures.append({'prefix':v['prefix_lens'].tolist(),'prefix_cpu':v['prefix_lens_cpu'].tolist(),'seq':v['seq_lens'].tolist(),'seq_cpu':v['seq_lens_cpu'].tolist(),'dtype':str(v['prefix_lens'].dtype),'seq_cpu_dtype':str(v['seq_lens_cpu'].dtype),'extend':v['extend_num_tokens'],'explicit_count':v['num_new_pages'],'CPU_count':common.get_num_new_pages(seq_lens=v['seq_lens_cpu'],page_size=a.page_size,prefix_lens=v['prefix_lens_cpu'])});v['prefix_lens']=DeviceLens.wrap(v['prefix_lens']);v['seq_lens']=DeviceLens.wrap(v['seq_lens']);return original(**v)
  a.alloc_extend=call
 def expect_boundary(fn,a,label):
  before=a.get_all_free_pages().clone();n=len(captures);reads=DeviceLens.reads;caught=False
  try:fn()
  except DeviceScalarRead:caught=True
  check(caught and DeviceLens.reads==reads+1,'exactly one scalar-read boundary reached');check(len(captures)==n+1,'one actual allocator call');v=captures[-1];oracle=sum((end+a.page_size-1)//a.page_size-(start+a.page_size-1)//a.page_size for start,end in zip(v['prefix_cpu'],v['seq_cpu']));check(v['prefix']==v['prefix_cpu'] and v['seq']==v['seq_cpu'],'source caller mirrors agree');check(v['CPU_count']==oracle,'CPU count equals independent integer-page oracle');check(v['explicit_count'] is None,'current caller leaves optional count absent/None');check(torch.equal(before,a.get_all_free_pages()),'observation stops before allocation mutation');result['cases'].append({**label,**v,'status':'PASS','outcome':'SCALAR_READ_OBSERVED_BEFORE_NATIVE','native_executed':False})
 def lengths(a,prefix):
  last=[];values=[]
  for n in prefix:
   ids=a.alloc(((n+a.page_size-1)//a.page_size)*a.page_size);values.append(ids[:n]);last.append(int(ids[n-1]) if n else -1)
  return torch.tensor(last,dtype=torch.int64),values
 stage='ordinary-caller'
 for ps in p['pages']:
  for bs in [2,3]:
   for pre in [0,127]:
    a=create(ps);prefix=[pre]*bs;seq=[512+i for i in range(bs)];last,_=lengths(a,prefix);observe(a);cache=types.SimpleNamespace(token_to_kv_pool_allocator=a);pt=torch.tensor(prefix,dtype=torch.int64);st=torch.tensor(seq,dtype=torch.int64);expect_boundary(lambda:env['alloc_paged_token_slots_extend'](cache,pt,pt.clone(),st,st.clone(),last,sum(e-s for s,e in zip(prefix,seq))),a,{'family':'ordinary-source-helper','page_size':ps,'batch':bs,'prefix_ref':pre})
 stage='PD-prealloc-caller'
 for ps in p['pages']:
  for pre in [0,127,128,129]:
   a=create(ps);last,values=lengths(a,[pre]);observe(a);req=types.SimpleNamespace(kv=types.SimpleNamespace(kv_allocated_len=pre));expect_boundary(lambda:env['alloc_for_decode_prealloc'](a,req=req,fill_len=512,delta_len=512-pre,prefix_len=pre,total_prefix_len=pre,prefix_indices=values[0],uses_swa_tail=False,swa_tail_len=0),a,{'family':'PD-source-prealloc','page_size':ps,'prefix_ref':pre});check(req.kv.kv_allocated_len==512,'PD producer stamps requested fill length before call')
 stage='EAGLE-source-producer-and-spec-caller'
 for ps in p['pages']:
  for bs in [2,3]:
   a=create(ps);reqs=[types.SimpleNamespace(kv=types.SimpleNamespace(kv_allocated_len=ps,kv_committed_len=ps-1),decode_batch_idx=0) for i in range(bs)];_,values=lengths(a,[ps]*bs);table=torch.full((bs,4*ps),-1,dtype=torch.int64)
   for i,v in enumerate(values):table[i,:ps]=v
   observe(a);cache=types.SimpleNamespace(token_to_kv_pool_allocator=a);batch=types.SimpleNamespace(reqs=reqs,device=torch.device('cpu'),tree_cache=cache,token_to_kv_pool_allocator=a,req_to_token_pool=types.SimpleNamespace(req_to_token=table),req_pool_indices=torch.arange(bs),sampling_info=types.SimpleNamespace(penalizer_orchestrator=types.SimpleNamespace(is_required=False)),batch_size=lambda:bs,maybe_evict_swa=lambda:None,model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace()));expect_boundary(lambda:env['eagle_prepare_for_decode'](batch),a,{'family':'EAGLE-source-prep-spec-helper-CPU-route','page_size':ps,'batch':bs,'reserve_fixture':8});check(captures[-1]['dtype']=='torch.int32','EAGLE caller retains int32 lengths');check(all(r.decode_batch_idx==1 for r in reqs),'EAGLE pre-call index mutation retained')
 stage='NPU-wrapper-nonDSV4'
 for ps in p['pages']:
  a=create(ps);last,_=lengths(a,[0,127]);observe(a);cache=types.SimpleNamespace(token_to_kv_pool_allocator=a);pt=torch.tensor([0,127],dtype=torch.int32);st=torch.tensor([512,512],dtype=torch.int32);batch=types.SimpleNamespace(model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace()));expect_boundary(lambda:env['alloc_paged_token_slots_extend_npu'](cache,pt,pt.clone(),st,st.clone(),last,897,batch=batch),a,{'family':'direct-NPU-wrapper-nonDSV4','page_size':ps})
 stage='existing-optional-count-early-reject'
 for ps in p['pages']:
  for dtype in [torch.int32,torch.int64]:
   a=create(ps,2);last,_=lengths(a,[127,127]);pt=torch.tensor([127,127],dtype=dtype);st=torch.tensor([512,512],dtype=dtype);count=common.get_num_new_pages(st,page_size=ps,prefix_lens=pt);reads=DeviceLens.reads;out=a.alloc_extend(DeviceLens.wrap(pt),pt,DeviceLens.wrap(st),st,last,770,num_new_pages=count);check(out is None and DeviceLens.reads==reads,'existing explicit count avoids device item before capacity rejection');check(a.available_size()==0,'rejection does not consume pages');result['cases'].append({'family':'existing-count-early-reject','page_size':ps,'dtype':str(dtype),'new_pages':count,'status':'PASS','native_success_proven':False})
 stage='existing-count-large-Torch-equivalence'
 for ps in p['pages']:
  for bs in [2,3]:
   a=create(ps,2048);b=create(ps,2048);pt=torch.zeros(bs,dtype=torch.int64);st=torch.full((bs,),32768,dtype=torch.int64);last=torch.full((bs,),-1,dtype=torch.int64);count=common.get_num_new_pages(st,page_size=ps,prefix_lens=pt);check(count>=200,'large reference uses existing Torch path');left=a.alloc_extend(pt,pt.clone(),st,st.clone(),last,32768*bs);reads=DeviceLens.reads;right=b.alloc_extend(DeviceLens.wrap(pt),pt.clone(),DeviceLens.wrap(st),st.clone(),last,32768*bs,num_new_pages=count);check(torch.equal(left,right) and left.dtype==right.dtype==torch.int32,'existing optional-count output equivalent');check(DeviceLens.reads==reads,'no tagged scalar read with explicit count');check(torch.equal(a.get_all_free_pages(),b.get_all_free_pages()),'identical page consumption');result['cases'].append({'family':'existing-count-large-equivalence','page_size':ps,'batch':bs,'new_pages':count,'status':'PASS','scope':'Existing capability, not candidate/native16K success'})
 check(not assignments,'no request-table writes after observation stop');result['source_scalar_observations']=DeviceLens.reads;result.update(probe_status='PASS',correctness='PASS',candidate_applied=False,whole_allocation_or_PD_or_Eagle_module_imported=False)
except Exception as exc:
 result.update(probe_status='UNAVAILABLE' if stage=='imports' else 'FAIL',correctness='NOT_EVALUATED' if stage=='imports' else 'FAIL',failed_stage=stage,error=repr(exc),traceback=traceback.format_exc())
print(json.dumps(result,ensure_ascii=False));sys.exit(0 if result['probe_status'] in ['PASS','UNAVAILABLE'] else 1)
