#!/usr/bin/env python3
"""Full installed NPU allocator import and genuine CPU metadata execution."""
import argparse,datetime,hashlib,importlib,json,os,pathlib,subprocess,sys,traceback
ap=argparse.ArgumentParser()
for name in ['source-root','expected-hashes','model-hashes','case-key','protocol','fixture-root']:ap.add_argument('--'+name,required=True)
ap.add_argument('--stdout',action='store_true');args=ap.parse_args();root=pathlib.Path('/sgl-workspace/sglang');snapshot=pathlib.Path(args.source_root);expected=json.loads(args.expected_hashes);models=json.loads(args.model_hashes);p=json.loads(pathlib.Path(args.protocol).read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0' and os.environ.get('SGLANG_USE_CPU_ENGINE')=='1';nodes=sorted({str(q) for pat in ['davinci*','devmm_svm','hisi_hdc','ascend*'] for q in pathlib.Path('/dev').glob(pat)});assert not nodes
head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();assert head==p['source_commit']
for f,h in expected.items():assert sha(root/f)==h==sha(snapshot/f);assert subprocess.check_output(['git','-C',str(root),'show',head+':'+f])==(root/f).read_bytes()
for f,h in models.items():assert sha(pathlib.Path('/model')/f)==h
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'probe_status':'PENDING','correctness':'NOT_EVALUATED','source_commit':head,'source_hashes':expected,'model_hashes':models,'script_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(pathlib.Path(args.protocol)),'accelerator_device_nodes':nodes,'actual_modules':[],'cases':[],'semantic_checks':0,'native_executed':False,'small_extend_native_kernel_executed':False,'performance':'NOT_MEASURED','scope':'Actual full NPU allocator module/class and unmodified inherited methods on Torch CPU tensors, with kvcache=None. Manages slot/page metadata only; no KV payload, device memory, native kernel, PD transfer, model inference or task-runtime factory claim.'};stage='full-module-import'
try:
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1);os.environ['SGLANG_DEBUG_MEMORY_POOL']='true'
 npu=importlib.import_module('sglang.srt.hardware_backend.npu.allocator_npu');Alloc=npu.NPUPagedTokenToKVPoolAllocator
 for name in ['sglang.srt.hardware_backend.npu.allocator_npu','sglang.srt.mem_cache.allocator.base','sglang.srt.mem_cache.allocator.paged','sglang.srt.mem_cache.allocator.token','sglang.kernels.ops.memory.allocator','sglang.srt.utils.common']:
  m=importlib.import_module(name);f=str(pathlib.Path(m.__file__).relative_to(root));assert sha(root/f)==expected[f];result['actual_modules'].append({'name':name,'path':f,'sha256':expected[f],'full_module_import':True})
 result['environment']={'python':sys.version,'torch':torch.__version__};result['full_module_import']='PASS';stage='actual-allocator-lifecycle'
 def check(value,message):
  assert value,message
  result['semantic_checks']+=1
 def create(ps,sort,pages=1024):return Alloc(pages*ps,ps,torch.float32,'cpu',None,sort)
 def pageset(a):return set(a.get_all_free_pages().tolist())
 for ps in p['page_sizes']:
  for sort in p['need_sort']:
   a=create(ps,sort,8);case={'family':'lifecycle','page_size':ps,'need_sort':sort};result['in_progress']=case;allpages=pageset(a);check(a.available_size()==8*ps and 0 not in allpages,'initial capacity excludes padded page');v=a.alloc(3*ps);check(v.numel()==3*ps and torch.unique(v//ps).numel()==3,'page allocation quantum');before=pageset(a);check(a.alloc(6*ps) is None and pageset(a)==before,'insufficient allocation leaves ownership');caught=False
   try:a.alloc(ps+1)
   except AssertionError:caught=True
   check(caught and pageset(a)==before,'debug aligned alloc rejects before mutation');caught=False
   try:a.free_segment(v[:1],start_pos=1)
   except AssertionError:caught=True
   check(caught and pageset(a)==before,'misaligned free rejects before mutation');a.free(v[:1]);check(a.available_size()==6*ps,'one slot free returns whole page');a.free_segment(v[ps:],start_pos=ps);check(a.available_size()==8*ps and pageset(a)==allpages,'aligned segment releases exact remaining pages');check(a.num_staged_pages==(3 if sort else 0),'PD sorted free staging')
   v=a.alloc(8*ps);saved=v.clone();check(v.numel()==8*ps and a.available_size()==0,'alloc merges staged pages');a.free_group_begin();first=v[:ps];second=v[ps:2*ps];a.free(first);a.free_segment(second,start_pos=ps);first.fill_(999999);second.fill_(999999);check(a.available_size()==0,'deferred free not exposed before group end');caught=False
   try:a.free_group_begin()
   except AssertionError:caught=True
   check(caught,'nested group rejected');a.free_group_end();check(a.available_size()==2*ps and pageset(a)==set((saved[:2*ps]//ps).tolist()),'deferred views are owned copies');a.free_segment(saved[2*ps:],start_pos=2*ps);check(a.available_size()==8*ps and pageset(a)==allpages,'group and remaining release conserve pages');result['cases'].append({**case,'status':'PASS'});result.pop('in_progress')
 stage='actual-npu-extend-large-Torch-branch'
 def oracle(ps,prefix,seq,last,free):
  out=[];cursor=0
  for pre,end,loc in zip(prefix,seq,last):
   boundary=((pre+ps-1)//ps)*ps;new=((end+ps-1)//ps)-((pre+ps-1)//ps)
   for pos in range(pre,end):out.append(loc+1+pos-pre if pos<boundary else free[cursor+(pos-boundary)//ps]*ps+pos%ps)
   cursor+=new
  return torch.tensor(out,dtype=torch.int32),cursor
 for ps in p['page_sizes']:
  for sort in p['need_sort']:
   for bs in p['batch_sizes']:
    for pre in p['prefixes']:
     length=32768;prefix=[pre]*bs;seq=[length]*bs;a=create(ps,sort,2048);last=[]
     for x in prefix:
      ids=a.alloc(((x+ps-1)//ps)*ps);last.append(int(ids[x-1]) if x else -1)
     before=a.available_size();free=a.free_pages.tolist();want,count=oracle(ps,prefix,seq,last,free);check(count>=200,'large case uses actual Torch branch');pt=torch.tensor(prefix,dtype=torch.int64);st=torch.tensor(seq,dtype=torch.int64);lt=torch.tensor(last,dtype=torch.int64);out=a.alloc_extend(pt,pt.clone(),st,st.clone(),lt,sum(end-start for start,end in zip(prefix,seq)));case={'family':'large-extend','page_size':ps,'need_sort':sort,'batch':bs,'raw_length':length,'prefix':pre,'new_pages':count};result['in_progress']=case
     check(out.dtype==torch.int32 and torch.equal(out,want),'extend equals independent per-position page oracle');check(a.available_size()==before-count*ps,'extend charges page units');check(torch.unique(out).numel()==out.numel(),'extend no overlapping output slots');check(not (set((out//ps).tolist())&pageset(a)),'output pages not free');result['cases'].append({**case,'status':'PASS'});result.pop('in_progress')
 # Boundaries around the literal native/Torch split; 199 succeeds only on NPU.
 for total in [199,200,201]:
  ps=128;seq=[100*ps,(total-100)*ps];a=create(ps,True,total-1 if total==199 else total);z=torch.zeros(2,dtype=torch.int64);st=torch.tensor(seq,dtype=torch.int64);last=torch.full((2,),-1,dtype=torch.int64);before=pageset(a)
  out=a.alloc_extend(z,z.clone(),st,st.clone(),last,sum(seq))
  if total==199:check(out is None and pageset(a)==before,'small branch exhaustion rejects before native import')
  else:
   want,count=oracle(ps,[0,0],seq,[-1,-1],sorted(before));check(out.dtype==torch.int32 and torch.equal(out,want),'200/201 pages use actual Torch branch');check(a.available_size()==0,'boundary pool consumed exactly')
  result['cases'].append({'family':'extend-threshold','requested_pages':total,'status':'PASS','native_branch_reached':False,'observation':'capacity rejection before import' if total==199 else 'actual Torch path'})
 stage='actual-npu-decode'
 for ps in p['page_sizes']:
  for sort in p['need_sort']:
   for mode in ['mixed','all-new','no-new','empty-prefix']:
    prefix={'mixed':[ps-1,ps,2*ps],'all-new':[ps,2*ps],'no-new':[1,2,ps-1],'empty-prefix':[0,ps-1,ps]}[mode];a=create(ps,sort,32);last=[]
    for x in prefix:
     v=a.alloc(((x+ps-1)//ps)*ps);last.append(int(v[x-1]) if x else -1)
    seq=[x+1 for x in prefix];free=a.free_pages.tolist();want,count=oracle(ps,prefix,seq,last,free);before=a.available_size();st=torch.tensor(seq,dtype=torch.int64);out=a.alloc_decode(st,st.clone(),torch.tensor(last,dtype=torch.int64));check(out.dtype==torch.int32 and torch.equal(out,want),'decode per-position oracle');check(a.available_size()==before-count*ps,'decode page charge');result['cases'].append({'family':'decode','page_size':ps,'need_sort':sort,'mode':mode,'new_pages':count,'status':'PASS'})
 a=create(128,True,1);v=a.alloc(128);st=torch.tensor([129,129],dtype=torch.int64);before=pageset(a);out=a.alloc_decode(st,st.clone(),torch.tensor([int(v[-1]),int(v[-1])],dtype=torch.int64));check(out is None and pageset(a)==before,'decode capacity rejection before index consumption');result['cases'].append({'family':'decode-exhaustion','status':'PASS'})
 result.update(probe_status='PASS',correctness='PASS',native_executed=False,small_extend_sufficient_capacity='NOT_EXECUTED: requires actual sgl_kernel_npu kernel',full_budget_module='NOT_ATTEMPTED',task_factory_runtime='UNCONFIRMED')
except Exception as exc:
 result.update(probe_status='UNAVAILABLE' if stage=='full-module-import' else 'FAIL',correctness='NOT_EVALUATED' if stage=='full-module-import' else 'FAIL',failed_stage=stage,error=repr(exc),traceback=traceback.format_exc())
print(json.dumps(result,ensure_ascii=False));sys.exit(0 if result['probe_status'] in ['PASS','UNAVAILABLE'] else 1)
