#!/usr/bin/env python3
"""Execute frozen kernel body as a bounds-checked scalar source model, not Triton."""
import ast,copy,datetime,hashlib,itertools,json,pathlib,platform,random,sys,types
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L015-R004-source-model';O.mkdir(parents=True,exist_ok=False);protocol_path=R/'specs/L015-source-model.json';p=json.loads(protocol_path.read_text());sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
for name,h in p['inputs'].items():assert sha(R/name)==h
kernel_path=R/p['kernel'];tree=ast.parse(kernel_path.read_text());defs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)};kernel=copy.deepcopy(defs['verify_tree_greedy_kernel']);body_hash=lambda n:hashlib.sha256(ast.dump(ast.Module(body=n.body,type_ignores=[]),include_attributes=False).encode()).hexdigest();original_body_sha=body_hash(kernel);kernel.decorator_list=[]
for a in kernel.args.args:a.annotation=None
assert body_hash(kernel)==original_body_sha
class Vec(list):
 def __add__(self,x):return Vec(a+(x[i] if isinstance(x,list) else x) for i,a in enumerate(self))
 __radd__=__add__
 def __sub__(self,x):return Vec(a-(x[i] if isinstance(x,list) else x) for i,a in enumerate(self))
 def __lt__(self,x):return Vec(a<x for a in self)
class MemoryBounds(Exception):pass
class Mem:
 def __init__(self,name,data,shape):self.name=name;self.data=list(data);self.shape=shape;self.writes=[]
 def __add__(self,index):return Ptr(self,index)
class Ptr:
 def __init__(self,mem,index):self.mem=mem;self.index=index
 def __add__(self,index):return Ptr(self.mem,self.index+index)
class ScalarTL:
 req=0
 @classmethod
 def program_id(cls,i):assert i==0;return cls.req
 @staticmethod
 def arange(start,end):return Vec(range(start,end))
 @staticmethod
 def load(ptr,mask=None,other=0):
  def read(i,enabled):
   if not enabled:return other
   if not 0<=i<len(ptr.mem.data):raise MemoryBounds(f'load {ptr.mem.name}[{i}] size{len(ptr.mem.data)}')
   return ptr.mem.data[i]
  if isinstance(ptr.index,list):return Vec(read(i,True if mask is None else mask[j]) for j,i in enumerate(ptr.index))
  return read(ptr.index,True if mask is None else mask)
 @staticmethod
 def store(ptr,value):
  i=ptr.index;assert isinstance(i,int)
  if not 0<=i<len(ptr.mem.data):raise MemoryBounds(f'store {ptr.mem.name}[{i}] size{len(ptr.mem.data)}')
  ptr.mem.data[i]=value;ptr.mem.writes.append({'program':ScalarTL.req,'offset':i,'value':value})
class AL:
 @staticmethod
 def get_element(v,index):
  i=index[0]
  if not 0<=i<len(v):raise MemoryBounds(f'get_element[{i}] size{len(v)}')
  return v[i]
env={'tl':ScalarTL,'al':AL};exec(compile(ast.Module(body=[kernel],type_ignores=[]),str(kernel_path),'exec'),env)
class Launch:
 def __getitem__(self,grid):
  def run(**kwargs):
   assert len(grid)==1
   for req in range(grid[0]):ScalarTL.req=req;env['verify_tree_greedy_kernel'](**kwargs)
  return run
wrapper=copy.deepcopy(defs['verify_tree_greedy']);wrapper_env={'verify_tree_greedy_kernel':Launch(),'triton':types.SimpleNamespace(next_power_of_2=lambda x:1<<(x-1).bit_length())};exec(compile(ast.Module(body=[wrapper],type_ignores=[]),str(kernel_path),'exec'),wrapper_env);run_wrapper=wrapper_env['verify_tree_greedy']
utils_tree=ast.parse((R/p['eagle_utils']).read_text());sample=next(n for n in utils_tree.body if isinstance(n,ast.FunctionDef) and n.name=='eagle_sample');ret=sample.body[-1];assert isinstance(ret,ast.Return) and 'num_correct_drafts + 1' in ast.unparse(ret)
finish=ast.parse('def finish(predict,num_correct_drafts,accept_index):\n pass').body[0];finish.body=[copy.deepcopy(ret)];ast.fix_missing_locations(finish);finish_env={};exec(compile(ast.Module(body=[finish],type_ignores=[]),'source-return','exec'),finish_env)
def oracle(candidates,target,child,sibling,retrieve,depth):
 bs=len(candidates);w=len(candidates[0]);pred=[-1]*(bs*w);indices=[-1]*(bs*depth);correct=[];bonuses=[]
 for req in range(bs):
  node=0;path=[]
  while True:
   assert len(path)<depth;path.append(node);pred[retrieve[req][node]]=target[req][node];next_node=child[req][node];seen=set()
   while next_node!=-1 and candidates[req][next_node]!=target[req][node]:
    assert next_node not in seen;seen.add(next_node);next_node=sibling[req][next_node]
   if next_node==-1:break
   node=next_node
  for i,node in enumerate(path):indices[req*depth+i]=retrieve[req][node]
  correct.append(len(path)-1);bonuses.append(target[req][path[-1]])
 return {'predict':pred,'accept_index':indices,'num_correct_drafts':correct,'accept_lens':[x+1 for x in correct],'bonus':bonuses}
def exercise(case):
 bs=case['batch'];w=case['width'];depth=case['depth'];prefixes=case.get('prefixes',[w-1]*bs);base=1000+case['serial']*500;cs=[[base+r*100+i for i in range(w)] for r in range(bs)];target=[[140000+r*100+i for i in range(w)] for r in range(bs)];children=[[i+1 if i+1<w else -1 for i in range(w)] for _ in range(bs)];siblings=[[-1]*w for _ in range(bs)];retrieve=[[r*w+i for i in range(w)] for r in range(bs)]
 for r,k in enumerate(prefixes):
  for i in range(k):target[r][i]=cs[r][i+1]
 if case['kind']=='branch-counterexample':
  assert w==4;children=[[1,3,-1,-1] for _ in range(bs)];siblings=[[-1,2,-1,-1] for _ in range(bs)]
  for r in range(bs):target[r][0]=cs[r][2]
 elif case['kind']=='bounds-counterexample':
  assert w==8;children=[[1,3,5,7,-1,-1,-1,-1] for _ in range(bs)];siblings=[[-1,2,-1,4,-1,6,-1,-1] for _ in range(bs)]
 flat=lambda values:[x for row in values for x in row]
 arrays={'predicts':Mem('predict',[-1]*(bs*w),(bs*w,)),'accept_index':Mem('accept_index',[-1]*(bs*depth),(bs,depth)),'accept_token_num':Mem('num_correct',[-1]*bs,(bs,)),'candidates':Mem('candidates',flat(cs),(bs,w)),'retrive_index':Mem('retrieve',flat(retrieve),(bs,w)),'retrive_next_token':Mem('child',flat(children),(bs,w)),'retrive_next_sibling':Mem('sibling',flat(siblings),(bs,w)),'target_predict':Mem('target',flat(target),(bs,w))};want=oracle(cs,target,children,siblings,retrieve,depth);error=None
 try:run_wrapper(**arrays)
 except MemoryBounds as exc:error=str(exc)
 got={'predict':arrays['predicts'].data,'accept_index':arrays['accept_index'].data,'num_correct_drafts':arrays['accept_token_num'].data};matches=not error and all(got[k]==want[k] for k in got)
 if case['kind']=='chain':
  assert matches,(case,got,want,error)
  _,lens,_=finish_env['finish'](None,Vec(got['num_correct_drafts']),None);assert list(lens)==want['accept_lens'];assert list(lens-1)==got['num_correct_drafts'];assert [got['predict'][got['accept_index'][r*depth+lens[r]-1]] for r in range(bs)]==want['bonus']
  return {'id':case['id'],'kind':'chain','batch':bs,'width':w,'depth':depth,'prefixes':prefixes,'status':'MATCH','num_correct_drafts':got['num_correct_drafts'],'accept_lens':list(lens),'bounds_error':None}
 assert not matches,'Counterexample did not distinguish source model from tree oracle'
 return {'id':case['id'],'kind':case['kind'],'status':'CONDITIONAL_COUNTEREXAMPLE','scope':'Not a demonstrated reachable native configuration','actual':got,'oracle':want,'bounds_error':error,'accept_index_writes':arrays['accept_index'].writes}
rows=[exercise(case) for case in p['cases']]
# Execute the actual NPU-default prefix and EAGLE local tail with explicit resolution fixtures.
npu_tree=ast.parse((R/p['npu_utils']).read_text());npu=next(n for n in npu_tree.body if isinstance(n,ast.FunctionDef) and n.name=='set_default_server_args');stop=next(i for i,n in enumerate(npu.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='npu_mem' for t in n.targets));prefix=copy.deepcopy(npu);prefix.body=prefix.body[:stop]
for a in prefix.args.args:a.annotation=None
hook_tree=ast.parse((R/p['spec_hook']).read_text());hook=next(n for n in hook_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_handle_eagle_family');tail=next(i for i,n in enumerate(hook.body) if isinstance(n,ast.If) and 'speculative_eagle_topk == 1' in ast.unparse(n.test) and 'speculative_num_draft_tokens' in ast.unparse(n.test));gate=ast.parse('def local_gate(args):\n cfg=resolving_view(args)').body[0];gate.body+=copy.deepcopy(hook.body[tail:]);ast.fix_missing_locations(gate)
def declare(args,owner,**fields):args.stash.update(fields)
gate_env={'resolving_view':lambda a:a.raw,'resolved_view':lambda a:types.SimpleNamespace(**{**vars(a.raw),**a.stash}),'declare_resolution':declare,'logger':types.SimpleNamespace(warning=lambda *a:None)};exec(compile(ast.Module(body=[prefix,gate],type_ignores=[]),'actual-source-gate-fragments','exec'),gate_env);gate_rows=[]
for topk,page,backend,width in itertools.product([1,2,4],[None,1,128],[None,'ascend','triton','flashinfer'],[4,8]):
 args=types.SimpleNamespace(raw=types.SimpleNamespace(speculative_eagle_topk=topk,speculative_num_steps=3,speculative_num_draft_tokens=width,page_size=page,attention_backend=backend),stash={});gate_env['set_default_server_args'](args);args.raw=types.SimpleNamespace(**{**vars(args.raw),**args.stash});args.stash={};error=None
 try:gate_env['local_gate'](args)
 except ValueError as exc:error=str(exc)
 final={**vars(args.raw),**args.stash};assert bool(error)==(topk>1 and page!=1);assert final['attention_backend']=='ascend';assert topk!=1 or final['speculative_num_draft_tokens']==4;gate_rows.append({'topk':topk,'input_page':page,'input_backend':backend,'input_width':width,'resolved':final,'tail_error':error,'scope':'Selected source statements in minimal resolution fixture, not full startup or backend support'})
used={n.id for n in ast.walk(kernel) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)};assert not ({'retrive_next_token','retrive_next_sibling'}&used)
greedy=copy.deepcopy(next(n for n in utils_tree.body if isinstance(n,ast.FunctionDef) and n.name=='_verify_uses_greedy'));g={};exec(compile(ast.Module(body=[greedy],type_ignores=[]),'actual-greedy-predicate','exec'),g);greedy_rows=[]
for values in itertools.product([False,True],repeat=5):
 flags=dict(zip(['is_all_greedy','is_cpu','is_hip','is_xpu','use_rejection_sampling'],values));out=g['_verify_uses_greedy'](**flags);greedy_rows.append({**flags,'greedy':out})
pipe=ast.parse((R/p['pipeline']).read_text());order={n.args[0].id:n.lineno for n in ast.walk(pipe) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_hook' and n.args and isinstance(n.args[0],ast.Name) and n.args[0].id in ['handle_pd_disaggregation','handle_npu_backends','handle_speculative_decoding']};assert order['handle_pd_disaggregation']<order['handle_npu_backends']<order['handle_speculative_decoding']
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'correctness':'PASS','probe_status':'PASS','source_commit':p['source_commit'],'protocol_sha256':sha(protocol_path),'source_hashes':p['inputs'],'script_sha256':sha(pathlib.Path(__file__)),'kernel_body_ast_sha256':original_body_sha,'kernel_body_unchanged':True,'wrapper_body_unchanged':True,'environment':{'python':sys.version,'machine':platform.machine()},'cases':rows,'gate_cases':gate_rows,'greedy_predicate_cases':greedy_rows,'pipeline_hook_lines':order,'counts':{'chain_matches':sum(r['kind']=='chain' for r in rows),'conditional_counterexamples':sum(r['kind']!='chain' for r in rows),'gate_rows':len(gate_rows),'gate_rejections':sum(bool(r['tail_error']) for r in gate_rows),'greedy_predicate_rows':len(greedy_rows)},'scope':'Bounds-checked pure-Python source model of unmodified kernel body and wrapper, with independent topology oracle. No Triton import/compile/interpreter, NPU execution, integer-dtype/stream behavior, native tree builder or model inference tested. Gate statements use a fixture, not real ServerArgs resolution. No performance claim.'};(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'counts':result['counts'],'correctness':'PASS','scope':'SOURCE_MODEL_CPU_ONLY'}))
