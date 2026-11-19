#!/usr/bin/env python3
"""Exercise hash-pinned candidate admission methods with synthetic CPU mocks.

This reads only explicit candidate source files, extracts selected Python AST
methods, and executes them with fake CPU tensors and cache nodes. It does not
run an NPU, serve requests, create user inputs, or verify historical imports.
"""
import argparse,ast,hashlib,json,sys,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/admission_063/synthetic.json'
PATHS={'decode':'disaggregation/decode.py','utils':'disaggregation/utils.py','common':'utils/common.py','base':'mem_cache/allocator/base.py','full':'mem_cache/unified_cache/components/full.py'}
SHAS={'decode':'39cf972b6bcc04cc8c150c650d845daaa68ccb511a69a71c5b0b8bdf7322a70d','utils':'a8cc425995171f5e65c54c05a1cdd474705dbd128ffb552972e9eadab646ee37','common':'fdb35eb4931f91cb0725fad87649c81927a80c735705d93b2517ea0185158261','base':'12047207f6112562861e6fa086377714a349e2818817ac9e61c0321aea388ee4','full':'467452ec980e79e3aed1d8e2d9f7d29a1d1977963a504e3de3f78e9829e2664b'}


def need(ok,msg):
 if not ok:raise ValueError(msg)


def extract(root):
 trees={};base=Path(root).resolve(strict=True)
 for key,rel in PATHS.items():
  path=base/rel;need(not path.is_symlink() and path.resolve(strict=True).is_relative_to(base),'unsafe candidate path');stat=path.stat();need(stat.st_size<5000000,'source too large');buf=path.read_bytes();after=path.stat();need((stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'source changed');need(hashlib.sha256(buf).hexdigest()==SHAS[key],'candidate SHA mismatch');trees[key]=ast.parse(buf)
 def f(key,name):
  found=[x for x in ast.walk(trees[key]) if isinstance(x,ast.FunctionDef) and x.name==name];need(len(found)==1,'ambiguous function');return found[0]
 def put(nodes,ns):
  doc=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[]);exec(compile(ast.fix_missing_locations(doc),'<candidate-synthetic>','exec'),ns)
 class Tensor:
  def __init__(self,x):self.x=x;self.device='cpu'
  def __add__(self,y):return Tensor(self.x+(y.x if isinstance(y,Tensor) else y))
  def __sub__(self,y):return Tensor(self.x-(y.x if isinstance(y,Tensor) else y))
  def __floordiv__(self,y):return Tensor(self.x//y)
  def to(self,*a):return self
  def item(self):return self.x
 ns={'torch':types.SimpleNamespace(tensor=lambda v,dtype=None:Tensor(v[0]),device=lambda x:x,sum=lambda v:v,int64='int64'),'CLIP_MAX_NEW_TOKEN':4096,'get_disagg':lambda:types.SimpleNamespace(disaggregation_decode_enable_radix_cache=True)}
 put([f('utils','pre_alloc_fill_len'),f('common','get_num_new_pages'),f('base','prealloc_fits')],ns)
 names=('_required_alloc_tokens','_required_admission_tokens','_allocatable_token_budgets','_need_space_for_single_req','_active_req_count','_active_reserved_tokens','_prealloc_reservation_fits')
 put([ast.ClassDef(name='Sim',bases=[],keywords=[],body=[f('decode',n) for n in names],decorator_list=[]),ast.ClassDef(name='Full',bases=[],keywords=[],body=[f('full','acquire_component_lock'),f('full','release_component_lock')],decorator_list=[])],ns)
 Sim=ns['Sim'];Sim._uses_swa_reservation=lambda self:False;Sim._radix_full_evictable=lambda self:self.evictable;Full=ns['Full'];Full.reclaimable_value_len=lambda self,node:len(node.component_data[0].value)
 class Pool:
  page_size=128
  prealloc_fits=ns['prealloc_fits']
  def available_size(self):return self.free
 class Node:
  def __init__(self,parent,refs):self.parent=parent;self.component_data={0:types.SimpleNamespace(value=range(8192),lock_ref=refs)}
 def req(n):return types.SimpleNamespace(origin_input_ids=range(n),output_ids=(),sampling_params=types.SimpleNamespace(max_new_tokens=1024))
 sim=Sim();sim.num_reserved_decode_tokens=512;sim.token_to_kv_pool_allocator=Pool();sim.transfer_queue=types.SimpleNamespace(queue=[]);sim.retracted_queue=[];sim.tree_cache=object();running=[req(n) for n in (2731,2731,2731,2731,2730,2730)];sim.scheduler=types.SimpleNamespace(enable_hisparse=False,last_batch=None,running_batch=types.SimpleNamespace(reqs=running),waiting_queue=[]);retractable=16384;results=[]
 for prelocked in (2,1,0):
  rootnode=object();parent=Node(rootnode,1 if prelocked>=1 else 0);leaf=Node(parent,1 if prelocked==2 else 0);new_locked=(2-prelocked)*8192
  state=types.SimpleNamespace(root_node=rootnode,component_evictable_size_={0:new_locked},component_protected_size_={0:prelocked*8192},evictable_device_leaves={parent,leaf},_update_evictable_leaf_sets=lambda x:None);comp=Full();comp.component_type=0;comp.tree_core=state
  sim.token_to_kv_pool_allocator.free=28160-new_locked;sim.evictable=new_locked;before=sim._allocatable_token_budgets(retractable_tokens=retractable);lock=types.SimpleNamespace(delta=None);comp.acquire_component_lock(leaf,lock)
  sim.evictable=state.component_evictable_size_[0];after=sim._allocatable_token_budgets(retractable_tokens=retractable);required=sim._required_admission_tokens(req(32768),sim._required_alloc_tokens(fill_len=32768,prefix_len=16384),16384,retractable);fit=sim._prealloc_reservation_fits(required,0,full_allocatable_tokens=after,swa_allocatable_tokens=None)
  comp.release_component_lock(leaf,None);sim.evictable=state.component_evictable_size_[0];restored=sim._allocatable_token_budgets(retractable_tokens=retractable)
  need((before,after,required,lock.delta,fit,restored)==(25088,25088-new_locked,16896,new_locked,new_locked<=8192,25088),'synthetic admission mismatch')
  results.append({'prelocked_slots':prelocked*8192,'newly_locked_slots':new_locked,'budget_before':before,'budget_after':after,'required_full':required,'independent_full_gate_fits':fit,'budget_after_unlock':restored})
 unaligned=sim._required_alloc_tokens(fill_len=256,prefix_len=129);naive=-(-(256-129)//128)*128;need(unaligned==0 and naive==128,'page helper mismatch')
 return {'schema':'sglang-candidate-synthetic-admission-v1','candidate_file_sha256':SHAS,'mock_inputs':{'fill_length':32768,'prefix_length':16384,'page_size':128,'reserve_tokens':512,'active_requests':6,'retractable_tokens':retractable},'cases':results,'unaligned_page_helper':{'fill_length':256,'prefix_length':129,'candidate_alloc':unaligned,'naive_difference_ceil':naive,'radix_branch_actual_reachability':'unknown'},'limits':['Executed selected hash-pinned candidate AST methods with synthetic CPU mocks only; not historical G181 runtime or real requests','Candidate helper page and budget logic; no real cache matching, device allocation, metadata/Mamba/LoRA, SWA or shared-byte allocator','Lock change equals reclaimable node values in this synthetic example only; actual K need not equal prefix length','A single independent full-pool gate result is not overall request admission, latency, or end-to-end root cause']}


def checks(doc):
 need(len(doc['cases'])==3,'case count');need([x['independent_full_gate_fits'] for x in doc['cases']]==[True,True,False],'case fit results');need(all(x['budget_before']==x['budget_after_unlock'] for x in doc['cases']),'unlock failure')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate-root',required=True);ap.add_argument('--write',action='store_true');ap.add_argument('--check',action='store_true');args=ap.parse_args();doc=extract(args.candidate_root);checks(doc);text=json.dumps(doc,ensure_ascii=False,indent=2)+'\n'
 if args.write:
  OUT.parent.mkdir(parents=True,exist_ok=True)
  with OUT.open('x',encoding='utf8') as f:f.write(text)
 if args.check:need(OUT.read_text(encoding='utf8')==text,'summary differs')
 print(json.dumps({'status':'pass','cases':len(doc['cases']),'sha256':hashlib.sha256(text.encode()).hexdigest(),'wrote':args.write,'checked':args.check}))


if __name__=='__main__':
 try:main()
 except Exception:raise SystemExit('FAILED_CLOSED: candidate source, synthetic model, or output validation')
