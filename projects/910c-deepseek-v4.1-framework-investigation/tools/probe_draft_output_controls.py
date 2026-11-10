#!/usr/bin/env python3
"""Original DSpark loop/guards on small finite tags; no framework/native math."""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
import ast,copy,json,struct
R=Path('audit/execution-search')
def main():
 sources=json.loads((R/'draft-output-dependencies.json').read_text())['sources'];records=[];excerpts=[]
 for s in sources:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
 def select(suf,owner,name):
  s=next(s for s in sources if s['runtime_path'].endswith(suf));p=Path(s['local_path']);tree=ast.parse(p.read_text());par=tree if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner);n=copy.deepcopy(next(n for n in par.body if getattr(n,'name','')==name));excerpts.append(dict(runtime_path=s['runtime_path'],sha256=s['runtime_sha256'],owner=owner,name=name,first=n.lineno,last=n.end_lineno,lines=p.read_text().splitlines()[n.lineno-1:n.end_lineno]));return n
 def compile_nodes(nodes,g):exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),'<original-draft-output>','exec'),g)
 def fn(suf,owner,name,g):
  n=select(suf,owner,name);n.decorator_list=[];compile_nodes([n],g);return g[name]
 # Exact original sequential Markov loop, all other numerical kernels are
 # explicit small finite fixture callbacks. Preserve column views/copies.
 merged=select('/vllm_ascend/spec_decode/llm_base_proposer.py','AscendSpecDecodeBaseProposer','_run_merged_draft');loop=next(n for n in ast.walk(merged) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='idx' and ast.unparse(n.iter)=='range(self.num_speculative_tokens)')
 class IDs:
  def __init__(self,seeds,K):self.rows=[[s]+[-9]*K for s in seeds]
  def __getitem__(self,key):return IDColumn(self,key[1])
 class IDColumn:
  def __init__(self,owner,col):self.owner,self.col=owner,col
  def values(self):return [r[self.col] for r in self.owner.rows]
  def copy_(self,values):
   for row,x in zip(self.owner.rows,values):row[self.col]=x
 class Logits:
  def __init__(self,B,K,V):self.rows=[[[0]*V for _ in range(K)] for _ in range(B)]
  def __getitem__(self,key):return ScoreColumn(self,key[1])
 class ScoreColumn:
  def __init__(self,owner,col):self.owner,self.col=owner,col
  def add_(self,bias):
   for r,b in zip(self.owner.rows,bias):r[self.col]=[a+x for a,x in zip(r[self.col],b)]
   return self
  def argmax(self,dim):assert dim==-1;return [max(range(len(r[self.col])),key=lambda i:r[self.col][i]) for r in self.owner.rows]
 for seeds in [[0,3],[7,2]]:
  calls=[];draft=IDs(seeds,5);logits=Logits(2,5,8)
  def embed(x):values=x.values();calls.append(values);return values
  model=NS(markov_embed=embed,markov_bias=lambda ids:[[5 if v==(x+1)%8 else 0 for v in range(8)] for x in ids])
  g=dict(self=NS(num_speculative_tokens=5,model=model),draft_token_ids=draft,logits=logits,use_probabilistic=False,dspark_has_vocab_mapping=False,sampling_metadata=None,dspark_probs_list=[])
  compile_nodes([copy.deepcopy(loop)],g);expected=[[(s+i)%8 for i in range(6)] for s in seeds];assert draft.rows==expected and calls==[[r[i] for r in expected] for i in range(5)]
  records.append(dict(kind='original_K5_sequential_loop',seeds=seeds,all_buffer_columns=draft.rows,Markov_input_IDs_in_order=calls,qualification='B2,V8 integer callbacks only; original loop and ID column mutation. Not native projection/add/argmax values, allocation or full merged-forward proof.'))
 # Original probability selector: frozen greedy-draft guard wins even if a
 # target request is random. Optional stochastic path must remain separate.
 calls=[]
 class FakeLogits:
  shape=(4,8)
  def argmax(self,dim):calls.append('argmax');return [1]*4
 g=dict(compute_probs_and_sample_next_token=lambda *a:calls.append('probability_RNG_sink') or ('ids','probs'))
 sample=fn('/vllm/vllm/v1/spec_decode/llm_base_proposer.py','SpecDecodeBaseProposer','_sample_from_logits',g)
 for enabled,allgreedy in [(False,True),(False,False),(True,True),(True,False)]:
  calls.clear();meta=NS(all_greedy=allgreedy,temperature=None);out=sample(NS(_enable_probabilistic_draft_probs=enabled,use_fp64_gumbel=False),FakeLogits(),meta);assert calls==(['probability_RNG_sink'] if enabled and not allgreedy else ['argmax']);records.append(dict(kind='original_draft_probability_guard',enabled=enabled,target_all_greedy=allgreedy,events=list(calls),returned_probability=out[1] is not None))
 # Exact constructor branch and exact caller guard, not merely a defined head.
 ctor=select('/vllm_ascend/spec_decode/dspark_proposer.py','AscendDSparkProposer','__init__');assign=next(n for n in ctor.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='dynamic_spec' for t in n.targets));guard=next(n for n in ctor.body if isinstance(n,ast.If) and ast.unparse(n.test)=="dynamic_spec_config.method == 'dspark'");invoke=next(n for n in ast.walk(merged) if isinstance(n,ast.If) and ast.unparse(n.test)=='self.dynamic_spec is not None')
 for method in [None,'dspark']:
  events=[];obj=NS(max_batch_size=4,num_speculative_tokens=5,model='opaque_model');g=dict(self=obj,dynamic_spec_config=NS(method=method,method_params={}),device='opaque_device',DynamicSpecScheduler=lambda **kw:NS(update=lambda **kw:events.append('dynamic_update')),last_hidden_states='hidden',draft_token_ids='ids',num_blk=4)
  compile_nodes([copy.deepcopy(assign),copy.deepcopy(guard),copy.deepcopy(invoke)],g);assert events==([] if method is None else ['dynamic_update']);records.append(dict(kind='original_dynamic_constructor_and_actual_caller_guard',method=method,events=events,current_frozen_case=method is None))
 get_probs=fn('/v1/worker/gpu_model_runner.py','GPUModelRunner','_get_spec_decode_draft_probs',{})
 assert get_probs(NS(_draft_probs=None,_draft_prob_req_ids=None),None) is None;records.append(dict(kind='original_draft_prob_none_consumer',result=None,probability_tensor_not_required=True))
 # Execute the exact original per-request loop from the Triton greedy kernel
 # with scalar load/store callbacks. Prologue vector loads/native compilation
 # are not simulated; request counts/start/masks are explicit fixture inputs.
 native=select('/ops/triton/reject_sample.py',None,'rejection_greedy_sample_triton');inner=next(n for n in native.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='pos')
 class Ptr:
  def __init__(self,data,index=0):self.data,self.index=data,index
  def __add__(self,n):return Ptr(self.data,self.index+n)
 def load(p):assert 0<=p.index<len(p.data);return p.data[p.index]
 def store(p,x):assert 0<=p.index<len(p.data);p.data[p.index]=x
 tl=NS(range=range,load=load,store=store);gbase=dict(tl=tl,get_element=lambda x,key:x[key[0]])
 renew=fn('/ops/triton/reject_sample.py',None,'bonus_renew',gbase)
 cases=[([5,5],[True,True],list(range(1,11)),list(range(1,11)),[[1,2,3,4,5,101],[6,7,8,9,10,102]]),([5,5],[True,True],list(range(1,11)),[1,2,99,4,5,6,7,8,9,10],[[1,2,99,-1,-1,-1],[6,7,8,9,10,102]]),([0,2],[True,True],[4,5],[4,5],[[101,-1,-1,-1,-1,-1],[4,5,102,-1,-1,-1]]),([2,0],[True,False],[4,5],[4,5],[[4,5,101,-1,-1,-1],[-1,-1,-1,-1,-1,-1]])]
 for counts,masks,draft,target,expected in cases:
  out=[-1]*12;starts=[0,counts[0]];g=dict(gbase,bonus_renew=renew,num_draft_tokens=counts,start_idx=starts,is_greedy_mask=masks,block_idx=0,BLOCK_SIZE=2,max_spec_len=5,SYNTHETIC_MODE=False,draft_token_ids_ptr=Ptr(draft),target_argmax_ptr=Ptr(target),bonus_token_ids_ptr=Ptr([101,102]),output_token_ids_ptr=Ptr(out))
  compile_nodes([copy.deepcopy(inner)],g);actual=[out[:6],out[6:]];assert actual==expected;records.append(dict(kind='original_native_greedy_inner_loop_scalar_control',counts=counts,greedy_mask=masks,complete_output_slots=actual,qualification='Original native source loop executed as scalar Python with explicit pointer callbacks; not Triton/native bits, vector-prologue masked-load proof or runtime count population.'))
 # Finite top1 packet/tie/pad controls. Dense argmax below is a mathematical
 # reference, not a stand-in claim about native NaN/tie or loaded kernels.
 for values,V in [([1,7,7,0],4),([2,2,1,99],3),([-float('inf')]*4,3),([4,0,3,2],4)]:
  full=max(range(V),key=lambda i:values[i]);candidates=[]
  for begin in [0,2]:
   ids=list(range(begin,min(begin+2,V)))
   if ids:candidates.append(max(ids,key=lambda i:(values[i],-i)))
  packet=max(candidates,key=lambda i:(values[i],-i));assert packet==full;records.append(dict(kind='finite_vocab_pair_tie_and_pad',scores=['-inf' if x==-float('inf') else x for x in values],valid_vocab=V,dense_id=full,packet_id=packet,qualification='Mathematical scalar ordering only; native NaN/tie/dtype/packet ABI remains.'))
 base=[2,1];bias=[0,2];assert max(range(2),key=lambda i:base[i])!=max(range(2),key=lambda i:base[i]+bias[i]);records.append(dict(kind='base_top1_is_not_biased_top1',base=base,bias=bias,base_top1=0,biased_top1=1))
 def bf(x):
  u=struct.unpack('<I',struct.pack('<f',x))[0];return (u+0x7fff+((u>>16)&1))>>16
 def val(b):return struct.unpack('<f',struct.pack('<I',b<<16))[0]
 exact_bias=1+2**-8;base=[-1,2**-9];old=[val(bf(base[0]+val(bf(exact_bias)))),base[1]];bad=[val(bf(base[0]+exact_bias)),base[1]];assert max(range(2),key=lambda i:old[i])==1 and max(range(2),key=lambda i:bad[i])==0
 records.append(dict(kind='bias_GEMM_BF16_boundary_changes_ID_if_erased',bias_exact=exact_bias,base=base,old_scores=old,omit_bias_rounding_scores=bad,original_id=1,collapsed_id=0,qualification='Finite IEEE formula only, not native/model witness. Original-rounding fused math remains possible.'))
 out=dict(status='PASS_BOUNDED_ORIGINAL_DRAFT_LOOP_AND_GUARDS_PLUS_FINITE_COUNTEREXAMPLES',source_pins=len(sources),controls=len(records),records=records,qualification='Original Python loop/guards on integer/opaque fixtures and finite IEEE scalars. Not21 full file reviews, native numerical/async completion or new admission.',every_replacement_bitwise=True,whole_chain_complete=False,new_primitive_admissions=0,new_framework_replacement_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest());(R/'draft-output-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(R/'draft-output-source-excerpts.json').write_text(json.dumps(dict(excerpts=excerpts),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(sources=len(sources),controls=len(records),new_admissions=0)))
if __name__=='__main__':main()
