#!/usr/bin/env python3
"""Original selector, row geometry and model-guard bodies; no frameworks.

Boolean row/alias and opaque callback controls do not validate native math/bits.
"""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS,MethodType
from typing import NamedTuple,cast,get_args
from functools import cache
from enum import Enum,auto
import ast,builtins,copy,json
ROOT=Path('audit/execution-search')
def main():
 sources=list({s['runtime_path']:s for n in ['model-gate-dependencies.json','model-gate-consumer-dependencies.json'] for s in json.loads((ROOT/n).read_text())['sources']}.values());assert len(sources)==27
 for s in sources:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
 excerpts=[];records=[]
 def select(suf,owner,name):
  s=next(s for s in sources if s['runtime_path'].endswith(suf));p=Path(s['local_path']);T=ast.parse(p.read_text());parent=T if owner is None else next(n for n in T.body if isinstance(n,ast.ClassDef) and n.name==owner)
  def named(n):return getattr(n,'name',None)==name or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets) or isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id==name
  n=copy.deepcopy(next(n for n in parent.body if named(n)));excerpts.append(dict(runtime_path=s['runtime_path'],sha256=s['runtime_sha256'],owner=owner,name=name,first=n.lineno,last=n.end_lineno,lines=p.read_text().splitlines()[n.lineno-1:n.end_lineno]));return n
 def execute(nodes,g):exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])),'<original-gate-body>','exec'),g)
 def function(suf,owner,name,g):
  n=select(suf,owner,name);n.decorator_list=[];execute([n],g);return g[name]
 # Exact generic selector construction + platform map, with only dependency
 # config/hardware getters and class resolver replaced by explicit opaque data.
 cfg=NS(cache_config=NS(user_specified_block_size=True,block_size=128),kv_transfer_config=None,speculative_config=NS(enable_adaptive_verification=False),attention_config=NS(use_non_causal=False,backend=None,backend_per_kind={}),parallel_config=NS(prefill_context_parallel_size=1,decode_context_parallel_size=1),model_config=NS(rswa_window=None))
 g=dict(NamedTuple=NamedTuple,torch=NS(dtype=str),CacheDType=str,AttentionType=NS(DECODER='decoder'),cast=cast,get_args=get_args,cache=cache,envs=NS(VLLM_BATCH_INVARIANT=False),logger=NS(info=lambda *a:None),get_ascend_config=lambda:NS(rl_config=NS(enabled=False,enable_training_consistency=False)),get_current_hardware_profile=lambda:NS(attention_backend_family='STANDARD'),AttentionBackendFamily=NS(COMPATIBILITY='COMPATIBILITY'))
 fake_platform=NS(); resolved=[]
 class AttentionBackend:pass
 class AscendAttentionBackend(AttentionBackend):pass
 class AscendMLABackend(AttentionBackend):pass
 class AscendDSABackend(AttentionBackend):pass
 def resolve(path):resolved.append(path);return {'AscendAttentionBackend':AscendAttentionBackend,'AscendMLABackend':AscendMLABackend,'AscendDSABackend':AscendDSABackend}[path.split('.')[-1]]
 def fake_import(name,*a,**k):
  if name=='vllm.config':return NS(get_current_vllm_config=lambda:cfg)
  if name=='vllm.platforms':return NS(current_platform=fake_platform)
  if name=='vllm_ascend.attention.attention_v1':return NS(AscendAttentionBackend=AscendAttentionBackend)
  if name=='__future__':return __import__(name,*a,**k)
  raise AssertionError('Unexpected import: '+name)
 g['__builtins__']={**vars(builtins),'__import__':fake_import};g['resolve_obj_by_qualname']=resolve
 function('/platform.py',None,'_validate_fa3_backend',g)
 platform_path=next(s['local_path'] for s in sources if s['runtime_path'].endswith('/platform.py'));P=ast.parse(Path(platform_path).read_text());owner=next(n.name for n in P.body if isinstance(n,ast.ClassDef) and any(getattr(m,'name',None)=='get_attn_backend_cls' for m in n.body));method=select('/platform.py',owner,'get_attn_backend_cls');method.decorator_list=[];execute([method],g);fake_platform.get_attn_backend_cls=lambda *a,**k:g['get_attn_backend_cls'](fake_platform,*a,**k)
 execute([select('/v1/attention/selector.py',None,'AttentionSelectorConfig'),select('/v1/attention/selector.py',None,'_cached_get_attn_backend'),select('/v1/attention/selector.py',None,'get_attn_backend')],g)
 updatable=function('/utils.py',None,'use_updatable_graph',g)
 assert 'use_compress' not in g['AttentionSelectorConfig']._fields
 for mla,mm in [(True,True),(True,False),(False,False)]:
  backend=g['get_attn_backend'](0,'bfloat16',None,use_mla=mla,use_sparse=False,use_mm_prefix=mm)
  assert updatable(backend)==(not mla);records.append(dict(kind='original_generic_selector_and_updatable_predicate',use_mla=mla,use_mm_prefix=mm,selected_path=resolved[-1],updatable=updatable(backend),qualification='Source constructor/map and original predicate; config/hardware dependency data and resolved classes are explicit stubs. GQA is a counterfactual. Declared sibling inheritance separately pinned, no loaded-object claim.'))
 # Rotation selector returns a path when optional metadata survives mapping.
 rotation=function('/utils.py',None,'get_rotation_path',{'Path':Path})
 for description,expected in [({},False),({'optional':{'quarot':{'rotation_map':{'global_rotation':'optional/quarot.safetensors'}}}},True)]:
  out=rotation(NS(quant_config=NS(quant_description=description),model_config=NS(model='/data/tiankuan/wio/model')));assert (out is not None)==expected;records.append(dict(kind='original_rotation_selector',optional_rotation_present=expected,returned_non_none=out is not None,qualification='Pure path selection only; no weight payload.'))
 # The source A3 set is a union of the standard set; it is not the A5 gate set.
 h={'Enum':Enum,'auto':auto};execute([select('/device/hardware_profile.py',None,'HardwareCapability'),select('/device/hardware_profile.py',None,'_STANDARD_CAPABILITIES'),select('/device/hardware_profile.py',None,'_A3_CAPABILITIES')],h)
 caps=h['_A3_CAPABILITIES'];H=h['HardwareCapability'];assert H.MOE_GATING_TOP_K_HASH_VISION in caps and H.ENGRAM_UNROTATED_GATE not in caps and H.FUSED_MOE_COMPATIBILITY not in caps
 records.append(dict(kind='actual_source_A3_capability_set',hash_vision_native=True,unrotated_Engram_gate=False,fused_MoE_compatibility=False,qualification='Original Enum/set declarations evaluated on host CPU. Not loaded operator/version or kernel population.'))
 q={};sp_policy=function('/config/parallel.py','ParallelConfig','use_sequence_parallel_moe',q)
 for tp_size,dp_size,ep,expected in [(8,2,True,True),(1,2,True,False),(8,1,True,False),(8,2,False,False)]:
  assert sp_policy(NS(all2all_backend='allgather_reducescatter',tensor_parallel_size=tp_size,data_parallel_size=dp_size,enable_expert_parallel=ep))==expected;records.append(dict(kind='original_SP_policy',all2all_backend='allgather_reducescatter',tp=tp_size,dp=dp_size,EP=ep,result=expected,qualification='Original property on explicit scalar config. Alternatives are counterfactuals; no live config read.'))
 # Follow the installed Ascend factory instead of using an upstream router's
 # padding helper as if it were automatically inherited.
 q={'get_current_hardware_profile':lambda:NS(supports=lambda cap:cap in caps),'HardwareCapability':H,'AscendFusedMoERouter':lambda **kw:NS(kind='AscendFusedTopKRouter',kwargs=kw),'AscendGroupedTopKRouter':lambda **kw:NS(kind='AscendGroupedTopKRouter',kwargs=kw),'CustomRoutingRouter':lambda **kw:NS(kind='CustomRoutingRouter',kwargs=kw)}
 function('/ops/fused_moe/router/router_factory.py',None,'check_npu_moe_gating_top_k',q);factory=function('/ops/fused_moe/router/router_factory.py',None,'create_ascend_fused_moe_router',q)
 for topk,experts,custom in [(6,384,None),(3,128,None),(6,384,object())]:
  out=factory(topk,experts,scoring_func='sqrtsoftplus',custom_routing_function=custom);assert out.kind==('CustomRoutingRouter' if custom is not None else 'AscendFusedTopKRouter');records.append(dict(kind='original_Ascend_router_factory',topk=topk,experts=experts,custom_function=custom is not None,selected=out.kind,qualification='Original selector with opaque constructor sinks; 6/384 matches target descriptor, alternatives are scalar controls. No native output values.'))
 # Row values are integer/boolean tags. They control exact source geometry and
 # functional custom-op output ownership, not Torch/native representations.
 class Rows:
  def __init__(self,rows,tail=(),owner=None):self.rows=list(rows);self.tail=tuple(tail);self.owner=owner or object();self.device=NS(type='npu');self.dtype='bf16'
  @property
  def shape(self):return (len(self.rows),*self.tail)
  @property
  def ndim(self):return len(self.shape)
  def __getitem__(self,i):return Rows(self.rows[i],self.tail,self.owner)
  def __setitem__(self,i,x):self.rows[i]=x.rows
  def clone(self):return Rows(self.rows,self.tail)
  def zero_(self):self.rows[:]=[0]*len(self.rows);return self
  def new_zeros(self,shape,*a,**k):
   shape=(shape,) if isinstance(shape,int) else shape;return Rows([False if k.get('dtype')=='bool' else 0]*shape[0],shape[1:])
  def new_empty(self,shape):return Rows(['opaque']*shape[0],shape[1:])
  def float(self):return self
  def __mul__(self,other):return self
 tp=NS(size=8,rank=0)
 def pad(x,pads,**kw):return Rows(x.rows+[kw.get('value',0)]*pads[-1],x.tail)
 g=dict(F=NS(pad=pad),get_tensor_model_parallel_world_size=lambda:tp.size,get_tensor_model_parallel_rank=lambda:tp.rank,torch=NS(bool='bool',ops=NS(vllm=NS())))
 mask=function('/models/common/ops/sequence_parallel.py',None,'_ascend_sp_padding_mask_impl',g);shard=function('/models/common/ops/sequence_parallel.py',None,'_ascend_sp_shard_impl',g);g['torch'].ops.vllm.ascend_sp_padding_mask_impl=mask
 front_mask=function('/models/common/ops/sequence_parallel.py',None,'sp_padding_mask',g)
 for T in [0,1,5,8,20,24,25,8192]:
  for rank in range(8):
   tp.rank=rank;x=Rows(range(T),(4,5120));m=Rows([False]*T);sx=shard(x);sm=front_mask(m,x);padded=(-T)%8;chunk=(T+padded)//8;expected=([False]*T+[True]*padded)[rank*chunk:(rank+1)*chunk]
   assert sm.rows==expected and sx.shape[0]==len(sm.rows) and sx.owner is not x.owner and sm.owner is not m.owner
   records.append(dict(kind='original_SP_mask_and_hidden_geometry',T=T,tp=8,rank=rank,local_rows=len(sm.rows),pad_rows=sum(sm.rows),functional_outputs_do_not_alias_inputs=True,qualification='Integer/boolean row and owner tags only; no native byte/allocator/graph proof.'))
 tp.rank=7;assert front_mask(None,Rows(range(8))).rows==[False]
 records.append(dict(kind='FIA_graph_padding_is_not_SP_modulo_mask',actual_tokens_example=6,model_shape=8,rank=7,global_row=7,mask_input=None,produced_mask=False,actual_domain_padding_would_be=True,consequence='SP helper has no actual-token count; padded graph rows are not automatically marked. No claim that example is an observed current frame.'))
 # Original attention guard still preprocesses global cache and projects output
 # with zero local query; a metadata=None profile call is a different path.
 ev=[];context=NS(attn_metadata={});g=dict(torch=NS(empty_like=lambda h:Rows([None]*h.shape[0],h.tail)),get_forward_context=lambda:context,get_tp_group=lambda:tp,restore_tp_heads=lambda x,group:ev.append('TP_exchange') or Rows(['exchanged']*(x.shape[0]*group.size),(8,512)))
 forward=function('/attention/dsa_v41.py','AscendDSAV41Impl','forward',g);project=function('/context_parallel/dsa_v41_cp.py','AscendDSAV41CPImpl','_project_output',g);g['AscendDSAV41Impl']=NS(_apply_rotary=lambda *a,**kw:ev.append('inverse_rotary'))
 for metadata_none,tokens in [(True,0),(False,0),(False,3)]:
  ev.clear();context.attn_metadata=None if metadata_none else {'target':'metadata'};metadata=NS(swa=NS(num_actual_tokens=tokens,cp_token_range=(0,3,3,24)),positions=Rows(range(24)),rope=lambda *a:ev.append('rope') or ('cos','sin'))
  impl=NS(_get_layer_metadata=lambda m:metadata,_prepare_inputs_and_caches=lambda *a:ev.append('global_cache_prepare') or ('q','qr',None),_select_sparse_indices=lambda *a:ev.append('select_indices') or None,_forward_attention=lambda *a:ev.append('attention') or Rows(range(tokens),(64,512)))
  impl._project_output=MethodType(project,impl);attn=NS(enable_dsa_cp=True,n_heads=64,n_local_heads=8,head_dim=512,rotary_emb=NS(layername='L'),dsa_attn=NS(dsa_attn=NS(impl=NS(_forward_o_proj=lambda *a:ev.append('O_projection')))))
  output=Rows([None]*24,(5120,));out=forward(impl,attn,'positions',Rows(range(24),(5120,)),output=output);assert out is output
  assert ('global_cache_prepare' in ev)==(not metadata_none);assert ('TP_exchange' in ev)==(not metadata_none);assert ('attention' in ev)==(tokens>0 and not metadata_none)
  records.append(dict(kind='original_attention_metadata_and_empty_query_guard',metadata_none=metadata_none,local_tokens=tokens,events=list(ev),qualification='Opaque original forward/CP projection callbacks and shape tags, not numerical operators or native collective completion.'))
 # Both rotation and the capability predicate are real gate conditions.
 ev=[];g=dict(get_current_hardware_profile=lambda:NS(supports=lambda cap:capability),HardwareCapability=NS(ENGRAM_UNROTATED_GATE='unrot'),fused_engram_gate=lambda *a:ev.append('fused_unrotated') or 'fused',engram_gate=lambda *a:ev.append('generic_gate') or 'generic')
 gate=function('/engram/layer.py','AscendEngram','forward',g)
 class KV:
  def split(self,*a):return NS(view=lambda *a:'key'),'value'
 for rotated in [False,True]:
  for capability in [False,True]:
   ev.clear();b=NS(wkv=lambda rows:KV(),hc_mult=4,dim=5120,q_weight=Rows([1]),k_weight=Rows([1]),eps=1e-20);out=gate(b,Rows(range(3),(4,5120)),'lookup','mask',object() if rotated else None);assert (out=='fused')==(not rotated and capability);records.append(dict(kind='original_Engram_gate_dispatch',rotation_non_none=rotated,unrotated_capability=capability,events=list(ev),qualification='Opaque gate sinks only. Rotated source path never calls the unrotated kernel even if capability were true.'))
 out=dict(status='PASS_BOUNDED_ORIGINAL_MODEL_GUARD_SELECTOR_GEOMETRY',source_pins=27,controls=len(records),records=records,qualification='Original Python bodies with explicit config/hardware/classes/row tags/callbacks only. Not loaded class/env cache, Torch/native numerical values, exact event completion or primitive admission.',every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],reader_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
 (ROOT/'model-gate-source-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(ROOT/'model-gate-control-source-excerpts.json').write_text(json.dumps(dict(qualification=out['qualification'],nodes=excerpts),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(controls=len(records),source_pins=27,new_admissions=0)))
if __name__=='__main__':main()
