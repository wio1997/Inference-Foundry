#!/usr/bin/env python3
"""Original dispatch/layout/scale/quant pipeline controls, never native-bit admission."""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS,MethodType
from contextlib import contextmanager
import ast,copy,json,struct
R=Path('audit/execution-search')
def main():
 sources=json.loads((R/'moe-output-dependencies.json').read_text())['sources'];excerpts=[];records=[]
 for s in sources:assert sha256(Path(s['local_path']).read_bytes()).hexdigest()==s['runtime_sha256']
 def select(suf,owner,name):
  s=next(s for s in sources if s['runtime_path'].endswith(suf));p=Path(s['local_path']);tree=ast.parse(p.read_text());par=tree if owner is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner);n=copy.deepcopy(next(n for n in par.body if getattr(n,'name','')==name));excerpts.append(dict(runtime_path=s['runtime_path'],sha256=s['runtime_sha256'],owner=owner,name=name,first=n.lineno,last=n.end_lineno,lines=p.read_text().splitlines()[n.lineno-1:n.end_lineno]));return n
 def fn(suf,owner,name,g):
  n=select(suf,owner,name);n.decorator_list=[];exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])),'<original-MoE-output>','exec'),g);return g[name]
 # Execute the installed factory wrapper, observing its original kwargs.
 cap={};cfg=NS(use_v2_model_runner=False);asc=NS(eplb_config=NS(dynamic_eplb=False,expert_map_path=None))
 def original_factory(*a,**kw):cap['factory']=kw;return NS(router=kw['router'])
 def make_router(**kw):cap['router']=kw;return NS()
 g=dict(get_ascend_config=lambda:asc,get_current_vllm_config=lambda:cfg,_original_FusedMoE=original_factory,create_ascend_fused_moe_router=make_router,_adapt_routed_experts_cls=lambda x:x,_adapt_eplb_router=lambda *a:None)
 factory=fn('/patch/platform/patch_fused_moe.py',None,'_ascend_FusedMoE',g)
 upstream=select('/fused_moe/layer.py',None,'FusedMoEFactory');call=next(n.value for n in upstream.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runner' for t in n.targets));expr=next(kw.value for kw in call.keywords if kw.arg=='routed_scaling_factor')
 for outside in [False,True]:
  factory(384,6,routed_scaling_factor=1.5,**({'apply_routed_scale_to_output':True} if outside else {}))
  runner_scale=eval(compile(ast.Expression(expr),'<original-runner-scalar-expression>','eval'),cap['factory'])
  assert runner_scale==(1.5 if outside else 1.0) and cap['router']['routed_scaling_factor']==(1.0 if outside else 1.5)
  records.append(dict(kind='original_factory_and_upstream_scalar_expression',explicit_output_scale=outside,router_scale=cap['router']['routed_scaling_factor'],runner_scale=runner_scale,current_frozen_case=not outside,qualification='Original wrapper with opaque constructors plus exact original runner keyword expression; no full factory or kernel arithmetic.'))
 # Original parallel selector protects shared expert layout from disable_tp.
 state=NS(sedp=True);g=dict(shared_expert_dp_enabled=lambda:state.sedp,get_tp_group=lambda:NS(rank_in_group=3,world_size=8),_get_column_parallel_op=lambda *a:None,_get_row_parallel_op=lambda *a:None)
 g['_is_shared_expert_layer']=fn('/ops/linear_op.py',None,'_is_shared_expert_layer',g);parallel=fn('/ops/linear_op.py',None,'get_parallel_op',g)
 for sedp,shared,disable in [(True,True,True),(False,True,True),(True,False,True),(False,False,False)]:
  state.sedp=sedp;prefix='layers.0.mlp.shared_experts.gate_up_proj' if shared else 'layers.0.mlp.gate_up_proj';out=parallel(disable,prefix,object(),'column');expected=1 if (shared and sedp or not shared and disable) else 8;assert out[2]==expected
  records.append(dict(kind='original_parallel_selector',SEDP=sedp,shared_prefix=shared,disable_tp=disable,tp_size=out[2],current_frozen_case=sedp and shared))
 Mode=NS(TENSOR_PARALLEL='TP',SEQUENCE_PARALLEL_SEDP='SP_SEDP',SEQUENCE_PARALLEL_ONLY='SP_ONLY',SHARED_EXPERT_DATA_PARALLEL_ONLY='SEDP_ONLY');PathKind=NS(A8_INT_FUSED='A8_INT',A8_MXFP_FUSED='A8_MXFP',LINEAR_WRAPPER='WRAPPER');Quant=NS(W8A8='W8A8',W4A8='W4A8',W4A8MXFP='W4A8MXFP')
 g=dict(SharedExpertParallelMode=Mode,SharedExpertMLPPath=PathKind,QuantType=Quant,has_lora=lambda x:x is not None)
 mode=fn('/ops/fused_moe/shared_experts.py','AscendSharedExperts','parallel_mode',g);path=fn('/ops/fused_moe/shared_experts.py','AscendSharedExperts','_select_mlp_path',g)
 for sp,sedp in [(True,True),(True,False),(False,True),(False,False)]:
  obj=NS(moe_config=NS(tp_group=NS(world_size=8),is_sequence_parallel=sp),weights_replicated=sedp)
  out=mode(obj);expected=Mode.SEQUENCE_PARALLEL_SEDP if sp and sedp else Mode.SEQUENCE_PARALLEL_ONLY if sp else Mode.SHARED_EXPERT_DATA_PARALLEL_ONLY if sedp else Mode.TENSOR_PARALLEL;assert out==expected
  records.append(dict(kind='original_shared_parallel_mode',SP=sp,SEDP=sedp,mode=out,current_frozen_case=sp and sedp))
 obj=NS(layer=NS(gate_up_proj=NS(weight_scale='scale'),down_proj=NS(weight_scale='scale')),lora_context=None,quant_type=Quant.W8A8);assert path(obj)==PathKind.A8_INT_FUSED;records.append(dict(kind='original_shared_quant_path',path=path(obj),qualification='W8A8 descriptors/typed old native sequence qualify frozen selection; opaque attributes are not loaded tensor values.'))
 # Run the complete original shared.forward and A8_INT stages with opaque
 # numerical sinks. Disabled stream switches must produce no waits/collectives.
 events=[]
 @contextmanager
 def switch(stream,enabled):events.append(('stream_switch',enabled));yield
 def mm(*a,**kw):events.append(('quant_matmul',kw['output_dtype'],kw.get('pertoken_scale')));return NS(dtype=kw['output_dtype'])
 def dq(x):events.append(('dynamic_quant',));return 'quantized','input_scale'
 def aq(**kw):events.append(('dequant_swiglu_quant',kw['quant_mode'],kw['swiglu_mode'],kw['clamp_limit']));return 'act_quantized','act_scale'
 g.update(npu_stream_switch=switch,shared_experts_calculation_stream=lambda:'shared_stream',torch_npu=NS(npu_dynamic_quant=dq,npu_quant_matmul=mm),torch=NS(int32='int32',float16='float16',ops=NS(_C_ascend=NS(npu_dequant_swiglu_quant=aq))),get_current_hardware_profile=lambda:NS(supports=lambda _:False),HardwareCapability=NS(FUSED_SWIGLU_TUNING_ARGS='tuning'),PreparedSharedExpertExecution=lambda **kw:NS(**kw),SubmittedSharedExpertGateUp=lambda **kw:NS(**kw))
 for name in ['_wait_for_milestone','_wait_for_routed_stage','_prepare_execution_input','_prepare_gate_up_input','enqueue_gate_up_after_router','_run_a8_int_mlp','_finish_shared_mlp','forward']:
  fn('/ops/fused_moe/shared_experts.py','AscendSharedExperts',name,g)
 layer=NS(gate_up_proj=NS(weight='w1',weight_scale='s1',weight_scale_fp32='s1fp32'),down_proj=NS(weight='w2',weight_scale='s2'))
 obj=NS(layer=layer,lora_context=None,quant_type=Quant.W8A8,multistream_overlap=False,weights_replicated=True,moe_config=NS(tp_group=NS(world_size=8),is_sequence_parallel=True),situ_activation=None,swiglu_limit=10.0)
 obj.parallel_mode=MethodType(mode,obj);obj._select_mlp_path=MethodType(path,obj)
 for name in ['_wait_for_milestone','_wait_for_routed_stage','_prepare_execution_input','_prepare_gate_up_input','enqueue_gate_up_after_router','_run_a8_int_mlp','_finish_shared_mlp']:
  setattr(obj,name,MethodType(g[name],obj))
 milestones=NS(shared_input_ready=None,router_output_ready=None,shared_activation_overlap_start=None,routed_combine_start=None,has_fine_grained_stage_events=False)
 out=g['forward'](obj,NS(hidden_states=NS(dtype='bfloat16'),is_gathered=False),milestones);assert out.dtype=='bfloat16';assert [e[0] for e in events if e[0]!='stream_switch']==['dynamic_quant','quant_matmul','dequant_swiglu_quant','quant_matmul'];assert not any(e[1] for e in events if e[0]=='stream_switch')
 records.append(dict(kind='original_complete_shared_forward_A8_SEDP_serial',events=events,numerical_outputs='opaque',qualification='Full original selected shared Python body/stages; no native math, waits, completion, bytes or admission.'))
 # Exact scalar IEEE finite counterexample to erasing the add BF16 boundary.
 def bf(x):
  bits=struct.unpack('<I',struct.pack('<f',x))[0];return (bits+0x7fff+((bits>>16)&1))>>16
 def val(b):return struct.unpack('<f',struct.pack('<I',b<<16))[0]
 a,b,post=1.,2**-8,3.;old=bf(val(bf(a+b))*post);collapsed=bf((a+b)*post);assert old==0x4040 and collapsed==0x4041
 records.append(dict(kind='finite_BF16_add_boundary_counterexample',shared=a,routed=b,post=post,original_hex=hex(old),omit_add_rounding_hex=hex(collapsed),stops='Universal algebraic collapse that erases BF16 ForeachAdd rounding, not an original-rounding fused primitive.',qualification='Finite IEEE formula only, no Ascend result or complete HC sum proof.'))
 result=dict(status='PASS_ORIGINAL_FACTORY_LAYOUT_AND_SHARED_PIPELINE_CONTROLS',source_pins=len(sources),controls=len(records),records=records,qualification='Original Python scalar/layout/opaque pipeline and finite IEEE counterexample only; no native bits/population or admission.',every_replacement_bitwise=True,new_primitive_admissions=0,new_framework_replacement_admissions=0,whole_chain_complete=False,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[],tool_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
 (R/'moe-output-controls.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(R/'moe-output-source-excerpts.json').write_text(json.dumps(dict(excerpts=excerpts),ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(controls=len(records),source_pins=len(sources),new_admissions=0)))
if __name__=='__main__':main()
