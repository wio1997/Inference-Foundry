#!/usr/bin/env python3
"""Run original reduction/transform/cursor controls with opaque tensor and TP ops."""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace as NS
from enum import Enum,auto
import ast,copy,json
R=Path('audit/execution-search')
D=[R/'moe-transform-path-dependencies.json',R/'moe-transform-binding-dependencies.json']
P={s['runtime_path']:s for p in D for s in json.loads(p.read_text())['sources']}
def path(suffix):return Path(next(s['local_path'] for s in P.values() if s['runtime_path'].endswith(suffix)))
def cls(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name)
def methods(c,names):
    return [copy.deepcopy(next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)) for name in names]
def execute(body,env,filename):
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+body,type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),filename,'exec'),env)
class T:
    def __init__(self,name,parent=None,view=None):self.name=name;self.parent=parent;self.index=view;self.shape=(3,5120);self.dtype='bfloat16'
    def __getitem__(self,k):return T(self.name+'-view',self,k)
    def view(self,*shape):return self
def main():
    events=[];context=NS(no_compile_layers={},all_moe_layers=None,moe_layer_index=0);extra=NS(moe_comm_type=None)
    def reduce(x):events.append('TP-reduce:'+x.name);return T('reduce('+x.name+')',x)
    stream=NS(wait_event=lambda e:events.append('wait-input:'+e))
    env=dict(Enum=Enum,auto=auto,get_forward_context=lambda:context,_EXTRA_CTX=extra,tensor_model_parallel_all_reduce=reduce,getattr=getattr,torch=NS(npu=NS(current_stream=lambda:stream)),PreparedSharedExpertInput=lambda x:NS(hidden_states=x,ready_event=None,is_gathered=False))
    comm=cls(path('/ascend_forward_context.py'),'MoECommType');mode=cls(path('/ops/fused_moe/shared_experts.py'),'SharedExpertParallelMode')
    execute([copy.deepcopy(comm),copy.deepcopy(mode)],env,'original-enums');C=env['MoECommType'];M=env['SharedExpertParallelMode']
    shared=ast.ClassDef(name='Shared',bases=[],keywords=[],body=methods(cls(path('/ops/fused_moe/shared_experts.py'),'AscendSharedExperts'),['parallel_mode']),decorator_list=[])
    base=ast.ClassDef(name='Base',bases=[],keywords=[],body=methods(cls(path('/runner/moe_runner.py'),'MoERunner'),['apply_routed_input_transform','apply_routed_output_transform','_encode_layer_name']),decorator_list=[])
    actual=ast.ClassDef(name='Actual',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=methods(cls(path('/ops/fused_moe/fused_moe.py'),'AscendMoERunner'),['_can_overlap_sp_shared_with','_get_shared_expert_parallel_mode','_should_reduce_routed_before_combine','_maybe_reduce_shared_expert_output','_maybe_reduce_routed_output_before_transform','_maybe_reduce_final_output','apply_routed_input_transform','apply_routed_output_transform']),decorator_list=[])
    execute([shared,base,actual],env,'original-MoERunner-Shared-methods');env['MoERunnerInterface']=env['Base'];env['_USE_LAYERNAME']=False
    nodes=ast.parse(path('/ops/register_custom_ops.py').read_text()).body
    funcs=[copy.deepcopy(next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name)) for name in ['_routed_output_is_reduced','_maybe_all_reduce_tensor_model_parallel_impl','_maybe_all_reduce_shared_expert_impl']]
    execute(funcs,env,'original-runtime-custom-op-implementations')
    env['torch'].ops=NS(vllm=NS(maybe_all_reduce_tensor_model_parallel=env['_maybe_all_reduce_tensor_model_parallel_impl'],maybe_all_reduce_shared_expert=env['_maybe_all_reduce_shared_expert_impl']))
    cursor=copy.deepcopy(next(n for n in ast.parse(path('/runner/moe_runner.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_layer_from_name'));execute([cursor],env,'original-layer-cursor')
    def runner(sp,replicated,transform=None,has_shared=True):
        r=env['Actual']();s=env['Shared']();s.moe_config=NS(tp_group=NS(world_size=8),is_sequence_parallel=sp);s.weights_replicated=replicated;s.multistream_overlap=False
        r.ascend_shared_experts=s if has_shared else None;r._shared_experts=object() if has_shared else None;r.routed_input_transform=None;r.routed_output_transform=transform;r.moe_config=NS(is_sequence_parallel=sp,tp_size=1,ep_size=16);r.layer_name='layer';context.no_compile_layers['layer']=r;return r
    cases=[];x=T('routed');s=T('shared');combined=T('combined')
    r=runner(True,True);assert r._get_shared_expert_parallel_mode() is M.SEQUENCE_PARALLEL_SEDP
    for comm in C:
        extra.moe_comm_type=comm
        for flag in [False,True]:
            events.clear();a,b=r._maybe_reduce_routed_output_before_transform(x,flag);assert a is x and b is flag;assert r._maybe_reduce_shared_expert_output(s,flag) is s;assert r._maybe_reduce_final_output(combined,None,flag) is combined;assert not events
    cases.append(dict(name='frozen-SP-SEDP-none-transform-four-runtime-comms-both-trace-flags',passed=True,property='All3 reduction hooks preserve input aliases; no native TP work to remove.'))
    r=runner(False,True)
    for comm,expected in [(C.FUSED_MC2,[]),(C.ALLGATHER,['TP-reduce:routed'])]:
        events.clear();extra.moe_comm_type=comm;a,b=r._maybe_reduce_routed_output_before_transform(x,True);assert b is True;assert (a is x)==(not expected);assert r._maybe_reduce_shared_expert_output(s,True) is s;r._maybe_reduce_final_output(combined,None,True);assert events==expected
    cases.append(dict(name='shared-DP-only-runtime-comm-transition-despite-captured-true',passed=True,property='ALLGATHER/nonSP still reduces routed; freezing an earlier reduced trace flag loses required communication. Alternative to frozenSP, not a measured frozen failure.'))
    r=runner(False,False)
    for comm,expected in [(C.FUSED_MC2,['TP-reduce:shared']),(C.ALLGATHER,['TP-reduce:combined'])]:
        extra.moe_comm_type=comm;events.clear();r._maybe_reduce_routed_output_before_transform(x,False);r._maybe_reduce_shared_expert_output(s,False);r._maybe_reduce_final_output(combined,None,False);assert events==expected
    cases.append(dict(name='ordinary-TP-shared-versus-final-reduction-route',passed=True,property='Fake alias result cannot replace real custom-op runtime communication.'))
    r=runner(False,False,transform=object());extra.moe_comm_type=C.FUSED_MC2;events.clear();r._maybe_reduce_routed_output_before_transform(x,False);r._maybe_reduce_shared_expert_output(s,False);assert events==['TP-reduce:shared'];v=r._maybe_reduce_final_output(combined,3,False);assert v.parent is combined and v.index==(Ellipsis,slice(None,3,None))
    cases.append(dict(name='latent-alternative-early-reduce-and-final-truncation-view',passed=True,property='Truncation view stays visible after original early-reduce policy; no native transform math tested.'))
    r=runner(True,True);events.clear();a,b=r.apply_routed_input_transform(x);assert a is b is x and r.apply_routed_output_transform(x) is x and not events
    r._shared_experts=None;a,b=r.apply_routed_input_transform(x);assert a is x and b is None
    cases.append(dict(name='frozen-no-transform-super-delegation-and-no-shared-return',passed=True,property='No latent projection or async shared gather exists on selected None path.'))
    r=runner(True,False);transformed=T('transformed');shared_input=T('gathered');r.routed_input_transform=lambda t:(transformed,T('ignored-bias'));r.routed_output_transform=lambda t:(transformed,T('ignored-bias'));r.ascend_shared_experts.multistream_overlap=True
    r.ascend_shared_experts.prepare_input_async=lambda t:NS(hidden_states=shared_input,ready_event='ready',is_gathered=True);r.ascend_shared_experts.wait_for_output=lambda:events.append('wait-output');events.clear();a,b=r.apply_routed_input_transform(x);assert a is transformed and b is shared_input;assert r.apply_routed_output_transform(x) is transformed;assert events==['wait-input:ready','wait-output']
    cases.append(dict(name='SP-only-latent-alternative-tuple-unpack-and-two-ready-edges',passed=True,property='Original tuple first output and shared-input/output waits retained. Outside selected frozen None path.'))
    r1=runner(True,True);r2=runner(True,True);context.no_compile_layers={'a':r1,'b':r2};context.all_moe_layers=['a','b'];context.moe_layer_index=0
    f=env['get_layer_from_name'];assert f('from_forward_context') is r1;assert f('from_forward_context') is r2
    try:f('from_forward_context');raise AssertionError('missing exhaustion error')
    except AssertionError as e:assert 'number of MOE layers' in str(e)
    assert f('a') is r1 and context.moe_layer_index==2
    cases.append(dict(name='legacy-layer-cursor-consumes-each-original-custom-op-slot',passed=True,property='Inlining or deleting a wrapper must preserve positional lookup/cursor protocol where this representation is selected.'))
    outer=ast.ClassDef(name='Outer',bases=[],keywords=[],body=methods(cls(path('/models/deepseek_v41/model.py'),'DeepseekV41MoE'),['forward']),decorator_list=[])
    env['torch'].float16='float16'
    def muls_add(a,b,factor):events.append('legacy-scale-add:'+str(factor));return T('legacy-scale-add')
    env['muls_add_triton']=muls_add;execute([outer],env,'original-DeepseekV41MoE-outer-forward')
    class Expert:
        is_internal_router=True
        def __init__(self,result):self.result=result
        def __call__(self,**kwargs):return self.result
        def maybe_all_reduce_tensor_model_parallel(self,x):events.append('legacy-TP');return x
    outer=env['Outer']();outer.is_sequence_parallel=True;outer.tp_size=8;outer.shared_experts=object();outer.routed_scaling_factor=1.5
    outer.experts=Expert(combined);events.clear();assert outer.forward(x,hidden_states_fp32=T('fp32'),already_sequence_parallel=True) is combined and not events
    outer.experts=Expert((s,x));outer.forward(x,hidden_states_fp32=T('fp32'),already_sequence_parallel=True);assert events==['legacy-scale-add:1.5','legacy-TP']
    cases.append(dict(name='outer-Tensor-return-versus-legacy-tuple-counterexample',passed=True,property='Returning internal shared/routed tuple activates outer routedscale/add andTP reduce. Replacement must retain original final Tensor interface; opaque arithmetic only.'))
    paths=[path('/ops/fused_moe/fused_moe.py'),path('/ops/fused_moe/shared_experts.py'),path('/runner/moe_runner.py'),path('/ops/register_custom_ops.py'),path('/ascend_forward_context.py'),path('/models/deepseek_v41/model.py')]
    out=dict(status='PASS_EIGHT_ORIGINAL_PYTHON_PROTOCOL_GROUPS',case_groups=cases,case_group_count=len(cases),method='Unmodified original method bodies assembled into Base/Actual classes with original super and properties. Opaque tensor/TP-reduce/event objects; no Torch/NPU import, arithmetic or communication execution.',qualification='Selected SP_SEDP identities and conditional alternatives/capture/cursor/outer-return protocols only; not native bit/capture/performance acceptance, not observed alternate populations.',whole_chain_complete=False,every_replacement_bitwise=True,new_primitive_admissions=0,new_NPU_work=0,remote_writes=[],inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in paths],probe_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'moe-transform-path-controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=out['status'],groups=len(cases),new_native_admissions=0)))
if __name__=='__main__':main()
