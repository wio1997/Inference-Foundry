"""Actual vLLM custom-op AST, CPU-only inference/TLS discriminator.

No model body, requests, NPU operations or performance claims.
"""
import ast,contextlib,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import torch
from torch.library import Library,infer_schema

class MoERunnerInterface:pass
class LayerName:
    def __init__(self,value):self.value=value

def device_state():
    mod=sys.modules.get('torch_npu')
    state=dict(NPU_initialized=False if mod is None else bool(mod.npu.is_initialized()),CUDA_initialized=bool(torch.cuda.is_initialized()))
    assert not any(state.values()),state
    return state

def load(path,names,ns):
    raw=path.read_bytes();nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return hashlib.sha256(raw).hexdigest()

def snapshot(where,x):
    try:version=x._version
    except RuntimeError:version='inference_tensor_no_version_counter'
    return dict(where=where,inference_mode=torch.is_inference_mode_enabled(),grad_enabled=torch.is_grad_enabled(),included=str(torch._C._dispatch_tls_local_include_set()),excluded=str(torch._C._dispatch_tls_local_exclude_set()),tensor_keys=torch._C._dispatch_key_set(x),tensor_is_inference=torch.is_inference(x),requires_grad=x.requires_grad,is_view=x._is_view(),version=version)

def main(runner,utility,dest):
    initial=device_state();records=[];rows=[];ctx=SimpleNamespace(moe_layer_index=0,all_moe_layers=['layer'],no_compile_layers={})
    ns=dict(torch=torch,Callable=Callable,Library=Library,infer_schema=infer_schema,_layer_name_type=str,LayerName=LayerName,_USE_LAYERNAME=False,MoERunnerInterface=MoERunnerInterface,ForwardContext=object,get_forward_context=lambda:ctx)
    hashes=dict(runner=load(runner,{'get_layer_from_name','_resolve_layer_name','_moe_forward_shared','_moe_forward_shared_fake'},ns),registration=load(utility,{'direct_register_custom_op'},ns))
    lib=Library('glm53_tls_cpu','FRAGMENT')
    def leaf(x:torch.Tensor)->torch.Tensor:
        records.append(snapshot('leaf_before_slice',x));y=x[:,1:3].transpose(0,1);records.append(snapshot('leaf_output',y));return y
    ns['direct_register_custom_op']('leaf',leaf,target_lib=lib,dispatch_key='CPU')
    class Body(MoERunnerInterface):
        def _forward_impl(self,hidden,router,shared,ids):
            records.append(snapshot('actual_handler_body',hidden))
            y=torch.ops.glm53_tls_cpu.leaf(hidden) if self.nested else leaf(hidden)
            return y,y
    body=Body();ctx.no_compile_layers['layer']=body
    ns['direct_register_custom_op']('moe_forward_shared',ns['_moe_forward_shared'],fake_impl=ns['_moe_forward_shared_fake'],target_lib=lib,dispatch_key='CPU',tags=(torch.Tag.needs_fixed_stride_order,))
    normal=torch.arange(12,dtype=torch.float32).reshape(3,4);requires=normal.clone().requires_grad_(True)
    with torch.inference_mode():inference=normal.clone()
    for mode in ['normal','no_grad','inference']:
        for input_name,x in [('normal',normal),('requires_grad',requires),('inference',inference)]:
            for entry in ['direct','boxed']:
                for nested in [False,True]:
                    guard=contextlib.nullcontext() if mode=='normal' else torch.no_grad() if mode=='no_grad' else torch.inference_mode()
                    with guard:
                        records.clear();records.append(snapshot('caller_before',x));before=x.clone();body.nested=nested
                        fn=ns['_moe_forward_shared'] if entry=='direct' else torch.ops.glm53_tls_cpu.moe_forward_shared
                        a,b=fn(x,x,None,None,'layer',0);records.append(snapshot('caller_after',x))
                        assert torch.equal(a,x[:,1:3].transpose(0,1)) and torch.equal(a,b) and torch.equal(x,before)
                        caller=records[0];inner=[s for s in records if s['where'] in ['actual_handler_body','leaf_before_slice']]
                        assert all(s['inference_mode']==caller['inference_mode'] and s['grad_enabled']==caller['grad_enabled'] for s in inner)
                        rows.append(dict(mode=mode,input=input_name,entry=entry,nested_boxed=nested,states=list(records),output_aliases_input=a.untyped_storage().data_ptr()==x.untyped_storage().data_ptr(),output_is_inference=torch.is_inference(a),output_requires_grad=a.requires_grad))
    result=dict(CPU_only=True,model_requests=0,NPU_operations=0,device_state_before=initial,device_state_after=device_state(),torch_version=torch.__version__,actual_source_sha256=hashes,cases=len(rows),states=rows,verdict='INFERENCE_BOOLEAN_AND_GRAD_STATE_PRESERVED',limits=['Actual vLLM handler/resolver/registration AST; body replaced with CPU views. Actual NPU PrivateUse1 dispatch TLS remains a separate source question.','CompositeAutograd and ADInplaceOrView names are not exclusive costs or proof of autograd graph construction. No latency/gain estimate.'])
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['states','limits']}))
    for row in rows:
        if row['mode']=='inference' and row['input']=='inference':print(row['entry'],row['nested_boxed'],[(s['where'],s['included'],s['excluded']) for s in row['states']])

if __name__=='__main__':main(*map(Path,sys.argv[1:]))
