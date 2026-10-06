"""CPU-only mechanical discriminator of one outer boxed->Python entry.

Real handler/lookup/registration AST, mocked model body, CPU dispatch key.
Never estimates PrivateUse1/NPU or model performance from this timing.
"""
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import ast,hashlib,json,statistics,sys,time
import torch
from torch.library import Library,infer_schema


class MoERunnerInterface:pass
class LayerName:
    def __init__(self,value):self.value=value


def state():
    npu=sys.modules.get('torch_npu')
    s=dict(torch_npu_auto_loaded=npu is not None,NPU_initialized=False if npu is None else bool(npu.npu.is_initialized()),CUDA_initialized=bool(torch.cuda.is_initialized()))
    assert not s['NPU_initialized'] and not s['CUDA_initialized'],s
    return s


def load_functions(path,names,ns):
    raw=path.read_bytes();tree=ast.parse(raw.decode())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return hashlib.sha256(raw).hexdigest()


def main(runner,utility,dest):
    before=state();ctx=SimpleNamespace(moe_layer_index=0,all_moe_layers=['layer']*2000,no_compile_layers={})
    ns=dict(torch=torch,Callable=Callable,Library=Library,infer_schema=infer_schema,
            _layer_name_type=str,LayerName=LayerName,_USE_LAYERNAME=False,
            MoERunnerInterface=MoERunnerInterface,ForwardContext=object,get_forward_context=lambda:ctx)
    runner_hash=load_functions(runner,{'get_layer_from_name','_resolve_layer_name','_moe_forward_shared','_moe_forward_shared_fake'},ns)
    utility_hash=load_functions(utility,{'direct_register_custom_op'},ns)
    class Body(MoERunnerInterface):
        def _forward_impl(self,*args):
            if self.audit:self.last_args=args
            return self.outputs
    body=Body();body.audit=True;body.outputs=(torch.zeros(2,6144,device='cpu'),torch.ones(2,6144,device='cpu'));ctx.no_compile_layers['layer']=body
    lib=Library('glm53_entry_cpu','FRAGMENT')
    ns['direct_register_custom_op']('moe_forward_shared',ns['_moe_forward_shared'],fake_impl=ns['_moe_forward_shared_fake'],target_lib=lib,dispatch_key='CPU',tags=(torch.Tag.needs_fixed_stride_order,))
    direct=ns['_moe_forward_shared'];boxed=torch.ops.glm53_entry_cpu.moe_forward_shared
    checks=0
    with torch.inference_mode():
        for n in (0,1,2,8):
            hidden=torch.empty(n,6144,device='cpu');router=torch.empty(n,256,device='cpu');ids=torch.arange(n,device='cpu')
            for shared in (None,hidden):
                for input_ids in (None,ids):
                    for name in ('layer','from_forward_context'):
                        args=(hidden,router,shared,input_ids,name,0)
                        ctx.moe_layer_index=0;a=boxed(*args);observed_a=body.last_args;index_a=ctx.moe_layer_index
                        ctx.moe_layer_index=0;b=direct(*args);observed_b=body.last_args;index_b=ctx.moe_layer_index
                        assert observed_a==observed_b and all(x is y for x,y in zip(observed_a,args[:4]))
                        assert all(x is y for x,y in zip(a,b)) and index_a==index_b
                        checks+=1
        body.audit=False;rows=[]
        for name in ('layer','from_forward_context'):
            args=(torch.empty(2,6144,device='cpu'),torch.empty(2,256,device='cpu'),None,None,name,0)
            funcs=[boxed,direct];batches=[[],[]]
            for f in funcs:
                ctx.moe_layer_index=0
                for _ in range(100):f(*args)
            for j in range(11):
                for i in ((0,1) if j%2==0 else (1,0)):
                    ctx.moe_layer_index=0;start=time.perf_counter_ns()
                    for _ in range(1000):funcs[i](*args)
                    batches[i].append((time.perf_counter_ns()-start)/1000/1000)
            rows.append(dict(layer_name_mode=name,boxed_median_us=statistics.median(batches[0]),direct_median_us=statistics.median(batches[1]),batches_us=batches))
    result=dict(status='completed',CPU_only=True,model_requests=0,NPU_operations=0,device_state_before=before,device_state_after=state(),torch_version=torch.__version__,runner_sha256=runner_hash,registration_utility_sha256=utility_hash,argument_output_lookup_checks=checks,mechanical_timings=rows,
        limits=['Actual handler, resolver, forward-context layer lookup and direct registration AST. Model body mocked with preallocated outputs; no model correctness established.',
        'CPU backend dispatch is used. PrivateUse1 key, graph/FakeTensor/TorchDispatch/subclass/LoRA contracts remain untested.',
        'This isolates only the outer dispatcher round-trip. Inner operator/MC2/MLA producers are retained by any such bypass and must not be credited to it.',
        'No CPU timing is an NPU, Decode critical-path or complete PD saving estimate.'])
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('limits','mechanical_timings')}))
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='batches_us'}))


if __name__=='__main__':main(*map(Path,sys.argv[1:]))
