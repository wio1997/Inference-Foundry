"""Full-rank CPU oracle for real prepare ASTs; explicit CPU allocations, no model/device work."""
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
import ast
import hashlib
import json
import statistics
import sys
import time
import torch
from torch import nn

@dataclass
class Output:
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    mc2_mask: torch.Tensor | None
    padded_hidden_states_shape: torch.Size
    pertoken_scale: torch.Tensor | None


def load(path, context, layout):
    raw = path.read_bytes()
    tree = ast.parse(raw.decode())
    names = {'PrepareAndFinalize', 'PrepareAndFinalizeWithAll2All', 'PrepareAndFinalizeWithMC2'}
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert len(nodes) == 3
    ns = dict(torch=torch,nn=nn,ABC=ABC,abstractmethod=abstractmethod,FusedMoEConfig=object,
        QuantType=SimpleNamespace(NONE=None),MoEPrepareOutput=Output,_EXTRA_CTX=context,
        get_dynamic_mx_quant_scale_alg=lambda:None,
        get_tensor_model_parallel_world_size=lambda:layout.tp,
        get_tensor_model_parallel_rank=lambda:layout.rank)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns['PrepareAndFinalizeWithMC2'], hashlib.sha256(raw).hexdigest()


def bits(x):
    if x.numel()==0:return b''
    return bytes(x.contiguous().view(torch.uint8).flatten().tolist())


def tensor(n, width, dtype, noncontiguous):
    data=torch.arange((n+2)*width*2,device='cpu',dtype=torch.float32).reshape(n+2,width*2)
    data=((data%101)-50)/7
    if dtype.is_floating_point and data.numel()>=6:
        data.view(-1)[width*2:width*2+6]=torch.tensor([0.,-0.,float('inf'),float('-inf'),float('nan'),1.25])
    data=data.to(dtype)
    return data[1:1+n,::2] if noncontiguous else data[:,:width].contiguous()[1:1+n]


def device_state():
    module=sys.modules.get('torch_npu')
    state=dict(torch_npu_auto_loaded=module is not None,
               NPU_initialized=False if module is None else bool(module.npu.is_initialized()),
               CUDA_initialized=bool(torch.cuda.is_initialized()))
    assert not state['NPU_initialized'] and not state['CUDA_initialized'], state
    return state


def main(original, candidate, dest):
    before_state=device_state()
    context=SimpleNamespace()
    layout=SimpleNamespace(tp=16,rank=0)
    old,oh=load(original,context,layout);new,nh=load(candidate,context,layout)
    cases=0
    retained=[]
    with torch.inference_mode():
        for tp in (1,2,4,16):
            layout.tp=tp
            specs={(n,((n+tp-1)//tp)*tp) for n in (0,1,2,3,7,8,15,16,17,31,32,33,257)}
            specs.update((n,n) for n in (1,3,7,17,33))
            specs.update((n,max(0,n-1)) for n in (1,3,17))
            for n,padded in sorted(specs):
                for dtype in (torch.float32,torch.float16,torch.bfloat16,torch.int8):
                    for noncontiguous in (False,True):
                        x=tensor(n,13,dtype,noncontiguous);router=tensor(n,11,torch.float32,noncontiguous)
                        context.padded_num_tokens=padded
                        context.mc2_mask=(torch.arange(padded,device='cpu')%3!=0)
                        before=(bits(x),bits(router),bits(context.mc2_mask))
                        padded_x=nn.functional.pad(x,(0,0,0,max(0,padded-n)))
                        padded_r=nn.functional.pad(router,(0,0,0,max(0,padded-n)))
                        # Independent oracle is the complete padded tensor and rank-coded expected row sequence.
                        reconstructed_x=[];reconstructed_r=[]
                        for rank in range(tp):
                            layout.rank=rank
                            assert x.shape[0] == router.shape[0]
                            a=old(object());b=new(object())
                            oa=a.prepare(x,router);ob=b.prepare(x,router)
                            q,rem=divmod(padded_x.shape[0],tp)
                            indices=list(range(rank*q+min(rank,rem),rank*q+min(rank,rem)+q+int(rank<rem)))
                            expect_x=padded_x[indices];expect_r=padded_r[indices]
                            assert oa.hidden_states.shape==ob.hidden_states.shape==expect_x.shape
                            assert oa.router_logits.shape==ob.router_logits.shape==expect_r.shape
                            assert bits(oa.hidden_states)==bits(ob.hidden_states)==bits(expect_x)
                            assert bits(oa.router_logits)==bits(ob.router_logits)==bits(expect_r)
                            mask_q,mask_rem=divmod(padded,tp)
                            mask_start=rank*mask_q+min(rank,mask_rem)
                            expect_mask=context.mc2_mask[mask_start:mask_start+mask_q+int(rank<mask_rem)] if tp>1 else context.mc2_mask
                            assert bits(oa.mc2_mask)==bits(ob.mc2_mask)==bits(expect_mask)
                            reconstructed_x.append(ob.hidden_states);reconstructed_r.append(ob.router_logits)
                            assert oa.hidden_states.stride()==ob.hidden_states.stride()
                            assert oa.router_logits.stride()==ob.router_logits.stride()
                            assert oa.mc2_mask.stride()==ob.mc2_mask.stride()
                            assert oa.mc2_mask.storage_offset()==ob.mc2_mask.storage_offset()
                            if padded>n and ob.hidden_states.numel():
                                assert ob.hidden_states.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
                                assert ob.router_logits.untyped_storage().data_ptr()!=router.untyped_storage().data_ptr()
                            assert oa.padded_hidden_states_shape==ob.padded_hidden_states_shape
                            assert oa.pertoken_scale is ob.pertoken_scale is None
                            assert a.num_tokens==b.num_tokens==n and a.replace_allreduce==b.replace_allreduce
                            assert before==(bits(x),bits(router),bits(context.mc2_mask))
                            if padded>n and padded%tp==0 and ob.hidden_states.numel():
                                assert ob.hidden_states.is_contiguous() and ob.router_logits.is_contiguous()
                            if cases%100==0:retained.append((ob.hidden_states,bits(ob.hidden_states)))
                            cases+=1
                        assert bits(torch.cat(reconstructed_x))==bits(padded_x)
                        assert bits(torch.cat(reconstructed_r))==bits(padded_r)
        # The SP branch is unchanged, including mask selection and local pad.
        for local in (0,1,2,3):
            layout.tp=16;context.padded_num_tokens=64
            context.mc2_mask=(torch.arange(64,device='cpu')%2==0)
            for rank in range(16):
                layout.rank=rank;x=tensor(local,13,torch.bfloat16,False);router=tensor(local,11,torch.float32,False)
                oa=old(object()).prepare(x,router,replace_allreduce=True)
                ob=new(object()).prepare(x,router,replace_allreduce=True)
                for k in ('hidden_states','router_logits','mc2_mask'):assert bits(getattr(oa,k))==bits(getattr(ob,k))
                assert oa.padded_hidden_states_shape==ob.padded_hidden_states_shape
                cases+=1
        for x,snapshot in retained:assert bits(x)==snapshot
        timings=[]
    after_state=device_state()
    result=dict(status='completed',CPU_only=True,model_requests=0,NPU_initialized=False,
        device_state_before=before_state,device_state_after=after_state,
        reconstructed_rank_order=True,original_sha256=oh,candidate_sha256=nh,torch_version=torch.__version__,cases=cases,
        bitwise_full_shards=True,input_preserved=True,retained_outputs=len(retained),SP_unchanged_cases=64,
        timings=timings,limits=['Independent CPU row-layout oracle covers noncontiguous, unequal, zero, NaN/Inf/signed-zero data.','Padding-path input isolation is retained through a local clone for full valid shards. Storage dimensions/offset change from global padded buffers to local buffers; native format/lifetime correctness still requires source/NPU validation.','CPU timing does not establish NPU/critical-path/PD/E2E gain.'])
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='timings'}))


if __name__=='__main__':main(*map(Path,sys.argv[1:]))
