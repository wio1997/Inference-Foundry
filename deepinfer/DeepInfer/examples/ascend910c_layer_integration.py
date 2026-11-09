"""Real layer-0 BF16 reference integration on two ranks, with a CPU golden.

One token per rank at position zero exercises embedding, shifted mHC, SWA/MLA,
grouped output projection, shared expert, routed experts and HCCL. Native FP8
activation casts and quantized KV storage are intentionally outside this probe;
it is not a full-model runner or an equivalence claim for the quantized model.
"""
import argparse
import ast
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from deepinfer import PLATFORM
from deepinfer.model_executor.layers.attention_backend import prefill_attn_varlen_sparse
from deepinfer.model_executor.models.v4_modules import mhc_ops as mhc
from deepinfer.npu_ops.checkpoint910c import ByteCheckpoint
from deepinfer.npu_ops.moe910c import dispatch_combine,route,swiglu
from ascend910c_moe_integration import cpu_matrix,expert


def assert_error(actual,expected,label,results):
    actual=actual.cpu().float();expected=expected.cpu().float()
    error=actual-expected
    relative=float(error.norm()/expected.norm().clamp_min(1e-10))
    maximum=float(error.abs().max())
    if not bool(actual.isfinite().all()) or relative>.02 or maximum>.03*float(expected.abs().max())+5e-4:
        raise AssertionError(f'{label}: relative_l2={relative}, max_abs={maximum}')
    results[label]={'relative_l2':relative,'max_abs_error':maximum}


def linear(x,w):return (x.float()@w.float().T).bfloat16()


def norm(x,w,eps):
    return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)*w.float()).bfloat16()


class LayerProbe:
    def __init__(self,reader,config,device,cpu=False):
        self.reader=reader;self.config=config;self.device=device;self.cpu=cpu
        if cpu:
            source=Path(os.environ['TILE_KERNELS_REFERENCE_SOURCE'])/'tile_kernels/torch/mhc.py'
            ref={};exec(compile(ast.parse(source.read_text()),str(source),'exec'),ref)
            self.op=lambda name:ref[name+'_ref']
        else:self.op=lambda name:getattr(mhc,name)

    def tensor(self,name):return self.reader.tensor('layers.0.'+name)[0].to(self.device)
    def matrix(self,name):
        name='layers.0.'+name+'.weight'
        return cpu_matrix(self.reader,name) if self.cpu else self.reader.matrix(name,self.device)

    def pre(self,trunk,previous,kind):
        c=self.config;m=c['hc_mult'];eps=c['hc_eps']
        value=self.op('mhc_pre_apply_mix')(trunk,previous)
        mixes=self.op('mhc_pre_norm_fn')(trunk,self.tensor(f'hc_{kind}_fn'),None,c['rms_norm_eps'])
        pre,post,comb=self.op('mhc_pre_split_mixes')(mixes,self.tensor(f'hc_{kind}_scale'),self.tensor(f'hc_{kind}_base'),m,2.,eps)
        comb=self.op('sinkhorn_normalize')(comb,repeat=c['hc_sinkhorn_iters'],eps=eps)
        return norm(value,self.tensor(('attn_norm' if kind=='attn' else 'ffn_norm')+'.weight'),c['rms_norm_eps']),pre,post,comb

    def attention(self,x):
        c=self.config;heads=c['num_attention_heads'];dim=c['head_dim'];groups=c['o_groups'];eps=c['rms_norm_eps']
        q=linear(norm(linear(x,self.matrix('attn.wq_a')),self.tensor('attn.q_norm.weight'),eps),self.matrix('attn.wq_b')).view(1,heads,dim)
        kv=norm(linear(x,self.matrix('attn.wkv')),self.tensor('attn.kv_norm.weight'),eps)
        sink=self.tensor('attn.attn_sink')
        # Layer 0 has no compressed KV. At position zero both forward and
        # conjugate RoPE are identity; there is exactly one visible SWA key.
        if self.cpu:
            logits=(q.float()*kv[:,None,:].float()).sum(-1)*dim**-.5
            context=(kv[:,None,:].float()*torch.sigmoid(logits-sink)[...,None]).bfloat16()
        else:
            context,_=prefill_attn_varlen_sparse(q,kv[:,None,:],torch.zeros((1,1),device=self.device,dtype=torch.int32),dim**-.5,attn_sink=sink)
        grouped=context.reshape(1,groups,-1)
        w=self.matrix('attn.wo_a').reshape(groups,c['o_lora_rank'],-1)
        projected=torch.einsum('bgr,gdr->bgd',grouped.float(),w.float()).bfloat16()
        return linear(projected.flatten(1),self.matrix('attn.wo_b'))

    def shared(self,x):
        a=linear(x,self.matrix('ffn.shared_experts.w1'));b=linear(x,self.matrix('ffn.shared_experts.w3'))
        limit=self.config['swiglu_limit']
        if self.cpu:
            a=a.float().clamp_max(limit);b=b.float().clamp(-limit,limit);hidden=(F.silu(a)*b).bfloat16()
        else:hidden=swiglu(a,b,torch.ones(x.shape[0],device=self.device),limit)
        return linear(hidden,self.matrix('ffn.shared_experts.w2'))

    def routed(self,x):
        c=self.config;logits=linear(x,self.matrix('ffn.gate'));bias=self.tensor('ffn.gate.bias')
        if self.cpu:
            scores=F.softplus(logits.float()).sqrt();indices=torch.topk(scores+bias,c['num_experts_per_tok'],sorted=True).indices
            weights=scores.gather(-1,indices);weights=weights/(weights.sum(-1,keepdim=True)+1e-20)*c['routed_scaling_factor']
            out=torch.zeros_like(x,dtype=torch.float32);cache={}
            for slot,i in enumerate(indices[0].tolist()):
                out+=expert(self.reader,i,x,weights[:,slot],0,self.device,c['swiglu_limit'],cache,cpu=True).float();cache.clear()
            out=out.bfloat16()
        else:
            indices,weights=route(logits,bias,c['num_experts_per_tok'],c['routed_scaling_factor']);cache={}
            callback=lambda i,t,w:expert(self.reader,i,t,w,0,self.device,c['swiglu_limit'],cache)
            out=dispatch_combine(x,indices,weights,c['n_routed_experts'],callback)
        return out,indices


def run(probe,embedding):
    m=probe.config['hc_mult'];x=embedding.to(probe.device).unsqueeze(0)
    trunk=probe.op('expand_to_mhc')(x,m)
    previous=torch.zeros((1,1,m,1),device=probe.device);previous[:,:,0]=1
    value,pre,post,comb=probe.pre(trunk,previous,'attn');attention=probe.attention(value.squeeze(0)).unsqueeze(0)
    trunk=probe.op('mhc_post')(attention,trunk,post,comb)
    value,next_pre,post,comb=probe.pre(trunk,pre,'ffn');value=value.squeeze(0)
    shared=probe.shared(value);routed,indices=probe.routed(value);ffn=(shared.float()+routed.float()).bfloat16()
    final=probe.op('mhc_post')(ffn.unsqueeze(0),trunk,post,comb)
    return {'attention':attention,'after_attention':trunk,'ffn_input':value,'shared_expert':shared,'routed_experts':routed,'output_trunk':final,'next_pre_mix':next_pre},indices


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('probe is bounded at two ranks')
    torch.npu.set_device(local);torch.set_num_threads(4)
    if not PLATFORM.is_910c_reference:raise RuntimeError('explicit 910C reference profile required')
    dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=240))
    reader=ByteCheckpoint(args.model_dir,max_read_bytes=64*1024**2)
    config=json.loads((args.model_dir/'config.json').read_text())['text_config']
    if not config.get('hc_shifted',True) or config['compress_ratios'][0]!=0 or config.get('use_qnorm',False):
        raise ValueError('probe requires shifted HC, uncompressed layer zero and no q_b norm')
    embedding=reader.rows('embed.weight',torch.tensor([1+rank]))
    actual,indices=run(LayerProbe(reader,config,torch.device(f'npu:{local}')),embedding)
    expected,cpu_indices=run(LayerProbe(reader,config,torch.device('cpu'),cpu=True),embedding)
    torch.testing.assert_close(indices.cpu(),cpu_indices,rtol=0,atol=0)
    errors={}
    for key,value in actual.items():assert_error(value,expected[key],key,errors)
    result={'status':'pass','rank':rank,'token_id':1+rank,'position':0,'layer':0,'selected_experts':indices.cpu().tolist(),'errors':errors,'boundary':'Real complete layer-0 arithmetic with BF16 activations/weights and two-rank routed MoE; no FP8 activation/quantized KV equivalence, later layers, full-model generation or performance claim.'}
    args.output_dir.mkdir(parents=True,exist_ok=True);(args.output_dir/f'layer-rank{rank}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
