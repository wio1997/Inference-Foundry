# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU rank DSA composition with explicit attention/INT8/sum stubs.

Only FLOAT KV-B contractions are computed here. No default attention, INT8
math, native imports/device work/collective, cache ownership or state commit.
The projected-output wrapper explicitly bypasses a second decoder O-proj.
"""
from dataclasses import dataclass,replace
import torch
from .absorbed_mla import AbsorbedMLAWeights
from .sfa_inputs import NoPESFAInputPlan,SFAInputIdentity,prepare_sfa_packet_cpu
from .sfa_weights import PreparedRankKVB
from .oproj_payload import PreparedOProjColumns
from .oproj_execution import execute_rank_oproj_cpu


def _identity(plan,expected):
    if type(plan) is not NoPESFAInputPlan or type(expected) is not SFAInputIdentity or expected!=plan.identity:
        raise ValueError('stale/foreign rank DSA identity')


def _positive(value,name):
    if type(value) is not int or value<1:raise ValueError('positive '+name+' required')


def _bf16(value,shape,name):
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.bfloat16 or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):
        raise ValueError('finite CPU BF16 '+name+' with shape '+str(shape)+' required')


@dataclass(frozen=True)
class CPURankAttentionResult:
    identity: SFAInputIdentity
    latent: torch.Tensor  # [T,local_heads,512], not head values or projected output


@dataclass(frozen=True)
class CPUProjectedDSAResult:
    identity: SFAInputIdentity
    hidden: torch.Tensor  # [T,4096], after explicit rank apply and sum callback


@dataclass(frozen=True)
class CPUProjectedDSAOperation:
    """Explicit decoder operation protocol; not an inferred output-shape mode.

    operation(step,private_state,plan=...,expected_identity=...) must return
    (CPUProjectedDSAResult,next_state). Decoder validates next_state separately.
    The outer request/stack owns transaction isolation and publication. Callable
    external effects are not rolled back by this wrapper or CPU tensor copies.
    """
    plan: NoPESFAInputPlan
    expected_identity: SFAInputIdentity
    operation: object
    query_payload: object = None
    query_backend: object = None

    def __post_init__(self):
        _identity(self.plan,self.expected_identity)
        if not callable(self.operation):raise ValueError('explicit projected DSA operation required')
        if (self.query_payload is None)!=(self.query_backend is None):raise ValueError('rank query payload/backend must be supplied together')
        if self.query_payload is not None:
            from .query_payload import PreparedRankQB
            if type(self.query_payload) is not PreparedRankQB or not callable(self.query_backend):raise ValueError('explicit prepared rank query/backend required')
            m=self.query_payload.manifest
            if (m.layer_id,m.heads,m.contract)!=(self.plan.layer_id,self.plan.heads,self.plan.contract):raise ValueError('rank query payload/plan differs')

    def project_input(self,mla,hidden):
        if self.query_payload is None:return mla.project(hidden)
        from .projections import RankMLAProjection
        p=self.plan
        value=mla.project_rank_cpu(hidden,payload=self.query_payload,backend=self.query_backend,plan=p,expected_identity=self.expected_identity)
        if type(value) is not RankMLAProjection or value.identity!=self.expected_identity:raise ValueError('rank MLA result identity/type differs')
        for tensor,shape,name in ((value.query_rank,(p.candidate.tokens,1536),'query rank'),(value.query,(p.candidate.tokens,p.heads.local_heads,256),'local query'),(value.latent,(p.candidate.tokens,512),'latent')):_bf16(tensor,shape,name)
        return value

    def validate_input(self,hidden,state,mla,layout):
        _identity(self.plan,self.expected_identity);p=self.plan
        if (mla.hidden_size,mla.num_heads,mla.qk_dim,mla.value_dim,mla.kv_rank,mla.q_rank)!=(4096,64,256,256,512,1536):raise ValueError('projected GLM decoder geometry differs')
        if (layout.latent_dim,layout.index_dim,layout.pool_size,layout.topk_tokens,layout.logical_block_size,layout.lookahead)!=(512,128,4,2048,p.candidate.block_size,0):raise ValueError('projected DSA cache layout differs')
        if state.tokens_seen!=p.candidate.start or len(hidden)!=p.candidate.tokens:raise ValueError('projected DSA candidate start/token count differs')
        _bf16(hidden,(p.candidate.tokens,4096),'decoder input')

    def __call__(self,step,state):
        _identity(self.plan,self.expected_identity);p=self.plan
        positions=step.positions
        if not isinstance(positions,torch.Tensor) or positions.device.type!='cpu' or positions.dtype!=torch.int64 or positions.shape!=(p.candidate.tokens,) or positions.tolist()!=list(range(p.candidate.start,p.candidate.end)) or step.scale!=p.contract.scale:
            raise ValueError('projected DSA step positions/scale differ')
        result,next_state=self.operation(step,state,plan=p,expected_identity=self.expected_identity)
        if type(result) is not CPUProjectedDSAResult or result.identity!=self.expected_identity:raise ValueError('projected DSA result identity/type differs')
        _bf16(result.hidden,(p.candidate.tokens,4096),'projected result')
        return result,next_state


@torch.inference_mode()
def execute_rank_dsa_cpu(query_qk,base_main_raw,stage,indices,*,plan,expected_identity,kv_b,o_proj,
                         expected_runtime_scale_sha256,attention,rank_apply,reduce_sum,
                         quantization_policy,bias_policy,index_namespace,
                         max_projection_work=8*1024**2,max_weight_bytes=4*1024**2,
                         max_cache_bytes=16*1024**2,max_packet_bytes=32*1024**2,
                         max_output_bytes=1024**2):
    """QK256 -> UK -> explicit SFA packet/callback -> UV256 -> O-proj/sum.

    One plan/identity binds every seam. The attention callback receives only a
    private CPUSFAInputPacket and must return CPURankAttentionResult. Its result
    is copied before UV. UK/UV are privately snapshotted before callbacks. No
    main/index/tail source backing is installed or mutated by this function.
    Empty work skips all tensor/payload access and callbacks after policy checks.
    Limits are logical reference bounds, not measured peak/throughput or callback
    budgets. Callback-side effects or real device completion are not covered.
    """
    _identity(plan,expected_identity)
    if not all(callable(v) for v in (attention,rank_apply,reduce_sum)):raise ValueError('explicit attention/rank backend/sum callbacks required')
    if quantization_policy!='rank_local_dynamic_int8' or bias_policy!='absent' or index_namespace!='logical_token':raise ValueError('explicit rank quant/bias/token namespace policies required')
    if not isinstance(expected_runtime_scale_sha256,str) or len(expected_runtime_scale_sha256)!=64 or any(v not in '0123456789abcdef' for v in expected_runtime_scale_sha256):raise ValueError('expected runtime scale digest required')
    for v,n in ((max_projection_work,'projection work'),(max_weight_bytes,'weight bytes'),(max_cache_bytes,'cache bytes'),(max_packet_bytes,'packet bytes'),(max_output_bytes,'output bytes')):_positive(v,n)
    t=plan.candidate.tokens;h=plan.heads.local_heads
    if h>4 or t*h*512*(256+256)>max_projection_work or t*4096*2>max_output_bytes:raise ValueError('rank DSA projection/output bound exceeded')
    if plan.skip_operator:return CPUProjectedDSAResult(plan.identity,torch.empty((0,4096),dtype=torch.bfloat16))
    if type(kv_b) is not PreparedRankKVB or type(o_proj) is not PreparedOProjColumns:raise ValueError('explicit prepared rank weights required')
    kv_b._require_plan(plan)
    if (o_proj.manifest.layer_id,o_proj.manifest.heads,o_proj.manifest.contract)!=(plan.layer_id,plan.heads,plan.contract):raise ValueError('O-proj and rank DSA plan differ')
    w=kv_b.weights;c=w.contract
    if (c.total_heads,c.head_ids,c.qk_dim,c.kv_rank,c.value_dim,c.pool_size,c.topk_tokens)!=(64,plan.heads.head_ids,256,512,256,4,2048):raise ValueError('KV-B projection contract differs')
    if h*512*(256+256)*2>max_weight_bytes:raise ValueError('rank DSA weight byte bound exceeded')
    _bf16(w.key_up,(h,256,512),'UK weight');_bf16(w.value_up,(h,512,256),'UV weight')
    owned=replace(kv_b,weights=AbsorbedMLAWeights(c,w.key_up.contiguous().clone(),w.value_up.contiguous().clone()))
    query=owned.absorb_query_cpu(query_qk,plan=plan)
    packet=prepare_sfa_packet_cpu(plan,base_main_raw,stage,query,indices,expected_identity=expected_identity,index_namespace=index_namespace,max_cache_bytes=max_cache_bytes,max_output_bytes=max_packet_bytes)
    attended=attention(packet)
    if type(attended) is not CPURankAttentionResult or attended.identity!=expected_identity:raise ValueError('attention latent identity/type differs')
    _bf16(attended.latent,(t,h,512),'attention latent')
    values=owned.value_up_cpu(attended.latent.contiguous().clone(),plan=plan)
    hidden=execute_rank_oproj_cpu(o_proj,values,plan=plan,expected_identity=expected_identity,expected_runtime_scale_sha256=expected_runtime_scale_sha256,quantization_policy=quantization_policy,bias_policy=bias_policy,rank_apply=rank_apply,reduce_sum=reduce_sum,max_output_bytes=max_output_bytes)
    return CPUProjectedDSAResult(plan.identity,hidden)
