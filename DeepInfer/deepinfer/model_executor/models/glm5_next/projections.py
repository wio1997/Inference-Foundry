# SPDX-License-Identifier: Apache-2.0
"""Explicit noRoPE GLM MLA/indexer projection seams, without attention execution.

MLA INT8 projections use the existing native loader only after an NPU-device
request; CPU dequantization is rejected before reader access. FLOAT indexer
projections may run on CPU for bounded correctness checks. The indexer matches
installed query/head-weight arithmetic and its retained-tail projection path;
this does not establish fused CANN key_pool numerical equivalence. CPU FLOAT
GEMMs explicitly accumulate in FP32 then round to BF16; target CPU torch BF16
GEMM failed the independent FP64 oracle. NPU retains its native BF16 GEMM path.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .cache import DSACacheLayout,append_dsa,dsa_checkpoint_manifest
from .mhc import rms_norm


@dataclass(frozen=True)
class MLAProjection:
    query_rank: torch.Tensor  # normalized q-lora, [T,q_rank], BF16
    query: torch.Tensor  # [T,heads,qk_nope], BF16
    latent: torch.Tensor  # normalized cache latent, [T,kv_rank], BF16


@dataclass(frozen=True)
class RankMLAProjection:
    identity: object
    query_rank: torch.Tensor  # full normalized [T,1536] for replicated indexer
    query: torch.Tensor  # rank-local [T,Hlocal,256], never global head indexed
    latent: torch.Tensor  # normalized [T,512]


@dataclass(frozen=True)
class IndexerProjection:
    query: torch.Tensor | None  # [T,index_heads,index_dim], BF16
    head_weights: torch.Tensor | None  # [T,index_heads], BF16, scaled once
    raw_keys: torch.Tensor  # [T,index_dim], FP32 LayerNorm output
    gate_scores: torch.Tensor  # [T,index_dim], BF16 GEMM boundary -> FP32


def _bf16_rows(x,width,name):
    if x.ndim!=2 or x.shape[1]!=width or x.dtype!=torch.bfloat16:
        raise ValueError(f'{name} requires BF16 [tokens,{width}]')


def _projection(module,x,width,name):
    y=module(x)
    _bf16_rows(y,width,name)
    if y.shape[0]!=x.shape[0] or y.device!=x.device:
        raise ValueError(f'{name} must preserve token count/device')
    return y


def _float_projection(x,weight):
    # CPU correctness reference: explicit FP32 accumulation and BF16 output.
    # Native BF16 GEMM accumulation/rounding remains hardware-unverified.
    if x.device.type=='cpu':
        return F.linear(x.float(),weight.float()).to(x.dtype)
    return F.linear(x,weight)


class _IndexerFloatLinear(nn.Module):
    def __init__(self,weight):
        super().__init__();self.register_buffer('weight',weight)

    def forward(self,x):
        return _float_projection(x,self.weight)


class GlmMLAProjections(nn.Module):
    def __init__(self,*,hidden_size,num_heads,qk_dim,value_dim,q_rank,kv_rank,
                 q_a,kv_a,q_b,kv_b,o_proj,q_norm,kv_norm,norm_eps=1e-5,rope_dim=0):
        super().__init__()
        if any(type(v) is not int or v<=0 for v in (hidden_size,num_heads,qk_dim,value_dim,q_rank,kv_rank)) or rope_dim!=0:
            raise ValueError('positive noRoPE MLA dimensions required')
        if not math.isfinite(norm_eps) or norm_eps<=0 or q_norm.shape!=(q_rank,) or kv_norm.shape!=(kv_rank,):
            raise ValueError('invalid MLA RMSNorm contract')
        if q_norm.dtype not in (torch.float32,torch.bfloat16) or kv_norm.dtype not in (torch.float32,torch.bfloat16) or q_norm.device!=kv_norm.device:
            raise ValueError('MLA norms require same-device floating weights')
        self.hidden_size,self.num_heads,self.qk_dim,self.value_dim=hidden_size,num_heads,qk_dim,value_dim
        self.q_rank,self.kv_rank,self.norm_eps=q_rank,kv_rank,norm_eps
        self.q_a,self.kv_a,self.q_b,self.kv_b,self.o_proj=q_a,kv_a,q_b,kv_b,o_proj
        # Match vLLM's replicated fused A GEMM for current FLOAT checkpoint
        # weights. Keep generic/quantized injected projections unchanged.
        from deepinfer.hugging_face.modelslim910c import FloatLinear
        self.fused_qkv_a = (FloatLinear(torch.cat((q_a.weight, kv_a.weight), dim=0))
                            if type(q_a) is FloatLinear and type(kv_a) is FloatLinear
                            and q_a.weight.dtype == kv_a.weight.dtype == torch.bfloat16
                            else None)
        self.register_buffer('q_norm',q_norm.clone());self.register_buffer('kv_norm',kv_norm.clone())

    @classmethod
    def from_checkpoint(cls,reader,layer_id,device,*,logical_block_size=16):
        # Actual GLM MLA contains W8A8_DYNAMIC projections. No CPU expansion or
        # speculative payload/native import is allowed before this rejection.
        if torch.device(device).type!='npu':
            raise ValueError('GLM MLA INT8 execution requires authorized NPU; no CPU dequantization fallback')
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
        config=Glm5NextConfig.from_checkpoint(reader.root)
        layout=DSACacheLayout.from_config(config,logical_block_size=logical_block_size)
        dsa_checkpoint_manifest(reader,config,layer_id,layout)
        d=config.mla_dimensions;root=f'model.language_model.layers.{layer_id}.self_attn.'
        modules=[reader.load_linear(root+name,device) for name in ('q_a_proj','kv_a_proj_with_mqa','q_b_proj','kv_b_proj','o_proj')]
        norms=[reader.tensor(root+name+'.weight')[0].to(device) for name in ('q_a_layernorm','kv_a_layernorm')]
        return cls(hidden_size=config.hidden_size,num_heads=d['num_attention_heads'],qk_dim=d['qk_nope_head_dim'],
                   value_dim=d['v_head_dim'],q_rank=d['q_lora_rank'],kv_rank=d['kv_lora_rank'],
                   q_a=modules[0],kv_a=modules[1],q_b=modules[2],kv_b=modules[3],o_proj=modules[4],
                   q_norm=norms[0],kv_norm=norms[1],norm_eps=config.hf_config['text_config']['rms_norm_eps'],rope_dim=d['qk_rope_head_dim']).eval()

    @torch.inference_mode()
    def project(self,hidden):
        _bf16_rows(hidden,self.hidden_size,'MLA hidden')
        if hidden.device!=self.q_norm.device:raise ValueError('MLA hidden/norm device mismatch')
        if not len(hidden):
            return MLAProjection(hidden.new_empty((0,self.q_rank)),hidden.new_empty((0,self.num_heads,self.qk_dim)),hidden.new_empty((0,self.kv_rank)))
        if self.fused_qkv_a is None:
            q_a = _projection(self.q_a,hidden,self.q_rank,'q_a')
            kv_a = _projection(self.kv_a,hidden,self.kv_rank,'kv_a')
        else:
            qkv = _projection(self.fused_qkv_a,hidden,self.q_rank+self.kv_rank,'fused_qkv_a')
            q_a,kv_a = qkv.split((self.q_rank,self.kv_rank),dim=-1)
        qr=rms_norm(q_a,self.q_norm,self.norm_eps)
        latent=rms_norm(kv_a,self.kv_norm,self.norm_eps)
        query=_projection(self.q_b,qr,self.num_heads*self.qk_dim,'q_b').reshape(-1,self.num_heads,self.qk_dim)
        return MLAProjection(qr,query,latent)

    @torch.inference_mode()
    def project_rank_cpu(self,hidden,*,payload,backend,plan,expected_identity):
        """Reuse A projections/norms once; replace full q_b with local rows.

        Explicit CPU backend contract only; no INT8 CPU implementation or native
        readiness claim. Full normalized qr is retained for replicated indexer.
        """
        from .query_payload import PreparedRankQB
        from .sfa_inputs import NoPESFAInputPlan
        if type(plan) is not NoPESFAInputPlan or expected_identity!=plan.identity:
            raise ValueError('stale/foreign rank MLA identity')
        if type(payload) is not PreparedRankQB or (payload.manifest.layer_id,payload.manifest.heads,payload.manifest.contract)!=(plan.layer_id,plan.heads,plan.contract):
            raise ValueError('rank MLA query payload differs')
        if (self.hidden_size,self.num_heads,self.qk_dim,self.value_dim,self.q_rank,self.kv_rank)!=(4096,64,256,256,1536,512):raise ValueError('rank MLA geometry differs')
        if not callable(backend):raise ValueError('explicit rank query backend required')
        _bf16_rows(hidden,4096,'rank MLA hidden')
        if hidden.device.type!='cpu' or self.q_norm.device.type!='cpu' or len(hidden)!=plan.candidate.tokens:raise ValueError('CPU rank MLA input differs')
        if not len(hidden):
            return RankMLAProjection(expected_identity,hidden.new_empty((0,1536)),hidden.new_empty((0,plan.heads.local_heads,256)),hidden.new_empty((0,512)))
        qr=rms_norm(_projection(self.q_a,hidden,1536,'q_a'),self.q_norm,self.norm_eps)
        latent=rms_norm(_projection(self.kv_a,hidden,512,'kv_a'),self.kv_norm,self.norm_eps)
        if not bool(torch.isfinite(qr).all()) or not bool(torch.isfinite(latent).all()):raise ValueError('nonfinite rank MLA projection')
        query=payload.project_cpu(qr,plan=plan,expected_identity=expected_identity,backend=backend,
            input_policy='rms_normalized_qr1536',quantization_policy='full_qr_dynamic_int8',bias_policy='absent')
        return RankMLAProjection(expected_identity,qr.contiguous().clone(),query.query,latent.contiguous().clone())

    @torch.inference_mode()
    def expand_kv(self,latent):
        """Explicit prefill/reference expansion; decode may bind absorbed MLA."""
        _bf16_rows(latent,self.kv_rank,'MLA latent')
        if not len(latent):
            return (latent.new_empty((0,self.num_heads,self.qk_dim)),latent.new_empty((0,self.num_heads,self.value_dim)))
        kv=_projection(self.kv_b,latent,self.num_heads*(self.qk_dim+self.value_dim),'kv_b')
        kv=kv.reshape(-1,self.num_heads,self.qk_dim+self.value_dim)
        return kv[...,:self.qk_dim],kv[...,self.qk_dim:]

    @torch.inference_mode()
    def finish(self,values):
        if values.ndim!=3 or values.shape[1:]!=(self.num_heads,self.value_dim) or values.dtype!=torch.bfloat16:
            raise ValueError('MLA attention values must be BF16 [tokens,heads,value_dim]')
        if not len(values):return values.new_empty((0,self.hidden_size))
        return _projection(self.o_proj,values.reshape(-1,self.num_heads*self.value_dim),self.hidden_size,'o_proj')


class GlmIndexerProjections(nn.Module):
    def __init__(self,*,wq_b,key_weight,head_weight,gate_weight,norm_weight,norm_bias,ape,hidden_size,q_rank,num_heads,layout,norm_eps=1e-6):
        super().__init__()
        if type(hidden_size) is not int or hidden_size<1 or type(q_rank) is not int or q_rank<1 or type(num_heads) is not int or num_heads<1:
            raise ValueError('indexer dimensions must be positive integers')
        width=layout.index_dim
        items={'key_weight':(key_weight,(width,hidden_size)),'head_weight':(head_weight,(num_heads,hidden_size)),
               'gate_weight':(gate_weight,(width,hidden_size)),'norm_weight':(norm_weight,(width,)),
               'norm_bias':(norm_bias,(width,)),'ape':(ape,(layout.pool_size,width))}
        if not math.isfinite(norm_eps) or norm_eps<=0:raise ValueError('invalid indexer LayerNorm epsilon')
        devices=set()
        for name,(tensor,shape) in items.items():
            if tensor.shape!=shape or tensor.dtype not in (torch.bfloat16,torch.float32):raise ValueError(f'invalid {name} shape/dtype')
            devices.add(tensor.device)
        if len(devices)!=1:raise ValueError('indexer weights must share a device')
        # Use BF16 model params_dtype for key/gate; promote head/norm weights
        # to FP32, matching the installed backend after checkpoint loading.
        self.register_buffer('key_weight',key_weight.bfloat16().clone())
        self.register_buffer('gate_weight',gate_weight.bfloat16().clone())
        for name in ('head_weight','norm_weight','norm_bias','ape'):
            self.register_buffer(name,items[name][0].float().clone())
        self.wq_b=wq_b;self.hidden_size,self.q_rank,self.num_heads=hidden_size,q_rank,num_heads
        self.layout,self.norm_eps=layout,norm_eps
        self.weight_scale=(width**-0.5)*(num_heads**-0.5)

    @classmethod
    def from_checkpoint(cls,reader,config,layer_id,device='cpu',*,logical_block_size=16):
        layout=DSACacheLayout.from_config(config,logical_block_size=logical_block_size)
        dsa_checkpoint_manifest(reader,config,layer_id,layout)
        root=f'model.language_model.layers.{layer_id}.self_attn.indexer.'
        def load(name):
            tensor,dtype=reader.tensor(root+name)
            if dtype not in ('BF16','F32'):raise ValueError('FLOAT indexer storage required')
            return tensor.to(device)
        # Query/key FLOAT projections execute using model BF16, matching the
        # model params_dtype even for a checkpoint with FLOAT F32 storage.
        return cls(wq_b=_IndexerFloatLinear(load('wq_b.weight').bfloat16()),key_weight=load('wk.weight').bfloat16(),
                   head_weight=load('weights_proj.weight').bfloat16(),gate_weight=load('index_kpool_compress_gate').bfloat16(),
                   norm_weight=load('k_norm.weight'),norm_bias=load('k_norm.bias'),ape=load('index_kpool_compress_ape'),
                   hidden_size=config.hidden_size,q_rank=config.mla_dimensions['q_lora_rank'],num_heads=config.indexer_config['index_n_heads'],layout=layout).eval()

    @torch.inference_mode()
    def forward(self,hidden,query_rank=None,*,compute_topk=True):
        _bf16_rows(hidden,self.hidden_size,'indexer hidden')
        if hidden.device!=self.key_weight.device:raise ValueError('indexer hidden/weight device mismatch')
        if compute_topk:
            if not isinstance(query_rank,torch.Tensor):raise ValueError('top-k requires unquantized query rank tensor')
            _bf16_rows(query_rank,self.q_rank,'indexer query rank')
            if query_rank.shape[0]!=hidden.shape[0] or query_rank.device!=hidden.device:raise ValueError('indexer token/device mismatch')
        if not len(hidden):
            q=hidden.new_empty((0,self.num_heads,self.layout.index_dim)) if compute_topk else None
            weights=hidden.new_empty((0,self.num_heads)) if compute_topk else None
            return IndexerProjection(q,weights,hidden.new_empty((0,self.layout.index_dim),dtype=torch.float32),hidden.new_empty((0,self.layout.index_dim),dtype=torch.float32))
        # This follows installed retained-tail arithmetic. Fused CANN key_pool
        # may have different accumulation/rounding and requires hardware tests.
        raw=_float_projection(hidden,self.key_weight).float()
        keys=F.layer_norm(raw,(self.layout.index_dim,),self.norm_weight,self.norm_bias,self.norm_eps)
        gates=_float_projection(hidden,self.gate_weight).float()
        query=weights=None
        if compute_topk:
            query=_projection(self.wq_b,query_rank,self.num_heads*self.layout.index_dim,'indexer wq_b').reshape(-1,self.num_heads,self.layout.index_dim)
            weights=F.linear(hidden.float(),self.head_weight).to(query.dtype)
            weights=weights*self.weight_scale
        return IndexerProjection(query,weights,keys,gates)

    def append_cache(self,latent,projection,state,*,start_position):
        return append_dsa(latent,projection.raw_keys,projection.gate_scores,self.ape,state,self.layout,start_position=start_position)
