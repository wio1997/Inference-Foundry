# SPDX-License-Identifier: Apache-2.0
"""Prepared GLM dense decoder entry: mHC -> KDA -> mHC -> dense SwiGLU.

Only dense KDA backbone layers are supported. Each sublayer's mHC post is
materialized immediately; this is algebraically the same seam as the installed
GLM decoder's deferred post/pre, with the same BF16 rounding boundary. Return
all residual streams for the next layer. No model registry, DSA/MoE, distributed
execution, multimodal or MTP support is advertised by this component.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .kda import Glm5NextKDA
from .mhc import GlmMHC, rms_norm


class GlmDenseMLP(nn.Module):
    def __init__(self,gate_proj,up_proj,down_proj,*,hidden_size,intermediate_size,limit=10.0):
        super().__init__()
        if limit is not None and (not math.isfinite(limit) or limit<=0):
            raise ValueError('SwiGLU limit must be finite and positive')
        self.gate_proj,self.up_proj,self.down_proj=gate_proj,up_proj,down_proj
        self.hidden_size,self.intermediate_size=hidden_size,intermediate_size
        self.limit=limit

    @classmethod
    def from_checkpoint(cls,reader,layer_id,device,*,down_partitions=1):
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
        c=Glm5NextConfig.from_checkpoint(reader.root);text=c.hf_config['text_config']
        if not 0<=layer_id<c.num_hidden_layers or c.layers[layer_id].mlp!='dense':
            raise ValueError('requested GLM layer is not dense')
        if text['hidden_act']!='silu':
            raise ValueError('GLM dense MLP requires SiLU')
        prefix=f'model.language_model.layers.{layer_id}.mlp'
        dims={'gate_proj':(text['intermediate_size'],c.hidden_size),
              'up_proj':(text['intermediate_size'],c.hidden_size),
              'down_proj':(c.hidden_size,text['intermediate_size'])}
        # Validate all projection contracts before any allocation. Quantized
        # matrices remain INT8. Never silently expand ModelSlim weights to BF16.
        for name,shape in dims.items():
            tensor_name=f'{prefix}.{name}.weight'
            if tuple(reader.descriptor(tensor_name)[2]['shape'])!=shape:
                raise ValueError('invalid GLM dense projection geometry')
            recipe=reader.quant.get(tensor_name)
            if recipe not in ('FLOAT','W8A8_DYNAMIC'):
                raise ValueError('unsupported GLM dense quantization recipe')
            if recipe=='W8A8_DYNAMIC' and torch.device(device).type!='npu':
                raise ValueError('INT8 dense execution requires an authorized NPU runtime; no CPU dequantization fallback')
        if type(down_partitions) is not int or down_partitions < 1 or text['intermediate_size'] % down_partitions:
            raise ValueError('down partitions must evenly divide intermediate width')
        projections=[reader.load_linear(f'{prefix}.{name}',device) for name in ('gate_proj','up_proj')]
        if down_partitions == 1:
            projections.append(reader.load_linear(f'{prefix}.down_proj',device))
        else:
            if torch.device(device).type != 'npu' or reader.quant.get(f'{prefix}.down_proj.weight') != 'W8A8_DYNAMIC':
                raise ValueError('partitioned down projection requires NPU and W8A8_DYNAMIC weights')
            from deepinfer.npu_ops.w8a8 import AscendW8A8InputPartitionedLinear
            weight,scale=reader.linear_payload(f'{prefix}.down_proj')
            projections.append(AscendW8A8InputPartitionedLinear(weight.to(device),scale.to(device),partitions=down_partitions))
        return cls(*projections,hidden_size=c.hidden_size,intermediate_size=text['intermediate_size'],
                   limit=text.get('swiglu_limit'))

    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]!=self.hidden_size:
            raise ValueError('expected dense input [tokens,hidden]')
        if x.shape[0]==0:
            return torch.empty_like(x)
        gate,up=self.gate_proj(x).float(),self.up_proj(x).float()
        if gate.shape!=(x.shape[0],self.intermediate_size) or up.shape!=gate.shape:
            raise ValueError('invalid dense gate/up projection outputs')
        if self.limit is not None:
            gate=gate.clamp_max(self.limit)
            up=up.clamp(-self.limit,self.limit)
        activated=(F.silu(gate)*up).to(x.dtype)
        return self.down_proj(activated)


class Glm5NextDenseDecoder(nn.Module):
    def __init__(self,*,layer_id,attn_hc,ffn_hc,input_norm,ffn_norm,attention,mlp,norm_eps=1e-5):
        super().__init__()
        if layer_id<0 or (attn_hc.streams,attn_hc.hidden_size)!=(ffn_hc.streams,ffn_hc.hidden_size):
            raise ValueError('incompatible decoder mHC components')
        if input_norm.shape!=(attn_hc.hidden_size,) or ffn_norm.shape!=input_norm.shape:
            raise ValueError('invalid decoder RMSNorm layout')
        self.layer_id,self.norm_eps=layer_id,norm_eps
        self.attn_hc,self.ffn_hc,self.attention,self.mlp=attn_hc,ffn_hc,attention,mlp
        self.register_buffer('input_norm',input_norm)
        self.register_buffer('ffn_norm',ffn_norm)

    @classmethod
    def from_checkpoint(cls,reader,layer_id,device,*,kda_backend='ascendc'):
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
        c=Glm5NextConfig.from_checkpoint(reader.root)
        if not 0<=layer_id<c.num_hidden_layers or (c.layers[layer_id].attention,c.layers[layer_id].mlp)!=('linear_attention','dense'):
            raise ValueError('this decoder entry supports only dense KDA backbone layers')
        # This also rejects a quantized CPU request before allocating KDA weights.
        mlp=GlmDenseMLP.from_checkpoint(reader,layer_id,device)
        prefix=f'model.language_model.layers.{layer_id}'
        norm=[]
        for suffix in ('input_layernorm.weight','post_attention_layernorm.weight'):
            value,dtype=reader.tensor(prefix+'.'+suffix)
            if dtype not in ('BF16','F32'):
                raise ValueError('decoder norms require floating storage')
            norm.append(value.to(device))
        return cls(layer_id=layer_id,attn_hc=GlmMHC.from_checkpoint(reader,layer_id,'attn',device),
                   ffn_hc=GlmMHC.from_checkpoint(reader,layer_id,'ffn',device),
                   input_norm=norm[0],ffn_norm=norm[1],
                   attention=Glm5NextKDA.from_checkpoint(reader,layer_id,device,backend=kda_backend),
                   mlp=mlp,norm_eps=c.hf_config['text_config']['rms_norm_eps']).eval()

    @torch.inference_mode()
    def forward(self,hidden_states,state=None,*,attention_forward=None,mlp_forward=None):
        if hidden_states.ndim==2:
            if self.layer_id!=0:
                raise ValueError('only the first backbone layer expands collapsed input')
            residual=self.attn_hc.expand(hidden_states)
        else:
            residual=hidden_states
        attn_input,mix=self.attn_hc.pre(residual)
        attention=self.attention if attention_forward is None else attention_forward
        output,next_state=attention(rms_norm(attn_input,self.input_norm,self.norm_eps),state)
        residual=self.attn_hc.post(output,residual,mix)
        ffn_input,mix=self.ffn_hc.pre(residual)
        mlp=self.mlp if mlp_forward is None else mlp_forward
        output=mlp(rms_norm(ffn_input,self.ffn_norm,self.norm_eps))
        return self.ffn_hc.post(output,residual,mix),next_state
