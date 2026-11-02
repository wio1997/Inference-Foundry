# SPDX-License-Identifier: Apache-2.0
"""GLM sigmoid/noaux_tc routing and portable unsharded dispatch preparation.

Correction bias affects selection only. Routed weights use the original
sigmoid scores, normalize over the selected experts, then apply the routed
scaling factor exactly once. The shared expert is outside this top-k and is
added once without that routed scaling factor. This component does not load
expert banks, dequantize weights, run collectives or register a full model.

Pack/combine are correctness paths for a single expert namespace, not an EP
implementation or a performance claim. Native router/grouped-FFN/MC2 behavior
and tie ordering require later authorized NPU verification.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MoERoutes:
    expert_ids: torch.Tensor  # [tokens, top_k], INT32
    weights: torch.Tensor  # [tokens, top_k], FP32; routed scale already applied


@dataclass(frozen=True)
class PackedRoutes:
    token_indices: torch.Tensor  # [tokens*top_k], INT64, expert-major order
    expert_ids: torch.Tensor  # [tokens*top_k], INT64
    weights: torch.Tensor  # [tokens*top_k], FP32, in the same packed order
    group_ends: torch.Tensor  # [num_experts], INT64 inclusive prefix counts
    num_tokens: int
    num_experts: int


def sigmoid_noaux_routes(logits,correction_bias,*,top_k,routed_scaling_factor=2.5):
    if logits.ndim!=2 or logits.dtype!=torch.float32:
        raise ValueError('GLM router logits must be FP32 [tokens,experts]')
    experts=logits.shape[1]
    if type(top_k)!=int or not 1<=top_k<=experts:
        raise ValueError('top_k must be an integer in [1,num_experts]')
    if correction_bias.shape!=(experts,) or correction_bias.dtype!=torch.float32 or correction_bias.device!=logits.device:
        raise ValueError('GLM correction bias must be FP32 [experts] on the logits device')
    if not math.isfinite(routed_scaling_factor) or routed_scaling_factor<=0:
        raise ValueError('routed scale must be finite and positive')
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(correction_bias).all()):
        raise ValueError('nonfinite GLM router input')
    scores=logits.sigmoid()
    # GLM has one expert group. For equal corrected scores, selection order is
    # intentionally delegated to topk; no cross-backend tie-order promise.
    ids=torch.topk(scores+correction_bias.unsqueeze(0),top_k,dim=-1,sorted=False).indices
    weights=scores.gather(1,ids)
    mass=weights.sum(-1,keepdim=True)
    if bool((mass<=0).any()):
        # Do not invent an epsilon or silently produce NaN on FP32 underflow.
        raise ValueError('selected sigmoid mass is zero in FP32')
    weights=weights/mass*routed_scaling_factor
    return MoERoutes(ids.to(torch.int32),weights)


class GlmMoERouter(nn.Module):
    def __init__(self,weight,correction_bias,*,top_k=8,routed_scaling_factor=2.5):
        super().__init__()
        if weight.ndim!=2 or not weight.shape[0] or not weight.shape[1] or not weight.is_floating_point():
            raise ValueError('router weight must be floating [experts,hidden]')
        if correction_bias.shape!=(weight.shape[0],) or not correction_bias.is_floating_point() or correction_bias.device!=weight.device:
            raise ValueError('invalid router correction bias')
        if type(top_k)!=int or not 1<=top_k<=weight.shape[0] or not math.isfinite(routed_scaling_factor) or routed_scaling_factor<=0:
            raise ValueError('invalid GLM routing configuration')
        if not bool(torch.isfinite(weight).all()) or not bool(torch.isfinite(correction_bias).all()):
            raise ValueError('nonfinite GLM router parameters')
        self.top_k,self.routed_scaling_factor=top_k,routed_scaling_factor
        self.register_buffer('weight',weight.float())
        self.register_buffer('correction_bias',correction_bias.float())

    @classmethod
    def from_checkpoint(cls,reader,layer_id,device='cpu'):
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
        c=Glm5NextConfig.from_checkpoint(reader.root);text=c.hf_config['text_config']
        if type(layer_id)!=int or not 0<=layer_id<c.num_hidden_layers or c.layers[layer_id].mlp!='sparse':
            raise ValueError('requested layer is not a GLM MoE backbone layer')
        expected={'scoring_func':'sigmoid','topk_method':'noaux_tc','norm_topk_prob':True,
                  'n_group':1,'topk_group':1,'moe_router_dtype':'float32'}
        if any(text.get(key)!=value for key,value in expected.items()):
            raise ValueError('unsupported GLM sigmoid/noaux_tc single-group routing contract')
        prefix=f'model.language_model.layers.{layer_id}.mlp.gate'
        values=[]
        for suffix in ('weight','e_score_correction_bias'):
            name=prefix+'.'+suffix
            if reader.quant.get(name)!='FLOAT':
                raise ValueError('GLM router requires explicit FLOAT parameters')
            value,dtype=reader.tensor(name)
            if dtype not in ('BF16','F32'):
                raise ValueError('GLM router requires BF16/F32 checkpoint storage')
            values.append(value.to(device))
        if values[0].shape!=(text['n_routed_experts'],c.hidden_size):
            raise ValueError('GLM router checkpoint dimensions disagree with config')
        return cls(*values,top_k=text['num_experts_per_tok'],routed_scaling_factor=text['routed_scaling_factor']).eval()

    def logits(self,hidden_states):
        if hidden_states.ndim!=2 or hidden_states.shape[1]!=self.weight.shape[1] or hidden_states.dtype not in (torch.bfloat16,torch.float32):
            raise ValueError('expected BF16/F32 router hidden states [tokens,hidden]')
        if hidden_states.device!=self.weight.device:
            raise ValueError('router inputs and parameters must share a device')
        return F.linear(hidden_states.float(),self.weight)

    @torch.inference_mode()
    def forward(self,hidden_states):
        return sigmoid_noaux_routes(self.logits(hidden_states),self.correction_bias,
                                   top_k=self.top_k,routed_scaling_factor=self.routed_scaling_factor)


def pack_routes(routes,*,num_experts):
    """Build expert-major rows and cumulative counts, including empty experts."""
    ids,weights=routes.expert_ids,routes.weights
    if type(num_experts)!=int or num_experts<1 or ids.ndim!=2 or ids.shape!=weights.shape or ids.shape[1]<1:
        raise ValueError('invalid route packing dimensions')
    if ids.dtype not in (torch.int32,torch.int64) or weights.dtype!=torch.float32 or weights.device!=ids.device:
        raise ValueError('route IDs must be integers with same-device FP32 weights')
    if bool(((ids<0)|(ids>=num_experts)).any()) or not bool(torch.isfinite(weights).all()) or bool((weights<0).any()):
        raise ValueError('invalid expert ID or route weight')
    sorted_ids=ids.sort(-1).values
    if bool((sorted_ids[:,1:]==sorted_ids[:,:-1]).any()):
        raise ValueError('a token cannot select one expert twice')
    tokens,k=ids.shape;flat=ids.reshape(-1).long()
    order=flat.argsort(stable=True)
    token_ids=torch.arange(tokens,device=ids.device,dtype=torch.int64).repeat_interleave(k)
    ends=torch.bincount(flat,minlength=num_experts).cumsum(0)
    return PackedRoutes(token_ids[order],flat[order],weights.reshape(-1)[order],ends,tokens,num_experts)


def dispatch_hidden(hidden_states,plan):
    if hidden_states.ndim!=2 or hidden_states.shape[0]!=plan.num_tokens or hidden_states.device!=plan.token_indices.device:
        raise ValueError('hidden states do not match packed routes')
    return hidden_states.index_select(0,plan.token_indices)


def combine_packed(expert_outputs,plan,*,shared_output=None):
    """Apply packed route weights once in FP32; add one unscaled shared output."""
    if expert_outputs.ndim!=2 or expert_outputs.shape[0]!=plan.token_indices.numel() or expert_outputs.device!=plan.weights.device:
        raise ValueError('expert outputs do not match packed routes')
    if expert_outputs.dtype not in (torch.bfloat16,torch.float32):
        raise ValueError('expert outputs must be BF16/F32')
    hidden=expert_outputs.shape[1]
    result=torch.zeros((plan.num_tokens,hidden),dtype=torch.float32,device=expert_outputs.device)
    weighted=expert_outputs.float()*plan.weights.unsqueeze(-1)
    if plan.num_tokens==1:
        # One token's expert-major contributions all target the same row.
        # Explicit sequential additions avoid an indexed multi-writer reduction
        # changing FP32 summation order before the final BF16 boundary.
        for contribution in weighted.unbind(0):result=result+contribution.unsqueeze(0)
    else:
        result.index_add_(0,plan.token_indices,weighted)
    if shared_output is not None:
        if shared_output.shape!=result.shape or shared_output.device!=result.device or shared_output.dtype not in (torch.bfloat16,torch.float32):
            raise ValueError('shared expert output must match token/hidden geometry and device')
        result=result+shared_output.float()
    return result.to(expert_outputs.dtype)
