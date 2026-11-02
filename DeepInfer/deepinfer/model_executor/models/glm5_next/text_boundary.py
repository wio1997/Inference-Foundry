# SPDX-License-Identifier: Apache-2.0
"""Bounded GLM text embedding/final contraction and explicit vocabulary shards.

No full vocabulary matrix, registry, logits gather, sampler or model generation.
Embedding reads only unique requested rows; output projection reads a bounded
explicit contiguous vocabulary range. This reader-backed policy is preparation,
not optimized production residency or tensor-parallel support. Native numerical
behavior remains unverified. CPU FLOAT dots use FP32 accumulation explicitly.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .mhc import rms_norm


@dataclass(frozen=True)
class TextBoundaryManifest:
    hidden_size: int
    vocab_size: int
    streams: int
    norm_eps: float
    logit_scale: float
    embedding_name: str = 'model.language_model.embed_tokens.weight'
    norm_name: str = 'model.language_model.norm.weight'
    head_name: str = 'lm_head.weight'


def text_boundary_manifest(reader,config):
    text=config.hf_config['text_config']
    if config.dtype!='bfloat16' or not text.get('mhc') or text.get('tie_word_embeddings',False):
        raise ValueError('prepared text boundary requires BF16 mHC and explicit untied tables')
    streams=text['hc_mult'];eps=text['rms_norm_eps'];scale=text.get('logit_scale',1.0)
    if type(streams) is not int or streams<1 or not math.isfinite(eps) or eps<=0 or not math.isfinite(scale) or scale<=0:
        raise ValueError('invalid text contraction/norm/logit contract')
    result=TextBoundaryManifest(config.hidden_size,config.vocab_size,streams,eps,scale)
    for name,shape in ((result.embedding_name,(result.vocab_size,result.hidden_size)),(result.head_name,(result.vocab_size,result.hidden_size)),(result.norm_name,(result.hidden_size,))):
        desc=reader.descriptor(name)[2]
        if reader.quant.get(name)!='FLOAT' or desc['dtype']!='BF16' or tuple(desc['shape'])!=shape:
            raise ValueError('text boundary requires explicit FLOAT BF16 table/norm geometry')
    return result


class GlmLogitsShard(nn.Module):
    """Columns [vocab_start,vocab_end), with explicit head precision and scale.

    BF16 head precision preserves a BF16 dot-output boundary before scaling;
    FP32 precision projects directly to FP32. These match explicit logits-
    processor branches, not an assumption about an engine's default head_dtype.
    No TP gather, soft cap, global argmax/top-k or padding is implied.
    """
    def __init__(self,weight,*,vocab_start,scale=1.0,head_dtype=torch.bfloat16,max_tokens=128):
        super().__init__()
        if weight.ndim!=2 or weight.dtype!=torch.bfloat16 or type(vocab_start) is not int or vocab_start<0:
            raise ValueError('head shard requires BF16 rows and a nonnegative global start')
        if head_dtype not in (torch.bfloat16,torch.float32) or not math.isfinite(scale) or scale<=0:
            raise ValueError('explicit BF16/FP32 head precision and positive scale required')
        self.register_buffer('weight',weight.clone())
        self.vocab_start,self.vocab_end=vocab_start,vocab_start+len(weight)
        if type(max_tokens) is not int or max_tokens<1:raise ValueError('max_tokens must be a positive explicit compute bound')
        self.scale,self.head_dtype,self.max_tokens=scale,head_dtype,max_tokens

    @torch.inference_mode()
    def forward(self,hidden):
        if hidden.ndim!=2 or hidden.shape[-1]!=self.weight.shape[-1] or hidden.dtype!=torch.bfloat16 or hidden.device!=self.weight.device:
            raise ValueError('head input must be same-device BF16 normalized hidden rows')
        if hidden.shape[0]>self.max_tokens:raise ValueError('head token count exceeds explicit compute bound')
        if not hidden.shape[0] or not self.weight.shape[0]:return torch.empty((len(hidden),len(self.weight)),dtype=self.head_dtype,device=hidden.device)
        if self.head_dtype==torch.float32 or hidden.device.type=='cpu':
            logits=F.linear(hidden.float(),self.weight.float()).to(self.head_dtype)
        else:
            logits=F.linear(hidden,self.weight)
        if self.scale!=1.0:logits=logits*self.scale
        return logits


class GlmTextBoundary(nn.Module):
    def __init__(self,reader,manifest,norm_weight,*,max_unique_tokens=32,max_tokens=128):
        super().__init__()
        if type(max_unique_tokens) is not int or max_unique_tokens<1:raise ValueError('max_unique_tokens must be a positive explicit read bound')
        if norm_weight.shape!=(manifest.hidden_size,) or norm_weight.dtype!=torch.bfloat16:
            raise ValueError('final norm must be BF16 hidden vector')
        if type(max_tokens) is not int or max_tokens<1:raise ValueError('max_tokens must be a positive explicit allocation bound')
        self.reader,self.manifest,self.max_unique_tokens,self.max_tokens=reader,manifest,max_unique_tokens,max_tokens
        self.register_buffer('norm_weight',norm_weight.clone())

    @classmethod
    def from_checkpoint(cls,reader,device='cpu',*,max_unique_tokens=32,max_tokens=128):
        if type(max_unique_tokens) is not int or max_unique_tokens<1 or type(max_tokens) is not int or max_tokens<1:raise ValueError('positive explicit text boundary bounds required')
        from deepinfer.hugging_face.glm5_next_config import Glm5NextConfig
        config=Glm5NextConfig.from_checkpoint(reader.root);manifest=text_boundary_manifest(reader,config)
        weight,dtype=reader.tensor(manifest.norm_name)
        if dtype!='BF16':raise ValueError('final norm storage must be BF16')
        return cls(reader,manifest,weight.to(device),max_unique_tokens=max_unique_tokens,max_tokens=max_tokens)

    @torch.inference_mode()
    def embed(self,token_ids):
        if token_ids.ndim!=1 or token_ids.dtype not in (torch.int32,torch.int64) or token_ids.device.type!='cpu':
            raise ValueError('bounded reader embedding takes CPU integer token vector')
        if token_ids.numel()>self.max_tokens:raise ValueError('embedding token count exceeds explicit allocation bound')
        if bool(((token_ids<0)|(token_ids>=self.manifest.vocab_size)).any()):raise ValueError('token ID out of vocabulary')
        ids,inverse=torch.unique(token_ids.long(),sorted=True,return_inverse=True)
        if ids.numel()>self.max_unique_tokens:raise ValueError('embedding exceeds explicit unique-token read bound')
        if not ids.numel():return torch.empty((0,self.manifest.hidden_size),dtype=torch.bfloat16,device=self.norm_weight.device)
        selected,dtype=self.reader.raw_rows(self.manifest.embedding_name,ids)
        if dtype!='BF16' or selected.shape!=(len(ids),self.manifest.hidden_size):raise ValueError('invalid embedding row payload')
        return selected[inverse].to(self.norm_weight.device)

    @torch.inference_mode()
    def final_hidden(self,streams):
        if streams.ndim!=3 or streams.shape[1:]!=(self.manifest.streams,self.manifest.hidden_size) or streams.dtype!=torch.bfloat16 or streams.device!=self.norm_weight.device:
            raise ValueError('final boundary requires BF16 materialized residual streams')
        # Installed GLM contracts its last materialized hc_post by averaging,
        # then normalizes the BF16 collapsed hidden; no learned hc_head exists.
        if len(streams)>self.max_tokens:raise ValueError('final hidden token count exceeds explicit compute bound')
        collapsed=streams.float().mean(dim=1).bfloat16()
        return rms_norm(collapsed,self.norm_weight,self.manifest.norm_eps)

    def head_shard(self,start,stop,*,max_rows=32,head_dtype=torch.bfloat16):
        if type(start) is not int or type(stop) is not int or not 0<=start<=stop<=self.manifest.vocab_size:
            raise ValueError('invalid global vocabulary row range')
        if type(max_rows) is not int or max_rows<1 or stop-start>max_rows:
            raise ValueError('output shard exceeds explicit row allocation bound')
        if head_dtype not in (torch.bfloat16,torch.float32):raise ValueError('head precision must be explicit BF16/FP32')
        weight,dtype=self.reader.row_slice(self.manifest.head_name,start,stop)
        if dtype!='BF16' or weight.shape!=(stop-start,self.manifest.hidden_size):raise ValueError('invalid output shard payload')
        return GlmLogitsShard(weight.to(self.norm_weight.device),vocab_start=start,scale=self.manifest.logit_scale,head_dtype=head_dtype,max_tokens=self.max_tokens)
