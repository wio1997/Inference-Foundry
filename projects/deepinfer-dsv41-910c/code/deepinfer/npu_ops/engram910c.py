"""Eager 910C Engram hash/gate contracts, following TileKernels 2.1.0."""
import torch


def hashes(tokens,multipliers,vocab_sizes,offsets,image_mask=None):
    if tokens.ndim!=2 or multipliers.ndim!=2 or tokens.shape[1]!=multipliers.shape[1]:
        raise ValueError('expected tokens [T,N] and per-layer multipliers [L,N]')
    layers,n=multipliers.shape
    if vocab_sizes.ndim!=3 or vocab_sizes.shape[:2]!=(layers,n-1) or offsets.shape!=(layers,(n-1)*vocab_sizes.shape[2]):
        raise ValueError('invalid vocabulary sizes or offsets')
    if tokens.dtype!=torch.int32 or multipliers.dtype!=torch.int64 or vocab_sizes.dtype!=torch.int32 or offsets.dtype!=torch.int32:
        raise ValueError('Engram hashes require int32 tokens/sizes/offsets and int64 multipliers')
    if len({v.device for v in (tokens,multipliers,vocab_sizes,offsets)})!=1 or bool((vocab_sizes<=0).any()):
        raise ValueError('invalid Engram hash devices or nonpositive vocabulary')
    product=tokens.long()[None]*multipliers[:,None];running=product[:,:,0];parts=[]
    for i in range(1,n):
        running=running.bitwise_xor(product[:,:,i])
        parts.append(torch.remainder(running[...,None],vocab_sizes[:,i-1,None,:].long()).int())
    output=torch.cat(parts,-1)+offsets[:,None,:]
    if image_mask is not None:
        if image_mask.shape!=tokens.shape[:1] or image_mask.dtype!=torch.bool:raise ValueError('invalid image mask')
        output=output.masked_fill(image_mask[None,:,None],-1)
    return output


def gate(hidden,kv,weight_hidden,weight_embed,clamp_value=1e-6,eps=1e-20,image_mask=None):
    if hidden.ndim!=3 or kv.shape!=(hidden.shape[0],hidden.shape[1]+1,hidden.shape[2]):
        raise ValueError('expected hidden [T,M,D] and KV [T,M+1,D]')
    if hidden.dtype!=torch.bfloat16 or kv.dtype!=torch.bfloat16 or weight_hidden.shape!=hidden.shape[1:] or weight_embed.shape!=weight_hidden.shape:
        raise ValueError('invalid BF16 Engram input or norm weights')
    if len({v.device for v in (hidden,kv,weight_hidden,weight_embed)})!=1:raise ValueError('Engram gate device mismatch')
    x=hidden.float();k=kv[:,:hidden.shape[1]].float();v=kv[:,-1].float()
    dot=(x*weight_hidden.float()*k*weight_embed.float()).sum(-1)
    dot=dot*torch.rsqrt(x.square().mean(-1)+eps)*torch.rsqrt(k.square().mean(-1)+eps)*hidden.shape[-1]**-.5
    score=torch.sigmoid(dot.sign()*dot.abs().clamp_min(clamp_value).sqrt())
    output=(x+score[...,None]*v[:,None,:]).bfloat16()
    if image_mask is not None:
        if image_mask.shape!=hidden.shape[:1] or image_mask.dtype!=torch.bool:raise ValueError('invalid image mask')
        output=torch.where(image_mask[:,None,None],hidden,output)
    return output
