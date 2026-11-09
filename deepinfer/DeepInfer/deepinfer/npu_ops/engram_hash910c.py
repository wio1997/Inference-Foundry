# SPDX-License-Identifier: Apache-2.0
# Follows the Apache-2.0 vLLM Ascend Engram hash kernel's exact mapping,
# multipliers, XOR/modulo and dead-token semantics. CPU provides only a
# bounded raw lookback window here; this is not a complete ZOS history path.
"""910C integer Engram hashes with immutable checkpoint-derived constants."""
import torch
import triton
import triton.language as tl
from deepinfer.multimodal_tokens import ImageTokenOffset


@triton.jit
def _hash_windows(context,token_map,multipliers,primes,offsets,output,valid,
                  TOKENS:tl.constexpr,NGRAM:tl.constexpr,HEADS:tl.constexpr,
                  HEADS_PER_ORDER:tl.constexpr,BLOCK_H:tl.constexpr,
                  MAP_SIZE:tl.constexpr,PAD_ID:tl.constexpr,
                  IMAGE_ID:tl.constexpr,IMAGE_PAD_ID:tl.constexpr,
                  COMPRESSED:tl.constexpr,IMAGE_SPECIALS:tl.constexpr):
    token=tl.program_id(0)
    layer=tl.program_id(1)
    head=tl.arange(0,BLOCK_H)
    order=head//HEADS_PER_ORDER+2
    rolling=tl.full((BLOCK_H,),0,tl.int64)
    blocked=tl.full((),False,tl.int1)
    if COMPRESSED:
        dead=tl.load(valid+token)==0
    else:
        current=tl.load(context+token*NGRAM)
        dead=(current==IMAGE_ID)|(current==IMAGE_PAD_ID)|((current>=MAP_SIZE)&(current<MAP_SIZE+IMAGE_SPECIALS))
    for shift in tl.static_range(NGRAM):
        raw=tl.load(context+token*NGRAM+shift)
        if COMPRESSED:
            # Original gather_engram_tokens_zos already applied vocabulary
            # compression, image barriers and left padding.
            value=raw.to(tl.int64)
        else:
            known=(raw>=0)&(raw<MAP_SIZE)
            blocked|=~known|(raw==IMAGE_ID)|(raw==IMAGE_PAD_ID)
            mapped=tl.load(token_map+tl.minimum(tl.maximum(raw,0),MAP_SIZE-1),known,other=PAD_ID)
            value=tl.where(blocked,PAD_ID,mapped).to(tl.int64)
        multiplier=tl.load(multipliers+layer*NGRAM+shift)
        rolling^=tl.where(shift<order,value*multiplier,0)
    prime=tl.load(primes+layer*HEADS+head,head<HEADS,other=1).to(tl.int64)
    offset=tl.load(offsets+layer*HEADS+head,head<HEADS,other=0)
    remainder=rolling%prime
    remainder=tl.where(remainder<0,remainder+prime,remainder)
    ids=tl.where(dead,-1,remainder+offset)
    tl.store(output+(layer*TOKENS+token)*HEADS+head,ids.to(tl.int32),head<HEADS)


def raw_contexts(histories,chunks):
    """Raw newest-first windows; keep exactly the CPU history commit API.

    No hashing, compression or table lookup runs on the CPU in this helper.
    Padding -1 denotes missing older tokens and is normalized by the kernel.
    """
    if len(histories)!=len(chunks) or not histories:
        raise ValueError('one nonempty history/chunk pair per request required')
    width=histories[0].cfg['engram_max_ngram_size']
    rows=[]
    pending=[]
    for history,chunk in zip(histories,chunks):
        if not chunk or any(type(x) is not int or not 0<=x<len(history.token_map)+ImageTokenOffset.NUM_SPECIAL for x in chunk):
            raise ValueError('expected nonempty valid integer token chunks')
        tail=history.history[-(width-1):] if width>1 else []
        joined=[-1]*(width-1-len(tail))+tail+list(chunk)
        rows.extend(joined[i:i+width][::-1] for i in range(len(chunk)))
        pending.append((history,chunk))
    for history,chunk in pending:history.history.extend(chunk)
    return rows


class DeviceEngramHash:
    def __init__(self,history,device):
        self.layers=tuple(history.layers)
        self.width=history.cfg['engram_max_ngram_size']
        self.pad_id=history.pad_id
        image_ids={value for value in history.image_ids-{None} if value<len(history.token_map)}
        if len(image_ids)>2:raise ValueError('at most two Engram image sentinel IDs')
        self.image_ids=tuple(sorted(image_ids))+(-1001,-1002)
        self.token_map=history.token_lookup.to(device=device,dtype=torch.int32)
        self.multipliers=history.multipliers.to(device)
        self.primes=history.primes.flatten(1).contiguous().to(device)
        self.offsets=history.offsets.to(device)
        self.heads_per_order=history.primes.shape[-1]
        self.heads=self.primes.shape[-1]
        if self.heads!=(self.width-1)*self.heads_per_order:
            raise ValueError('invalid Engram order/head layout')

    def compute(self,context,output=None):
        if context.ndim!=2 or context.shape[1]!=self.width or context.dtype!=torch.int32 or not context.is_contiguous():
            raise ValueError('expected contiguous raw [tokens,max_ngram] int32 windows')
        shape=(len(self.layers),context.shape[0],self.heads)
        if output is None:output=torch.empty(shape,dtype=torch.int32,device=context.device)
        if output.shape!=shape or output.dtype!=torch.int32 or not output.is_contiguous():
            raise ValueError('expected contiguous [layers,tokens,heads] int32 output')
        if context.shape[0]:
            _hash_windows[(context.shape[0],len(self.layers))](context,self.token_map,
                self.multipliers,self.primes,self.offsets,output,self.token_map,TOKENS=context.shape[0],
                NGRAM=self.width,HEADS=self.heads,HEADS_PER_ORDER=self.heads_per_order,
                BLOCK_H=triton.next_power_of_2(self.heads),MAP_SIZE=self.token_map.numel(),
                PAD_ID=self.pad_id,IMAGE_ID=self.image_ids[0],IMAGE_PAD_ID=self.image_ids[1],
                COMPRESSED=False,IMAGE_SPECIALS=int(ImageTokenOffset.NUM_SPECIAL),num_warps=4)
        return {layer:output[i] for i,layer in enumerate(self.layers)}

    def compute_compressed(self, tokens, valid, output=None):
        """Consume the unchanged original ngram metadata directly on NPU."""
        if (tokens.ndim != 2 or tokens.shape[1] != self.width
                or tokens.dtype != torch.int32 or not tokens.is_contiguous()
                or valid.shape != tokens.shape[:1] or valid.dtype != torch.bool
                or tokens.device != self.token_map.device or valid.device != tokens.device):
            raise ValueError('expected original device ngrams and per-token validity')
        shape = (len(self.layers), tokens.shape[0], self.heads)
        if output is None:
            output = torch.empty(shape, dtype=torch.int32, device=tokens.device)
        if (output.shape != shape or output.dtype != torch.int32
                or output.device != tokens.device or not output.is_contiguous()):
            raise ValueError('expected owned [layers,tokens,heads] int32 output')
        if tokens.shape[0]:
            _hash_windows[(tokens.shape[0],len(self.layers))](tokens,self.token_map,
                self.multipliers,self.primes,self.offsets,output,valid,TOKENS=tokens.shape[0],
                NGRAM=self.width,HEADS=self.heads,HEADS_PER_ORDER=self.heads_per_order,
                BLOCK_H=triton.next_power_of_2(self.heads),MAP_SIZE=self.token_map.numel(),
                PAD_ID=self.pad_id,IMAGE_ID=self.image_ids[0],IMAGE_PAD_ID=self.image_ids[1],
                COMPRESSED=True,IMAGE_SPECIALS=int(ImageTokenOffset.NUM_SPECIAL),num_warps=4)
        return {layer:output[i] for i,layer in enumerate(self.layers)}
