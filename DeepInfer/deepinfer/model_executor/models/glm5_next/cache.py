# SPDX-License-Identifier: Apache-2.0
"""GLM noRoPE MLA/keypool contracts and a bounded correctness cache.

This is an unpaged, single-request state reference, not a serving allocator or
attention kernel. Main MLA and compressed indexer share logical block IDs;
raw key/gate rings are request-private and never prefix-shared. The reference
retains latent rows to test continuation; serving must bind its paged buffers
separately. It neither projects hidden states nor executes sparse attention.
"""
from dataclasses import dataclass

import torch


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


@dataclass(frozen=True)
class DSACacheLayout:
    latent_dim: int
    index_dim: int
    pool_size: int
    topk_tokens: int
    logical_block_size: int
    lookahead: int = 0

    def __post_init__(self):
        for name in ('latent_dim','index_dim','pool_size','topk_tokens','logical_block_size'):
            _positive(getattr(self,name),name)
        if self.pool_size <= 1 or self.logical_block_size % self.pool_size or self.topk_tokens % self.pool_size:
            raise ValueError('block size/topk must be divisible by a pool size greater than one')
        if type(self.lookahead) is not int or self.lookahead < 0:
            raise ValueError('lookahead must be a nonnegative integer')

    @classmethod
    def from_config(cls,config,*,logical_block_size,lookahead=0):
        dims=config.mla_dimensions;idx=config.indexer_config;text=config.hf_config['text_config']
        if dims['qk_rope_head_dim'] != 0 or not text.get('mla_use_nope'):
            raise ValueError('this GLM cache contract requires explicit noRoPE MLA')
        if not idx.get('index_kpool_compress') or not idx.get('index_kpool_always_select_tail'):
            raise ValueError('compressed keypool with causal raw tail is required')
        return cls(dims['kv_lora_rank'],idx['index_head_dim'],idx['index_kpool'],
                   idx['index_topk'],logical_block_size,lookahead)

    @property
    def compressed_block_size(self):
        return self.logical_block_size // self.pool_size

    @property
    def tail_capacity(self):
        return self.pool_size + self.lookahead

    def storage_shapes(self,main_blocks,tail_blocks):
        # Distinct allocator namespaces: tail_blocks is not main_blocks.
        _positive(main_blocks,'main_blocks');_positive(tail_blocks,'tail_blocks')
        return {'main':(main_blocks,self.logical_block_size,self.latent_dim),
                'index':(main_blocks,self.compressed_block_size,self.index_dim),
                'tail':(tail_blocks,2,self.tail_capacity,self.index_dim)}


def compressed_slot_mapping(slots,positions,layout):
    if slots.ndim!=1 or slots.shape!=positions.shape or slots.device!=positions.device:
        raise ValueError('slots/positions must be matching same-device vectors')
    if slots.dtype not in (torch.int32,torch.int64) or positions.dtype not in (torch.int32,torch.int64):
        raise ValueError('slots/positions must be integers')
    if bool((positions<0).any()):
        raise ValueError('positions must be nonnegative')
    safe=slots.long().clamp_min(0)
    valid=(slots>=0)&((positions.long()+1)%layout.pool_size==0)
    target=(safe//layout.logical_block_size)*layout.compressed_block_size+(safe%layout.logical_block_size)//layout.pool_size
    return torch.where(valid,target,-1)


def expand_pool_indices(pool_ids,positions,layout):
    """Expand ranked complete pools, then pack causal raw tail without holes.

    Input rows contain a compact valid prefix of distinct pool IDs followed by
    -1. Every selected pool must be complete at that query. Short histories must
    select all completed pools, matching the PKI/CANN packed-prefix contract.
    """
    k=layout.topk_tokens//layout.pool_size
    if pool_ids.ndim!=2 or pool_ids.shape!=(positions.numel(),k) or positions.ndim!=1:
        raise ValueError('invalid selected pool/position shapes')
    if pool_ids.dtype not in (torch.int32,torch.int64) or positions.dtype not in (torch.int32,torch.int64) or pool_ids.device!=positions.device:
        raise ValueError('selected pools/positions must be same-device integers')
    if bool((positions<0).any()):raise ValueError('positions must be nonnegative')
    out=torch.full((len(positions),layout.topk_tokens+layout.pool_size-1),-1,dtype=torch.int32,device=pool_ids.device)
    offsets=torch.arange(layout.pool_size,device=pool_ids.device)
    for row,pos in enumerate(positions.tolist()):
        complete=(pos+1)//layout.pool_size;count=min(complete,k);ids=pool_ids[row,:count]
        if bool(((ids<0)|(ids>=complete)).any()) or ids.unique().numel()!=count or bool((pool_ids[row,count:]!=-1).any()):
            raise ValueError('selected pools must be distinct causal complete pools in a packed prefix')
        expanded=(ids[:,None]*layout.pool_size+offsets).reshape(-1)
        out[row,:expanded.numel()]=expanded.to(torch.int32)
        tail_start=complete*layout.pool_size;tail_count=pos+1-tail_start
        out[row,expanded.numel():expanded.numel()+tail_count]=torch.arange(tail_start,pos+1,device=out.device,dtype=torch.int32)
    return out


@dataclass(frozen=True)
class DSAState:
    latent: torch.Tensor  # [tokens_seen,kv_lora_rank], BF16 normalized MLA latent
    compressed: torch.Tensor  # [tokens_seen//pool,index_head_dim], BF16
    tail: torch.Tensor  # [2,pool+lookahead,index_head_dim], FP32 raw K/gates
    tail_positions: torch.Tensor  # [capacity], INT64; never share across requests
    tokens_seen: int

    def clone(self):
        return DSAState(self.latent.clone(),self.compressed.clone(),self.tail.clone(),self.tail_positions.clone(),self.tokens_seen)


def validate_dsa_state(state,layout):
    n=state.tokens_seen
    if type(n) is not int or n<0:raise ValueError('invalid DSA token count')
    if state.latent.shape!=(n,layout.latent_dim) or state.compressed.shape!=(n//layout.pool_size,layout.index_dim):
        raise ValueError('invalid DSA latent/compressed history')
    if state.latent.dtype!=torch.bfloat16 or state.compressed.dtype!=torch.bfloat16:
        raise ValueError('main/index cache must be BF16')
    if state.tail.shape!=(2,layout.tail_capacity,layout.index_dim) or state.tail.dtype!=torch.float32:
        raise ValueError('raw key/gate ring must be FP32')
    if state.tail_positions.shape!=(layout.tail_capacity,) or state.tail_positions.dtype!=torch.int64:
        raise ValueError('invalid raw ring position tags')
    if len({x.device for x in (state.latent,state.compressed,state.tail,state.tail_positions)})!=1:
        raise ValueError('DSA state tensors must share a device')
    # Prefix restores can omit old completed pools' raw history, but must have
    # every raw row in the current incomplete pool before continuation.
    for pos in range(n//layout.pool_size*layout.pool_size,n):
        if state.tail_positions[pos%layout.tail_capacity].item()!=pos:
            raise ValueError('missing request-private incomplete-pool history')


def state_from_aligned_prefix(latent,compressed,layout):
    n=latent.shape[0]
    if n%layout.pool_size:
        raise ValueError('prefix reuse must end at a complete pool; raw tails cannot be shared')
    state=DSAState(latent.clone(),compressed.clone(),torch.zeros(2,layout.tail_capacity,layout.index_dim,dtype=torch.float32,device=latent.device),
                   torch.full((layout.tail_capacity,),-1,dtype=torch.int64,device=latent.device),n)
    validate_dsa_state(state,layout)
    return state


@torch.inference_mode()
def append_dsa(latent,raw_keys,gate_scores,ape,state,layout,*,start_position):
    """FP32 channel-wise softmax(gate+APE), BF16 compressed-cache boundary.

    Produces a private branch; input state is unchanged. No rollback-by-truncate
    or MTP transaction protocol is implied by supporting an oversized tail ring.
    """
    validate_dsa_state(state,layout)
    if type(start_position) is not int or start_position!=state.tokens_seen:
        raise ValueError('DSA continuation position must equal committed token count')
    rows=latent.shape[0]
    if latent.shape!=(rows,layout.latent_dim) or latent.dtype!=torch.bfloat16:
        raise ValueError('latent input must be BF16 [tokens,kv_lora_rank]')
    if raw_keys.shape!=(rows,layout.index_dim) or gate_scores.shape!=raw_keys.shape or ape.shape!=(layout.pool_size,layout.index_dim):
        raise ValueError('invalid raw key/gate/APE geometry')
    if raw_keys.dtype not in (torch.float32,torch.bfloat16) or gate_scores.dtype!=torch.float32 or ape.dtype!=torch.float32:
        raise ValueError('raw keys must be floating; gate/APE must be FP32')
    if any(t.device!=state.latent.device for t in (latent,raw_keys,gate_scores,ape)):
        raise ValueError('DSA append must preserve device')
    if any(not bool(torch.isfinite(t).all()) for t in (latent,raw_keys,gate_scores,ape)):
        raise ValueError('DSA inputs must be finite')
    tail=state.tail.clone();tags=state.tail_positions.clone();pools=[]
    for row in range(rows):
        pos=start_position+row;slot=pos%layout.tail_capacity
        tail[0,slot]=raw_keys[row].float();tail[1,slot]=gate_scores[row];tags[slot]=pos
        if (pos+1)%layout.pool_size==0:
            positions=torch.arange(pos+1-layout.pool_size,pos+1,device=tail.device)
            slots=positions%layout.tail_capacity
            if not torch.equal(tags[slots],positions):raise ValueError('raw ring lost a completed pool')
            weights=torch.softmax(tail[1,slots]+ape,dim=0)
            pools.append((weights*tail[0,slots]).sum(0).to(torch.bfloat16))
    addition=torch.stack(pools) if pools else state.compressed[:0]
    return DSAState(torch.cat((state.latent,latent)),torch.cat((state.compressed,addition)),tail,tags,start_position+rows)


@dataclass(frozen=True)
class DSACheckpointSpec:
    layer_id: int
    qk_scale: float
    tensors: tuple  # (checkpoint suffix, shape, storage dtype, recipe)


def dsa_checkpoint_manifest(reader,config,layer_id,layout):
    """Metadata only: GLM's 256-wide QK is distinct from its 512-wide latent.

    Parameter storage may be BF16/F32 for FLOAT; runtime APE is promoted to F32.
    Quantized projections retain signed INT8 with explicit per-channel metadata.
    """
    if type(layer_id) is not int or not 0<=layer_id<len(config.layers) or config.layers[layer_id].attention!='deepseek_sparse_attention':
        raise ValueError('requested layer is not a GLM DSA backbone layer')
    expected=DSACacheLayout.from_config(config,logical_block_size=layout.logical_block_size,lookahead=layout.lookahead)
    if expected!=layout:raise ValueError('layout disagrees with checkpoint configuration')
    d=config.mla_dimensions;hidden=config.hidden_size;idx=config.indexer_config
    heads=d['num_attention_heads'];q=d['q_lora_rank'];kv=d['kv_lora_rank'];qk=d['qk_nope_head_dim'];v=d['v_head_dim']
    ih=idx['index_n_heads'];width=layout.index_dim
    shapes={
        'q_a_proj.weight':((q,hidden),'W8A8_DYNAMIC'),
        'q_b_proj.weight':((heads*qk,q),'W8A8_DYNAMIC'),
        'kv_a_proj_with_mqa.weight':((kv,hidden),'W8A8_DYNAMIC'),
        'kv_b_proj.weight':((heads*(qk+v),kv),'FLOAT'),
        'o_proj.weight':((hidden,heads*v),'W8A8_DYNAMIC'),
        'q_a_layernorm.weight':((q,),'FLOAT'),'kv_a_layernorm.weight':((kv,),'FLOAT'),
        'indexer.wk.weight':((width,hidden),'FLOAT'),
        'indexer.weights_proj.weight':((ih,hidden),'FLOAT'),
        'indexer.wq_b.weight':((ih*width,q),'FLOAT'),
        'indexer.k_norm.weight':((width,),'FLOAT'),'indexer.k_norm.bias':((width,),'FLOAT'),
        'indexer.index_kpool_compress_ape':((layout.pool_size,width),'FLOAT'),
        'indexer.index_kpool_compress_gate':((width,hidden),'FLOAT'),
    }
    prefix=f'model.language_model.layers.{layer_id}.self_attn.';result=[]
    for suffix,(shape,recipe) in shapes.items():
        name=prefix+suffix;desc=reader.descriptor(name)[2]
        actual_recipe=reader.quant.get(name)
        # Current checkpoints may retain these projections in FLOAT; preserve
        # the exact checkpoint recipe, and validate its matching storage below.
        allowed=('W8A8_DYNAMIC','FLOAT') if recipe=='W8A8_DYNAMIC' else ('FLOAT',)
        if tuple(desc['shape'])!=shape or actual_recipe not in allowed:
            raise ValueError(f'{suffix}: GLM DSA geometry/quantization mismatch')
        recipe=actual_recipe
        if desc['dtype'] not in (('I8',) if recipe=='W8A8_DYNAMIC' else ('BF16','F32')):
            raise ValueError(f'{suffix}: GLM DSA storage dtype mismatch')
        if recipe=='W8A8_DYNAMIC':
            stem=name.removesuffix('.weight')
            for kind in ('weight_scale','weight_offset'):
                meta=reader.descriptor(stem+'.'+kind)[2]
                if meta['dtype']!='F32' or tuple(meta['shape'])!=(shape[0],1):
                    raise ValueError('GLM DSA quantization metadata must be per output channel')
        result.append((suffix,shape,desc['dtype'],recipe))
    return DSACheckpointSpec(layer_id,qk**-0.5,tuple(result))
