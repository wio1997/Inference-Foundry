# SPDX-License-Identifier: Apache-2.0
"""CPU source-role packets for one-request BF16 NoPE sparse MLA preparation.

No native import/call, device transfer, allocator lease, completion receipt or
runtime readiness. A CPU copy installs staged MAIN pages only to inspect the
source call's inputs. This is not native page installation. The indexer and
producer remain explicit callers; packet checks do not prove their arithmetic.
C8/TQ/SMLA/CP/MTP/graphs are outside this conservative source-derived domain.
"""
from dataclasses import asdict,dataclass,field
import hashlib
import json
import torch
from .private_pages import PrivateDSAPlan,PrivateDSAStage


def _int(value,name,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError('invalid '+name)


@dataclass(frozen=True)
class SFAHeadPartition:
    total_heads: int
    world_size: int
    rank: int
    head_ids: tuple[int,...]

    def __post_init__(self):
        for v,n in ((self.total_heads,'total heads'),(self.world_size,'world size')):_int(v,n,1)
        _int(self.rank,'rank')
        if self.total_heads%self.world_size or self.rank>=self.world_size:raise ValueError('even explicit head partition required')
        width=self.total_heads//self.world_size
        if not isinstance(self.head_ids,tuple) or any(type(h) is not int for h in self.head_ids) or self.head_ids!=tuple(range(self.rank*width,(self.rank+1)*width)):
            raise ValueError('complete contiguous rank-local heads required;sample heads are not a partition')

    @property
    def local_heads(self):return len(self.head_ids)


@dataclass(frozen=True)
class NoPESFAContract:
    total_heads: int
    original_qk_dim: int
    latent_dim: int
    pool_size: int
    topk_tokens: int
    reference_source_sha256: str
    profile: str = 'c_ascend_nope_bf16_source'
    sink_policy: str = 'absent'

    def __post_init__(self):
        if any(type(v) is not int for v in (self.total_heads,self.original_qk_dim,self.latent_dim,self.pool_size,self.topk_tokens)) or (self.total_heads,self.original_qk_dim,self.latent_dim,self.pool_size,self.topk_tokens)!=(64,256,512,4,2048):
            raise ValueError('this prepared GLM source contract requires64/256/512/pool4/topk2048')
        if self.profile!='c_ascend_nope_bf16_source' or self.sink_policy!='absent':raise ValueError('unsupported C8/TQ/SMLA/profile or sink policy')
        if not isinstance(self.reference_source_sha256,str) or len(self.reference_source_sha256)!=64 or any(c not in '0123456789abcdef' for c in self.reference_source_sha256):
            raise ValueError('explicit source evidence SHA256 required;not an ABI attestation')

    @classmethod
    def from_config(cls,config,*,reference_source_sha256):
        d=config.mla_dimensions;t=config.hf_config['text_config'];i=config.indexer_config
        if config.model_type!='glm5_next' or d['qk_rope_head_dim']!=0 or t.get('mla_use_nope') is not True:raise ValueError('explicit GLM NoPE configuration required')
        return cls(d['num_attention_heads'],d['qk_nope_head_dim'],d['kv_lora_rank'],i['index_kpool'],i['index_topk'],reference_source_sha256)

    @property
    def scale(self):return self.original_qk_dim**-.5
    @property
    def index_width(self):return self.topk_tokens+self.pool_size-1


@dataclass(frozen=True)
class SFAInputIdentity:
    request_id: str
    layer_id: int
    revision: int
    metadata_sha256: str
    _owner: object = field(repr=False)


@dataclass(frozen=True)
class NoPESFAInputPlan:
    request_id: str
    layer_id: int
    revision: int
    candidate: PrivateDSAPlan
    heads: SFAHeadPartition
    contract: NoPESFAContract
    _owner: object = field(default_factory=object,repr=False)

    def __post_init__(self):
        if not isinstance(self.request_id,str) or not self.request_id:raise ValueError('explicit request ID required')
        _int(self.layer_id,'layer ID');_int(self.revision,'revision')
        if self.layer_id not in range(3,44,4):raise ValueError('prepared GLM backbone DSA layer required')
        if type(self.candidate) is not PrivateDSAPlan or type(self.heads) is not SFAHeadPartition or type(self.contract) is not NoPESFAContract:raise ValueError('explicit validated candidate/heads/contract required')
        p=self.candidate
        if self.heads.total_heads!=self.contract.total_heads or p.pool_size!=self.contract.pool_size:raise ValueError('candidate/partition/contract geometry differs')
        if p.tokens>128 or p.end>2147483647 or len(p.kernel_read_table)>4096:raise ValueError('bounded non-speculative INT32 SFA metadata required')

    @property
    def identity(self):
        value={'request_id':self.request_id,'layer_id':self.layer_id,'revision':self.revision,'candidate':asdict(self.candidate),'heads':asdict(self.heads),'contract':asdict(self.contract)}
        digest=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return SFAInputIdentity(self.request_id,self.layer_id,self.revision,digest,self._owner)

    @property
    def skip_operator(self):return self.candidate.tokens==0
    @property
    def query_shape(self):return (self.candidate.tokens,self.heads.local_heads,self.contract.latent_dim)
    @property
    def cache_shape(self):return (self.candidate.num_main_pages*(self.candidate.block_size//128),128,1,self.contract.latent_dim)
    @property
    def indices_shape(self):return (self.candidate.tokens,1,self.contract.index_width)
    @property
    def query_ends(self):return (self.candidate.tokens,)  # no leading0 for this source branch
    @property
    def kv_lengths(self):return (self.candidate.end,)  # RAW tokens,not compressed pools


@dataclass(frozen=True)
class CPUSFAInputPacket:
    identity: SFAInputIdentity
    query: torch.Tensor
    cache: torch.Tensor
    sparse_indices: torch.Tensor
    block_table: torch.Tensor
    query_ends: torch.Tensor
    kv_lengths: torch.Tensor
    scale: float

    @property
    def native_ABI_validated(self):return False

    def source_kwargs_cpu(self):
        """CPU fixture roles only. No device conversion or dispatch is provided."""
        return dict(query=self.query,key=self.cache,value=self.cache,sparse_indices=self.sparse_indices,
            scale_value=self.scale,sparse_block_size=1,block_table=self.block_table,actual_seq_lengths_query=self.query_ends,
            actual_seq_lengths_kv=self.kv_lengths,query_rope=None,key_rope=None,layout_query='TND',layout_kv='PA_BSND',
            sparse_mode=3,attention_mode=2,return_softmax_lse=False)


@torch.inference_mode()
def prepare_sfa_packet_cpu(plan,base_main_raw,stage,query_latent,indices,*,expected_identity,index_namespace,
                            max_cache_bytes=16*1024**2,max_output_bytes=32*1024**2):
    """Clone base CPU cache,install staged main bytes into that CLONE,pack roles.

    Candidate identity is an owner-local stale-work guard,not a real page lease.
    Accept only the completed-history+causal-tail token policy already used by
    this prepared indexer path. Selection scores/order need separate validation.
    Empty work returns None before reading/allocating tensor inputs.
    """
    if type(plan) is not NoPESFAInputPlan:raise ValueError('explicit SFA input plan required')
    if type(expected_identity) is not SFAInputIdentity or expected_identity!=plan.identity:raise ValueError('stale/foreign candidate request/layer/revision/table/partition identity')
    if index_namespace!='logical_token':raise ValueError('explicit logical_token indices required,not pool or physical IDs')
    if plan.skip_operator:return None
    for value,name in ((max_cache_bytes,'cache budget'),(max_output_bytes,'output budget')):_int(value,name,1)
    if type(stage) is not PrivateDSAStage or stage.plan!=plan.candidate:raise ValueError('staged main pages belong to a different candidate')
    tensors=(base_main_raw,stage.main,query_latent,indices)
    if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in tensors):raise ValueError('source-role packet is CPU-only')
    p=plan.candidate;c=plan.contract;page_bytes=p.block_size*c.latent_dim*2;cache_bytes=p.num_main_pages*page_bytes
    if cache_bytes>max_cache_bytes:raise ValueError('cache byte budget exceeded')
    if base_main_raw.dtype!=torch.uint8 or base_main_raw.ndim!=1 or not base_main_raw.is_contiguous() or base_main_raw.storage_offset()!=0 or base_main_raw.numel()!=cache_bytes:
        raise ValueError('dense zero-offset BF16 main backing required')
    if stage.main.dtype!=torch.uint8 or stage.main.shape!=(len(p.private_ids),page_bytes) or not stage.main.is_contiguous() or stage.main.storage_offset()!=0:raise ValueError('compact staged main geometry differs')
    if query_latent.shape!=plan.query_shape or query_latent.dtype!=torch.bfloat16:raise ValueError('query must be absorbed BF16 [T,local_heads,latent512]')
    if indices.shape!=plan.indices_shape or indices.dtype!=torch.int32:raise ValueError('native source roles require INT32 [T,1,2051] token indices')
    cost=cache_bytes+query_latent.numel()*2+indices.numel()*4+(len(p.kernel_read_table)+2)*4
    if cost>max_output_bytes:raise ValueError('packet output byte budget exceeded')
    if not bool(torch.isfinite(query_latent).all()):raise ValueError('nonfinite live absorbed query')
    if bool((indices < -1).any()):raise ValueError('only -1 token padding supported')
    selected=[]
    for row,pos in enumerate(range(p.start,p.end)):
        ids=indices[row,0];valid=ids[ids>=0].long();n=valid.numel();tail_start=(pos+1)//c.pool_size*c.pool_size
        history=min(tail_start,c.topk_tokens);tail_count=pos+1-tail_start
        if n!=history+tail_count or bool((ids[:n]<0).any()) or bool((ids[n:]!=-1).any()):raise ValueError('packed completed-history and causal-tail counts differ')
        if valid.unique().numel()!=n or bool(((valid>=p.end)|(valid>pos)).any()) or bool((valid[:history]>=tail_start).any()):raise ValueError('distinct causal logical TOKEN IDs required')
        if valid[history:].tolist()!=list(range(tail_start,pos+1)):raise ValueError('raw causal tail must appear once at the packed suffix')
        selected.append(valid)
    # This is CPU fixture installation,not an allocator/device ownership receipt.
    raw=base_main_raw.clone()
    for row,page in enumerate(p.private_ids):raw.reshape(p.num_main_pages,page_bytes)[page].copy_(stage.main[row])
    cache=raw.view(torch.bfloat16).reshape(plan.cache_shape)
    logical=raw.view(torch.bfloat16).reshape(p.num_main_pages,p.block_size,c.latent_dim)
    for ids in selected:
        for logical_page in torch.div(ids,p.block_size,rounding_mode='floor').unique().tolist():
            offsets=ids[torch.div(ids,p.block_size,rounding_mode='floor')==logical_page]%p.block_size
            if not bool(torch.isfinite(logical[p.read_table[logical_page],offsets]).all()):raise ValueError('nonfinite selected candidate latent rows')
    return CPUSFAInputPacket(plan.identity,query_latent.contiguous().clone(),cache,indices.contiguous().clone(),
        torch.tensor([p.kernel_read_table],dtype=torch.int32),torch.tensor(plan.query_ends,dtype=torch.int32),torch.tensor(plan.kv_lengths,dtype=torch.int32),c.scale)
