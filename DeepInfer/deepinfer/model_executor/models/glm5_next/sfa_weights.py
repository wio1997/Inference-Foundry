# SPDX-License-Identifier: Apache-2.0
"""Bounded FLOAT rank-local KV-B row loader and explicit CPU projection seams.

No full matrix materialization, INT8 dequantization, native import/dispatch or
model/TP registration. Selected contiguous heads are checked against the SFA
input plan. CPU FP32 contractions and BF16 boundaries are correctness references,
not validation of native BF16 GEMM, ND/NZ layouts or production residency.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
import torch
from .absorbed_mla import AbsorbedMLAContract,split_kv_b_cpu
from .sfa_inputs import SFAHeadPartition,NoPESFAContract,NoPESFAInputPlan


def _positive(value,name):
    if type(value) is not int or value<1:raise ValueError('positive '+name+' required')


@dataclass(frozen=True)
class RankKVBSlice:
    layer_id: int
    heads: SFAHeadPartition
    contract: NoPESFAContract
    tensor_name: str
    source_dtype: str
    source_shape: tuple[int,int]
    first_row: int
    stop_row: int
    checkpoint_bytes: int
    runtime_bytes: int
    descriptor_sha256: str

    def to_dict(self):return asdict(self)


def rank_kv_b_manifest(reader,config,layer_id,heads,contract,*,max_local_heads=4,max_checkpoint_bytes=8*1024**2,max_runtime_bytes=4*1024**2):
    """Validate all metadata/bounds before the first row read. No tensors loaded."""
    for value,name in ((max_local_heads,'head bound'),(max_checkpoint_bytes,'checkpoint bound'),(max_runtime_bytes,'runtime bound')):_positive(value,name)
    if type(heads) is not SFAHeadPartition or type(contract) is not NoPESFAContract:raise ValueError('explicit head partition/source contract required')
    if max_local_heads>4 or heads.local_heads>max_local_heads:raise ValueError('bounded CPU KV-B supports at most4 rank-local heads;no full matrix read')
    if heads.total_heads!=contract.total_heads or NoPESFAContract.from_config(config,reference_source_sha256=contract.reference_source_sha256)!=contract:raise ValueError('head/config/source contract mismatch')
    if type(layer_id) is not int or not 0<=layer_id<len(config.layers) or config.layers[layer_id].layer_id!=layer_id or config.layers[layer_id].attention!='deepseek_sparse_attention':
        raise ValueError('explicit backbone DSA layer required')
    if config.mla_dimensions['v_head_dim']!=256:raise ValueError('prepared GLM KV-B requires value width256')
    name=f'model.language_model.layers.{layer_id}.self_attn.kv_b_proj.weight'
    if reader.quant.get(name)!='FLOAT':raise ValueError('KV-B must be explicitly FLOAT;no INT8 CPU fallback')
    path,header,desc=reader.descriptor(name);dtype=desc.get('dtype');shape=desc.get('shape');offsets=desc.get('data_offsets')
    full=(contract.total_heads*(contract.original_qk_dim+256),contract.latent_dim)
    if dtype not in ('BF16','F32') or not isinstance(shape,(list,tuple)) or any(type(v) is not int for v in shape) or tuple(shape)!=full:raise ValueError('KV-B FLOAT matrix descriptor geometry differs')
    element=2 if dtype=='BF16' else 4
    if type(header) is not int or header<0 or not isinstance(offsets,(list,tuple)) or len(offsets)!=2 or any(type(v) is not int for v in offsets) or not 0<=offsets[0]<=offsets[1] or offsets[1]-offsets[0]!=full[0]*full[1]*element:
        raise ValueError('KV-B descriptor byte range differs')
    first=heads.head_ids[0]*(contract.original_qk_dim+256);stop=(heads.head_ids[-1]+1)*(contract.original_qk_dim+256)
    payload=(stop-first)*contract.latent_dim*element;resident=(stop-first)*contract.latent_dim*2
    _positive(reader.max_read_bytes,'reader bound')
    if payload>min(max_checkpoint_bytes,reader.max_read_bytes) or resident>max_runtime_bytes:raise ValueError('rank KV-B checkpoint/runtime byte budget exceeded')
    source={'path':str(path),'header':header,'dtype':dtype,'shape':list(shape),'offsets':list(offsets),'recipe':'FLOAT'}
    digest=hashlib.sha256(json.dumps(source,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return RankKVBSlice(layer_id,heads,contract,name,dtype,full,first,stop,payload,resident,digest)


@dataclass(frozen=True)
class PreparedRankKVB:
    manifest: RankKVBSlice
    weights: object
    checkpoint_sha256: str
    runtime_sha256: str

    def _require_plan(self,plan):
        m=self.manifest
        if type(plan) is not NoPESFAInputPlan or plan.layer_id!=m.layer_id or plan.heads!=m.heads or plan.contract!=m.contract:raise ValueError('KV-B layer/head partition/source contract differs from SFA plan')

    @torch.inference_mode()
    def _project(self,value,plan,kind,max_tokens,max_output_bytes):
        self._require_plan(plan);_positive(max_tokens,'token bound');_positive(max_output_bytes,'projection output bound')
        c=self.weights.contract;t=plan.candidate.tokens;input_width=c.qk_dim if kind=='query' else c.kv_rank;output_width=c.kv_rank if kind=='query' else c.value_dim
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.bfloat16 or value.shape!=(t,c.heads,input_width):raise ValueError('explicit plan-matched CPU BF16 projection input required')
        if t>max_tokens or t*c.heads*output_width*2>max_output_bytes:raise ValueError('projection token/output byte budget exceeded')
        if not t:return torch.empty((0,c.heads,output_width),dtype=torch.bfloat16)
        if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite projection input')
        if kind=='query':result=torch.einsum('thp,hpl->thl',value.float(),self.weights.key_up.float()).bfloat16()
        else:result=torch.einsum('thl,hlv->thv',value.float(),self.weights.value_up.float()).bfloat16()
        if not bool(torch.isfinite(result).all()):raise ValueError('nonfinite projection result')
        return result

    def absorb_query_cpu(self,query,*,plan,max_tokens=128,max_output_bytes=2*1024**2):
        """Original QK256 -> latent512;packet retains scale based on QK256."""
        return self._project(query,plan,'query',max_tokens,max_output_bytes)

    def value_up_cpu(self,attention_latent,*,plan,max_tokens=128,max_output_bytes=2*1024**2):
        """Latent512 -> rank-local head values256;no o_proj or TP reduction."""
        return self._project(attention_latent,plan,'value',max_tokens,max_output_bytes)


@torch.inference_mode()
def load_rank_kv_b_cpu(reader,config,layer_id,heads,contract,*,max_local_heads=4,max_checkpoint_bytes=8*1024**2,max_runtime_bytes=4*1024**2):
    """One bounded contiguous row_slice only;own private BF16 W_UK_T/W_UV.

    Metadata is checked again after reading;this detects descriptor/recipe drift,
    not arbitrary in-place file mutation with identical metadata. Raw/runtime
    digests record read bytes,not a trusted expected hash or native attestation.
    Budgets are logical payload/output limits,not a measured process peak.
    """
    bounds=dict(max_local_heads=max_local_heads,max_checkpoint_bytes=max_checkpoint_bytes,max_runtime_bytes=max_runtime_bytes)
    m=rank_kv_b_manifest(reader,config,layer_id,heads,contract,**bounds)
    value,dtype=reader.row_slice(m.tensor_name,m.first_row,m.stop_row)
    expected_dtype=torch.bfloat16 if m.source_dtype=='BF16' else torch.float32
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=expected_dtype or dtype!=m.source_dtype or value.shape!=(m.stop_row-m.first_row,contract.latent_dim):raise ValueError('returned KV-B rows differ from FLOAT descriptor')
    if rank_kv_b_manifest(reader,config,layer_id,heads,contract,**bounds)!=m:raise ValueError('checkpoint metadata changed during row read')
    if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite KV-B source rows')
    raw_sha=hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    runtime=value.bfloat16()
    if not bool(torch.isfinite(runtime).all()):raise ValueError('KV-B invalid after BF16 rounding')
    runtime_sha=hashlib.sha256(runtime.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    ac=AbsorbedMLAContract.from_config(config,heads.head_ids)
    weights=split_kv_b_cpu(runtime,ac,max_weight_bytes=max_runtime_bytes)
    return PreparedRankKVB(m,weights,raw_sha,runtime_sha)
