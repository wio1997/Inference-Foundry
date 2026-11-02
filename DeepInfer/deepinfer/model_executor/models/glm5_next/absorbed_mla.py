# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU reference for NoPE SFA absorption, not a native/runtime fallback.

The attention dimension becomes latent512 after absorbing W_UK, but the score
scale remains based on original QK256. BF16 boundaries after query absorption
and latent reduction differ from expanded-K/V attention. No sink is modeled;
the separate SMLA metadata/sinks branch must not silently use this reference.
No paged cache write, allocator, native import, TP collective or decoder binding.
"""
import ast
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class AbsorbedMLAContract:
    total_heads: int
    head_ids: tuple[int,...]
    qk_dim: int
    kv_rank: int
    value_dim: int
    pool_size: int = 4
    topk_tokens: int = 2048

    def __post_init__(self):
        if any(type(v) is not int or v<1 for v in (self.total_heads,self.qk_dim,self.kv_rank,self.value_dim,self.pool_size,self.topk_tokens)):
            raise ValueError('positive integer MLA dimensions required')
        if not isinstance(self.head_ids,tuple) or not 1<=len(self.head_ids)<=4 or any(type(h) is not int or not 0<=h<self.total_heads for h in self.head_ids) or len(set(self.head_ids))!=len(self.head_ids):
            raise ValueError('one to four distinct explicit global head IDs required')
        if self.topk_tokens%self.pool_size:raise ValueError('topk must contain complete pools')

    @classmethod
    def from_config(cls,config,head_ids):
        d=config.mla_dimensions
        if d['qk_rope_head_dim']!=0 or config.hf_config['text_config'].get('mla_use_nope') is not True:
            raise ValueError('absorbed reference supports explicit NoPE only')
        return cls(d['num_attention_heads'],head_ids,d['qk_nope_head_dim'],d['kv_lora_rank'],d['v_head_dim'],
                   config.indexer_config['index_kpool'],config.indexer_config['index_topk'])

    @property
    def heads(self):return len(self.head_ids)
    @property
    def scale(self):return self.qk_dim**-.5
    @property
    def output_width(self):return self.topk_tokens+self.pool_size-1


@dataclass(frozen=True)
class AbsorbedMLAWeights:
    contract: AbsorbedMLAContract
    key_up: torch.Tensor  # [selected_heads,QK,latent]
    value_up: torch.Tensor  # [selected_heads,latent,V]


@dataclass(frozen=True)
class AbsorbedMLAResult:
    query_latent: torch.Tensor  # [T,H,latent], BF16, inactive rows zero
    attention_latent: torch.Tensor  # [T,H,latent], BF16
    values: torch.Tensor  # [T,H,V], BF16, before o_proj
    lse: torch.Tensor  # [T,H], FP32; -inf for empty selections


def _cpu(*tensors):
    if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in tensors):raise ValueError('absorbed reference is CPU-only')


def _positive(v,name):
    if type(v) is not int or v<1:raise ValueError(f'{name} must be a positive integer')


@torch.inference_mode()
def split_kv_b_cpu(weight,contract,*,max_weight_bytes=8*1024**2):
    """Caller supplies ONLY explicit selected heads in head_ids order.

    FLOAT checkpoint storage must be rounded to runtime BF16 by the caller.
    No automatic slicing/dequantization, head partitioning or TP is inferred.
    Returned weights own storage, never alias the reader or each other.
    """
    _cpu(weight);_positive(max_weight_bytes,'max_weight_bytes')
    h=contract.heads;p=contract.qk_dim;v=contract.value_dim;l=contract.kv_rank
    if weight.dtype!=torch.bfloat16 or weight.shape!=(h*(p+v),l):raise ValueError('selected kv_b weight requires BF16 [heads*(QK+V),latent]')
    if weight.numel()*weight.element_size()>max_weight_bytes:raise ValueError('selected weight exceeds byte budget')
    if not bool(torch.isfinite(weight).all()):raise ValueError('nonfinite selected weights')
    blocks=weight.reshape(h,p+v,l)
    return AbsorbedMLAWeights(contract,blocks[:,:p].contiguous().clone(),blocks[:,p:].transpose(1,2).contiguous().clone())


@torch.inference_mode()
def absorbed_sparse_mla_cpu(query,latent,indices,positions,weights,*,sink_policy,max_tokens=128,max_latent_tokens=4096,max_work_elements=32*1024**2):
    """One request's shared BF16 latent history, selected logical TOKEN IDs.

    CPU reference uses FP32 contractions then BF16 storage at native seams.
    Packed unique causal selections may be ranked; no sorting is required for
    the C_ascend sparse-mode3 source branch. Empty/padded rows have zero output.
    The bound counts contraction products, not measured memory or performance.
    """
    if sink_policy!='absent':raise ValueError('only explicit absent sink is supported; SMLA sink semantics unverified')
    for v,n in ((max_tokens,'max_tokens'),(max_latent_tokens,'max_latent_tokens'),(max_work_elements,'max_work_elements')):_positive(v,n)
    _cpu(query,latent,indices,positions,weights.key_up,weights.value_up)
    c=weights.contract;h=c.heads;p=c.qk_dim;l=c.kv_rank;v=c.value_dim
    if query.ndim!=3 or query.shape[1:]!=(h,p) or query.dtype!=torch.bfloat16:raise ValueError('query requires selected-head BF16 QK vectors')
    t=query.shape[0]
    if latent.ndim!=2 or latent.shape[1]!=l or latent.dtype!=torch.bfloat16:raise ValueError('cache must be BF16 shared latent, not expanded K/V')
    if indices.ndim==3 and indices.shape[1]==1:indices=indices[:,0]
    if indices.ndim!=2 or indices.shape[0]!=t or indices.shape[1]>c.output_width or indices.dtype not in (torch.int32,torch.int64):raise ValueError('bounded token indices required')
    if positions.shape!=(t,) or positions.dtype not in (torch.int32,torch.int64):raise ValueError('positions require integer token vector')
    if weights.key_up.shape!=(h,p,l) or weights.value_up.shape!=(h,l,v) or any(w.dtype!=torch.bfloat16 for w in (weights.key_up,weights.value_up)):
        raise ValueError('absorbed weight geometry/dtype differs')
    if t>max_tokens or len(latent)>max_latent_tokens:raise ValueError('token/history bound exceeded')
    if bool((indices < -1).any()):raise ValueError('only -1 padding supported')
    selected=int((indices>=0).sum());work=t*h*l*(p+v)+2*selected*h*l
    if work>max_work_elements:raise ValueError('absorbed work exceeds explicit CPU bound')
    selections=[];active=[]
    for row in range(t):
        ids=indices[row];valid=ids[ids>=0].long();n=valid.numel()
        if bool((ids[:n]<0).any()) or bool((ids[n:]!=-1).any()):raise ValueError('selected token IDs must be a packed prefix')
        if n:
            if positions[row]<0 or valid.unique().numel()!=n or bool(((valid>=len(latent))|(valid>positions[row])).any()):raise ValueError('selected IDs must be unique causal logical tokens')
            if any(not bool(torch.isfinite(x).all()) for x in (query[row],latent[valid])):raise ValueError('nonfinite live operands')
            active.append(row)
        selections.append(valid)
    if any(not bool(torch.isfinite(x).all()) for x in (weights.key_up,weights.value_up)):raise ValueError('nonfinite absorbed weights')
    ql=torch.zeros(t,h,l,dtype=torch.bfloat16);out=torch.zeros_like(ql);lse=torch.full((t,h),float('-inf'))
    if active:ql[active]=torch.einsum('thp,hpl->thl',query[active].float(),weights.key_up.float()).bfloat16()
    if active and not bool(torch.isfinite(ql[active]).all()):raise ValueError('query absorption overflow')
    for row,ids in enumerate(selections):
        if not ids.numel():continue
        logits=torch.einsum('hl,sl->hs',ql[row].float(),latent[ids].float())*c.scale
        if not bool(torch.isfinite(logits).all()):raise ValueError('latent attention logit overflow')
        out[row]=torch.einsum('hs,sl->hl',torch.softmax(logits,dim=-1),latent[ids].float()).bfloat16()
        lse[row]=torch.logsumexp(logits,dim=-1)
    values=torch.einsum('thl,hlv->thv',out.float(),weights.value_up.float()).bfloat16()
    if not bool(torch.isfinite(values).all()):raise ValueError('value projection overflow')
    return AbsorbedMLAResult(ql,out,values,lse)


def inspect_sfa_source_contract(source):
    """AST source ledger, NOT a binary schema, ABI or native execution check."""
    tree=ast.parse(source)
    def method(name):
        found=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(found)!=1:raise ValueError('missing/ambiguous source method '+name)
        return found[0]
    def call(name,root):
        found=[n for n in ast.walk(root) if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
        if len(found)!=1:raise ValueError('missing/ambiguous source call '+name)
        return found[0]
    native=call('torch.ops._C_ascend.npu_sparse_flash_attention',method('sparse_mla'))
    expected={'query':'query.contiguous()','key':'cache','value':'cache','sparse_indices':'indices','scale_value':'scale','sparse_block_size':'1',
        'block_table':'metadata.block_table','actual_seq_lengths_query':'metadata.query_start_loc[1:].to(torch.int32)',
        'actual_seq_lengths_kv':'metadata.seq_lens.to(torch.int32)','query_rope':'None','key_rope':'None','layout_query':"'TND'",'layout_kv':"'PA_BSND'",'sparse_mode':'3','attention_mode':'2','return_softmax_lse':'False'}
    if native.args or {k.arg:ast.unparse(k.value) for k in native.keywords}!=expected:raise ValueError('NoPE native call roles/layout drift')
    scatter=call('torch_npu.npu_scatter_nd_update_',method('exec_kv'))
    if [ast.unparse(a) for a in scatter.args]!=['cache.view(-1, self.kv_lora_rank)','slots[:values.shape[0]].view(-1, 1)','values.to(cache.dtype)']:
        raise ValueError('main latent cache write contract drift')
    f=method('forward');order=[call(n,f).lineno for n in ('self.exec_kv','self.indexer','notify_kv_cache_written','self._execute_sparse_flash_attention_process')]
    if order!=sorted(order) or len(set(order))!=4:raise ValueError('cache/indexer/notification/attention source order drift')
    dispatch=method('_execute_sparse_flash_attention_process');call('sparse_mla',dispatch)
    q=method('_q_proj_and_k_up_proj');k=call('torch.bmm',q)
    if [ast.unparse(a) for a in k.args]!=['q_nope','self.W_UK_T']:raise ValueError('query absorption orientation drift')
    vv=call('torch.bmm',method('_v_up_proj'))
    if [ast.unparse(a) for a in vv.args]!=['x','self.W_UV']:raise ValueError('value projection orientation drift')
    return {'source_only':True,'native_ABI_validated':False,'native_call':expected,
            'cache_write':'in-place normalized latent scatter BEFORE selection/attention',
            'order':['exec_kv','indexer','notify_kv_cache_written','attention'],
            'deferred_transaction_implemented':False,'sink_scope':'C_ascend branch absent;SMLA sinks unverified'}
