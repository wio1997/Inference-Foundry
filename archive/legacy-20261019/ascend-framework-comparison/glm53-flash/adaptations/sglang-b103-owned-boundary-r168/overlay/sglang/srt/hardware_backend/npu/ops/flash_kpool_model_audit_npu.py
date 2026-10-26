# SPDX-License-Identifier: Apache-2.0
"""Read-only SGLang bridge to the common CPU64 KPool model oracle."""
from contextvars import ContextVar
from types import SimpleNamespace
import pathlib,hashlib,json,os,collections
import torch
from sglang.srt.hardware_backend.npu.ops import flash_indexer_projection_audit as common
_LAYER=ContextVar('flash_sglang_indexer_layer',default=None)
_SCORE_COUNTS=collections.Counter()


def before(indexer,layer_id,hidden,q_c,projected,raw_weights,key,gate,query,weights,
           cache,tail_key,tail_score,positions,slots,index_locations,tail_slots,query_ends):
    _LAYER.set(int(layer_id))
    if not pathlib.Path('/run-evidence/audit-enabled').exists() or torch.npu.current_device()!=0:return None
    # Materialize only diagnostic views; the native parameters/state are never
    # registered, replaced, or written by this adapter.
    owner=SimpleNamespace(k_cache=SimpleNamespace(prefix=f'model.layers.{layer_id}.indexer'),
        index_kpool=4,head_dim=128,n_head=32,softmax_scale=indexer.softmax_scale,
        _wk_weight_f32=torch.cat((indexer.wk.weight.float(),indexer.weights_proj.weight.float()),dim=0),
        _gate_weight_f32=indexer.index_kpool_compress_gate.float(),
        _norm_weight_f32=indexer.k_norm.weight.float() if indexer.k_norm.weight is not None else None,
        _norm_bias_f32=indexer.k_norm.bias.float() if indexer.k_norm.bias is not None else None,
        k_norm=indexer.k_norm,wq_b=indexer.wq_b,index_kpool_compress_ape=indexer.index_kpool_compress_ape)
    metadata=SimpleNamespace(positions=positions,cum_query_lens=query_ends,slot_mapping=index_locations,query_metadata=None)
    tail_metadata=SimpleNamespace(block_table=slots.unsqueeze(1),slot_mapping=tail_slots)
    return common.before(owner,hidden,hidden,q_c,torch.cat((projected,raw_weights),dim=-1),key,gate,query,weights,cache,torch.stack((tail_key,tail_score),dim=1),metadata,tail_metadata)


def after(check,cache,tail_key,tail_score):
    if check is not None:common.after(check,cache,torch.stack((tail_key,tail_score),dim=1))


def audit_scores(query,weights,cache,ends,pool_lens,table,positions,scores,start,rows,max_pool_seq_len,pool):
    layer=_LAYER.get()
    if layer is None or not pathlib.Path('/run-evidence/audit-enabled').exists() or torch.npu.current_device()!=0:return
    if max_pool_seq_len*pool<=2048 or _SCORE_COUNTS[layer]>=4:return
    assert query.dtype==weights.dtype==cache.dtype==torch.bfloat16
    ends_cpu=ends.detach().cpu().tolist();pos=positions.detach().cpu().tolist();eligible=[i for i in range(start,start+rows) if i<ends_cpu[-1] and pos[i]>=2048]
    if not eligible:return
    sample=eligible[-4:];q=query[sample].detach().cpu().double();w=weights[sample].detach().cpu().double();actual=scores[[i-start for i in sample]].detach().cpu();tab=table.detach().cpu();lens=pool_lens.detach().cpu().tolist();checks=[];max_error=0.
    for i,token in enumerate(sample):
        request=sum(token>=end for end in ends_cpu);visible=min((pos[token]+1)//pool,lens[request],max_pool_seq_len);npage=(visible+cache.shape[1]-1)//cache.shape[1];pages=tab[request,:npage].long().to(query.device);keys=cache.index_select(0,pages).detach().cpu().reshape(-1,query.shape[-1])[:visible].double()
        reference=(torch.matmul(q[i],keys.T).relu()*w[i].unsqueeze(1)).sum(0);value=actual[i,:visible].double();assert torch.isfinite(reference).all() and torch.isfinite(value).all();torch.testing.assert_close(value,reference,atol=1e-4,rtol=1e-4);assert (actual[i,visible:]==torch.finfo(torch.float32).min).all()
        error=float((value-reference).abs().max()) if visible else 0.;max_error=max(max_error,error);selected=min(512,visible);ref_top=torch.topk(reference,min(selected+1,visible));got=set(torch.topk(value,selected).indices.tolist());wanted=set(ref_top.indices[:selected].tolist());margin=float(ref_top.values[selected-1]-ref_top.values[selected]) if visible>selected else None;ambiguous=margin is not None and margin<=2e-4*(1+abs(float(ref_top.values[selected-1])));assert got==wanted or ambiguous
        checks.append({'position':pos[token],'visible_pools':visible,'pool_sets_equal':got==wanted,'ambiguous_boundary':ambiguous,'boundary_margin':margin})
    _SCORE_COUNTS[layer]+=1;record={'kind':'sglang_actual_indexer_score_CPU64','layer_id':layer,'device_id':0,'pid':os.getpid(),'audit_call':_SCORE_COUNTS[layer],'query_dtype':str(query.dtype),'head_weights_dtype':str(weights.dtype),'cache_dtype':str(cache.dtype),'max_abs_fp32_score_error':max_error,'sample_checks':checks,'passed':True,'atol':1e-4,'rtol':1e-4,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()};print('FLASH_SGLANG_INDEXER_SCORE_CHECK '+json.dumps(record),flush=True)
