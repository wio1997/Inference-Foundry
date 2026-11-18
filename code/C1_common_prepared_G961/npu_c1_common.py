"""Prepared, NOT deployed: preserve native C1 block policy, remove dense slot maps.

Install only after the existing freqs_install has patched native decode_scores.
Native DenseBlocks publish/consume/top-k functions and their function aliases stay.
Actual block/slot/graph correctness and end-to-end effect remain unverified.
"""
from pathlib import Path
import hashlib,inspect,json,os,types
import torch

_DONE=False
_ROOT=None
_SEEN=set()

def _record(kind,inputs,**detail):
    key=(kind,inputs.layer_id,inputs.compress_ratio,inputs.positions.shape[0])
    if key in _SEEN:return
    _SEEN.add(key)
    value={'kind':kind,'pid':os.getpid(),'layer_id':inputs.layer_id,
           'ratio':inputs.compress_ratio,'query_rows':inputs.positions.shape[0],
           'scope':'Pythonconstruction/pathonly; notactualreplay/quality/gainproof',**detail}
    with (_ROOT/f'c1_common_hits.pid{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(value)+'\n')

def _decode_scores(*,inputs,token_to_kv_pool,req_to_token,candidate_mask=None,compute_topk_plan=True):
    if inputs.x.device.type!='npu':
        return _c1_prior_decode(inputs=inputs,token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,candidate_mask=candidate_mask,compute_topk_plan=compute_topk_plan)
    # Keep the original nonpaged branch's block-policy semantics. In that branch
    # candidate_mask and compute_topk_plan are unused; the consumer selects among
    # published blocks later. Do not silently route through Hopper PagedDecodeScores.
    ratio=inputs.compress_ratio;assert ratio in (1,2)
    req,pos=inputs.req_rows,inputs.positions;batch=req.shape[0]
    assert pos.shape[0]==batch
    inputs.reset_outputs()
    if not batch:return None
    max_len=min(inputs.paged_metadata.max_compressed_seq_len,req_to_token.shape[1]//ratio)
    if not max_len:return None
    lens=(pos+1)//ratio
    q=inputs.indexer.queries(inputs.q_lora,_c1_freqs_gather(inputs.freqs_cis,pos))
    weights=inputs.indexer.head_weights(inputs.x)
    table=token_to_kv_pool.get_index_k_with_scale_buffer(inputs.layer_id)
    scores=_c1_score_paged(q,weights,req,req_to_token,lens,table,table.shape[1]//68,max_len,ratio)
    _c1_record('score',inputs,capacity=max_len,slots_materialized=False,
               candidate_mask_present=candidate_mask is not None,
               candidate_mask_policy='SameasnativeNPU nonpaged branch: blockselection remains downstream')
    # A DecodeScores subclass deliberately keeps existing native isinstance
    # dispatch. Only write_decode reads .slots; that call is replaced below.
    return _c1_scores_type(bs=batch,lmax=max_len,lens=lens,slots=None,scores=scores,
                          req=req,req_to_token=req_to_token,ratio=ratio)

def _write_decode(inputs,d,idx):
    if not isinstance(d,_c1_scores_type):return _c1_prior_write(inputs,d,idx)
    ordered=idx.sort(dim=-1).values.to(torch.int32)
    pages=inputs.out_page_indices;k=ordered.shape[1]
    _c1_map_slots[(pages.shape[0],)](ordered,d.lens,d.req,d.req_to_token,pages,
        d.bs,k,d.lmax,pages.shape[1],ordered.stride(0),d.req_to_token.stride(0),
        pages.stride(0),d.ratio,_c1_next_power_of_2(max(k,pages.shape[1])),num_warps=4)
    _c1_record('write',inputs,selected_width=k,capacity=d.lmax,
               selection='OriginalnativeC1block/topkpolicy andtorchsort, selectedKphysicalmappingonly')

def install(root):
    global _DONE,_ROOT
    if _DONE:return
    import triton
    import freqs_install
    from freqs_compat import gather
    import sglang.srt.layers.attention.dsv4.v41_indexer.scoring as scoring
    import sglang.srt.layers.attention.dsv4.v41_indexer.dense_blocks as dense
    import sglang.kernels.ops.attention.dsv4.candidate_blocks as blocks
    from npu_paged_indexer import score_paged,_map_sorted_selected_slots
    assert freqs_install._DONE,'Run after the original bit-preserving frequency adaptation'
    pins=json.loads((root/'native_C1_source_pins_g961.json').read_text())
    for module,key in [(scoring,'scoring'),(dense,'dense_blocks'),(blocks,'candidate_blocks')]:
        assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==pins[key]
    assert hashlib.sha256(Path(inspect.getsourcefile(gather)).read_bytes()).hexdigest()==pins['freqs_compat']
    assert dense.decode_scores is scoring.decode_scores and dense.write_decode is scoring.write_decode
    prior_decode=types.FunctionType(scoring.decode_scores.__code__,scoring.decode_scores.__globals__,
        'prior_native_frequency_patched_decode',scoring.decode_scores.__defaults__,scoring.decode_scores.__closure__)
    prior_decode.__kwdefaults__=scoring.decode_scores.__kwdefaults__
    prior_write=types.FunctionType(scoring.write_decode.__code__,scoring.write_decode.__globals__,
        'prior_native_write',scoring.write_decode.__defaults__,scoring.write_decode.__closure__)
    class NPUScores(scoring.DecodeScores,frozen=True):
        req:torch.Tensor
        req_to_token:torch.Tensor
        ratio:int
    _ROOT=Path(root)
    scoring.__dict__.update(_c1_prior_decode=prior_decode,_c1_prior_write=prior_write,
        _c1_freqs_gather=gather,_c1_score_paged=score_paged,_c1_scores_type=NPUScores,
        _c1_map_slots=_map_sorted_selected_slots,_c1_next_power_of_2=triton.next_power_of_2,
        _c1_record=_record)
    assert not scoring.decode_scores.__code__.co_freevars and not _decode_scores.__code__.co_freevars
    assert not scoring.write_decode.__code__.co_freevars and not _write_decode.__code__.co_freevars
    # Preserve native function identity so imported aliases in every consumer
    # see the same change. Prefill helpers and C1 block policy are untouched.
    scoring.decode_scores.__code__=_decode_scores.__code__
    scoring.write_decode.__code__=_write_decode.__code__
    _DONE=True
    (_ROOT/f'c1_common_install.pid{os.getpid()}.json').write_text(json.dumps({
        'source_pins':pins,'native_alias_identity_preserved':True,
        'scope':'PreparedC1shareddecodeadapter; requiresactualcandidateblock/score/slot/graphgate/fullmodelvalidation. Noqualityorperformanceacceptance.'},indent=2)+'\n')
