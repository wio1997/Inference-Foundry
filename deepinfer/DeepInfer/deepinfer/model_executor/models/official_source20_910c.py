"""Reuse the archived full source-20 C1/cache routine."""

def prestore_full_source(model, layer, state, trunk, previous):
    """Official consumer_full_hidden duty, before query compaction at layer20.

    Retain the original full-layout norm and C1 compression/index-K math.
    This function does not project Q, write SWA, or advance any position.
    """
    from .w8a8_compressor910c import compress_batch
    from .w8a8_math910c import rms_norm_mixed, rotate_positions
    c = model.cfg
    # Bounded, opt-in correctness provenance for the dedicated functional
    # session. No tracing or extra device reads in ordinary serving.
    import os
    trace_root=os.environ.get('DEEPINFER_910C_FUNCTIONAL_SOURCE_TRACE')
    if trace_root:
        from pathlib import Path
        root=Path(trace_root)
        if not root.is_relative_to('/data/tiankuan/wio/910c-deepinfer-172/diagnostics/module-validation-20261107'):
            raise ValueError('functional source trace outside owned diagnostics')
        number=getattr(model,'_functional_source_traces',0)
        limit=int(os.environ.get('DEEPINFER_910C_FUNCTIONAL_SOURCE_TRACE_LIMIT','4'))
        if not 1<=limit<=8:raise ValueError('functional trace limit must be 1..8')
        if number<limit:
            state.boundary20_trace_dir=root/f'forward-{number:03d}'
            state.boundary20_trace_dir.mkdir(parents=True,exist_ok=True)
            model._functional_source_traces=number+1
    boundary = next(i for i,r in enumerate(c['compress_ratios']) if r == 1)
    if (boundary != 20 or layer.layer_id != boundary or layer.ratio != 1 or
            max(c['kv_source_layer_ids']) != boundary or state.decode_frame is not None or
            not state.sequence_parallel or state.context_query is None or
            getattr(state,'pre_stored_source',None) is not None):
        raise ValueError('unsupported full-source prestore contract')
    from deepinfer import envs
    if layer.fused_mhc or envs.DEEPINFER_BATCH_INVARIANT:
        x, _, _, _ = layer._hc_pre(trunk,previous,'attn')
    else:
        x, _, _, _ = layer._pre(trunk,previous,'attn')
    x = rms_norm_mixed(x,layer.attn_norm_weight,c['rms_norm_eps'])[0]
    # The original source layer gathers full normalized hidden before its
    # compressor. Preserve that input shape and ordering in this adapter.
    x = layer.sp.gather(x,sum(state.batch.query_lengths))
    latent,position_list,request_list = compress_batch(x,state.batch,state.slots,layer.ratio,
        layer.compressor_wkv_weight,getattr(layer,'compressor_wgate_weight',None),
        layer.compressor_norm_weight,c['rms_norm_eps'],model.pending.setdefault(boundary,{}))
    expected,positions,plans = state.source_metadata[boundary]
    if tuple(position_list) != expected or len(request_list) != len(expected):
        raise RuntimeError('full source coordinates differ from full prepared plan')
    actual_requests = [slot for slot,q in zip(state.slots,state.batch.query_lengths) for _ in range(q)]
    if list(request_list) != actual_requests or latent.shape[0] != sum(state.batch.query_lengths):
        raise RuntimeError('C1 full-source row/request layout changed')
    key,scale = layer.indexer.key(latent,positions)
    rotated = rotate_positions(latent,positions,layer.rope_freq)
    model.cache.compressed[boundary].write(rotated[:,None],plans[0])
    model.cache.index_key[boundary].write(key,plans[1])
    model.cache.index_scale[boundary].write(scale[:,None,:],plans[2])
    state.pre_stored_source=boundary
    proof=None
    if getattr(state,'boundary20_trace_dir',None) is not None:
        proof=verify_source_cache(state,model,rotated,key,scale,plans)
    record_full_source(state,layer.layer_id,x,latent,rotated,key,scale,positions,stage='prestore',cache_proof=proof)
    return boundary


def sample_rows(query_lengths):
    samples=set();offset=0
    for q in query_lengths:
        samples.update(offset+i for i in [0,1,q//2,q-141,q-140,q-129,q-128,q-2,q-1] if 0<=i<q)
        offset+=q
    return sorted(samples)


def verify_source_cache(state,model,rotated,key,scale,plans):
    """Read sample cache rows written by the actual full source producer."""
    import torch
    samples=sample_rows(state.batch.query_lengths);proof=[]
    for name,value,plan in zip(('compressed','index_key','index_scale'),
            (rotated[:,None],key,scale[:,None,:]),plans,strict=True):
        plane=getattr(model.cache,name)[20]
        index=torch.tensor(samples,dtype=torch.int64,device=value.device)
        physical=plan.physical_rows.index_select(0,index)
        actual=plane.cache.flatten(0,1).index_select(0,physical).cpu().contiguous()
        expected=value.index_select(0,index).cpu().contiguous()
        if not torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)):
            raise RuntimeError('source20 cache payload was not stored at prepared physical rows')
        proof.append(dict(plane=name,full_rows=value.shape[0],plan_rows=plan.count,
            sampled_rows=samples,physical_rows=physical.cpu().tolist(),sample_bits_identical=True))
    return proof


def record_index_contract(state,layer_id):
    root=getattr(state,'boundary20_trace_dir',None)
    if root is None or layer_id!=20:return
    import json
    import torch.distributed as dist
    cp=state.context_query
    record=dict(rank=dist.get_rank(),layer=layer_id,query_lengths=list(state.batch.query_lengths),
        local_query_rows=cp.end-cp.start,selected_shape=list(state.selected.shape),
        candidates_none=state.candidates is None,
        candidates_shape=None if state.candidates is None else list(state.candidates.shape),
        actual_index_source=state.index_source)
    if state.selected.shape[0]!=cp.end-cp.start:
        raise RuntimeError('source20 selected indices do not belong to current CP rows')
    if state.candidates is not None and state.candidates.shape[0]!=cp.end-cp.start:
        raise RuntimeError('source20 candidates do not belong to current CP rows')
    with (root/f'boundary20-index-rank{dist.get_rank()}.json').open('x') as f:json.dump(record,f,indent=2)


def record_full_source(state, layer_id, x, latent, rotated, key, scale, positions, *, stage,cache_proof=None):
    """Optional prefill-only provenance; never enabled inside startup graphs."""
    root = getattr(state,'boundary20_trace_dir',None)
    if root is None or layer_id != 20:
        return
    from pathlib import Path
    import hashlib,json
    import torch
    import torch.distributed as dist
    root = Path(root)
    if not root.is_relative_to('/data/tiankuan/wio/910c-deepinfer-172'):
        raise ValueError('trace path outside wio')
    rank = dist.get_rank()
    # Include original query starts, compact starts and page/window edges.
    samples=sample_rows(state.batch.query_lengths);raw={};descriptions={}
    for name,value in [('full_normalized_hidden',x),('latent',latent),('rotated_KV',rotated),('index_key',key),('index_scale',scale)]:
        cpu=value.detach().cpu().contiguous()
        bits=cpu.reshape(-1).view(torch.uint8)
        descriptions[name]={'shape':list(value.shape),'dtype':str(value.dtype),
            'sha256':hashlib.sha256(bits.numpy().tobytes()).hexdigest()}
        selected=cpu.index_select(0,torch.tensor(samples,dtype=torch.int64)).contiguous()
        raw[name]=selected.reshape(len(samples),-1).view(torch.uint8).clone()
    pt=root/f'boundary20-fullsource-rank{rank}.pt'
    with pt.open('xb') as f:torch.save(raw,f)
    report={'state':'SOURCE20_FULL_PAYLOAD_CAPTURED','rank':rank,'stage':stage,
        'candidates_none_before_consumer_entry':state.candidates is None,
        'query_lengths':list(state.batch.query_lengths),'slots':list(state.slots),
        'source_positions':positions.detach().cpu().tolist(),'sample_rows':samples,
        'tensors':descriptions,'cache_write_proof':cache_proof,
        'raw_file':pt.name,'raw_sha256':hashlib.sha256(pt.read_bytes()).hexdigest(),
        'scope':'Full tensor digests and saved sample bits; compare identical full input across native/mask/compact'}
    with (root/f'boundary20-fullsource-rank{rank}.json').open('x') as f:json.dump(report,f,indent=2)
