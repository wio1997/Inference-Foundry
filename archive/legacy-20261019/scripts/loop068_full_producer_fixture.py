"""One-shot same-prestate private full DSA producer -> Sparse value gate.

Runs before the original layer2 live producer update. Every arm restores one
private alias-preserving bank from the still-unmodified live cache. No timing
or Product inference is made here.
"""

import json
import os
import traceback
from pathlib import Path

import torch

from scripts.loop066_owner_fixture import _comparison, _rows_by_slot


def _private_cache(kv_cache):
    groups = {}
    copies = []
    for tensor in kv_cache:
        if tensor is None:
            copies.append(None)
            continue
        storage = tensor.untyped_storage()
        if not storage.nbytes():
            copies.append(tensor.clone())
            continue
        ptr = storage.data_ptr()
        if ptr not in groups:
            live = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (storage.nbytes(),), (1,))
            private = live.clone()
            groups[ptr] = (live, private)
        private = groups[ptr][1]
        copy = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
            private.untyped_storage(), tensor.storage_offset(), tensor.size(), tensor.stride())
        if copy.untyped_storage().data_ptr() != private.untyped_storage().data_ptr():
            raise RuntimeError('private typed alias lost')
        copies.append(copy)
    if len(groups) != 3:
        raise RuntimeError(f'expected three unique layer2 physical stores, saw {len(groups)}')
    return tuple(copies), list(groups.values())


def _restore(groups):
    for live, private in groups:
        private.copy_(live)


def _owner_metadata(impl, meta, qsl, req_slice, ratio):
    from vllm_ascend.device.device_op import DeviceOperator
    req = meta.req_metadata
    qsl_b = (qsl[req_slice.start:req_slice.stop + 1] - qsl[req_slice.start]).contiguous()
    start_b = req.start_pos[req_slice].contiguous()
    bt_b = req.block_table[req_slice].contiguous()
    cos, sin, slots = torch.ops._C_ascend.compressor_metadata(
        req.full_compress_cos.view(req.full_compress_cos.shape[0], -1),
        req.full_compress_sin.view(req.full_compress_sin.shape[0], -1),
        qsl_b, start_b, bt_b, req.block_size,
        DeviceOperator.get_dsa_compressor_slot_mapping_format(),
        ratio, min(16, 16 // ratio + 2), 2)
    return qsl_b, start_b, bt_b, cos, sin, slots


def _current_state_snapshot(state, meta, starts_tensor, owners):
    req = meta.req_metadata
    starts = starts_tensor.cpu().tolist()
    table = req.block_table.cpu().tolist()
    block_size = int(req.block_size)
    coords = set()
    for request in owners:
        for pos in range(starts[request], starts[request] + 8):
            page = int(table[request][pos // block_size])
            slot = pos % block_size
            if page < 0 or page >= state.shape[0] or slot >= state.shape[1]:
                raise RuntimeError(f'invalid state page/slot {request} {page} {slot}')
            coords.add((page, slot))
    coords = sorted(coords)
    return torch.stack([state[page, slot].clone() for page, slot in coords]), len(coords)


def _cache_slot_snapshot(cache, slots):
    rows, duplicate = _rows_by_slot(slots)
    if duplicate or not rows:
        raise RuntimeError(f'invalid current cache slot map: rows={len(rows)} dup={duplicate[:2]}')
    values = []
    for slot in sorted(rows):
        if len(slot) != 2:
            raise RuntimeError(f'expected [page, offset] slot: {slot}')
        page, offset = slot
        if page >= cache.shape[0] or offset >= cache.shape[1]:
            raise RuntimeError(f'cache slot outside tensor: {slot}')
        values.append(cache[page, offset].clone())
    return torch.stack(values), len(rows)


def _arm(impl, private_cache, *, owner, x_full, x_local, q, qr,
         q_per_token_scale, attn_metadata, layer_name, owners,
         prepared_index=None, prepared_main=None):
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.attention.context_parallel.dsa_cp import rotate_activation
    (cmp_cache, swa_cache, main_state, _, _, _) = DeviceOperator.unpack_dsa_forward_kv_cache(
        private_cache, impl.compress_ratio)
    (index_state, index_key, index_scale, index_full) = DeviceOperator.unpack_dsa_indexer_kv_cache(
        private_cache)
    if index_full is not None:
        raise RuntimeError('non-A5 six-tensor cache expected')
    cmp_meta, main_state_meta, index_state_meta, index_scale_meta, swa_meta = attn_metadata
    req = cmp_meta.req_metadata
    cp = req.cp_metadata
    swa_req = swa_meta.req_metadata
    full_qsl = req.query_start_loc
    req_slice = slice(owners[0], owners[1] + 1)
    x = x_full[owners[0] * 8:(owners[1] + 1) * 8].contiguous() if owner else x_full
    cos = req.cos[layer_name][owners[0] * 8:(owners[1] + 1) * 8] if owner else req.cos[layer_name]
    sin = req.sin[layer_name][owners[0] * 8:(owners[1] + 1) * 8] if owner else req.sin[layer_name]

    # WKV -> SWA: this is the first of the three gathered-hidden producers.
    kv = impl.kv_norm(impl.wkv(x)).view(-1, 1, impl.nope_head_dim + impl.rope_head_dim)
    torch.ops._C_ascend.inplace_partial_rotary_mul(
        kv.unsqueeze(1), cos[:kv.shape[0]], sin[:kv.shape[0]],
        rotary_mode='interleave', partial_slice=[impl.nope_head_dim, impl.head_dim])
    swa_slots = (swa_req.slot_mapping[owners[0] * 8:(owners[1] + 1) * 8]
                 if owner else swa_req.slot_mapping)
    DeviceOperator.dsa_kv_compress_scatter(swa_cache, kv, swa_slots)

    # Indexer Compressor -> typed key/scale -> native QLI, with identical local
    # query and metadata in both arms.
    if prepared_index is not None:
        iqsl, istart, ibt, icos, isin, islots = prepared_index
    elif owner:
        iqsl, istart, ibt, icos, isin, islots = _owner_metadata(
            impl, index_scale_meta, full_qsl, req_slice, impl.compress_ratio)
        ibt = index_state_meta.req_metadata.block_table[req_slice].contiguous()
    else:
        iqsl, istart, ibt = full_qsl, index_scale_meta.req_metadata.start_pos, index_state_meta.req_metadata.block_table
        icos, isin, islots = impl._compute_compressor_metadata(index_scale_meta.req_metadata)
    ikv = torch.ops._C_ascend.compressor(
        x, impl.indexcom_wkv.weight, impl.indexcom_wgate.weight,
        index_state.squeeze(-2), impl.indexcom_ape, impl.indexcom_norm.weight,
        isin.view(-1, isin.shape[-1]), icos.view(-1, icos.shape[-1]),
        state_block_table=ibt, cu_seqlens=iqsl, seqused=None, start_pos=istart,
        rope_head_dim=impl.rope_head_dim, cmp_ratio=impl.compress_ratio,
        coff=2 if impl.compressor_overlap else 1,
        norm_eps=impl.compressor_norm_eps, rotary_mode=2, cache_mode=1)
    if ikv.numel():
        if impl.indexer.compressor.rotate:
            ikv = rotate_activation(ikv, index_scale_meta.hadamard)
        _, ikv_scale = DeviceOperator.indexer_quant_scatter_part1(
            ikv, index_key, None, islots)
        if ikv_scale is not None:
            DeviceOperator.dsa_indexer_scatter_scale_part3(ikv_scale, index_scale, islots)
    topk = impl._indexer_select_topk(
        x=x_local, qr=qr, kv_cache=private_cache, attn_metadata=attn_metadata,
        cos=cp.local_cos[layer_name], sin=cp.local_sin[layer_name],
        actual_seq_lengths_query=cp.local_query_start_loc,
        actual_seq_lengths_key=cp.local_seq_lens,
        qr_pertoken_scale=q_per_token_scale)

    # Main Compressor -> compressed KV scatter.
    if prepared_main is not None:
        mqsl, mstart, mbt, mcos, msin, mslots = prepared_main
    elif owner:
        mqsl, mstart, mbt, mcos, msin, mslots = _owner_metadata(
            impl, cmp_meta, full_qsl, req_slice, impl.compress_ratio)
        # The state cache can use a different table from compressed KV.
        mbt = main_state_meta.req_metadata.block_table[req_slice].contiguous()
    else:
        mqsl, mstart, mbt = full_qsl, req.start_pos, main_state_meta.req_metadata.block_table
        mcos, msin, mslots = impl._compute_compressor_metadata(req)
    ckv = torch.ops._C_ascend.compressor(
        x, impl.compressor_wkv.weight, impl.compressor_wgate.weight,
        main_state.squeeze(-2), impl.compressor_ape, impl.compressor_norm.weight,
        msin.view(-1, msin.shape[-1]), mcos.view(-1, mcos.shape[-1]),
        state_block_table=mbt, cu_seqlens=mqsl, seqused=None, start_pos=mstart,
        rope_head_dim=impl.rope_head_dim, cmp_ratio=impl.compress_ratio,
        coff=2 if impl.compressor_overlap else 1,
        norm_eps=impl.compressor_norm_eps, rotary_mode=2, cache_mode=1)
    DeviceOperator.dsa_kv_compress_scatter(cmp_cache, ckv if ckv.numel() else None, mslots)

    # The same Q and original local sparse metadata consume each private cache.
    extra = DeviceOperator.get_dsa_sparse_attn_base_kwargs()
    if swa_req.dspark_swa_indices is not None:
        extra['ori_sparse_indices'] = swa_req.dspark_swa_indices
    sparse_kwargs = dict(
        cu_seqlens_q=cp.local_query_start_loc,
        seqused_kv=cp.local_seq_lens,
        sinks=impl.attn_sink, softmax_scale=impl.softmax_scale,
        cmp_ratio=impl.compress_ratio, ori_mask_mode=4,
        ori_win_left=impl.window_size - 1 if swa_req.ori_win_left is None else swa_req.ori_win_left,
        ori_win_right=0 if swa_req.ori_win_right is None else swa_req.ori_win_right,
        layout_q='TND', layout_kv='PA_ND', **extra)
    DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
        sparse_kwargs, cu_seqlens_cmp_kv=req.cu_cmp_seqlen_list)
    sparse = DeviceOperator.get_dsa_sparse_attn_op()(
        q.clone(), ori_kv=swa_cache, cmp_kv=cmp_cache,
        cmp_sparse_indices=topk, ori_block_table=swa_req.block_table,
        cmp_block_table=req.block_table, metadata=req.sas_metadata,
        cmp_mask_mode=3, **sparse_kwargs)[0]
    return dict(
        sparse=sparse.clone(), topk=topk.clone(),
        swa_slots=swa_slots, main_slots=mslots, index_slots=islots,
        swa_cache=swa_cache, cmp_cache=cmp_cache,
        index_key=index_key, index_scale=index_scale,
        index_state=index_state, main_state=main_state,
        index_state_meta=index_state_meta, main_state_meta=main_state_meta)


def run_private_full_producer(impl, *, x_full, x_local, q, qr,
                              q_per_token_scale, kv_cache, attn_metadata,
                              layer_name):
    from vllm.forward_context import get_forward_context
    from vllm_ascend.device.device_op import DeviceOperator
    out_dir = os.getenv('EXTREME_FULL_PRODUCER_DIR')
    if not out_dir or getattr(impl, '_extreme_full_producer_done', False):
        return
    if layer_name != 'model.layers.2.self_attn.attn' or tuple(x_full.shape) != (96, 4096):
        return
    context = get_forward_context()
    if getattr(context, 'capturing', False) or torch.npu.is_current_stream_capturing():
        return
    if getattr(context, 'is_draft_model', False) or not context.additional_kwargs.get(
            'extreme_full_producer_runtime', False):
        return
    impl._extreme_full_producer_done = True
    rank = int(impl.tp_rank)
    result = {'rank': rank, 'stage': 'preflight', 'pass': False,
              'scope': 'one real Target layer2 private A/A/B/A same-prestate full producer->Sparse; no timing or Product claim'}
    path = Path(out_dir) / f'rank{rank}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if len(kv_cache) != 6 or impl.compress_ratio != 4 or tuple(q.shape[:1]) != (12,):
            raise RuntimeError('frozen c4 six-cache/local Q ABI changed')
        owners = [(rank * 12) // 8, (rank * 12 + 11) // 8]
        if owners[1] != owners[0] + 1 or attn_metadata[0].req_metadata.query_start_loc.cpu().tolist() != list(range(0, 97, 8)):
            raise RuntimeError('owner/qsl geometry changed')
        result['owners'] = owners
        private_cache, groups = _private_cache(kv_cache)
        result['private_store_bytes'] = [private.numel() for _, private in groups]
        result['private_store_count'] = len(groups)
        baseline = None
        arm_results = []
        for name in ('A', 'A2', 'B', 'A3'):
            _restore(groups)
            arm = _arm(impl, private_cache, owner=(name == 'B'), x_full=x_full,
                       x_local=x_local, q=q, qr=qr, q_per_token_scale=q_per_token_scale,
                       attn_metadata=attn_metadata, layer_name=layer_name, owners=owners)
            torch.npu.synchronize()
            # Current-owner typed writes are the smallest persistent state gate.
            owner_swa = attn_metadata[-1].req_metadata.slot_mapping[
                owners[0] * 8:(owners[1] + 1) * 8]
            owner_cmp = _owner_metadata(impl, attn_metadata[0],
                                        attn_metadata[0].req_metadata.query_start_loc,
                                        slice(owners[0], owners[1] + 1), 4)[-1]
            owner_index = _owner_metadata(impl, attn_metadata[3],
                                          attn_metadata[0].req_metadata.query_start_loc,
                                          slice(owners[0], owners[1] + 1), 4)[-1]
            snapshot = {'sparse': arm['sparse'], 'topk': arm['topk']}
            for label, cache, slots in (
                ('swa', arm['swa_cache'], owner_swa),
                ('main', arm['cmp_cache'], owner_cmp),
                ('index_key', arm['index_key'], owner_index),
                ('index_scale', arm['index_scale'], owner_index)):
                snapshot[label], count = _cache_slot_snapshot(cache, slots)
                result[f'{label}_owner_slots'] = count
            for label in ('index', 'main'):
                state, meta = ((arm['index_state'], arm['index_state_meta']) if label == 'index'
                               else (arm['main_state'], arm['main_state_meta']))
                starts = (attn_metadata[3].req_metadata.start_pos if label == 'index'
                          else attn_metadata[0].req_metadata.start_pos)
                snapshot[f'{label}_state'], count = _current_state_snapshot(
                    state, meta, starts, owners)
                result[f'{label}_state_current_slots'] = count
            if baseline is None:
                baseline = {key: value.clone() for key, value in snapshot.items()}
            comparisons = {key: bool(torch.equal(value, baseline[key]))
                           for key, value in snapshot.items()}
            arm_results.append({'arm': name, 'exact_to_first_A': comparisons,
                                'sparse_shape': list(arm['sparse'].shape),
                                'topk_shape': list(arm['topk'].shape)})
        result['arms'] = arm_results
        result['stage'] = 'complete'
        result['pass'] = all(all(x['exact_to_first_A'].values()) for x in arm_results)
        if result['pass'] and os.getenv('EXTREME_FULL_PRODUCER_GRAPH_DIR'):
            from scripts.loop068_full_producer_graph_fixture import graph_screen
            graph_screen(impl, private_cache=private_cache, groups=groups,
                         x_full=x_full, x_local=x_local, q=q, qr=qr,
                         q_per_token_scale=q_per_token_scale,
                         attn_metadata=attn_metadata, layer_name=layer_name,
                         owners=owners, eager_sparse=baseline['sparse'],
                         eager_topk=baseline['topk'], eager_snapshot=baseline)
    except Exception as exc:
        result['stage'] = 'error'
        result['pass'] = False
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()[-6000:]
    finally:
        try:
            torch.npu.synchronize()
        except Exception as exc:
            result['final_sync_error'] = repr(exc)
            result['pass'] = False
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(f'EXTREME_FULL_PRODUCER rank={rank} stage={result["stage"]} pass={result["pass"]}', flush=True)
    if not result['pass']:
        raise RuntimeError(f'private full producer gate failed on rank{rank}: {result.get("error", "value mismatch")}')
