"""One-shot real layer2 c4 owner-compact Compressor fixture, eager diagnostic only.

Runs on private state tensors before the original indexer update. It intentionally
stops at Compressor output and recursive state; shared typed key/scale scatter,
QLI, Sparse and cross-cycle lifetime are separate gates.
"""
import json
import os
from pathlib import Path

import torch


def _private_alias_views(state, key, scale):
    views = (state, key, scale)
    storage = state.untyped_storage()
    if any(t.untyped_storage().data_ptr() != storage.data_ptr() for t in views):
        raise RuntimeError('indexer state/key/scale no longer share one untyped storage')
    nbytes = storage.nbytes()
    raw_source = torch.empty(0, dtype=torch.uint8, device=state.device).set_(
        storage, 0, (nbytes,), (1,))
    raw_private = raw_source.clone()
    copies = tuple(torch.empty(0, dtype=t.dtype, device=t.device).set_(
        raw_private.untyped_storage(), t.storage_offset(), t.size(), t.stride())
        for t in views)
    if any(t.untyped_storage().data_ptr() != raw_private.untyped_storage().data_ptr() for t in copies):
        raise RuntimeError('private typed aliases lost')
    return copies, raw_private


def _compress(impl, x, state, block_table, qsl, start_pos, cos, sin):
    return torch.ops._C_ascend.compressor(
        x, impl.indexcom_wkv.weight, impl.indexcom_wgate.weight,
        state.squeeze(-2), impl.indexcom_ape, impl.indexcom_norm.weight,
        sin.view(-1, sin.shape[-1]), cos.view(-1, cos.shape[-1]),
        state_block_table=block_table, cu_seqlens=qsl, seqused=None,
        start_pos=start_pos, rope_head_dim=impl.rope_head_dim,
        cmp_ratio=impl.compress_ratio, coff=2 if impl.compressor_overlap else 1,
        norm_eps=impl.compressor_norm_eps, rotary_mode=2, cache_mode=1,
    )


def _rows_by_slot(slots):
    cpu_slots = slots.cpu().tolist()
    result = {}
    duplicates = []
    for idx, slot in enumerate(cpu_slots):
        key = tuple(slot) if isinstance(slot, list) else (int(slot),)
        if any(v < 0 for v in key):
            continue
        if key in result:
            duplicates.append(key)
        result[key] = idx
    return result, duplicates


def _comparison(full, full_slots, other, other_slots, expected_other_slots=None):
    full_rows, duplicate_full = _rows_by_slot(full_slots)
    other_rows, duplicate_other = _rows_by_slot(other_slots)
    common = sorted(set(full_rows) & set(other_rows))
    missing = sorted(set(other_rows) - set(full_rows))
    expected_rows, _ = _rows_by_slot(expected_other_slots) if expected_other_slots is not None else (None, [])
    complete = (not missing and not duplicate_full and not duplicate_other and
                (expected_rows is None or set(other_rows)==set(expected_rows)))
    full_ids = [full_rows[k] for k in common]
    other_ids = [other_rows[k] for k in common]
    if full_ids:
        a = full[full_ids].float()
        b = other[other_ids].float()
        diff = (a-b).abs()
        max_abs = float(diff.max().item())
        exact = bool(torch.equal(a,b))
    else:
        max_abs = None
        exact = False
    return {'full_valid_slots':len(full_rows),'other_valid_slots':len(other_rows),
            'common_slots':len(common),'missing_other_slots':missing[:8],
            'duplicate_full_slots':duplicate_full[:8],
            'duplicate_other_slots':duplicate_other[:8],
            'expected_other_valid_slots':len(expected_rows) if expected_rows is not None else None,
            'complete_slot_coverage':complete,
            'matched_values_exact':exact,'matched_values_max_abs':max_abs}


def run_owner_fixture(impl, *, x, kv_cache, attn_metadata, qsl, layer_name):
    from vllm.forward_context import get_forward_context
    from vllm_ascend.device.device_op import DeviceOperator
    out_dir = os.getenv('EXTREME_OWNER_FIXTURE_DIR')
    if not out_dir or getattr(impl, '_extreme_owner_fixture_done', False):
        return
    if layer_name != 'model.layers.2.self_attn.attn' or tuple(x.shape) != (96, 4096):
        return
    context = get_forward_context()
    if getattr(context, 'is_draft_model', False):
        return
    if not context.additional_kwargs.get('extreme_owner_fixture_runtime', False):
        return
    impl._extreme_owner_fixture_done = True
    rank = int(impl.tp_rank)
    req0 = (rank * 12) // 8
    req1 = (rank * 12 + 11) // 8
    if req1 != req0 + 1:
        raise RuntimeError(f'expected two adjacent owners: rank{rank} {req0},{req1}')
    qsl_cpu = qsl.cpu().tolist()
    if qsl_cpu != list(range(0, 97, 8)):
        raise RuntimeError(f'owner fixture qsl changed: {qsl_cpu}')
    (_, _, state_meta, scale_meta, _) = attn_metadata
    state_req = state_meta.req_metadata
    scale_req = scale_meta.req_metadata
    state, key, scale, _ = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
    (state_a, key_a, scale_a), raw_a = _private_alias_views(state, key, scale)
    (state_a2, key_a2, scale_a2), raw_a2 = _private_alias_views(state, key, scale)
    (state_b, key_b, scale_b), raw_b = _private_alias_views(state, key, scale)
    (state_a3, key_a3, scale_a3), raw_a3 = _private_alias_views(state, key, scale)
    full_cos, full_sin, full_slots = impl._compute_compressor_metadata(scale_req)
    a = _compress(impl, x, state_a, state_req.block_table, qsl,
                  scale_req.start_pos, full_cos, full_sin)
    a2 = _compress(impl, x, state_a2, state_req.block_table, qsl,
                   scale_req.start_pos, full_cos, full_sin)
    req_slice = slice(req0, req1+1)
    x_b = x[req0*8:(req1+1)*8].contiguous()
    qsl_b = (qsl[req0:req1+2] - qsl[req0]).contiguous()
    start_b = scale_req.start_pos[req_slice].contiguous()
    state_bt_b = state_req.block_table[req_slice].contiguous()
    scale_bt_b = scale_req.block_table[req_slice].contiguous()
    ncmp = min(16, 16 // impl.compress_ratio + 2)
    b_cos, b_sin, b_slots = torch.ops._C_ascend.compressor_metadata(
        scale_req.full_compress_cos.view(scale_req.full_compress_cos.shape[0],-1),
        scale_req.full_compress_sin.view(scale_req.full_compress_sin.shape[0],-1),
        qsl_b, start_b, scale_bt_b, scale_req.block_size,
        DeviceOperator.get_dsa_compressor_slot_mapping_format(),
        impl.compress_ratio, ncmp, 2)
    b = _compress(impl, x_b, state_b, state_bt_b, qsl_b,
                  start_b, b_cos, b_sin)
    a3 = _compress(impl, x, state_a3, state_req.block_table, qsl,
                   scale_req.start_pos, full_cos, full_sin)
    torch.npu.synchronize()
    pages = torch.unique(state_bt_b).long()
    pages = pages[pages > 0]
    total_pages = state.shape[0]
    page_stride_bytes = state.stride(0) * state.element_size()
    state_bytes_per_page = state.shape[1] * state.shape[2] * state.shape[3] * state.element_size()
    if (raw_a.numel() != total_pages * page_stride_bytes or state.storage_offset() != 0 or
            state_bytes_per_page > page_stride_bytes):
        raise RuntimeError('unrecognized shared indexer page layout')
    owner_state_a = raw_a.view(total_pages,page_stride_bytes)[pages,:state_bytes_per_page]
    owner_state_a2 = raw_a2.view(total_pages,page_stride_bytes)[pages,:state_bytes_per_page]
    owner_state_b = raw_b.view(total_pages,page_stride_bytes)[pages,:state_bytes_per_page]
    owner_state_a3 = raw_a3.view(total_pages,page_stride_bytes)[pages,:state_bytes_per_page]
    expected_owner_slots = full_slots[req0*2:(req1+1)*2]
    result = {
        'rank':rank,'layer':layer_name,'owners':[req0,req1],
        'qsl':qsl_cpu,'x_shape':list(x.shape),'owner_x_shape':list(x_b.shape),
        'indexer_typed_abi':[
            {'name':name,'dtype':str(t.dtype),'shape':list(t.shape),'stride':list(t.stride()),
             'storage_offset':t.storage_offset(),'storage_nbytes':t.untyped_storage().nbytes(),
             'same_storage_as_state':t.untyped_storage().data_ptr()==state.untyped_storage().data_ptr()}
            for name,t in (('state',state),('key',key),('scale',scale))],
        'private_storage_nbytes':raw_a.untyped_storage().nbytes(),
        'private_alias_preserved':all(t.untyped_storage().data_ptr()==raw_a.untyped_storage().data_ptr()
                                      for t in (state_a,key_a,scale_a)),
        'full_output_shape':list(a.shape),'owner_output_shape':list(b.shape),
        'full_slot_shape':list(full_slots.shape),'owner_slot_shape':list(b_slots.shape),
        'owner_state_pages':int(pages.numel()),
        'page_stride_bytes':page_stride_bytes,'state_bytes_per_page':state_bytes_per_page,
        'A_A_output':_comparison(a,full_slots,a2,full_slots),
        'A_B_owner_output':_comparison(a,full_slots,b,b_slots,expected_owner_slots),
        'A_after_B_output':_comparison(a,full_slots,a3,full_slots),
        'A_A_full_storage_bytes_exact':bool(torch.equal(raw_a,raw_a2)),
        'A_B_owner_state_bytes_exact':bool(torch.equal(owner_state_a,owner_state_b)),
        'A_after_B_full_storage_bytes_exact':bool(torch.equal(raw_a,raw_a3)),
        'scope':'private Compressor output/state only; no typed key/scale scatter, QLI/Sparse, full Graph or cross-cycle proof',
    }
    dest=Path(out_dir);dest.mkdir(parents=True,exist_ok=True)
    # Run295 attribution: distinguish bytes B actually changed from A-only writes
    # in historical/shared pages. This is a diagnostic mask, not a complete
    # native write-set proof (a write that reproduces the prestate is invisible).
    raw_live = torch.empty(0, dtype=torch.uint8, device=state.device).set_(
        state.untyped_storage(), 0, (raw_a.numel(),), (1,))
    owner_live = raw_live.view(total_pages, page_stride_bytes)[pages, :state_bytes_per_page]
    ab_diff = owner_state_a != owner_state_b
    a_changed = owner_state_a != owner_live
    b_changed = owner_state_b != owner_live
    page_any = ab_diff.any(dim=1)
    first_diff = None
    if bool(page_any.any().item()):
        first_row = int(torch.nonzero(page_any)[0, 0].item())
        first_byte = int(torch.nonzero(ab_diff[first_row])[0, 0].item())
        first_page = int(pages[first_row].item())
        aliases = torch.nonzero(state_req.block_table == first_page).cpu().tolist()
        first_diff = {
            'physical_page': first_page, 'byte_offset_in_state': first_byte,
            'A_byte': int(owner_state_a[first_row, first_byte].item()),
            'B_byte': int(owner_state_b[first_row, first_byte].item()),
            'prestate_byte': int(owner_live[first_row, first_byte].item()),
            'block_table_aliases_request_block': aliases[:32],
            'alias_count': len(aliases),
        }
    bt_cpu = state_req.block_table.cpu().tolist()
    owner_page_ids = {int(v) for req in (req0, req1) for v in bt_cpu[req] if int(v) > 0}
    nonowner_page_ids = {int(v) for req, row in enumerate(bt_cpu)
                         if req not in (req0, req1) for v in row if int(v) > 0}
    result['owner_state_delta'] = {
        'A_B_diff_bytes': int(ab_diff.sum().item()),
        'A_B_diff_pages': int(page_any.sum().item()),
        'B_changed_bytes_vs_prestate': int(b_changed.sum().item()),
        'A_changed_bytes_vs_prestate': int(a_changed.sum().item()),
        'B_changed_bytes_different_in_A': int((b_changed & ab_diff).sum().item()),
        'A_only_changed_bytes': int((a_changed & ~b_changed).sum().item()),
        'B_only_changed_bytes': int((b_changed & ~a_changed).sum().item()),
        'shared_owner_nonowner_block_table_pages': len(owner_page_ids & nonowner_page_ids),
        'first_A_B_diff': first_diff,
        'start_pos_by_request': scale_req.start_pos.cpu().tolist(),
        'state_block_size': int(state_req.block_size),
        'A_B_diff_physical_pages_first256': pages[page_any][:256].cpu().tolist(),
        'B_changed_physical_pages_first256': pages[b_changed.any(dim=1)][:256].cpu().tolist(),
        'A_changed_physical_pages_first256': pages[a_changed.any(dim=1)][:256].cpu().tolist(),
        'mask_limit': 'prestate equality cannot detect a write of identical bytes',
    }
    result['gate_pass'] = (
        result['A_A_output']['matched_values_exact'] and
        result['A_A_output']['complete_slot_coverage'] and
        result['A_A_output']['full_valid_slots']==24 and
        result['A_B_owner_output']['matched_values_exact'] and
        result['A_B_owner_output']['complete_slot_coverage'] and
        result['A_B_owner_output']['other_valid_slots']==4 and
        result['A_after_B_output']['matched_values_exact'] and
        result['A_after_B_output']['complete_slot_coverage'] and
        result['A_A_full_storage_bytes_exact'] and
        result['A_B_owner_state_bytes_exact'] and
        result['A_after_B_full_storage_bytes_exact'])
    (dest/f'rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'EXTREME_OWNER_FIXTURE rank={rank} gate={result["gate_pass"]}',flush=True)
