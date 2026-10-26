"""Private real-layer owner16 state/scatter/QLI gate; never writes live cache."""
import json
import os
import traceback
from pathlib import Path

import torch

from scripts.loop066_owner_fixture import (_private_alias_views, _compress,
                                           _comparison, _rows_by_slot)


def _save(rank, result):
    path = Path(os.environ['EXTREME_OWNER_CONSUMER_FIXTURE_DIR'])
    path.mkdir(parents=True, exist_ok=True)
    (path / f'rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'EXTREME_OWNER_CONSUMER rank={rank} stage={result.get("stage")} gate={result.get("gate_pass")}', flush=True)


def _state_domains(state_req, starts, owners, state, raw):
    bt = state_req.block_table.cpu().tolist()
    block_size = int(state_req.block_size)
    page_stride = state.stride(0) * state.element_size()
    state_bytes = state.shape[1] * state.shape[2] * state.shape[3] * state.element_size()
    if not (block_size == 2 and state_bytes == 4096 and page_stride == 4160 and state.storage_offset() == 0):
        raise RuntimeError(f'unexpected state ABI: block={block_size} state={state_bytes} stride={page_stride}')
    if raw.numel() != state.shape[0] * page_stride:
        raise RuntimeError('private state backing size changed')
    per_token = state_bytes // block_size
    domains = {}
    for name, relative in (('write', range(0, 8)), ('live_read', range(-8, 0))):
        coords = set()
        zero_pages = 0
        for req in owners:
            start = int(starts[req])
            for delta in relative:
                pos = start + delta
                if pos < 0:
                    continue
                logical_block = pos // block_size
                if logical_block >= len(bt[req]):
                    raise RuntimeError(f'state table too short req{req} block{logical_block}')
                page = int(bt[req][logical_block])
                if page < 0 or page >= state.shape[0]:
                    raise RuntimeError(f'invalid state page req{req} block{logical_block} page{page}')
                if page == 0:
                    zero_pages += 1
                coords.add((page, pos % block_size))
        domains[name] = {'coords': sorted(coords), 'page0_positions': zero_pages,
                         'per_token_bytes': per_token, 'page_stride_bytes': page_stride}
    return domains


def _domain_bytes(raw, domain):
    per_token = domain['per_token_bytes']
    stride = domain['page_stride_bytes']
    return torch.cat([raw.narrow(0, page * stride + index * per_token, per_token)
                      for page, index in domain['coords']])


def _slot_cache_compare(key_a, scale_a, key_b, scale_b, slots):
    b_rows, duplicates = _rows_by_slot(slots)
    if duplicates or len(b_rows) != 4:
        raise RuntimeError(f'owner slot mapping invalid: {len(b_rows)} {duplicates[:3]}')
    mismatches = []
    for slot in sorted(b_rows):
        if len(slot) != 2:
            raise RuntimeError(f'unexpected slot format {slot}')
        page, offset = slot
        if page >= key_a.shape[0] or offset >= key_a.shape[1]:
            raise RuntimeError(f'owner slot out of range {slot}')
        if not torch.equal(key_a[page, offset], key_b[page, offset]) or not torch.equal(
                scale_a[page, offset], scale_b[page, offset]):
            mismatches.append(list(slot))
    return {'valid_owner_slots': len(b_rows), 'key_scale_exact': not mismatches,
            'mismatches': mismatches[:8]}


def prepare_owner_consumer(impl, *, x, kv_cache, attn_metadata, qsl, layer_name):
    from vllm.forward_context import get_forward_context
    from vllm_ascend.device.device_op import DeviceOperator
    from vllm_ascend.attention.context_parallel.dsa_cp import rotate_activation
    out = os.getenv('EXTREME_OWNER_CONSUMER_FIXTURE_DIR')
    if not out or getattr(impl, '_extreme_owner_consumer_done', False) or getattr(
            impl, '_extreme_owner_consumer_pending', None) is not None:
        return
    if layer_name != 'model.layers.2.self_attn.attn' or tuple(x.shape) != (96, 4096):
        return
    context = get_forward_context()
    if getattr(context, 'is_draft_model', False) or not context.additional_kwargs.get(
            'extreme_owner_consumer_runtime', False):
        return
    rank = int(impl.tp_rank)
    result = {'rank': rank, 'layer': layer_name, 'stage': 'prepare', 'gate_pass': False,
              'scope': 'private eager one-layer same-prestate state/scatter/QLI only'}
    try:
        owners = [(rank * 12) // 8, (rank * 12 + 11) // 8]
        if owners[1] != owners[0] + 1 or qsl.cpu().tolist() != list(range(0, 97, 8)):
            raise RuntimeError('frozen owner/qsl geometry changed')
        result['owners'] = owners
        (_, _, state_meta, scale_meta, _) = attn_metadata
        state_req, scale_req = state_meta.req_metadata, scale_meta.req_metadata
        state, key, scale, full = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        if full is not None or len(kv_cache) != 6:
            raise RuntimeError('fixture supports current non-A5 six-tensor cache ABI only')
        arms = []
        for name in ('A', 'A2', 'B', 'A3'):
            (private_state, private_key, private_scale), raw = _private_alias_views(state, key, scale)
            arms.append({'name': name, 'state': private_state, 'key': private_key,
                         'scale': private_scale, 'raw': raw})
        full_cos, full_sin, full_slots = impl._compute_compressor_metadata(scale_req)
        req_slice = slice(owners[0], owners[1] + 1)
        x_b = x[owners[0] * 8:(owners[1] + 1) * 8].contiguous()
        qsl_b = (qsl[owners[0]:owners[1] + 2] - qsl[owners[0]]).contiguous()
        start_b = scale_req.start_pos[req_slice].contiguous()
        state_bt_b = state_req.block_table[req_slice].contiguous()
        scale_bt_b = scale_req.block_table[req_slice].contiguous()
        ncmp = min(16, 16 // impl.compress_ratio + 2)
        b_cos, b_sin, b_slots = torch.ops._C_ascend.compressor_metadata(
            scale_req.full_compress_cos.view(scale_req.full_compress_cos.shape[0], -1),
            scale_req.full_compress_sin.view(scale_req.full_compress_sin.shape[0], -1),
            qsl_b, start_b, scale_bt_b, scale_req.block_size,
            DeviceOperator.get_dsa_compressor_slot_mapping_format(),
            impl.compress_ratio, ncmp, 2)
        for arm in arms:
            is_b = arm['name'] == 'B'
            arm['slots'] = b_slots if is_b else full_slots
            arm['kv'] = _compress(impl, x_b if is_b else x, arm['state'],
                                  state_bt_b if is_b else state_req.block_table,
                                  qsl_b if is_b else qsl,
                                  start_b if is_b else scale_req.start_pos,
                                  b_cos if is_b else full_cos,
                                  b_sin if is_b else full_sin)
        torch.npu.synchronize()
        a, a2, b, a3 = arms
        result['output_A_A2'] = _comparison(a['kv'], full_slots, a2['kv'], full_slots)
        result['output_A_B'] = _comparison(a['kv'], full_slots, b['kv'], b_slots,
                                            full_slots[owners[0] * 2:(owners[1] + 1) * 2])
        result['output_A_A3'] = _comparison(a['kv'], full_slots, a3['kv'], full_slots)
        result['pre_scatter_full_raw_self_exact'] = bool(torch.equal(a['raw'], a2['raw']) and
                                                        torch.equal(a['raw'], a3['raw']))
        starts = scale_req.start_pos.cpu().tolist()
        domains = _state_domains(state_req, starts, owners, state, a['raw'])
        result['state_domains'] = {}
        for name, domain in domains.items():
            a_bytes = _domain_bytes(a['raw'], domain)
            result['state_domains'][name] = {
                'token_locations': len(domain['coords']), 'page0_positions': domain['page0_positions'],
                'A_B_exact': bool(torch.equal(a_bytes, _domain_bytes(b['raw'], domain))),
                'A_A2_exact': bool(torch.equal(a_bytes, _domain_bytes(a2['raw'], domain))),
                'A_A3_exact': bool(torch.equal(a_bytes, _domain_bytes(a3['raw'], domain))),
            }
        result['state_block_size'] = int(state_req.block_size)
        result['start_pos_by_request'] = starts
        result['storage_nbytes'] = a['raw'].numel()
        if impl.indexer.compressor.rotate:
            for arm in arms:
                arm['kv'] = rotate_activation(arm['kv'], scale_meta.hadamard)
        for arm in arms:
            _, kv_scale = DeviceOperator.indexer_quant_scatter_part1(
                arm['kv'], arm['key'], None, arm['slots'])
            if kv_scale is not None:
                DeviceOperator.dsa_indexer_scatter_scale_part3(
                    kv_scale, arm['scale'], arm['slots'])
        torch.npu.synchronize()
        result['post_scatter_full_raw_self_exact'] = bool(torch.equal(a['raw'], a2['raw']) and
                                                         torch.equal(a['raw'], a3['raw']))
        result['post_scatter_state_domains'] = {}
        for name, domain in domains.items():
            a_bytes = _domain_bytes(a['raw'], domain)
            result['post_scatter_state_domains'][name] = {
                'A_B_exact': bool(torch.equal(a_bytes, _domain_bytes(b['raw'], domain))),
                'A_A2_exact': bool(torch.equal(a_bytes, _domain_bytes(a2['raw'], domain))),
                'A_A3_exact': bool(torch.equal(a_bytes, _domain_bytes(a3['raw'], domain))),
            }
        result['owner_typed_slot'] = _slot_cache_compare(a['key'], a['scale'],
                                                          b['key'], b['scale'], b_slots)
        impl._extreme_owner_consumer_pending = {'arms': arms, 'result': result}
    except Exception as exc:
        result['stage'] = 'prepare_error'
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()[-4000:]
        impl._extreme_owner_consumer_done = True
        _save(rank, result)


def complete_owner_consumer(impl, *, q, q_scale, weights, qli_metadata,
                            block_table, local_qsl, local_ksl):
    from vllm_ascend.device.device_op import DeviceOperator
    pending = getattr(impl, '_extreme_owner_consumer_pending', None)
    if pending is None or getattr(impl, '_extreme_owner_consumer_done', False):
        return
    rank = int(impl.tp_rank)
    result, arms = pending['result'], pending['arms']
    try:
        if not isinstance(qli_metadata, torch.Tensor):
            raise RuntimeError('QLI metadata is not a tensor')
        qsl_values = local_qsl.cpu().tolist()
        ksl_values = local_ksl.cpu().tolist()
        expected_delta = [max(0, min(8 * (req + 1), 12 * rank + 12) -
                              max(8 * req, 12 * rank)) for req in range(12)]
        actual_delta = [right - left for left, right in zip(qsl_values, qsl_values[1:])]
        if (len(qsl_values) != 13 or qsl_values[0] != 0 or qsl_values[-1] != 12 or
                actual_delta != expected_delta or q.shape[0] != 12 or
                len(ksl_values) != 12 or any(ksl_values[req] <= 0 for req in result['owners'])):
            raise RuntimeError(f'unexpected real local QLI geometry q={tuple(q.shape)} qsl={qsl_values} ksl={ksl_values}')
        result['qli_local_qsl'] = qsl_values
        result['qli_local_ksl'] = ksl_values
        result['qli_expected_query_delta'] = expected_delta
        meta_before = qli_metadata.clone()
        prepared_weights = DeviceOperator.prepare_dsa_indexer_weights(weights)
        out_by_name = {}
        for arm in arms:
            topk, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
                query=q.clone(), key=arm['key'], weights=prepared_weights.clone(),
                query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale).clone(),
                key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(arm['scale']),
                actual_seq_lengths_query=local_qsl[1:].clone(),
                actual_seq_lengths_key=local_ksl.clone(),
                block_table=block_table.clone(), metadata=qli_metadata.clone(),
                query_quant_mode=0, key_quant_mode=0, layout_query='TND', layout_key='PA_BSND',
                sparse_count=impl.index_topk, sparse_mode=3,
                pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
                cmp_ratio=4, return_value=False)
            out_by_name[arm['name']] = topk.clone()
        torch.npu.synchronize()
        result['qli_output_shape'] = list(out_by_name['A'].shape)
        if out_by_name['A'].numel() == 0 or out_by_name['A'].shape[0] != 12:
            raise RuntimeError(f'empty or unexpected QLI output {tuple(out_by_name["A"].shape)}')
        result['qli_A_A2_exact'] = bool(torch.equal(out_by_name['A'], out_by_name['A2']))
        result['qli_A_B_exact'] = bool(torch.equal(out_by_name['A'], out_by_name['B']))
        result['qli_A_A3_exact'] = bool(torch.equal(out_by_name['A'], out_by_name['A3']))
        result['original_qli_metadata_unchanged'] = bool(torch.equal(meta_before, qli_metadata))
        result['stage'] = 'complete'
        result['gate_pass'] = (
            all(result[f'output_{pair}']['matched_values_exact'] and
                result[f'output_{pair}']['complete_slot_coverage']
                for pair in ('A_A2', 'A_B', 'A_A3')) and
            result['pre_scatter_full_raw_self_exact'] and
            result['post_scatter_full_raw_self_exact'] and
            all(v['A_B_exact'] and v['A_A2_exact'] and v['A_A3_exact']
                for v in result['state_domains'].values()) and
            all(v['A_B_exact'] and v['A_A2_exact'] and v['A_A3_exact']
                for v in result['post_scatter_state_domains'].values()) and
            result['owner_typed_slot']['key_scale_exact'] and
            result['qli_A_A2_exact'] and result['qli_A_B_exact'] and
            result['qli_A_A3_exact'] and result['original_qli_metadata_unchanged'])
    except Exception as exc:
        result['stage'] = 'complete_error'
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()[-4000:]
        result['gate_pass'] = False
    impl._extreme_owner_consumer_done = True
    impl._extreme_owner_consumer_pending = None
    _save(rank, result)
