"""Real layer2 private owner16 chain timing after Run296 semantic closure.

All stores are private, reset from the same prestate outside event windows. This
is a native-chain capacity screen, not a live-path or Product timing sample.
"""
import json
import os
import statistics
import traceback
from pathlib import Path

import torch

from scripts.loop066_owner_fixture import _private_alias_views, _compress
from scripts.loop066_owner_consumer_fixture import prepare_owner_consumer, complete_owner_consumer


def prepare_timing(impl, *, x, kv_cache, attn_metadata, qsl, layer_name):
    from vllm_ascend.device.device_op import DeviceOperator
    prepare_owner_consumer(impl, x=x, kv_cache=kv_cache,
                           attn_metadata=attn_metadata, qsl=qsl,
                           layer_name=layer_name)
    pending = getattr(impl, '_extreme_owner_consumer_pending', None)
    if pending is None:
        return
    rank = int(impl.tp_rank)
    owners = pending['result']['owners']
    (_, _, state_meta, scale_meta, _) = attn_metadata
    state_req, scale_req = state_meta.req_metadata, scale_meta.req_metadata
    state, key, scale, _ = DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
    (_, _, _), raw_prestate = _private_alias_views(state, key, scale)
    full_cos, full_sin, full_slots = impl._compute_compressor_metadata(scale_req)
    req_slice = slice(owners[0], owners[1] + 1)
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
    if not torch.equal(b_slots, pending['arms'][2]['slots']):
        raise RuntimeError('timing owner metadata differs from parity fixture')
    impl._extreme_owner_timing_setup = {
        'rank': rank, 'arms': {a['name']: a for a in pending['arms']},
        'raw_prestate': raw_prestate,
        'full': (x, state_req.block_table, qsl, scale_req.start_pos, full_cos, full_sin, full_slots),
        'owner': (x[owners[0] * 8:(owners[1] + 1) * 8].contiguous(),
                  state_bt_b, qsl_b, start_b, b_cos, b_sin, b_slots),
        'hadamard': scale_meta.hadamard,
    }


def complete_timing(impl, *, q, q_scale, weights, qli_metadata,
                    block_table, local_qsl, local_ksl):
    from vllm_ascend.device.device_op import DeviceOperator
    setup = getattr(impl, '_extreme_owner_timing_setup', None)
    complete_owner_consumer(impl, q=q, q_scale=q_scale, weights=weights,
                            qli_metadata=qli_metadata, block_table=block_table,
                            local_qsl=local_qsl, local_ksl=local_ksl)
    if setup is None or getattr(impl, '_extreme_owner_timing_done', False):
        return
    impl._extreme_owner_timing_done = True
    rank = setup['rank']
    parity_path = Path(os.environ['EXTREME_OWNER_CONSUMER_FIXTURE_DIR']) / f'rank{rank}.json'
    parity = json.loads(parity_path.read_text())
    result = {'rank': rank, 'stage': 'timing', 'pass': False,
              'scope': 'private real-layer2 eager chain event timing; no live-path or product speed inference',
              'parity_gate_pass': bool(parity.get('gate_pass')),
              'warmup_triplets': 2, 'measured_triplets': 5,
              'sequence': ['A', 'B', 'A2']}
    out_dir = Path(os.environ['EXTREME_OWNER_TIMING_DIR'])
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not result['parity_gate_pass']:
            raise RuntimeError('Run296 semantic gate failed in timing run')
        query_weights = DeviceOperator.prepare_dsa_indexer_weights(weights)
        query_scale = DeviceOperator.prepare_dsa_indexer_query_scale(q_scale)
        qsl_qli = local_qsl[1:].clone()
        ksl_qli = local_ksl.clone()
        results = []
        baseline_topk = None
        for triplet in range(7):
            for arm_name in ('A', 'B', 'A2'):
                arm = setup['arms'][arm_name]
                arm['raw'].copy_(setup['raw_prestate'])
                torch.npu.synchronize()
                params = setup['owner'] if arm_name == 'B' else setup['full']
                x, state_bt, qsl, start, cos, sin, slots = params
                metadata_copy = qli_metadata.clone()
                start_event = torch.npu.Event(enable_timing=True)
                update_end = torch.npu.Event(enable_timing=True)
                qli_end = torch.npu.Event(enable_timing=True)
                start_event.record()
                kv = _compress(impl, x, arm['state'], state_bt, qsl, start, cos, sin)
                if impl.indexer.compressor.rotate:
                    from vllm_ascend.attention.context_parallel.dsa_cp import rotate_activation
                    kv = rotate_activation(kv, setup['hadamard'])
                _, kv_scale = DeviceOperator.indexer_quant_scatter_part1(
                    kv, arm['key'], None, slots)
                if kv_scale is not None:
                    DeviceOperator.dsa_indexer_scatter_scale_part3(
                        kv_scale, arm['scale'], slots)
                update_end.record()
                topk, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
                    query=q, key=arm['key'], weights=query_weights,
                    query_dequant_scale=query_scale,
                    key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(arm['scale']),
                    actual_seq_lengths_query=qsl_qli,
                    actual_seq_lengths_key=ksl_qli,
                    block_table=block_table, metadata=metadata_copy,
                    query_quant_mode=0, key_quant_mode=0,
                    layout_query='TND', layout_key='PA_BSND',
                    sparse_count=impl.index_topk, sparse_mode=3,
                    pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
                    cmp_ratio=4, return_value=False)
                qli_end.record()
                qli_end.synchronize()
                if baseline_topk is None:
                    baseline_topk = topk.clone()
                parity_exact = bool(torch.equal(topk, baseline_topk))
                results.append({'triplet': triplet, 'arm': arm_name,
                                'warmup': triplet < 2,
                                'update_ms': start_event.elapsed_time(update_end),
                                'qli_tail_ms': update_end.elapsed_time(qli_end),
                                'through_qli_ms': start_event.elapsed_time(qli_end),
                                'topk_exact_to_first_A': parity_exact,
                                'topk_shape': list(topk.shape)})
        result['samples'] = results
        measured = [x for x in results if not x['warmup']]
        result['median_ms_by_arm'] = {
            arm: {metric: statistics.median(x[metric] for x in measured if x['arm'] == arm)
                  for metric in ('update_ms', 'qli_tail_ms', 'through_qli_ms')}
            for arm in ('A', 'B', 'A2')}
        result['pass'] = all(x['topk_exact_to_first_A'] and x['topk_shape'] == [12, 1, 512]
                             for x in results)
        result['stage'] = 'complete'
    except Exception as exc:
        result['stage'] = 'error'
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()[-4000:]
    (out_dir / f'rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'EXTREME_OWNER_TIMING rank={rank} stage={result["stage"]} pass={result["pass"]}', flush=True)
    impl._extreme_owner_timing_setup = None
