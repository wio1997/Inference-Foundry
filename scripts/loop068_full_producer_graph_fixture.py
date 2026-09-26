"""Private full producer A/B Graph replay cost screen after eager value gate."""

import json
import os
import statistics
import traceback
from pathlib import Path

import torch

from scripts.loop068_full_producer_fixture import (
    _arm, _cache_slot_snapshot, _current_state_snapshot,
    _owner_metadata, _restore)


def _persistent_owner_snapshot(private_cache, attn_metadata, owners, impl):
    from vllm_ascend.device.device_op import DeviceOperator
    (cmp_cache, swa_cache, main_state, _, _, _) = DeviceOperator.unpack_dsa_forward_kv_cache(
        private_cache, impl.compress_ratio)
    (index_state, index_key, index_scale, _) = DeviceOperator.unpack_dsa_indexer_kv_cache(
        private_cache)
    cmp_meta, main_state_meta, index_state_meta, index_scale_meta, swa_meta = attn_metadata
    qsl = cmp_meta.req_metadata.query_start_loc
    request_slice = slice(owners[0], owners[1] + 1)
    swa_slots = swa_meta.req_metadata.slot_mapping[owners[0] * 8:(owners[1] + 1) * 8]
    main_slots = _owner_metadata(impl, cmp_meta, qsl, request_slice, 4)[-1]
    index_slots = _owner_metadata(impl, index_scale_meta, qsl, request_slice, 4)[-1]
    snapshot = {}
    for label, cache, slots in (
            ('swa', swa_cache, swa_slots), ('main', cmp_cache, main_slots),
            ('index_key', index_key, index_slots), ('index_scale', index_scale, index_slots)):
        snapshot[label], _ = _cache_slot_snapshot(cache, slots)
    snapshot['index_state'], _ = _current_state_snapshot(
        index_state, index_state_meta, index_scale_meta.req_metadata.start_pos, owners)
    snapshot['main_state'], _ = _current_state_snapshot(
        main_state, main_state_meta, cmp_meta.req_metadata.start_pos, owners)
    return snapshot


def _params(impl, attn_metadata, owners):
    cmp_meta, main_state_meta, index_state_meta, index_scale_meta, _ = attn_metadata
    qsl = cmp_meta.req_metadata.query_start_loc
    request_slice = slice(owners[0], owners[1] + 1)
    icos, isin, islots = impl._compute_compressor_metadata(index_scale_meta.req_metadata)
    mcos, msin, mslots = impl._compute_compressor_metadata(cmp_meta.req_metadata)
    full_index = (qsl, index_scale_meta.req_metadata.start_pos,
                  index_state_meta.req_metadata.block_table, icos, isin, islots)
    full_main = (qsl, cmp_meta.req_metadata.start_pos,
                 main_state_meta.req_metadata.block_table, mcos, msin, mslots)
    owner_index = list(_owner_metadata(impl, index_scale_meta, qsl, request_slice, 4))
    owner_index[2] = index_state_meta.req_metadata.block_table[request_slice].contiguous()
    owner_main = list(_owner_metadata(impl, cmp_meta, qsl, request_slice, 4))
    owner_main[2] = main_state_meta.req_metadata.block_table[request_slice].contiguous()
    return {'A': (full_index, full_main), 'B': (tuple(owner_index), tuple(owner_main))}


def graph_screen(impl, *, private_cache, groups, x_full, x_local, q, qr,
                 q_per_token_scale, attn_metadata, layer_name, owners,
                 eager_sparse, eager_topk, eager_snapshot):
    rank = int(impl.tp_rank)
    out_dir = Path(os.environ['EXTREME_FULL_PRODUCER_GRAPH_DIR'])
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {'rank': rank, 'stage': 'preflight', 'pass': False,
              'scope': 'one real layer2 private Graph full producer through Sparse; no live rank rendezvous or Product E2E',
              'sequence': ['A', 'B', 'A2'], 'warmup_triplets': 3, 'measured_triplets': 10}
    graphs = {}
    try:
        params = _params(impl, attn_metadata, owners)
        result['memory_before_capture_bytes'] = int(torch.npu.memory_allocated())
        for name in ('A', 'B'):
            _restore(groups)
            torch.npu.synchronize()
            index_params, main_params = params[name]
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = _arm(
                    impl, private_cache, owner=(name == 'B'),
                    x_full=x_full, x_local=x_local, q=q, qr=qr,
                    q_per_token_scale=q_per_token_scale,
                    attn_metadata=attn_metadata, layer_name=layer_name,
                    owners=owners, prepared_index=index_params,
                    prepared_main=main_params)
            graphs[name] = (graph, output['sparse'], output['topk'])
        torch.npu.synchronize()
        result['memory_after_capture_bytes'] = int(torch.npu.memory_allocated())
        samples = []
        for triplet in range(13):
            for arm_name in ('A', 'B', 'A2'):
                key = 'B' if arm_name == 'B' else 'A'
                graph, sparse, topk = graphs[key]
                _restore(groups)
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                sparse_exact = bool(torch.equal(sparse, eager_sparse))
                topk_exact = bool(torch.equal(topk, eager_topk))
                persistent = _persistent_owner_snapshot(
                    private_cache, attn_metadata, owners, impl)
                persistent_exact = {key: bool(torch.equal(value, eager_snapshot[key]))
                                    for key, value in persistent.items()}
                samples.append({'triplet': triplet, 'arm': arm_name,
                                'warmup': triplet < 3,
                                'replay_ms': start.elapsed_time(end),
                                'sparse_exact_to_eager_A': sparse_exact,
                                'topk_exact_to_eager_A': topk_exact,
                                'persistent_owner_exact_to_eager_A': persistent_exact})
                if not sparse_exact or not topk_exact or not all(persistent_exact.values()):
                    raise RuntimeError(f'Graph {arm_name} value/persistent parity failed triplet{triplet}')
        measured = [x for x in samples if not x['warmup']]
        result['samples'] = samples
        result['median_replay_ms_by_arm'] = {
            name: statistics.median(x['replay_ms'] for x in measured if x['arm'] == name)
            for name in ('A', 'B', 'A2')}
        result['strict_B_faster_than_both_A_pairs'] = sum(
            next(x['replay_ms'] for x in measured if x['triplet'] == triplet and x['arm'] == 'B') <
            min(x['replay_ms'] for x in measured if x['triplet'] == triplet and x['arm'] in ('A', 'A2'))
            for triplet in range(3, 13))
        result['stage'] = 'complete'
        result['pass'] = True
    except Exception as exc:
        result['stage'] = 'error'
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()[-6000:]
    finally:
        try:
            torch.npu.synchronize()
        except Exception as exc:
            result['final_sync_error'] = repr(exc)
            result['pass'] = False
        try:
            result['capture_still_active'] = bool(torch.npu.is_current_stream_capturing())
            if result['capture_still_active']:
                result['pass'] = False
        except Exception as exc:
            result['capture_state_error'] = repr(exc)
            result['pass'] = False
        graphs.clear()
        (out_dir / f'rank{rank}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(f'EXTREME_FULL_PRODUCER_GRAPH rank={rank} stage={result["stage"]} pass={result["pass"]}', flush=True)
    if not result['pass']:
        raise RuntimeError(f'private full producer Graph failed rank{rank}: {result.get("error")}')
