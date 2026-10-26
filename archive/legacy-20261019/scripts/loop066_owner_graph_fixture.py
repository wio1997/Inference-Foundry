"""Private real-layer c4 owner16 Graph replay screen after typed consumer parity."""
import json
import os
import statistics
import traceback
from pathlib import Path

import torch

from scripts.loop066_owner_fixture import _compress
from scripts.loop066_owner_consumer_fixture import (
    complete_owner_consumer, _state_domains, _domain_bytes, _slot_cache_compare)
from scripts.loop066_owner_timing_fixture import prepare_timing


def prepare_graph(impl, *, x, kv_cache, attn_metadata, qsl, layer_name):
    prepare_timing(impl, x=x, kv_cache=kv_cache,
                   attn_metadata=attn_metadata, qsl=qsl,
                   layer_name=layer_name)
    setup = getattr(impl, '_extreme_owner_timing_setup', None)
    if setup is not None:
        (_, _, state_meta, scale_meta, _) = attn_metadata
        setup['state_req'] = state_meta.req_metadata
        setup['starts'] = scale_meta.req_metadata.start_pos.cpu().tolist()
        setup['owners'] = [(int(impl.tp_rank) * 12) // 8,
                           (int(impl.tp_rank) * 12 + 11) // 8]


def _chain(impl, arm, params, hadamard, q, q_scale, weights,
           metadata, block_table, local_qsl, local_ksl):
    from vllm_ascend.device.device_op import DeviceOperator
    x, state_bt, qsl, start, cos, sin, slots = params
    kv = _compress(impl, x, arm['state'], state_bt, qsl, start, cos, sin)
    if impl.indexer.compressor.rotate:
        from vllm_ascend.attention.context_parallel.dsa_cp import rotate_activation
        kv = rotate_activation(kv, hadamard)
    _, kv_scale = DeviceOperator.indexer_quant_scatter_part1(
        kv, arm['key'], None, slots)
    if kv_scale is not None:
        DeviceOperator.dsa_indexer_scatter_scale_part3(kv_scale, arm['scale'], slots)
    topk, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(
        query=q, key=arm['key'], weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(arm['scale']),
        actual_seq_lengths_query=local_qsl,
        actual_seq_lengths_key=local_ksl,
        block_table=block_table, metadata=metadata,
        query_quant_mode=0, key_quant_mode=0,
        layout_query='TND', layout_key='PA_BSND',
        sparse_count=impl.index_topk, sparse_mode=3,
        pre_tokens=(1 << 63) - 1, next_tokens=(1 << 63) - 1,
        cmp_ratio=4, return_value=False)
    return topk, kv


def complete_graph(impl, *, q, q_scale, weights, qli_metadata,
                   block_table, local_qsl, local_ksl):
    from vllm.forward_context import get_forward_context
    from vllm_ascend.device.device_op import DeviceOperator
    setup = getattr(impl, '_extreme_owner_timing_setup', None)
    complete_owner_consumer(impl, q=q, q_scale=q_scale, weights=weights,
                            qli_metadata=qli_metadata, block_table=block_table,
                            local_qsl=local_qsl, local_ksl=local_ksl)
    if setup is None or getattr(impl, '_extreme_owner_graph_done', False):
        return
    impl._extreme_owner_graph_done = True
    rank = setup['rank']
    parity_path = Path(os.environ['EXTREME_OWNER_CONSUMER_FIXTURE_DIR']) / f'rank{rank}.json'
    parity = json.loads(parity_path.read_text())
    result = {'rank': rank, 'stage': 'preflight', 'pass': False,
              'parity_gate_pass': bool(parity.get('gate_pass')),
              'scope': 'private same-prestate real-layer2 Graph replay; not product FULL Graph or E2E',
              'sequence': ['A', 'B', 'A2'],
              'warmup_triplets': 3, 'measured_triplets': 10}
    out_dir = Path(os.environ['EXTREME_OWNER_GRAPH_DIR'])
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs = {}
    try:
        context = get_forward_context()
        if not result['parity_gate_pass'] or bool(getattr(context, 'capturing', False)) or \
                torch.npu.is_current_stream_capturing():
            raise RuntimeError('invalid parity or nested/startup capture context')
        if tuple(q.shape[:1]) != (12,) or local_qsl.numel() != 13:
            raise RuntimeError('real local QLI geometry changed')
        prepared_weights = DeviceOperator.prepare_dsa_indexer_weights(weights)
        prepared_query_scale = DeviceOperator.prepare_dsa_indexer_query_scale(q_scale)
        qsl_qli = local_qsl[1:].clone()
        ksl_qli = local_ksl.clone()
        block_table_static = block_table.clone()
        meta_pre = qli_metadata.clone()
        meta = {name: meta_pre.clone() for name in ('A', 'B', 'A2')}
        result['memory_before_capture_bytes'] = int(torch.npu.memory_allocated())
        # Independent eager full96 oracle on fourth private bank.
        oracle_arm = setup['arms']['A3']
        oracle_arm['raw'].copy_(setup['raw_prestate'])
        torch.npu.synchronize()
        eager_topk, eager_kv = _chain(
            impl, oracle_arm, setup['full'], setup['hadamard'],
            q, prepared_query_scale, prepared_weights, meta_pre.clone(),
            block_table_static, qsl_qli, ksl_qli)
        torch.npu.synchronize()
        eager_topk = eager_topk.clone()
        result['eager_topk_shape'] = list(eager_topk.shape)
        result['eager_kv_shape'] = list(eager_kv.shape)
        if result['eager_topk_shape'] != [12, 1, 512]:
            raise RuntimeError('eager oracle topk shape changed')
        # Capture is allowed to write private state; each capture starts from
        # the immutable prestate and each later replay restores it again.
        for name in ('A', 'B', 'A2'):
            arm = setup['arms'][name]
            arm['raw'].copy_(setup['raw_prestate'])
            meta[name].copy_(meta_pre)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                graph_topk, graph_kv = _chain(
                    impl, arm, setup['owner'] if name == 'B' else setup['full'],
                    setup['hadamard'], q, prepared_query_scale, prepared_weights,
                    meta[name], block_table_static, qsl_qli, ksl_qli)
            graphs[name] = {'graph': graph, 'topk': graph_topk, 'kv': graph_kv}
        torch.npu.synchronize()
        result['memory_after_capture_bytes'] = int(torch.npu.memory_allocated())
        result['captured_kv_shapes'] = {name: list(v['kv'].shape) for name, v in graphs.items()}
        result['captured_topk_shapes'] = {name: list(v['topk'].shape) for name, v in graphs.items()}
        samples = []
        for triplet in range(13):
            for name in ('A', 'B', 'A2'):
                arm = setup['arms'][name]
                arm['raw'].copy_(setup['raw_prestate'])
                meta[name].copy_(meta_pre)
                torch.npu.synchronize()
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                graphs[name]['graph'].replay()
                end.record()
                end.synchronize()
                exact = bool(torch.equal(graphs[name]['topk'], eager_topk))
                samples.append({'triplet': triplet, 'arm': name,
                                'warmup': triplet < 3,
                                'replay_ms': start.elapsed_time(end),
                                'topk_exact_to_eager_A3': exact})
        result['samples'] = samples
        measured = [s for s in samples if not s['warmup']]
        result['median_replay_ms_by_arm'] = {
            name: statistics.median(s['replay_ms'] for s in measured if s['arm'] == name)
            for name in ('A', 'B', 'A2')}
        domains = _state_domains(setup['state_req'], setup['starts'], setup['owners'],
                                 oracle_arm['state'], oracle_arm['raw'])
        result['post_replay_state_domains'] = {}
        for name, domain in domains.items():
            oracle_bytes = _domain_bytes(oracle_arm['raw'], domain)
            result['post_replay_state_domains'][name] = {
                arm_name: bool(torch.equal(oracle_bytes, _domain_bytes(setup['arms'][arm_name]['raw'], domain)))
                for arm_name in ('A', 'B', 'A2')}
        result['post_replay_full_raw_self_exact'] = all(
            bool(torch.equal(oracle_arm['raw'], setup['arms'][name]['raw']))
            for name in ('A', 'A2'))
        result['owner_typed_slot'] = _slot_cache_compare(
            oracle_arm['key'], oracle_arm['scale'],
            setup['arms']['B']['key'], setup['arms']['B']['scale'],
            setup['owner'][-1])
        result['original_qli_metadata_unchanged'] = bool(torch.equal(qli_metadata, meta_pre))
        result['pass'] = (
            all(s['topk_exact_to_eager_A3'] for s in samples) and
            all(shape == [12, 1, 512] for shape in result['captured_topk_shapes'].values()) and
            all(all(v.values()) for v in result['post_replay_state_domains'].values()) and
            result['post_replay_full_raw_self_exact'] and
            result['owner_typed_slot']['key_scale_exact'] and
            result['original_qli_metadata_unchanged'])
        result['stage'] = 'complete'
    except Exception as exc:
        result['stage'] = 'error'
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()[-4000:]
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
        print(f'EXTREME_OWNER_GRAPH rank={rank} stage={result["stage"]} pass={result["pass"]}', flush=True)
        impl._extreme_owner_timing_setup = None
    if (result['stage'] == 'error' or result.get('final_sync_error') or
            result.get('capture_state_error') or result.get('capture_still_active')):
        raise RuntimeError(f'private Graph device/capture failure rank{rank}: '
                           f'{result.get("error", result.get("final_sync_error", result.get("capture_state_error")))}')
