"""Same-prestate dynamic metadata Graph diagnostic for one real c4 Target layer.

Capture metadata generation and the full DSA producer together, then replay
against the original and a one-token-shifted start position. All producer
writes use private alias-preserving cache storage; live metadata is restored.
"""

import json
import os
import traceback
from pathlib import Path

import torch

from scripts.loop068_full_producer_fixture import (
    _arm, _owner_metadata, _private_cache, _restore,
)
from scripts.loop068_full_producer_graph_fixture import _persistent_owner_snapshot


def _full_native_metadata(metadata, ratio):
    """Bypass the forward-context Python cache during private Graph capture."""
    from vllm_ascend.device.device_op import DeviceOperator
    req = metadata.req_metadata
    return torch.ops._C_ascend.compressor_metadata(
        req.full_compress_cos.view(req.full_compress_cos.shape[0], -1),
        req.full_compress_sin.view(req.full_compress_sin.shape[0], -1),
        req.query_start_loc, req.start_pos, req.block_table,
        req.block_size,
        DeviceOperator.get_dsa_compressor_slot_mapping_format(),
        ratio, req.num_compressed_tokens, req.num_reqs_actual,
    )


def _params(impl, metadata, owners, owner):
    cmp_meta, main_state_meta, index_state_meta, index_scale_meta, _ = metadata
    qsl = cmp_meta.req_metadata.query_start_loc
    req_slice = slice(owners[0], owners[1] + 1)
    if owner:
        index = list(_owner_metadata(impl, index_scale_meta, qsl, req_slice, 4))
        main = list(_owner_metadata(impl, cmp_meta, qsl, req_slice, 4))
        index[2] = index_state_meta.req_metadata.block_table[req_slice].contiguous()
        main[2] = main_state_meta.req_metadata.block_table[req_slice].contiguous()
    else:
        icos, isin, islots = _full_native_metadata(index_scale_meta, 4)
        mcos, msin, mslots = _full_native_metadata(cmp_meta, 4)
        index = [qsl, index_scale_meta.req_metadata.start_pos,
                 index_state_meta.req_metadata.block_table, icos, isin, islots]
        main = [qsl, cmp_meta.req_metadata.start_pos,
                main_state_meta.req_metadata.block_table, mcos, msin, mslots]
    return tuple(index), tuple(main)


def _metadata_exact(impl, metadata, owners, captured, owner):
    cmp_meta, main_state_meta, index_state_meta, index_scale_meta, _ = metadata
    qsl = cmp_meta.req_metadata.query_start_loc
    req_slice = slice(owners[0], owners[1] + 1)
    if owner:
        expected_index = list(_owner_metadata(impl, index_scale_meta, qsl, req_slice, 4))
        expected_main = list(_owner_metadata(impl, cmp_meta, qsl, req_slice, 4))
        expected_index[2] = index_state_meta.req_metadata.block_table[req_slice].contiguous()
        expected_main[2] = main_state_meta.req_metadata.block_table[req_slice].contiguous()
    else:
        icos, isin, islots = _full_native_metadata(index_scale_meta, 4)
        mcos, msin, mslots = _full_native_metadata(cmp_meta, 4)
        expected_index = [qsl, index_scale_meta.req_metadata.start_pos,
                          index_state_meta.req_metadata.block_table,
                          icos, isin, islots]
        expected_main = [qsl, cmp_meta.req_metadata.start_pos,
                         main_state_meta.req_metadata.block_table,
                         mcos, msin, mslots]
    fields = ('qsl', 'start', 'state_table', 'cos', 'sin', 'slots')
    return {
        f'{which}_{field}': bool(torch.equal(actual, expected))
        for which, actuals, expecteds in (
            ('index', captured[0], expected_index),
            ('main', captured[1], expected_main),
        )
        for field, actual, expected in zip(fields, actuals, expecteds)
    }


def run_probe(impl, *, x_full, x_local, q, qr, q_per_token_scale,
              kv_cache, attn_metadata, layer_name):
    from vllm.forward_context import get_forward_context
    out_dir = os.getenv('EXTREME_DYNAMIC_METADATA_DIR')
    if not out_dir or getattr(impl, '_extreme_dynamic_metadata_done', False):
        return
    if layer_name != 'model.layers.2.self_attn.attn' or tuple(x_full.shape) != (96, 4096):
        return
    context = get_forward_context()
    if (getattr(context, 'capturing', False)
            or torch.npu.is_current_stream_capturing()
            or getattr(context, 'is_draft_model', False)
            or not context.additional_kwargs.get('extreme_dynamic_metadata_runtime', False)):
        return
    impl._extreme_dynamic_metadata_done = True
    rank = int(impl.tp_rank)
    result = {'rank': rank, 'pass': False, 'stage': 'preflight',
              'scope': 'one layer, one real prestate, private Graph with dynamic metadata; not Product E2E'}
    path = Path(out_dir) / f'rank{rank}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    try:
        if (impl.tp_size != 8 or impl.compress_ratio != 4
                or tuple(x_local.shape) != (12, 4096)
                or attn_metadata[0].req_metadata.query_start_loc.cpu().tolist()
                   != list(range(0, 97, 8))):
            raise RuntimeError('frozen c12/TP8/c4 owner geometry changed')
        owners = [(rank * 12) // 8, (rank * 12 + 11) // 8]
        result['owners'] = owners
        private_cache, groups = _private_cache(kv_cache)
        result['private_store_bytes'] = [int(x.numel()) for _, x in groups]
        seen = set()
        for meta in (attn_metadata[0], attn_metadata[3]):
            start = meta.req_metadata.start_pos
            if start.data_ptr() not in seen:
                seen.add(start.data_ptr())
                saved.append((start, start.clone()))
        graphs = {}
        for name, owner in (('A', False), ('B', True)):
            _restore(groups)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                params = _params(impl, attn_metadata, owners, owner)
                output = _arm(
                    impl, private_cache, owner=owner, x_full=x_full,
                    x_local=x_local, q=q, qr=qr,
                    q_per_token_scale=q_per_token_scale,
                    attn_metadata=attn_metadata, layer_name=layer_name,
                    owners=owners, prepared_index=params[0],
                    prepared_main=params[1])
            graphs[name] = (graph, output, params)
        torch.npu.synchronize()
        samples = []
        result['samples'] = samples
        first_owner_metadata = None
        first_reference_metadata = None
        for delta in (0, -1, -4):
            for start, original in saved:
                if delta and int(original.min().item()) < -delta:
                    raise RuntimeError(f'cannot shift start position by {delta}')
                start.copy_(original + delta)
            result.setdefault('start_mod4_by_delta', {})[str(delta)] = [
                start[:12].remainder(4).cpu().tolist() for start, _ in saved]
            fresh_owner = _params(impl, attn_metadata, owners, True)
            fresh_observed = {
                f'{which}_{field}': tensor.clone()
                for which, group in (('index', fresh_owner[0]), ('main', fresh_owner[1]))
                for field, tensor in zip(('cos', 'sin', 'slots'), group[3:])
            }
            if first_reference_metadata is None:
                first_reference_metadata = fresh_observed
            else:
                result.setdefault('reference_metadata_changed_by_delta', {})[str(delta)] = {
                    key: not bool(torch.equal(value, first_reference_metadata[key]))
                    for key, value in fresh_observed.items()
                }
            # Independent eager A oracle from the same private live prestate.
            _restore(groups)
            torch.npu.synchronize()
            eager_params = _params(impl, attn_metadata, owners, False)
            eager_output = _arm(
                impl, private_cache, owner=False, x_full=x_full,
                x_local=x_local, q=q, qr=qr,
                q_per_token_scale=q_per_token_scale,
                attn_metadata=attn_metadata, layer_name=layer_name,
                owners=owners, prepared_index=eager_params[0],
                prepared_main=eager_params[1])
            torch.npu.synchronize()
            eager_snapshot = _persistent_owner_snapshot(
                private_cache, attn_metadata, owners, impl)
            control = {'topk': eager_output['topk'].clone(),
                       'sparse': eager_output['sparse'].clone(),
                       **{k: v.clone() for k, v in eager_snapshot.items()}}
            graph_control = None
            for arm in ('A', 'B', 'A2'):
                graph, output, params = graphs['B' if arm == 'B' else 'A']
                _restore(groups)
                torch.npu.synchronize()
                graph.replay()
                torch.npu.synchronize()
                snapshot = _persistent_owner_snapshot(private_cache, attn_metadata, owners, impl)
                value = {'topk': output['topk'].clone(),
                         'sparse': output['sparse'].clone(),
                         **{k: v.clone() for k, v in snapshot.items()}}
                eager_exact = {key: bool(torch.equal(value[key], control[key]))
                               for key in control}
                if graph_control is None:
                    graph_control = {key: tensor.clone() for key, tensor in value.items()}
                graph_exact = {key: bool(torch.equal(value[key], graph_control[key]))
                               for key in graph_control}
                sparse_diff_eager = (value['sparse'].float() - control['sparse'].float()).abs()
                sparse_diff_graph = (value['sparse'].float() - graph_control['sparse'].float()).abs()
                sparse_first_eager = torch.nonzero(sparse_diff_eager.reshape(-1))
                sparse_first_graph = torch.nonzero(sparse_diff_graph.reshape(-1))
                sparse_finite = bool(torch.isfinite(value['sparse']).all().item())
                sparse_finite_eager = bool(torch.isfinite(control['sparse']).all().item())
                meta_exact = _metadata_exact(
                    impl, attn_metadata, owners, params, arm == 'B')
                if arm == 'B':
                    observed = {
                        f'{which}_{field}': tensor.clone()
                        for which, group in (('index', params[0]), ('main', params[1]))
                        for field, tensor in zip(('cos', 'sin', 'slots'), group[3:])
                    }
                    if first_owner_metadata is None:
                        first_owner_metadata = observed
                    else:
                        result.setdefault('captured_metadata_changed_by_delta', {})[str(delta)] = {
                            key: not bool(torch.equal(value, first_owner_metadata[key]))
                            for key, value in observed.items()
                        }
                samples.append({'delta': delta, 'arm': arm,
                                'value_exact_to_eager_A': eager_exact,
                                'value_exact_to_graph_A': graph_exact,
                                'sparse_max_abs_eager_A': float(sparse_diff_eager.max().item()),
                                'sparse_changed_count_eager_A': int(torch.count_nonzero(sparse_diff_eager).item()),
                                'sparse_first_flat_index_eager_A': (None if sparse_first_eager.numel() == 0
                                                                    else int(sparse_first_eager[0, 0].item())),
                                'sparse_max_abs_graph_A': float(sparse_diff_graph.max().item()),
                                'sparse_changed_count_graph_A': int(torch.count_nonzero(sparse_diff_graph).item()),
                                'sparse_first_flat_index_graph_A': (None if sparse_first_graph.numel() == 0
                                                                    else int(sparse_first_graph[0, 0].item())),
                                'sparse_finite': sparse_finite,
                                'sparse_finite_eager_A': sparse_finite_eager,
                                'owner_metadata_exact': meta_exact})
                if (not all(graph_exact.values()) or not all(meta_exact.values())
                        or not sparse_finite or not sparse_finite_eager
                        or any(not value for key, value in eager_exact.items()
                               if key != 'sparse')
                        or (delta == 0 and not eager_exact['sparse'])):
                    raise RuntimeError(f'dynamic owner graph parity failed delta={delta} arm={arm}')
        reference_changed = result.get('reference_metadata_changed_by_delta', {}).get('-4', {})
        captured_changed = result.get('captured_metadata_changed_by_delta', {}).get('-4', {})
        if not any(reference_changed.values()):
            raise RuntimeError('synthetic start shift did not change fresh reference metadata')
        if captured_changed != reference_changed:
            raise RuntimeError('captured metadata dynamic-response mask differs from reference')
        result['stage'] = 'complete'
        result['pass'] = True
    except Exception as exc:
        result['stage'] = 'error'
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()[-6000:]
    finally:
        for start, original in saved:
            try:
                start.copy_(original)
            except Exception as exc:
                result.setdefault('restore_errors', []).append(repr(exc))
                result['pass'] = False
        try:
            torch.npu.synchronize()
        except Exception as exc:
            result['sync_error'] = repr(exc)
            result['pass'] = False
        try:
            result['capture_still_active'] = bool(torch.npu.is_current_stream_capturing())
            if result['capture_still_active']:
                result['pass'] = False
        except Exception as exc:
            result['capture_state_error'] = repr(exc)
            result['pass'] = False
        path.write_text(json.dumps(result, indent=2) + '\n')
        print(f'EXTREME_DYNAMIC_METADATA rank={rank} pass={result["pass"]}', flush=True)
    if not result['pass']:
        raise RuntimeError(f'dynamic metadata private Graph failed rank{rank}: {result.get("error")}')
