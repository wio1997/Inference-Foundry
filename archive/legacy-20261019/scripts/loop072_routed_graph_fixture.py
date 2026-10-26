"""Private all-rank complete MoE Graph A/A/B/A screen at a real parked cycle."""

import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist


def _difference(reference, candidate):
    if reference.numel() == 0:
        return {'elements': 0, 'exact': True, 'finite': True, 'count_nonzero': 0,
                'max_abs': 0.0, 'mean_abs': 0.0, 'rms': 0.0,
                'p50_abs': 0.0, 'p95_abs': 0.0, 'p99_abs': 0.0,
                'mean_signed': 0.0}
    a = reference.float().cpu()
    b = candidate.float().cpu()
    signed = b - a
    absolute = signed.abs().reshape(-1)
    return {'elements': int(absolute.numel()),
            'exact': bool(torch.equal(a, b)),
            'finite': bool(torch.isfinite(a).all().item() and
                           torch.isfinite(b).all().item()),
            'count_nonzero': int(torch.count_nonzero(absolute).item()),
            'max_abs': float(absolute.max().item()),
            'mean_abs': float(absolute.mean().item()),
            'rms': float(torch.sqrt(torch.mean(signed.square())).item()),
            'p50_abs': float(torch.quantile(absolute, 0.50).item()),
            'p95_abs': float(torch.quantile(absolute, 0.95).item()),
            'p99_abs': float(torch.quantile(absolute, 0.99).item()),
            'mean_signed': float(signed.mean().item())}


def install(target_handoff, state, rank):
    root = os.getenv('EXTREME_PARKED_MOE_DIR')
    if not root:
        return
    if rank is None or not 0 <= int(rank) < 8:
        raise RuntimeError('routed Graph fixture requires TP8 rank')
    from vllm.config import CUDAGraphMode
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context
    from vllm_ascend.quantization.methods import w4a8 as w4a8_module

    module = target_handoff.model.model.layers[4].mlp
    old = getattr(module, '_extreme_parked_moe_original_forward', None)
    if old is not None:
        module.forward = old
    original = module.forward
    module._extreme_parked_moe_original_forward = original
    cohort = int(getattr(module, '_extreme_parked_moe_cohort', 0)) + 1
    module._extreme_parked_moe_cohort = cohort
    path = Path(root) / f'rank{rank}_cohort{cohort}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = False

    def wrapped(hidden_states, input_ids=None):
        nonlocal done
        if done:
            return original(hidden_states, input_ids)
        mask = state.active_mask.to(torch.int32).contiguous()
        key = torch.cat((mask, torch.tensor([int(state.cycle_index)], dtype=torch.int32,
                                            device=mask.device)))
        gathered = torch.empty((8, 13), dtype=torch.int32, device=mask.device)
        dist.all_gather_into_tensor(gathered, key, group=get_tp_group().device_group)
        if not torch.equal(gathered, key.unsqueeze(0).expand_as(gathered)):
            done = True
            path.write_text(json.dumps({'rank': int(rank), 'cohort': cohort,
                                        'stage': 'mask_or_cycle_mismatch',
                                        'all_keys': gathered.cpu().tolist()}, indent=2) + '\n')
            return original(hidden_states, input_ids)
        if int(mask.sum().item()) != 6:
            return original(hidden_states, input_ids)
        done = True
        report = {'rank': int(rank), 'cohort': cohort,
                  'cycle': int(state.cycle_index), 'active_mask': mask.cpu().tolist(),
                  'stage': 'preflight', 'pass': False,
                  'scope': 'one layer4 private complete MoE Graph endpoint, not live Target or Product E2E',
                  'warmup_triplets': 3, 'measured_triplets': 10,
                  'sequence': ['A', 'A_repeat', 'B', 'A2'],
                  'capture_design': 'A and A2 are independent full-MoE Graph captures; A_repeat replays A',
                  'numeric_screen_rule': 'For each nonempty rank, compare B active max_abs/rms/signed-mean to the largest paired A/A_repeat or A/A2 control across measured triplets; this is a diagnostic envelope, not frozen correctness.'}
        local_error = None
        try:
            context = get_forward_context()
            parallel = target_handoff.vllm_config.parallel_config
            quant = module.experts._quant_method
            if (tuple(hidden_states.shape) != (12, 4096) or
                    int(context.num_tokens) != 96 or
                    int(context.pad_size) != 0 or
                    int(context.padded_num_tokens) != 96 or
                    tuple(context.input_ids.shape) != (96,) or
                    not context.flash_comm_v1_enabled or
                    str(context.moe_comm_type) != 'MoECommType.ALLGATHER' or
                    module.is_sequence_parallel or module.hash or module.layer_idx != 4 or
                    module.experts.dynamic_eplb or parallel.enable_eplb or
                    getattr(quant, 'tid2eid', None) is not None or
                    type(getattr(quant, 'quant_method', None)).__name__ !=
                    'AscendW4A8DynamicFusedMoEMethod' or
                    parallel.data_parallel_size != 1 or parallel.tensor_parallel_size != 8 or
                    parallel.prefill_context_parallel_size != 1 or
                    parallel.decode_context_parallel_size != 1 or
                    target_handoff.vllm_config.lora_config is not None or
                    target_handoff.aclgraph_runtime_mode != CUDAGraphMode.NONE or
                    not target_handoff.skip_compiled):
                raise RuntimeError('frozen layer4 W4A8/ALLGATHER/eager contract changed')
        except Exception as exc:
            local_error = repr(exc)
        flag = torch.tensor([int(local_error is not None)], dtype=torch.int32,
                            device=mask.device)
        dist.all_reduce(flag, group=get_tp_group().device_group)
        if int(flag.item()) != 0:
            report.update({'stage': 'preflight_failed', 'local_error': local_error,
                           'failed_rank_count': int(flag.item())})
            path.write_text(json.dumps(report, indent=2) + '\n')
            return original(hidden_states, input_ids)

        quant_apply = quant.apply
        select_experts = w4a8_module.select_experts
        original_moe_index = int(context.moe_layer_index)
        global_idx = torch.nonzero(mask.repeat_interleave(8), as_tuple=False).flatten().long()
        local_idx = torch.nonzero(mask.repeat_interleave(8)[rank * 12:(rank + 1) * 12],
                                  as_tuple=False).flatten().long()
        report['local_active_rows'] = int(local_idx.numel())
        graphs = {}
        routing = {}
        failure = None

        def compact_apply(*args, **kwargs):
            if (args or kwargs.get('layer') is not module.experts.routed_experts or
                    'x' not in kwargs or 'router_logits' not in kwargs):
                raise RuntimeError('unexpected routed apply call')
            x = kwargs['x']
            logits = kwargs['router_logits']
            scale = kwargs.get('pertoken_scale')
            if (tuple(x.shape) != (96, 4096) or int(logits.shape[0]) != 96 or
                    kwargs.get('mc2_mask') is not None or
                    kwargs.get('enable_force_load_balance')):
                raise RuntimeError('routed apply is not the global96 ALLGATHER input')
            smaller = dict(kwargs)
            smaller['x'] = x.index_select(0, global_idx).contiguous()
            smaller['router_logits'] = logits.index_select(0, global_idx).contiguous()
            if scale is not None:
                if int(scale.shape[0]) != 96:
                    raise RuntimeError('unexpected per-token scale')
                smaller['pertoken_scale'] = scale.index_select(0, global_idx).contiguous()
            result = quant_apply(**smaller)
            routed = result.routed_out
            if tuple(routed.shape) != (48, 4096):
                raise RuntimeError('compact routed output shape invalid')
            restored = routed.new_zeros((96, 4096))
            restored.index_copy_(0, global_idx, routed)
            result.routed_out = restored
            return result

        def capture(arm):
            seen = []

            def capture_select(*args, **kwargs):
                weight, expert_id = select_experts(*args, **kwargs)
                seen.append((weight, expert_id))
                return weight, expert_id

            context.moe_layer_index = original_moe_index
            w4a8_module.select_experts = capture_select
            quant.apply = compact_apply if arm == 'B' else quant_apply
            try:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = original(hidden_states, input_ids)
            finally:
                quant.apply = quant_apply
                w4a8_module.select_experts = select_experts
            torch.npu.synchronize()
            if len(seen) != 1 or tuple(output.shape) != (12, 4096):
                raise RuntimeError(f'Graph {arm} routing capture/output invalid: {len(seen)}')
            expected_rows = 48 if arm == 'B' else 96
            if tuple(seen[0][1].shape) != (expected_rows, 6):
                raise RuntimeError(f'Graph {arm} routing ID shape invalid: {tuple(seen[0][1].shape)}')
            graphs[arm] = (graph, output)
            routing[arm] = seen[0]

        try:
            report['memory_before_capture_bytes'] = int(torch.npu.memory_allocated())
            # Warm each shape before capture; these calls have no KV mutation.
            context.moe_layer_index = original_moe_index
            eager_a = original(hidden_states, input_ids).index_select(0, local_idx).clone()
            torch.npu.synchronize()
            context.moe_layer_index = original_moe_index
            quant.apply = compact_apply
            try:
                original(hidden_states, input_ids)
            finally:
                quant.apply = quant_apply
            torch.npu.synchronize()
            capture('A')
            capture('B')
            capture('A2')
            report['memory_after_capture_bytes'] = int(torch.npu.memory_allocated())
            samples = []
            report['samples'] = samples
            for triplet in range(13):
                values = {}
                for arm in ('A', 'A_repeat', 'B', 'A2'):
                    graph_key = 'A' if arm == 'A_repeat' else arm
                    graph, output = graphs[graph_key]
                    weight, expert_id = routing[graph_key]
                    context.moe_layer_index = original_moe_index
                    # All ranks enter every replay in identical order; the barrier is
                    # outside the measured complete-MoE endpoint.
                    dist.barrier(group=get_tp_group().device_group)
                    torch.npu.synchronize()
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    host_start = time.perf_counter_ns()
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    host_end = time.perf_counter_ns()
                    values[arm] = (output.index_select(0, local_idx).clone(),
                                   weight.clone() if graph_key == 'B' else weight.index_select(0, global_idx).clone(),
                                   expert_id.clone() if graph_key == 'B' else expert_id.index_select(0, global_idx).clone())
                    samples.append({'triplet': triplet, 'arm': arm,
                                    'warmup': triplet < 3,
                                    'device_ms': float(start.elapsed_time(end)),
                                    'host_start_ns': host_start, 'host_end_ns': host_end})
                a, ar, b, a2 = (values[k] for k in ('A', 'A_repeat', 'B', 'A2'))
                comparisons = {'A_A_repeat': _difference(a[0], ar[0]),
                               'A_A2': _difference(a[0], a2[0]),
                               'A_eager': _difference(eager_a, a[0]),
                               'B_active_vs_A': _difference(a[0], b[0]),
                               'router_weights_A_vs_B': _difference(a[1], b[1]),
                               'router_ids_exact': bool(torch.equal(a[2], b[2])),
                               'router_ids_A_vs_A2_exact': bool(torch.equal(a[2], a2[2]))}
                samples[-1]['comparisons'] = comparisons
                if not all(x['finite'] for x in comparisons.values() if isinstance(x, dict)):
                    raise RuntimeError(f'nonfinite Graph output/routing at triplet{triplet}')
            measured = [x for x in samples if not x['warmup']]
            report['median_device_ms_by_arm'] = {
                arm: statistics.median(x['device_ms'] for x in measured if x['arm'] == arm)
                for arm in ('A', 'A_repeat', 'B', 'A2')}
            report['stage'] = 'complete'
            report['pass'] = True
        except Exception as exc:
            failure = exc
            report.update({'stage': 'error', 'pass': False, 'error': repr(exc),
                           'traceback': traceback.format_exc()[-6000:]})
        finally:
            quant.apply = quant_apply
            w4a8_module.select_experts = select_experts
            context.moe_layer_index = original_moe_index
            try:
                torch.npu.synchronize()
            except Exception as exc:
                report['final_sync_error'] = repr(exc)
                failure = exc
                report['stage'] = 'error'
                report['pass'] = False
            try:
                report['capture_still_active'] = bool(torch.npu.is_current_stream_capturing())
                if report['capture_still_active']:
                    failure = failure or RuntimeError('Graph capture remained active')
                    report['stage'] = 'error'
                    report['pass'] = False
            except Exception as exc:
                report['capture_state_error'] = repr(exc)
                failure = failure or exc
                report['stage'] = 'error'
                report['pass'] = False
            graphs.clear()
            path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'EXTREME_ROUTED_GRAPH rank={rank} cohort={cohort} stage={report["stage"]}', flush=True)
        if failure is not None:
            raise RuntimeError(f'private routed Graph failed after collective rank{rank}') from failure
        return original(hidden_states, input_ids)

    module.forward = wrapped
