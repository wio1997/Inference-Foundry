"""One real warmed prefill layer4: unchanged eager/Graph/eager complete MoE gate.

Runs inside the first measured padded88 ModelRunner forward, on all TP8 workers.
It never substitutes the Graph result into the serving forward.
"""

import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist


def difference(a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return {'shape_equal': False, 'a_shape': list(a.shape), 'b_shape': list(b.shape)}
    delta = b - a
    return {'shape_equal': True, 'exact': bool(torch.equal(a, b)),
            'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            'max_abs': float(delta.abs().max()),
            'rms': float(torch.sqrt(torch.mean(delta.square()))),
            'mean_signed': float(delta.mean())}


def install_for_one_forward(runner, output_dir):
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context
    from vllm.config import CUDAGraphMode

    rank = int(get_tp_group().rank_in_group)
    module = runner.get_model().model.layers[4].mlp
    original = module.forward
    path = Path(output_dir) / f'rank{rank}.json'
    used = False
    report = {'rank': rank, 'stage': 'not_entered', 'pass': False,
              'scope': 'private unchanged full-MoE eager A / Graph B / eager A2 on original warm prefill layer4; no live Graph result'}

    def wrapped(hidden_states, input_ids=None):
        nonlocal used
        if used:
            return original(hidden_states, input_ids)
        used = True
        context = get_forward_context()
        report.update({'stage': 'preflight', 'input_shape': list(hidden_states.shape),
                       'input_dtype': str(hidden_states.dtype),
                       'context_num_tokens': int(context.num_tokens),
                       'context_padded_num_tokens': int(context.padded_num_tokens),
                       'context_pad_size': int(context.pad_size),
                       'context_mode': str(context.cudagraph_runtime_mode),
                       'moe_comm_type': str(context.moe_comm_type),
                       'flash_comm_v1_enabled': bool(context.flash_comm_v1_enabled),
                       'input_ids_shape': None if input_ids is None else list(input_ids.shape)})
        local_bad = int(
            tuple(hidden_states.shape) != (11, 4096) or
            int(context.padded_num_tokens) != 88 or
            int(context.num_tokens) > 88 or
            context.cudagraph_runtime_mode != CUDAGraphMode.NONE or
            not context.flash_comm_v1_enabled or
            str(context.moe_comm_type) != 'MoECommType.ALLGATHER' or
            runner.vllm_config.parallel_config.data_parallel_size != 1 or
            runner.vllm_config.parallel_config.tensor_parallel_size != 8 or
            module.layer_idx != 4)
        flag = torch.tensor([local_bad], dtype=torch.int32, device=hidden_states.device)
        dist.all_reduce(flag, group=get_tp_group().device_group)
        if int(flag.item()) != 0:
            report.update({'stage': 'preflight_failed', 'bad_ranks': int(flag.item())})
            path.write_text(json.dumps(report, indent=2) + '\n')
            return original(hidden_states, input_ids)

        old_layer_index = int(context.moe_layer_index)
        graph = None
        failure = None
        try:
            input_before = hidden_states.detach().clone()
            # Both arms use the exact original module and immutable layer input.
            # The complete MoE output includes shared/routed join and final RS.
            context.moe_layer_index = old_layer_index
            original(hidden_states, input_ids)
            torch.npu.synchronize()
            context.moe_layer_index = old_layer_index
            graph = torch.npu.NPUGraph()
            capture_start = time.perf_counter_ns()
            with torch.npu.graph(graph):
                graph_output = original(hidden_states, input_ids)
            torch.npu.synchronize()
            report['capture_wall_ms'] = (time.perf_counter_ns() - capture_start) / 1e6
            report['graph_output_shape'] = list(graph_output.shape)
            if tuple(graph_output.shape) != (11, 4096):
                raise RuntimeError('Graph output shape changed')
            report['stage'] = 'measuring'
            samples = []
            report['samples'] = samples
            for triplet in range(13):
                outputs = {}
                for arm in ('A', 'B', 'A2'):
                    context.moe_layer_index = old_layer_index
                    dist.barrier(group=get_tp_group().device_group)
                    torch.npu.synchronize()
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    host_start = time.perf_counter_ns()
                    cpu_start = time.thread_time_ns()
                    start.record()
                    if arm == 'B':
                        graph.replay()
                        output = graph_output
                    else:
                        output = original(hidden_states, input_ids)
                    host_submitted = time.perf_counter_ns()
                    cpu_submitted = time.thread_time_ns()
                    end.record()
                    end.synchronize()
                    host_completed = time.perf_counter_ns()
                    outputs[arm] = output.detach().clone()
                    samples.append({'triplet': triplet, 'arm': arm,
                                    'warmup': triplet < 3,
                                    'stream_endpoint_ms': float(start.elapsed_time(end)),
                                    'host_submit_ms': (host_submitted-host_start)/1e6,
                                    'host_submit_thread_cpu_ms': (cpu_submitted-cpu_start)/1e6,
                                    'host_complete_ms': (host_completed-host_start)/1e6,
                                    'host_start_ns': host_start,
                                    'host_completed_ns': host_completed})
                cmp = {'A_A2': difference(outputs['A'], outputs['A2']),
                       'A_B': difference(outputs['A'], outputs['B'])}
                samples[-1]['comparisons'] = cmp
                if not all(x.get('finite') and x.get('shape_equal') for x in cmp.values()):
                    raise RuntimeError(f'nonfinite or shape mismatch at triplet {triplet}')
            measured = [x for x in samples if not x['warmup']]
            report['median_by_arm'] = {
                arm: {key: statistics.median(x[key] for x in measured if x['arm'] == arm)
                      for key in ('stream_endpoint_ms', 'host_submit_ms', 'host_complete_ms')}
                for arm in ('A', 'B', 'A2')}
            torch.npu.synchronize()
            report['input_unchanged'] = bool(torch.equal(input_before, hidden_states))
            if not report['input_unchanged']:
                raise RuntimeError('private fixture mutated live MoE input')
            report['stage'] = 'complete'
            report['pass'] = True
        except Exception as exc:
            failure = exc
            report.update({'stage': 'error', 'error': repr(exc),
                           'traceback': traceback.format_exc()[-6000:]})
        finally:
            context.moe_layer_index = old_layer_index
            try:
                torch.npu.synchronize()
                if torch.npu.is_current_stream_capturing():
                    raise RuntimeError('capture still active')
            except Exception as exc:
                report.update({'stage': 'error', 'final_sync_error': repr(exc), 'pass': False})
                failure = failure or exc
            path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'EXTREME_PREFILL_MOE_GRAPH rank={rank} stage={report["stage"]}', flush=True)
        if failure is not None:
            raise RuntimeError(f'private prefill MoE Graph gate failed rank{rank}') from failure
        return original(hidden_states, input_ids)

    module.forward = wrapped
    return module, original, report


def restore(module, original):
    module.forward = original
