"""Private TP8 all-active gate-placement Graph screen at one real layer4 call."""

import json
import os
import statistics
import time
import traceback
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from scripts.loop072_routed_graph_fixture import _difference


def install(target_handoff, state, rank):
    root = os.getenv('EXTREME_GATE_PLACEMENT_DIR')
    if not root:
        return
    if rank is None or not 0 <= int(rank) < 8:
        raise RuntimeError('gate placement requires TP8')
    from vllm.config import CUDAGraphMode
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context
    from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
    from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEPrepareOutput
    from vllm_ascend.quantization.methods import w4a8 as w4a8_module

    module = target_handoff.model.model.layers[4].mlp
    original = module.forward
    module._extreme_gate_original_forward = original
    cohort = int(getattr(module, '_extreme_gate_cohort', 0)) + 1
    module._extreme_gate_cohort = cohort
    path = Path(root) / f'rank{rank}_cohort{cohort}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    done = False

    def wrapped(hidden_states, input_ids=None):
        nonlocal done
        if done:
            return original(hidden_states, input_ids)
        mask = state.active_mask.to(torch.int32).contiguous()
        key = torch.cat((mask, torch.tensor([int(state.cycle_index),
                                            int(hidden_states.shape[0]),
                                            int(hidden_states.shape[1])],
                                           dtype=torch.int32, device=mask.device)))
        gathered = torch.empty((8, 15), dtype=torch.int32, device=mask.device)
        dist.all_gather_into_tensor(gathered, key, group=get_tp_group().device_group)
        if not torch.equal(gathered, key.unsqueeze(0).expand_as(gathered)):
            done = True
            path.write_text(json.dumps({'rank': int(rank), 'stage': 'mask_or_cycle_mismatch',
                                        'all_keys': gathered.cpu().tolist()}, indent=2) + '\n')
            return original(hidden_states, input_ids)
        if (int(mask.sum().item()) != 12 or int(state.cycle_index) < 64 or
                int(key[-2].item()) != 12 or int(key[-1].item()) != 4096):
            return original(hidden_states, input_ids)
        done = True
        report = {'rank': int(rank), 'cohort': cohort, 'cycle': int(state.cycle_index),
                  'active_mask': mask.cpu().tolist(), 'stage': 'preflight', 'pass': False,
                  'scope': 'one private full-active layer4 complete MoE Graph; not live Target or formal E2E',
                  'warmup_triplets': 3, 'measured_triplets': 10,
                  'sequence': ['A', 'A_repeat', 'B', 'A2'],
                  'capture_design': 'A and A2 independent original Graphs; A_repeat replays A',
                  'B_schedule': 'existing hidden AllGather, then replicated global96 FP32 gate; no logits AllGather or local gate; shared TP gather begins only after B hidden gather, and shared gate-up waits for B gate event',
                  'numeric_screen_rule': 'B full96 logits/routing and local12 output against A/A_repeat/independent A2 control; finite, IDs exact, output max/rms/signed envelope diagnostic only, not frozen correctness'}
        context = get_forward_context()
        inner = module.experts
        gate = module.gate
        parallel = target_handoff.vllm_config.parallel_config
        quant = inner._quant_method
        comm = fused_moe_module._EXTRA_CTX.moe_comm_method
        report['shared_tp_gather_mode'] = bool(inner._flashcomm_uses_tp_shared_experts)
        local_error = None
        try:
            if (tuple(hidden_states.shape) != (12, 4096) or
                    int(context.num_tokens) != 96 or int(context.pad_size) != 0 or
                    int(context.padded_num_tokens) != 96 or
                    tuple(context.input_ids.shape) != (96,) or
                    not context.flash_comm_v1_enabled or
                    str(context.moe_comm_type) != 'MoECommType.ALLGATHER' or
                    module.is_sequence_parallel or module.hash or module.layer_idx != 4 or
                    inner.dynamic_eplb or parallel.enable_eplb or
                    inner._shared_experts is None or
                    inner.routed_input_transform is not None or
                    type(comm).__name__ != 'AllGatherCommImpl' or
                    type(comm.prepare_finalize).__name__ != 'PrepareAndFinalizeWithAllGather' or
                    not inner.is_internal_router or gate.bias is not None or
                    tuple(gate.weight_fp32.shape) != (256, 4096) or
                    gate.weight_fp32.dtype != torch.float32 or
                    getattr(quant, 'tid2eid', None) is not None or
                    type(getattr(quant, 'quant_method', None)).__name__ !=
                    'AscendW4A8DynamicFusedMoEMethod' or
                    parallel.data_parallel_size != 1 or parallel.tensor_parallel_size != 8 or
                    parallel.prefill_context_parallel_size != 1 or
                    parallel.decode_context_parallel_size != 1 or
                    target_handoff.vllm_config.lora_config is not None or
                    target_handoff.aclgraph_runtime_mode != CUDAGraphMode.NONE or
                    not target_handoff.skip_compiled or comm.moe_config.pcp_size != 1):
                raise RuntimeError('frozen full-active gate/ALLGATHER contract changed')
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

        # A true loaded-weight equality check, not merely a ReplicatedLinear type check.
        ref_weight = gate.weight_fp32.detach().clone()
        dist.broadcast(ref_weight, src=0, group=get_tp_group().device_group)
        report['gate_weight_exact_to_rank0'] = bool(torch.equal(ref_weight, gate.weight_fp32))
        mismatch = torch.tensor([int(not report['gate_weight_exact_to_rank0'])],
                                dtype=torch.int32, device=mask.device)
        dist.all_reduce(mismatch, group=get_tp_group().device_group)
        if int(mismatch.item()) != 0:
            report.update({'stage': 'weight_mismatch', 'failed_rank_count': int(mismatch.item())})
            path.write_text(json.dumps(report, indent=2) + '\n')
            return original(hidden_states, input_ids)
        del ref_weight

        original_prepare = comm.prepare
        original_shared = inner.shared_forward_impl
        original_select = w4a8_module.select_experts
        original_moe_index = int(context.moe_layer_index)
        graphs = {}
        capture_values = {}
        b_hidden_event = {}
        b_gate_event = {}
        failure = None

        def b_prepare(*args, **kwargs):
            if args or tuple(kwargs['hidden_states'].shape) != (12, 4096):
                raise RuntimeError('unexpected B prepare ABI')
            x = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                kwargs['hidden_states'], True, True)
            if tuple(x.shape) != (96, 4096):
                raise RuntimeError('B gathered hidden shape invalid')
            b_hidden_event['value'] = torch.npu.current_stream().record_event()
            logits = F.linear(x.float(), gate.weight_fp32)
            b_gate_event['value'] = torch.npu.current_stream().record_event()
            comm.prepare_finalize.num_tokens = x.shape[0]
            capture_values['logits'].append(logits)
            return MoEPrepareOutput(hidden_states=x, router_logits=logits,
                                    mc2_mask=None, padded_hidden_states_shape=None,
                                    pertoken_scale=None)

        def b_shared(inner_self, hidden_states, router_logits, shared_experts_input=None):
            shared_hidden = shared_experts_input if shared_experts_input is not None else hidden_states
            # B prepare ignores this placeholder and computes logits on gathered x.
            result = inner_self.no_shared_forward_impl(
                hidden_states, hidden_states, return_with_event=True)
            if 'value' not in b_hidden_event or 'value' not in b_gate_event:
                raise RuntimeError('B hidden/gate event missing')
            shared = inner_self._forward_shared_experts(
                shared_hidden,
                fused_moe_module.FusedMoEEvents(
                    before_routed_experts=b_hidden_event['value'],
                    after_routed_experts=b_gate_event['value'],
                    before_dispatch=result.before_dispatch_evt,
                    before_gmm2=result.before_gmm2_evt,
                    before_combine=result.before_combine_evt,
                    swiglu_limit=result.swiglu_limit,
                    swiglu_alpha=result.swiglu_alpha,
                    swiglu_beta=result.swiglu_beta))
            return shared, result.routed_out

        def capture(arm):
            seen = []
            capture_values['logits'] = []

            def observed_prepare(*args, **kwargs):
                result = original_prepare(*args, **kwargs)
                capture_values['logits'].append(result.router_logits)
                return result

            def observed_select(*args, **kwargs):
                weight, expert_id = original_select(*args, **kwargs)
                seen.append((weight, expert_id))
                return weight, expert_id

            context.moe_layer_index = original_moe_index
            comm.prepare = b_prepare if arm == 'B' else observed_prepare
            inner.shared_forward_impl = types.MethodType(b_shared, inner) if arm == 'B' else original_shared
            w4a8_module.select_experts = observed_select
            try:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    output = original(hidden_states, input_ids)
            finally:
                comm.prepare = original_prepare
                inner.shared_forward_impl = original_shared
                w4a8_module.select_experts = original_select
            torch.npu.synchronize()
            if (len(seen) != 1 or len(capture_values['logits']) != 1 or
                    tuple(output.shape) != (12, 4096) or
                    tuple(capture_values['logits'][0].shape) != (96, 256) or
                    tuple(seen[0][1].shape) != (96, 6)):
                raise RuntimeError(f'Graph {arm} capture ABI mismatch')
            graphs[arm] = (graph, output, capture_values['logits'][0],
                           seen[0][0], seen[0][1])

        try:
            report['memory_before_capture_bytes'] = int(torch.npu.memory_allocated())
            context.moe_layer_index = original_moe_index
            eager = original(hidden_states, input_ids).clone()
            torch.npu.synchronize()
            # Shape-specific B warmup without altering authoritative output.
            comm.prepare = b_prepare
            inner.shared_forward_impl = types.MethodType(b_shared, inner)
            capture_values['logits'] = []
            context.moe_layer_index = original_moe_index
            try:
                original(hidden_states, input_ids)
            finally:
                comm.prepare = original_prepare
                inner.shared_forward_impl = original_shared
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
                    key_arm = 'A' if arm == 'A_repeat' else arm
                    graph, output, logits, weight, expert_id = graphs[key_arm]
                    context.moe_layer_index = original_moe_index
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
                    values[arm] = (output.clone(), logits.clone(), weight.clone(), expert_id.clone())
                    samples.append({'triplet': triplet, 'arm': arm,
                                    'warmup': triplet < 3,
                                    'device_ms': float(start.elapsed_time(end)),
                                    'host_start_ns': host_start, 'host_end_ns': host_end})
                a, ar, b, a2 = (values[k] for k in ('A', 'A_repeat', 'B', 'A2'))
                comp = {'A_A_repeat_output': _difference(a[0], ar[0]),
                        'A_A2_output': _difference(a[0], a2[0]),
                        'B_A_output': _difference(a[0], b[0]),
                        'A_eager_output': _difference(eager, a[0]),
                        'A_A2_logits': _difference(a[1], a2[1]),
                        'B_A_logits': _difference(a[1], b[1]),
                        'A_A2_router_ids_exact': bool(torch.equal(a[3], a2[3])),
                        'B_A_router_ids_exact': bool(torch.equal(a[3], b[3])),
                        'B_A_router_weights': _difference(a[2], b[2])}
                samples[-1]['comparisons'] = comp
                if not all(x['finite'] for x in comp.values() if isinstance(x, dict)):
                    raise RuntimeError(f'nonfinite gate Graph triplet{triplet}')
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
            comm.prepare = original_prepare
            inner.shared_forward_impl = original_shared
            w4a8_module.select_experts = original_select
            context.moe_layer_index = original_moe_index
            try:
                torch.npu.synchronize()
            except Exception as exc:
                report['final_sync_error'] = repr(exc)
                failure = failure or exc
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
            print(f'EXTREME_GATE_PLACEMENT rank={rank} stage={report["stage"]}', flush=True)
        if failure is not None:
            raise RuntimeError(f'private gate placement failed rank{rank}') from failure
        return original(hidden_states, input_ids)

    module.forward = wrapped
