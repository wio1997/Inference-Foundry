"""Private same-prestate post-AllGather routed-only active-row MoE screen.

The live Target result remains the original module call. Candidate B only
wraps this layer's quant method during one extra private MoE invocation.
"""

import json
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist


def _metrics(reference, candidate):
    finite_ref = bool(torch.isfinite(reference).all().item())
    finite_candidate = bool(torch.isfinite(candidate).all().item())
    if reference.numel() == 0:
        return {'rows': 0, 'exact': True, 'finite_reference': finite_ref,
                'finite_candidate': finite_candidate, 'max_abs': 0.0,
                'max_rel': 0.0, 'count_nonzero': 0,
                'allclose_atol1e-3_rtol1e-3': True,
                'allclose_atol1e-2_rtol1e-2': True}
    ref = reference.float()
    got = candidate.float()
    diff = (got - ref).abs()
    return {
        'rows': int(reference.shape[0]),
        'exact': bool(torch.equal(reference, candidate)),
        'finite_reference': finite_ref,
        'finite_candidate': finite_candidate,
        'max_abs': float(diff.max().item()),
        'max_rel': float((diff / ref.abs().clamp_min(1e-6)).max().item()),
        'count_nonzero': int(torch.count_nonzero(diff).item()),
        'allclose_atol1e-3_rtol1e-3': bool(torch.allclose(reference, candidate,
                                                         atol=1e-3, rtol=1e-3)),
        'allclose_atol1e-2_rtol1e-2': bool(torch.allclose(reference, candidate,
                                                         atol=1e-2, rtol=1e-2)),
    }


def install(target_handoff, state, rank):
    root = os.getenv('EXTREME_PARKED_MOE_DIR')
    if not root:
        return
    if rank is None or not 0 <= int(rank) < 8:
        raise RuntimeError('post-gather routed fixture requires TP8 rank')
    from vllm.config import CUDAGraphMode
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context

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
                                        'local_key': key.cpu().tolist(),
                                        'all_keys': gathered.cpu().tolist()}, indent=2) + '\n')
            return original(hidden_states, input_ids)
        if int(mask.sum().item()) != 6:
            return original(hidden_states, input_ids)
        done = True
        report = {'rank': int(rank), 'cohort': cohort,
                  'cycle': int(state.cycle_index),
                  'active_mask': mask.cpu().tolist(),
                  'active_count': 6, 'stage': 'preflight',
                  'scope': 'private one-layer full MoE A/A_repeat/B/A2, routed-only active48 after existing gather; not Product E2E'}

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
                    module.is_sequence_parallel or module.hash or
                    module.layer_idx != 4 or
                    bool(module.experts.dynamic_eplb) or
                    getattr(quant, 'tid2eid', None) is not None or
                    parallel.enable_eplb or
                    parallel.data_parallel_size != 1 or
                    parallel.tensor_parallel_size != 8 or
                    parallel.prefill_context_parallel_size != 1 or
                    parallel.decode_context_parallel_size != 1 or
                    target_handoff.vllm_config.lora_config is not None or
                    target_handoff.aclgraph_runtime_mode != CUDAGraphMode.NONE or
                    not target_handoff.skip_compiled):
                raise RuntimeError('real layer4/TP8/ALLGATHER/eager contract changed')
            if not hasattr(quant, 'apply'):
                raise RuntimeError('quant method has no apply hook')
            report['quant_method_type'] = type(quant).__name__
            report['quant_scheme_type'] = type(getattr(quant, 'quant_method', None)).__name__
            report['tid2eid_is_none'] = True
        except Exception as exc:
            local_error = repr(exc)
        preflight = torch.tensor([int(local_error is not None)], dtype=torch.int32,
                                 device=mask.device)
        dist.all_reduce(preflight, group=get_tp_group().device_group)
        if int(preflight.item()) != 0:
            report.update({'stage': 'preflight_failed',
                           'failed_rank_count': int(preflight.item()),
                           'local_error': local_error})
            path.write_text(json.dumps(report, indent=2) + '\n')
            return original(hidden_states, input_ids)

        failure = None
        original_moe_index = int(context.moe_layer_index)
        original_apply = quant.apply
        applied = []
        global_idx = torch.nonzero(mask.repeat_interleave(8), as_tuple=False).flatten().long()
        local_idx = torch.nonzero(mask.repeat_interleave(8)[rank * 12:(rank + 1) * 12],
                                  as_tuple=False).flatten().long()

        def compact_apply(*args, **kwargs):
            if args or 'x' not in kwargs or 'router_logits' not in kwargs:
                raise RuntimeError('unexpected quant apply signature')
            if kwargs.get('layer') is not module.experts.routed_experts:
                raise RuntimeError('quant apply layer is not hooked layer4 routed experts')
            x = kwargs['x']
            router_logits = kwargs['router_logits']
            scale = kwargs.get('pertoken_scale')
            if (tuple(x.shape) != (96, 4096) or
                    int(router_logits.shape[0]) != 96 or
                    kwargs.get('mc2_mask') is not None or
                    kwargs.get('enable_force_load_balance')):
                raise RuntimeError('routed apply is not global96 ALLGATHER input')
            smaller = dict(kwargs)
            smaller['x'] = x.index_select(0, global_idx).contiguous()
            smaller['router_logits'] = router_logits.index_select(0, global_idx).contiguous()
            if scale is not None:
                if int(scale.shape[0]) != 96:
                    raise RuntimeError('unexpected per-token scale shape')
                smaller['pertoken_scale'] = scale.index_select(0, global_idx).contiguous()
            result = original_apply(**smaller)
            routed = result.routed_out
            if tuple(routed.shape) != (48, 4096):
                raise RuntimeError(f'unexpected routed output shape {tuple(routed.shape)}')
            restored = routed.new_zeros((96, 4096))
            restored.index_copy_(0, global_idx, routed)
            result.routed_out = restored
            applied.append({'full_input': list(x.shape),
                            'compact_input': list(smaller['x'].shape),
                            'routed_compact': list(routed.shape),
                            'routed_restored': list(restored.shape),
                            'per_token_scale': None if scale is None else list(scale.shape)})
            return result

        try:
            torch.npu.synchronize()
            context.moe_layer_index = original_moe_index
            full_a = original(hidden_states, input_ids).clone()
            torch.npu.synchronize()
            context.moe_layer_index = original_moe_index
            full_repeat = original(hidden_states, input_ids).clone()
            torch.npu.synchronize()
            report['A_A_repeat'] = _metrics(full_a, full_repeat)
            context.moe_layer_index = original_moe_index
            quant.apply = compact_apply
            try:
                candidate_b = original(hidden_states, input_ids).clone()
            finally:
                quant.apply = original_apply
            torch.npu.synchronize()
            context.moe_layer_index = original_moe_index
            full_a2 = original(hidden_states, input_ids).clone()
            torch.npu.synchronize()
            if len(applied) != 1 or tuple(candidate_b.shape) != (12, 4096):
                raise RuntimeError(f'candidate routed hook count/shape invalid: {applied}')
            report.update({'stage': 'complete', 'pass': True,
                           'global_active_rows': global_idx.cpu().tolist(),
                           'local_active_rows': local_idx.cpu().tolist(),
                           'apply': applied[0],
                           'A_A2': _metrics(full_a, full_a2),
                           'B_active_vs_A': _metrics(full_a.index_select(0, local_idx),
                                                     candidate_b.index_select(0, local_idx)),
                           'B_all_finite': bool(torch.isfinite(candidate_b).all().item()),
                           'outer_context_restored': get_forward_context() is context})
        except Exception as exc:
            failure = exc
            report.update({'stage': 'error', 'error': repr(exc),
                           'traceback': traceback.format_exc()[-6000:]})
        finally:
            quant.apply = original_apply
            context.moe_layer_index = original_moe_index
            path.write_text(json.dumps(report, indent=2) + '\n')
            print(f'EXTREME_ROUTED_MOE rank={rank} cohort={cohort} stage={report["stage"]}',
                  flush=True)
        if failure is not None:
            raise RuntimeError(f'private routed MoE failed after collective rank{rank}') from failure
        return original(hidden_states, input_ids)

    module.forward = wrapped
