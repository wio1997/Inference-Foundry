"""Read-only layer4 MoE row-ownership census at a real six-active cycle."""

import json
import os
import traceback
from pathlib import Path

import torch


def install(target_handoff, state, rank):
    root = os.getenv('EXTREME_PARKED_MOE_DIR')
    if not root:
        return
    if rank is None or not 0 <= int(rank) < 8:
        raise RuntimeError('parked MoE census requires TP8 rank')
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
        mask = state.active_mask
        if int(mask.sum().item()) != 6:
            return original(hidden_states, input_ids)
        done = True
        report = {'rank': int(rank), 'cohort': cohort,
                  'cycle': int(state.cycle_index),
                  'active_mask': mask.to(torch.int32).cpu().tolist(),
                  'scope': 'read-only real layer4 MoE ownership census; no candidate execution',
                  'stage': 'error'}
        try:
            context = get_forward_context()
            parallel = target_handoff.vllm_config.parallel_config
            full_tokens = int(context.num_tokens)
            local_rows = int(hidden_states.shape[0])
            if (tuple(hidden_states.shape) != (12, 4096) or
                    full_tokens != 96 or full_tokens // 8 != local_rows or
                    tuple(context.input_ids.shape) != (96,) or
                    parallel.data_parallel_size != 1 or
                    parallel.tensor_parallel_size != 8 or
                    module.layer_idx != 4 or module.hash):
                raise RuntimeError('observed layout differs from frozen TP8 layer4 contract')
            start = int(rank) * local_rows
            global_rows = list(range(start, start + local_rows))
            request_ids = [row // 8 for row in global_rows]
            active_flags = [bool(report['active_mask'][request]) for request in request_ids]
            method = getattr(context, 'moe_comm_method', None)
            report.update({'stage': 'complete', 'pass': True,
                'hidden_shape': list(hidden_states.shape),
                'context_num_tokens': full_tokens,
                'context_pad_size': int(context.pad_size),
                'context_padded_num_tokens': int(context.padded_num_tokens),
                'context_flash_comm_v1_enabled': bool(context.flash_comm_v1_enabled),
                'context_moe_comm_type': str(context.moe_comm_type),
                'context_moe_comm_method': type(method).__name__,
                'context_layer_idx': str(context.layer_idx),
                'context_moe_layer_index': int(context.moe_layer_index),
                'module_is_sequence_parallel': bool(module.is_sequence_parallel),
                'parallel': {'dp': int(parallel.data_parallel_size),
                    'tp': int(parallel.tensor_parallel_size),
                    'pcp': int(parallel.prefill_context_parallel_size),
                    'dcp': int(parallel.decode_context_parallel_size)},
                'global_rows': global_rows,
                'request_ids': request_ids,
                'local_active_flags': active_flags,
                'local_active_rows': int(sum(active_flags))})
        except Exception as exc:
            report.update({'error': repr(exc), 'traceback': traceback.format_exc()[-4000:]})
        path.write_text(json.dumps(report, indent=2) + '\n')
        print(f'EXTREME_PARKED_MOE_CENSUS rank={rank} cohort={cohort} stage={report["stage"]}', flush=True)
        return original(hidden_states, input_ids)

    module.forward = wrapped
