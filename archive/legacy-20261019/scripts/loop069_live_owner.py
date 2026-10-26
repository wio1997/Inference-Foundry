"""Layer2 owner16 producer helpers for a reversible Target integration diagnostic.

Only the frozen c12 decode geometry is eligible. The caller retains the
original hidden AllGather, Q path, QLI query and Sparse consumer.
"""

import os
import json
from pathlib import Path

import torch


def enabled(impl, *, layer_name, hidden_states_cache, hidden_states_local,
            has_prefill):
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    descriptor = getattr(context, 'batch_descriptor', None)
    graph_key = (getattr(descriptor, 'num_tokens', None),
                 getattr(descriptor, 'num_reqs', None),
                 getattr(descriptor, 'uniform', None),
                 getattr(descriptor, 'has_lora', None),
                 getattr(descriptor, 'num_active_loras', None))
    selected = (os.getenv('EXTREME_LIVE_OWNER_LAYER2') == '1'
            and layer_name == 'model.layers.2.self_attn.attn'
            and impl.compress_ratio == 4 and not has_prefill
            and impl.tp_size == 8 and 0 <= impl.tp_rank < 8
            and graph_key == (96, 12, True, False, 0)
            and not getattr(context, 'is_draft_model', False)
            and tuple(hidden_states_cache.shape) == (96, 4096)
            and tuple(hidden_states_local.shape) == (12, 4096))
    if selected and os.getenv('EXTREME_LIVE_OWNER_TRACE_DIR'):
        capture = bool(getattr(context, 'capturing', False)
                       or torch.npu.is_current_stream_capturing())
        seen = getattr(impl, '_extreme_live_owner_logged', set())
        if capture not in seen:
            rank = int(impl.tp_rank)
            path = Path(os.environ['EXTREME_LIVE_OWNER_TRACE_DIR'])
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'rank{rank}.jsonl').open('a') as handle:
                handle.write(json.dumps({'rank': rank, 'capturing': capture,
                                         'phase': 'selected',
                                         'graph_key': list(graph_key),
                                         'owner_requests': [(rank * 12) // 8,
                                                            (rank * 12 + 11) // 8],
                                         'owner_rows': [16, 4096],
                                         'full_shape': list(hidden_states_cache.shape),
                                         'local_shape': list(hidden_states_local.shape)}) + '\n')
            seen.add(capture)
            impl._extreme_live_owner_logged = seen
    return selected


def record_producer_complete(owner_ctx):
    trace_dir = os.getenv('EXTREME_LIVE_OWNER_TRACE_DIR')
    if not trace_dir:
        return
    from vllm.forward_context import get_forward_context
    context = get_forward_context()
    capture = bool(getattr(context, 'capturing', False)
                   or torch.npu.is_current_stream_capturing())
    seen = getattr(owner_ctx.impl, '_extreme_live_owner_complete_logged', set())
    if capture in seen:
        return
    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    desc = context.batch_descriptor
    with (path / f'rank{owner_ctx.rank}.jsonl').open('a') as handle:
        handle.write(json.dumps({
            'rank': owner_ctx.rank, 'capturing': capture,
            'phase': 'producer_complete',
            'graph_key': [desc.num_tokens, desc.num_reqs, desc.uniform,
                          desc.has_lora, desc.num_active_loras],
            'owner_requests': [owner_ctx.req0, owner_ctx.req1],
            'owner_rows': list(owner_ctx.x.shape),
        }) + '\n')
    seen.add(capture)
    owner_ctx.impl._extreme_live_owner_complete_logged = seen


def verify_handoff(inputs, state, config):
    """Fail before replay if its dispatch key or fixed request rows can differ."""
    if os.getenv('EXTREME_LIVE_OWNER_LAYER2') != '1':
        return
    desc = inputs.target.batch_descriptor
    key = (desc.num_tokens, desc.num_reqs, desc.uniform,
           desc.has_lora, desc.num_active_loras)
    if key != (96, 12, True, False, 0):
        raise RuntimeError(f'owner replay graph key changed: {key}')
    if (config.batch_size, config.speculative_tokens) != (12, 7):
        raise RuntimeError('owner replay fixed contract changed')
    rank, size = inputs.target_tp_rank, inputs.target_tp_size
    if size != 8 or rank not in range(8):
        raise RuntimeError('owner replay TP geometry changed')
    if (tuple(state.target_input_ids.shape) != (96,)
            or tuple(state.target_positions.shape) != (96,)
            or tuple(state.target_query_start_loc.shape) != (13,)):
        raise RuntimeError('owner replay target tensor shape changed')
    query = state.target_query_start_loc.to('cpu').tolist()
    if query != list(range(0, 97, 8)):
        raise RuntimeError(f'owner replay qsl changed: {query}')
    from vllm.config import CUDAGraphMode
    if (inputs.target.aclgraph_runtime_mode != CUDAGraphMode.FULL
            or inputs.target.skip_compiled):
        raise RuntimeError('owner replay requires compiled target FULL Graph')
    trace_dir = Path(os.environ['EXTREME_LIVE_OWNER_TRACE_DIR'])
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f'handoff_rank{rank}.json').write_text(json.dumps({
        'rank': rank, 'graph_key': list(key), 'query_start_loc': query,
        'owner_requests': [(rank * 12) // 8, (rank * 12 + 11) // 8],
        'owner_rows': [16, 4096], 'target_graph_requested': True,
    }, indent=2) + '\n')


def verify_replay_dispatch(handoff, num_tokens):
    """Check the key at the actual Runtime target call before Graph dispatch."""
    desc = handoff.batch_descriptor
    key = (desc.num_tokens, desc.num_reqs, desc.uniform,
           desc.has_lora, desc.num_active_loras)
    from vllm.config import CUDAGraphMode
    if (num_tokens != 96 or key != (96, 12, True, False, 0)
            or handoff.skip_compiled or handoff.num_actual_tokens != 96
            or handoff.aclgraph_runtime_mode != CUDAGraphMode.FULL):
        raise RuntimeError(f'owner Runtime replay dispatch mismatch: {num_tokens}, {key}')
    if getattr(handoff, '_extreme_live_owner_replay_logged', False):
        return
    import torch.distributed as dist
    rank = dist.get_rank()
    path = Path(os.environ['EXTREME_LIVE_OWNER_TRACE_DIR'])
    path.mkdir(parents=True, exist_ok=True)
    (path / f'replay_rank{rank}.json').write_text(json.dumps({
        'rank': rank, 'graph_key': list(key), 'num_tokens': num_tokens,
        'num_actual_tokens': handoff.num_actual_tokens,
        'compiled': True,
    }, indent=2) + '\n')
    handoff._extreme_live_owner_replay_logged = True


def verify_generic_graph_dispatch(batch_desc, graph_mode, num_reqs,
                                  num_tokens, scheduled_tokens):
    """Guard all ordinary runner replays of the owner-specialized Graph key."""
    key = (batch_desc.num_tokens, batch_desc.num_reqs, batch_desc.uniform,
           batch_desc.has_lora, batch_desc.num_active_loras)
    if key != (96, 12, True, False, 0):
        return
    from vllm.config import CUDAGraphMode
    schedule = [int(x) for x in scheduled_tokens[:num_reqs]]
    if (graph_mode != CUDAGraphMode.FULL or num_reqs != 12
            or num_tokens != 96 or schedule != [8] * 12):
        raise RuntimeError(f'owner ordinary Graph replay geometry changed: {key}, {schedule}')
    import torch.distributed as dist
    rank = dist.get_rank()
    path = Path(os.environ['EXTREME_LIVE_OWNER_TRACE_DIR'])
    path.mkdir(parents=True, exist_ok=True)
    marker = path / f'generic_rank{rank}.json'
    if not marker.exists():
        marker.write_text(json.dumps({
            'rank': rank, 'graph_key': list(key),
            'schedule': schedule, 'mode': 'FULL',
        }, indent=2) + '\n')


class OwnerContext:
    def __init__(self, impl, *, layer_name, full_x, attn_metadata):
        from vllm_ascend.device.device_op import DeviceOperator
        from scripts.loop068_full_producer_fixture import _owner_metadata
        self.impl = impl
        self.rank = int(impl.tp_rank)
        self.req0 = (self.rank * 12) // 8
        self.req1 = (self.rank * 12 + 11) // 8
        if self.req1 != self.req0 + 1:
            raise RuntimeError('owner request geometry changed')
        self.req_slice = slice(self.req0, self.req1 + 1)
        self.row_slice = slice(self.req0 * 8, (self.req1 + 1) * 8)
        self.x = full_x[self.row_slice].contiguous()
        cmp_meta, main_state_meta, index_state_meta, index_scale_meta, swa_meta = attn_metadata
        req = cmp_meta.req_metadata
        self.cos = req.cos[layer_name][self.row_slice]
        self.sin = req.sin[layer_name][self.row_slice]
        self.swa_slots = swa_meta.req_metadata.slot_mapping[self.row_slice]
        self.qsl = req.query_start_loc
        self.cmp_meta = cmp_meta
        self.main_state_meta = main_state_meta
        self.index_state_meta = index_state_meta
        self.index_scale_meta = index_scale_meta
        self._metadata = _owner_metadata
        self._device_op = DeviceOperator

    def indexer_update(self, kv_cache):
        from vllm_ascend.attention.context_parallel.dsa_cp import rotate_activation
        state, key, scale, full = self._device_op.unpack_dsa_indexer_kv_cache(kv_cache)
        if full is not None:
            raise RuntimeError('non-A5 indexer cache ABI changed')
        qsl, start, _, cos, sin, slots = self._metadata(
            self.impl, self.index_scale_meta, self.qsl, self.req_slice, 4)
        state_bt = self.index_state_meta.req_metadata.block_table[self.req_slice].contiguous()
        kv = torch.ops._C_ascend.compressor(
            self.x, self.impl.indexcom_wkv.weight, self.impl.indexcom_wgate.weight,
            state.squeeze(-2), self.impl.indexcom_ape, self.impl.indexcom_norm.weight,
            sin.view(-1, sin.shape[-1]), cos.view(-1, cos.shape[-1]),
            state_block_table=state_bt, cu_seqlens=qsl, seqused=None,
            start_pos=start, rope_head_dim=self.impl.rope_head_dim,
            cmp_ratio=4, coff=2 if self.impl.compressor_overlap else 1,
            norm_eps=self.impl.compressor_norm_eps, rotary_mode=2, cache_mode=1)
        if kv.numel() == 0:
            return
        if self.impl.indexer.compressor.rotate:
            kv = rotate_activation(kv, self.index_scale_meta.hadamard)
        _, kv_scale = self._device_op.indexer_quant_scatter_part1(kv, key, None, slots)
        if kv_scale is not None:
            self._device_op.dsa_indexer_scatter_scale_part3(kv_scale, scale, slots)

    def main_metadata(self):
        qsl, start, _, cos, sin, slots = self._metadata(
            self.impl, self.cmp_meta, self.qsl, self.req_slice, 4)
        state_bt = self.main_state_meta.req_metadata.block_table[self.req_slice].contiguous()
        return qsl, start, state_bt, cos, sin, slots
