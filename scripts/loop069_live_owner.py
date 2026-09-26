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
    selected = (os.getenv('EXTREME_LIVE_OWNER_LAYER2') == '1'
            and layer_name == 'model.layers.2.self_attn.attn'
            and impl.compress_ratio == 4 and not has_prefill
            and tuple(hidden_states_cache.shape) == (96, 4096)
            and tuple(hidden_states_local.shape) == (12, 4096))
    if selected and os.getenv('EXTREME_LIVE_OWNER_TRACE_DIR'):
        from vllm.forward_context import get_forward_context
        capture = bool(getattr(get_forward_context(), 'capturing', False)
                       or torch.npu.is_current_stream_capturing())
        seen = getattr(impl, '_extreme_live_owner_logged', set())
        if capture not in seen:
            rank = int(impl.tp_rank)
            path = Path(os.environ['EXTREME_LIVE_OWNER_TRACE_DIR'])
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'rank{rank}.jsonl').open('a') as handle:
                handle.write(json.dumps({'rank': rank, 'capturing': capture,
                                         'full_shape': list(hidden_states_cache.shape),
                                         'local_shape': list(hidden_states_local.shape)}) + '\n')
            seen.add(capture)
            impl._extreme_live_owner_logged = seen
    return selected


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
