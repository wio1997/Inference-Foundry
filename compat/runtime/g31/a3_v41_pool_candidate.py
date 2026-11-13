"""Isolated, UNVALIDATED A3 V4.1 {0,1,2} cache adapter candidate.

This file does not install itself. Integrator must select both pool and allocator
below in kv_cache_configurator for the target model only. BF16 latents / INT8
index keys follow the working A3 vLLM donor, not CUDA FP4 cache semantics.
Requires the matching CANN backend adapter; never use generic CUDA pack/read
operations on these buffers. No model or NPU validation has been performed.
"""

from __future__ import annotations

import torch
import torch_npu

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
    DSV4NPUTokenToKVPool,
    NPUDeepSeekV4IndexerPool,
    NPUDeepSeekV4SingleKVPool,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout


class A3V41IndexerPool(NPUDeepSeekV4IndexerPool):
    """CANN PA_BBND K/scale pair without an unused CUDA packed shadow buffer."""

    storage_contract = "ascend_a3_int8_fp16_pa_bbnd"

    def _create_buffer(self):
        if is_npu_arch35():
            raise ValueError("A3V41IndexerPool is specific to Ascend A3")
        if self.use_fp4_indexer:
            raise ValueError("CUDA FP4 index cache cannot back the A3 INT8 ABI")
        kp = self._kernel_page_size
        num_pages = (self.size + kp + kp - 1) // kp
        self.index_k_with_scale_buffer = None
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.index_k_buffer = [
                torch.zeros(num_pages, kp, 1, self.index_head_dim,
                            dtype=torch.int8, device=self.device)
                for _ in range(self.layer_num)
            ]
            self.index_scale_buffer = [
                torch.zeros(num_pages, kp, 1, 1,
                            dtype=torch.float16, device=self.device)
                for _ in range(self.layer_num)
            ]

    def get_bytes_per_token(self):
        return self.index_head_dim + torch.float16.itemsize

    def get_index_k_with_scale_buffer(self, layer_id):
        raise RuntimeError("Use get_index_kv_npu: A3 stores separate INT8 K/FP16 scale")


class A3V41TokenToKVPool(DSV4NPUTokenToKVPool):
    """Retain generic source mapping and C2 state; bind storage to A3 PA_ND."""

    storage_contract = "ascend_a3_bf16_pa_nd_int8_index"

    def __init__(self, *args, **kwargs):
        ratios = kwargs.get("compression_ratios")
        if ratios is None or not set(ratios) <= {0, 1, 2}:
            raise ValueError("Candidate accepts keyword compression_ratios in {0,1,2}")
        if kwargs.get("dtype") != torch.bfloat16 or is_npu_arch35():
            raise ValueError("Candidate requires Ascend A3 and BF16 KV")
        super().__init__(*args, **kwargs)
        if self._unified_kv or self.request_window is not None:
            raise ValueError("Candidate requires the existing paged SWA allocator")

    def _init_compressed_pools(self, *, stage_ratios, page_size, dtype, device,
                               enable_memory_saver, enable_hisparse, kv_pool_cls):
        if enable_hisparse or kv_pool_cls is not DeepSeekV4SingleKVPool:
            raise ValueError("Candidate requires standard BF16 A3 pools")
        if self.full_size is None or self.full_size % page_size:
            raise ValueError("full_size must be a page-aligned full-slot capacity")
        self.kv_pools = {}
        self.index_pools = {}
        for ratio in (1, 2):
            sources = self.sources_by_ratio.get(ratio, [])
            if not sources:
                continue
            if page_size % ratio:
                raise ValueError("Full page size must be divisible by compression ratio")
            physical_page = page_size // ratio
            # Real full slots start at Pfull, so compressed page zero is reserved
            # too. Both buffers allocate capacity + one reserved physical page.
            compressed_size = self.full_size // ratio
            self.kv_pools[ratio] = NPUDeepSeekV4SingleKVPool(
                compressed_size, physical_page, dtype,
                self.qk_nope_head_dim, self.qk_rope_head_dim,
                len(sources), device, enable_memory_saver,
                kernel_page_size=physical_page,
                global_page_size=physical_page,
                kv_layout=KVLayout.V4,
            )
            self.index_pools[ratio] = self._make_indexer_pool(
                compressed_size, physical_page, dtype, self.indexer_head_dim,
                len(sources), device, enable_memory_saver,
                force_fp4=True, global_page_size=physical_page,
                index_k_page8=False,
            )
        self.c4_kv_pool = self.kv_pools.get(4)
        self.c128_kv_pool = self.kv_pools.get(128)
        self.c4_indexer_kv_pool = self.index_pools.get(4)

    def _make_indexer_pool(self, size, page_size, dtype, index_head_dim,
                           layer_num, device, enable_memory_saver,
                           force_fp4=False, global_page_size=None,
                           index_k_page8=False):
        if index_k_page8:
            raise ValueError("AITER page8 packing is not the A3 cache ABI")
        pool = A3V41IndexerPool(
            size, page_size, dtype, index_head_dim, layer_num, device,
            enable_memory_saver, use_fp4_indexer=False,
            kernel_page_size=page_size, global_page_size=global_page_size,
        )
        # Explicitly expose the representation difference; no FP4 compatibility
        # claim. The matched NPU backend must quantize raw RoPE'd K to INT8.
        pool.generic_requested_fp4 = force_fp4
        return pool

    def get_key_buffer(self, layer_id):
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return self.get_swa_buffer(layer_id)
        return item.compress_kv_pool.kv_buffer[item.compress_layer_id]

    def get_extra_key_buffer(self, layer_id):
        item = self.layer_mapping[layer_id]
        if item.compress_ratio == 0:
            return None
        return self.get_key_buffer(layer_id)

    def get_index_kv_npu(self, layer_id):
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        pool = self._indexer_pool(item.compress_ratio)
        return (pool.get_index_k(item.compress_layer_id),
                pool.get_index_scale(item.compress_layer_id))

    def set_extra_key_buffer_fused(self, layer_id, loc, cache_k, freqs_cis=None):
        if freqs_cis is not None:
            raise ValueError("A3 cache writes require already RoPE'd BF16 K")
        buf = self.get_extra_key_buffer(layer_id)
        if buf is None:
            raise ValueError("No compressed cache for ratio zero")
        self._scatter_pa(buf, loc, cache_k)

    @staticmethod
    def _scatter_pa(buf, loc, values):
        width = buf.shape[-1]
        # Negative graph-padding slots target reserved physical row zero.
        # Real full page zero is never allocated; repeated padding writes there
        # have no observable result. No dynamic-shape boolean/nonzero op.
        slots = loc.reshape(-1, 1).clamp_min(0).long()
        rows = values.reshape(-1, 1, width).to(buf.dtype)
        torch_npu.npu_scatter_nd_update_(buf.view(-1, 1, width), slots, rows)

    def set_index_kv_npu(self, layer_id, loc, key_int8, scale_fp16):
        if key_int8.dtype != torch.int8 or scale_fp16.dtype != torch.float16:
            raise ValueError("A3 index writes require quantized INT8 K and FP16 scale")
        key, scale = self.get_index_kv_npu(layer_id)
        self._scatter_pa(key, loc, key_int8)
        self._scatter_pa(scale, loc, scale_fp16)

    def set_compress_buffer(self, layer_id, loc, kv, kv_scale, from_indexer):
        if from_indexer:
            if kv_scale is None:
                raise ValueError("A3 index writes need explicit quantization scale")
            return self.set_index_kv_npu(layer_id, loc, kv, kv_scale)
        return self.set_extra_key_buffer_fused(layer_id, loc, kv)

    def get_contiguous_buf_infos(self):
        buffers = []
        for ratio in (1, 2):
            if ratio in self.kv_pools:
                buffers += self.kv_pools[ratio].kv_buffer
                buffers += self.index_pools[ratio].index_k_buffer
                buffers += self.index_pools[ratio].index_scale_buffer
        return ([b.data_ptr() for b in buffers], [b.nbytes for b in buffers],
                [b[0].nbytes for b in buffers])


class A3V41TokenToKVPoolAllocator(SWATokenToKVPoolAllocator):
    """Reuse full/SWA NPUPaged allocators; no unused C128 allocation/sidecar.

    Accept NPU request-pool free hooks so C2 state is cleared before slot reuse.
    Return plain full-slot tensors; existing DSV4 bundle hooks then no-op.
    """

    def alloc_extend(self, *args, req_pool_indices=None, req_to_token_pool=None,
                     **kwargs):
        return super().alloc_extend(*args, **kwargs)

    def alloc_decode(self, *args, req_pool_indices=None, req_to_token_pool=None,
                     **kwargs):
        return super().alloc_decode(*args, **kwargs)

    def alloc_extend_swa_tail(self, *args, req_pool_indices=None,
                              req_to_token_pool=None, **kwargs):
        return super().alloc_extend_swa_tail(*args, **kwargs)

    def free(self, free_index=None, *, req=None, req_to_token_pool=None):
        if free_index is not None:
            super().free(free_index)
        if req is not None and req.kv.req_pool_idx is not None:
            self.get_kvcache().clear_request_scoped_state(int(req.kv.req_pool_idx))
