"""910B BF16 low-ratio pools using native ownership and native FP4 storage.

Standalone integration candidate; no active service imports this module yet.
"""
import torch
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
    DSV4NPUTokenToKVPool, NPUDeepSeekV4SingleKVPool,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool
from sglang.srt.mem_cache.deepseek_v4_buffer_info import DSV4PagedBufferInfo
from fp4_index_pool import make_low_ratio_index_pool


class LowRatioBF16Pool(NPUDeepSeekV4SingleKVPool):
    def get_bytes_per_token(self):
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) * 2

    def paged_buffer_info(self):
        # A physical C2 page holds two logical compressed pages. Describe
        # zero-copy logical-page views, retaining the same underlying storage.
        views = [buf.view(-1, self.page_size, 1, self.kv_cache_total_dim)
                 for buf in self.kv_buffer]
        return DSV4PagedBufferInfo.from_buffers(
            views, page_slots=self.page_size, layout="pa_nd")


class LowRatioNPUTokenToKVPool(DSV4NPUTokenToKVPool):
    def _make_kv_pool(self, *, size, page_size, dtype, layer_num, device,
                      enable_memory_saver, global_page_size,
                      cls=DeepSeekV4SingleKVPool, kv_layout=KVLayout.V4):
        if page_size not in (global_page_size, global_page_size // 2):
            return super()._make_kv_pool(
                size=size, page_size=page_size, dtype=dtype, layer_num=layer_num,
                device=device, enable_memory_saver=enable_memory_saver,
                global_page_size=global_page_size, cls=cls, kv_layout=kv_layout)
        assert cls is DeepSeekV4SingleKVPool and kv_layout is KVLayout.V4
        assert dtype == torch.bfloat16, "This candidate is scoped to BF16 910B KV"
        return LowRatioBF16Pool(
            size, page_size, dtype, self.qk_nope_head_dim, self.qk_rope_head_dim,
            layer_num, device, enable_memory_saver,
            global_page_size=global_page_size, kernel_page_size=global_page_size)

    def _make_indexer_pool(self, size, page_size, dtype, index_head_dim,
                          layer_num, device, enable_memory_saver,
                          force_fp4=False, global_page_size=None,
                          index_k_page8=False):
        if not force_fp4:
            assert not index_k_page8
            return super()._make_indexer_pool(
                size, page_size, dtype, index_head_dim, layer_num, device,
                enable_memory_saver, global_page_size=global_page_size)
        assert not index_k_page8, "HIP page-8 layout is not an NPU storage format"
        return make_low_ratio_index_pool(
            size, page_size, dtype, index_head_dim, layer_num, device,
            enable_memory_saver, global_page_size=global_page_size)

    def get_key_buffer(self, layer_id):
        self.wait_layer_transfer(layer_id)
        mapping = self.layer_mapping[layer_id]
        if mapping.compress_ratio in (1, 2):
            return mapping.compress_kv_pool.kv_buffer[mapping.compress_layer_id]
        return super().get_key_buffer(layer_id)

    def get_compress_buffer(self, layer_id, from_indexer=False, loc=None):
        mapping = self.layer_mapping[layer_id]
        if mapping.compress_ratio not in (1, 2):
            return super().get_compress_buffer(layer_id, from_indexer, loc)
        self.wait_layer_transfer(layer_id)
        if from_indexer:
            return self.get_low_ratio_index_k_dequant(layer_id, loc)
        buf = mapping.compress_kv_pool.kv_buffer[mapping.compress_layer_id]
        return buf if loc is None else buf.flatten(0, 1)[loc]

    def get_contiguous_buf_infos(self):
        ptrs, lengths, items = super().get_contiguous_buf_infos()
        # Retain native C4 order (and separate native C128 state group), then
        # append source-owned C1/C2 groups in native registry insertion order.
        for ratio, pool in self.kv_pools.items():
            if ratio not in (1, 2):
                continue
            rows = self.page_size // ratio
            for buf in pool.kv_buffer:
                ptrs.append(buf.data_ptr())
                lengths.append(buf.nbytes)
                items.append(rows * buf.shape[-1] * buf.element_size())
            indexer = self.index_pools[ratio]
            assert rows % indexer.page_size == 0
            for buf in indexer.contiguous_page_row_buffers():
                ptrs.append(buf.data_ptr())
                lengths.append(buf.nbytes)
                items.append(buf[0].nbytes * (rows // indexer.page_size))
        return ptrs, lengths, items

    def get_c128_kv_buf_infos(self):
        if self.c128_kv_pool is None:
            return [], [], []
        return super().get_c128_kv_buf_infos()

    def set_extra_key_buffer_fused(self, layer_id, loc, cache_k, freqs_cis=None):
        item=self.layer_mapping[layer_id]
        if item.compress_ratio not in (1,2):
            return super().set_extra_key_buffer_fused(layer_id,loc,cache_k,freqs_cis)
        assert freqs_cis is None, 'BF16 low-ratio store receives native post-RoPE FP4 values'
        buf=item.compress_kv_pool.kv_buffer[item.compress_layer_id]
        buf.view(-1,512)[loc.long()]=cache_k.reshape(-1,512).to(torch.bfloat16)

    def get_swa_buffer(self, layer_id, loc=None):
        self.wait_layer_transfer(layer_id)
        buf=self.get_swa_raw_buffer(layer_id)
        return buf if loc is None else buf.view(-1,1,512)[loc]

    def get_state_buf_infos(self):
        if self.request_window is None:
            return super().get_state_buf_infos()
        buffers=self.request_window.state.kv_buffer
        assert all(b.dtype==torch.bfloat16 and b.shape[-1]==512 for b in buffers)
        return ([b.data_ptr() for b in buffers],[b.nbytes for b in buffers],
                [self.swa_page_size*1024 for b in buffers])
