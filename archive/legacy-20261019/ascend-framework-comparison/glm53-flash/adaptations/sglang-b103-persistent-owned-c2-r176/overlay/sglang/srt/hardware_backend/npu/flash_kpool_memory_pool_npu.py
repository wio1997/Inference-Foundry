# SPDX-License-Identifier: Apache-2.0
"""Candidate B103 Flash BF16 index/FP32 tail storage, pending NPU tests.

Uses B103's NPU MLA allocation (Apache-2.0). Hybrid disaggregation already
registers main KV, DSA index state and request-local DSA tails separately;
therefore compressed index buffers must not also appear as main KV here.
Initial scope: TP16/DP1/CP1, NoPE, eager, no draft/cache offload or relocation.
"""
import torch
import os
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool


class FlashNPUKPool(NPUMLATokenToKVPool):
    _flash_npu_kpool = True
    index_k_with_scale_buffer_dtype = torch.bfloat16
    quant_block_size = 128

    def __init__(self, size, page_size, kv_lora_rank, dtype, qk_rope_head_dim,
                 layer_num, device, index_head_dim, enable_memory_saver,
                 kv_cache_dim, start_layer=None, end_layer=None,
                 index_buf_size=None, index_kpool=4, index_kpool_compress=True,
                 tail_extra_slots=0, max_running_requests=None,
                 skip_topk_layers=None):
        if enable_memory_saver:
            raise ValueError('Flash offload/reallocation requires separate hybrid snapshot qualification')
        if dtype != torch.bfloat16 or qk_rope_head_dim != 0:
            raise ValueError('Initial Flash pool requires BF16 latent KV and NoPE')
        if kv_lora_rank != 512 or kv_cache_dim != 512 or index_head_dim != 128:
            raise ValueError('Unexpected Flash latent/index dimensions')
        if index_kpool != 4 or not index_kpool_compress or tail_extra_slots:
            raise ValueError('Initial Flash pool is compressed KPool4 without MTP')
        if size <= 0 or page_size <= 0 or page_size % 4:
            raise ValueError('Token pages must preserve complete KPool4 groups')
        if max_running_requests is None or max_running_requests < 1:
            raise ValueError('Request-local tail capacity is required')
        if skip_topk_layers is not None and (len(skip_topk_layers) != layer_num or any(skip_topk_layers)):
            raise ValueError('Initial Flash pool requires all full-attention indexers')
        if index_buf_size is not None and index_buf_size != size:
            raise ValueError('Independent index capacity needs separate page mapping qualification')
        super().__init__(size=size,page_size=page_size,dtype=dtype,
            kv_lora_rank=kv_lora_rank,qk_rope_head_dim=0,layer_num=layer_num,
            device=device,enable_memory_saver=enable_memory_saver,
            index_head_dim=128,index_size=size//4,index_page_size=page_size//4,
            start_layer=start_layer,end_layer=end_layer,kv_cache_dim=512)
        if self.dcp_size != 1 or self.use_fia_nz or self.enable_sparsity_driven_kv_offload:
            raise ValueError('Initial Flash pool requires DCP1, ordinary layout, native cache')
        self.index_kpool = 4
        self.index_kpool_compress = True
        self.kpool_use_compress = True
        self.tail_extra_slots = 0
        self.slots_per_page = page_size
        self.skip_topk_layers = [False] * layer_num
        self.req_pool_size = max_running_requests + 1
        # Separate contiguous buffers preserve the existing DSA_TAIL wire
        # contract: all layer keys, then all layer scores, indexed by request.
        shape = (layer_num,self.req_pool_size,4,128)
        self._flash_persistent_owned = os.environ.get('FLASH_B103_PERSISTENT_OWNED_STORAGE','0') == '1'
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            if self._flash_persistent_owned:
                # Allocate independent logical per-layer tensors BEFORE native
                # SDK registration. Same fresh factory as R168's proven mirror;
                # no per-step full-cache copies or copy-back. Initial content
                # remains all zeros and the wire descriptor order is unchanged.
                packed_index = self.index_k_buffer
                if packed_index is None or len(packed_index)!=layer_num:
                    raise ValueError('Missing dense native index layers')
                index_shape = tuple(packed_index.shape[1:])
                self.index_k_buffer = [torch.empty(index_shape,dtype=packed_index.dtype,
                    device=packed_index.device,memory_format=torch.contiguous_format).zero_()
                    for _ in range(layer_num)]
                del packed_index
                self._flash_tail_key = [torch.empty(shape[1:],dtype=torch.float32,
                    device=device,memory_format=torch.contiguous_format).zero_()
                    for _ in range(layer_num)]
                self._flash_tail_score = [torch.empty(shape[1:],dtype=torch.float32,
                    device=device,memory_format=torch.contiguous_format).zero_()
                    for _ in range(layer_num)]
            else:
                self._flash_tail_key = torch.zeros(shape,dtype=torch.float32,device=device)
                self._flash_tail_score = torch.zeros(shape,dtype=torch.float32,device=device)
        self._finalize_allocation_log(size)

    def get_kv_size_bytes(self):
        size = super().get_kv_size_bytes()
        for name in ['_flash_tail_key','_flash_tail_score']:
            value = getattr(self,name,None)
            if value is not None: size += sum(t.nbytes for t in value) if isinstance(value,list) else value.nbytes
        return size

    def get_contiguous_buf_infos(self):
        # Hybrid's DSA state inventory owns index_k; RoPE has zero bytes.
        buffers = list(self.k_buffer)
        return ([x.data_ptr() for x in buffers], [x.nbytes for x in buffers],
                [x[0].nbytes for x in buffers])

    def get_dcp_remote_decode_layout(self):
        return [False] * self.layer_num

    def get_kv_layer_ids(self):
        return list(range(self.start_layer,self.start_layer+self.layer_num))

    def get_compress_tail_buffers(self, layer_id):
        offset = layer_id-self.start_layer
        if not 0 <= offset < self.layer_num: raise ValueError('Wrong dense Flash layer ID')
        return self._flash_tail_key[offset], self._flash_tail_score[offset]

    def get_npu_kpool_buffers(self, layer_id):
        key,score = self.get_compress_tail_buffers(layer_id)
        return self.get_index_k_buffer(layer_id),key,score

    def get_compress_tail_buf_infos(self):
        buffers = list(self._flash_tail_key) + list(self._flash_tail_score)
        return ([x.data_ptr() for x in buffers], [x.nbytes for x in buffers],
                [x[0].nbytes for x in buffers])

    def reset_flash_request_tail(self, request_slot):
        if not 0 <= request_slot < self.req_pool_size:
            raise ValueError('Wrong Flash tail request slot')
        if self._flash_persistent_owned:
            for key,score in zip(self._flash_tail_key,self._flash_tail_score):
                key[request_slot].zero_();score[request_slot].zero_()
        else:
            self._flash_tail_key[:,request_slot].zero_()
            self._flash_tail_score[:,request_slot].zero_()

    def move_kv_cache(self, *args, **kwargs):
        raise NotImplementedError('Flash compressed page relocation requires a separate tested mapping')

    def get_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError('Flash hybrid cache snapshots require index and FP32 tails')

    def load_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError('Flash hybrid cache restore is not qualified')
