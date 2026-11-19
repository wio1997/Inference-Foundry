# SPDX-License-Identifier: Apache-2.0
"""NPU MLA storage with an explicit compressed-key and request-tail contract.

The parent HybridLinearKVPool owns global->dense layer routing. This child
keeps the NPU MLA K/V allocation/writer and existing get_* buffer registration.
"""
import torch
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.runtime_context import get_parallel


class NPUKPoolTokenToKVPool(NPUMLATokenToKVPool):
    glm53_index_in_owner_kv = True
    adapter_generation = 2
    def __init__(self, *, size, page_size, dtype, kv_lora_rank, qk_rope_head_dim,
                 layer_num, device, index_head_dim, enable_memory_saver,
                 kv_cache_dim=None, index_kpool=4, index_kpool_compress=True,
                 tail_extra_slots=0, max_running_requests=None, skip_topk_layers=None,
                 start_layer=None, end_layer=None, is_draft_worker=False, **kwargs):
        if kwargs:
            raise TypeError(f'Unknown pool contract fields: {sorted(kwargs)}')
        if index_kpool != 4 or not index_kpool_compress or index_head_dim != 128:
            raise ValueError('Adapter is scoped to the frozen GLM compressed KPool4 config')
        if dtype != torch.bfloat16 or get_parallel().attn_dcp_size != 1:
            raise NotImplementedError('First integration gate requires BF16 KV and symmetric DCP1; other KV/DCP paths unverified')
        if max_running_requests is None or page_size % index_kpool:
            raise ValueError('KPool requires request-owned tails and pages divisible by ratio')
        self.index_kpool = index_kpool
        self.index_kpool_compress = self.kpool_use_compress = True
        self.tail_extra_slots = tail_extra_slots
        ids = tuple(i for i in range(layer_num) if skip_topk_layers is None or not skip_topk_layers[i])
        super().__init__(size=size,page_size=page_size,dtype=dtype,kv_lora_rank=kv_lora_rank,
                         qk_rope_head_dim=qk_rope_head_dim,layer_num=layer_num,device=device,
                         enable_memory_saver=enable_memory_saver,index_head_dim=index_head_dim,
                         index_size=size//index_kpool,index_page_size=page_size//index_kpool,
                         indexer_layer_ids=ids,start_layer=start_layer,end_layer=end_layer,
                         kv_cache_dim=kv_cache_dim,is_draft_worker=is_draft_worker)
        capacity=index_kpool+tail_extra_slots
        # Separate contiguous arrays preserve the existing DSA_TAIL wire ordering.
        self._compress_tail_k=[torch.zeros(max_running_requests+1,capacity,128,device=device,dtype=torch.float32) for _ in range(self.num_indexer_layers)]
        self._compress_tail_score=[torch.zeros_like(t) for t in self._compress_tail_k]

    def get_compress_tail_buffers(self,layer_id):
        slot=self._get_indexer_slot(layer_id)
        return self._compress_tail_k[slot],self._compress_tail_score[slot]

    def get_compress_tail_buf_infos(self):
        buffers=self._compress_tail_k+self._compress_tail_score
        return ([t.data_ptr() for t in buffers],[t.nbytes for t in buffers],[t[0].nbytes for t in buffers])

    def get_index_k_with_scale_buffer(self,layer_id):
        # Compatibility accessor only; these are native BF16 keys, without scales.
        return self.get_index_k_buffer(layer_id)

    def get_kv_size_bytes(self):
        result=super().get_kv_size_bytes()
        return result+sum(t.nbytes for t in getattr(self,'_compress_tail_k',[]))+sum(t.nbytes for t in getattr(self,'_compress_tail_score',[]))

    @staticmethod
    def _index_plan(indices):
        """Stable unique pool order plus input offsets, independent of page IDs."""
        ids = indices.detach().to(device="cpu",dtype=torch.long).flatten()
        if ids.numel() == 0:
            return ids, ids, ids
        if bool((ids < 0).any()):
            raise ValueError("Cache lifecycle copies require physical nonnegative slots")
        groups = ids // 4
        first = torch.cat((torch.ones(1,dtype=torch.bool),groups[1:] != groups[:-1]))
        offsets = first.nonzero().flatten()
        return groups[offsets], offsets, ids.remainder(4)

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        # Retraction is a lifecycle operation, outside the inference hot path.
        # Save compressed groups once, not once per raw token.
        torch.npu.synchronize()
        groups, offsets, mod = self._index_plan(indices)
        group_device = groups.to(device=self.index_k_buffer.device)
        raw = indices.to(device=self.k_buffer.device,dtype=torch.long)
        result = {"schema":"glm53_npu_kpool_g2", "count":raw.numel(),
                  "pool_offsets":offsets, "raw_mod4":mod, "kv":[], "index":[],
                  "tail_k":None, "tail_score":None}
        chunk = self.cpu_offloading_chunk_size
        for layer in range(self.layer_num):
            buffers=(self.k_buffer[layer].flatten(0,1),self.v_buffer[layer].flatten(0,1))
            result["kv"].append([[b.index_select(0,raw[i:i+chunk]).cpu() for b in buffers]
                                  for i in range(0,raw.numel(),chunk)])
        for slot in range(self.num_indexer_layers):
            b=self.index_k_buffer[slot].flatten(0,1)
            result["index"].append([b.index_select(0,group_device[i:i+chunk]).cpu()
                                    for i in range(0,groups.numel(),chunk)])
        if req_pool_index is not None:
            result["tail_k"]=[b[req_pool_index].cpu().clone() for b in self._compress_tail_k]
            result["tail_score"]=[b[req_pool_index].cpu().clone() for b in self._compress_tail_score]
        torch.npu.synchronize()
        return result

    def load_cpu_copy(self, saved, indices, mamba_indices=None, req_pool_index=None):
        if saved.get("schema") != "glm53_npu_kpool_g2":
            raise ValueError("Unexpected cache snapshot schema")
        groups, offsets, mod = self._index_plan(indices)
        if indices.numel()!=saved["count"] or not torch.equal(mod,saved["raw_mod4"]) or not torch.equal(offsets,saved["pool_offsets"]):
            raise ValueError("Restore changed compressed-pool alignment/grouping")
        raw=indices.to(device=self.k_buffer.device,dtype=torch.long)
        gi=groups.to(device=self.index_k_buffer.device)
        chunk=self.cpu_offloading_chunk_size
        for layer in range(self.layer_num):
            buffers=(self.k_buffer[layer].flatten(0,1),self.v_buffer[layer].flatten(0,1))
            for n,i in enumerate(range(0,raw.numel(),chunk)):
                for b,cpu in zip(buffers,saved["kv"][layer][n],strict=True):
                    b.index_copy_(0,raw[i:i+chunk],cpu.to(device=b.device))
        for slot,b in enumerate(self.index_k_buffer):
            b=b.flatten(0,1)
            for n,i in enumerate(range(0,gi.numel(),chunk)):
                b.index_copy_(0,gi[i:i+chunk],saved["index"][slot][n].to(device=b.device))
        if saved["tail_k"] is not None:
            if req_pool_index is None:
                raise ValueError("Request-tail restore requires a destination request slot")
            for dest,cpu in zip(self._compress_tail_k,saved["tail_k"],strict=True):
                dest[req_pool_index]=cpu.to(device=dest.device)
            for dest,cpu in zip(self._compress_tail_score,saved["tail_score"],strict=True):
                dest[req_pool_index]=cpu.to(device=dest.device)
        torch.npu.synchronize()

    def move_kv_cache(self, tgt_loc, src_loc):
        # Whole aligned pools move atomically with the raw K/V rows. Inference
        # never reconstructs a compressed vector from arbitrary mixed tokens.
        if tgt_loc.shape != src_loc.shape:
            raise ValueError("Source and destination token spans differ")
        src_groups,src_offsets,src_mod=self._index_plan(src_loc)
        dst_groups,dst_offsets,dst_mod=self._index_plan(tgt_loc)
        if not torch.equal(src_offsets,dst_offsets) or not torch.equal(src_mod,dst_mod):
            raise ValueError("Relocation must preserve pool membership and offsets")
        counts=torch.diff(torch.cat((src_offsets,torch.tensor([src_loc.numel()]))))
        if bool((counts != 4).any()) or bool((src_mod[src_offsets] != 0).any()):
            if torch.equal(tgt_loc,src_loc):
                return
            raise ValueError("Relocation requires complete aligned four-token pools")
        src=src_loc.long();dst=tgt_loc.long()
        for buffers in (self.k_buffer,self.v_buffer):
            for layer in buffers:
                b=layer.flatten(0,1)
                b.index_copy_(0,dst,b.index_select(0,src))
        sg=src_groups.to(device=self.index_k_buffer.device)
        dg=dst_groups.to(device=self.index_k_buffer.device)
        for layer in self.index_k_buffer:
            b=layer.flatten(0,1)
            b.index_copy_(0,dg,b.index_select(0,sg))

    def set_kv_buffer(self,layer,loc_info,cache_k,cache_v):
        if self.qk_rope_head_dim != 0:
            return super().set_kv_buffer(layer,loc_info,cache_k,cache_v)
        from sglang.srt.mem_cache.memory_pool import unwrap_write_loc
        import torch_npu
        loc,_,_=unwrap_write_loc(loc_info)
        dst=self.k_buffer[layer.layer_id-self.start_layer].reshape(-1,1,self.kv_lora_rank)
        values=cache_k.reshape(cache_k.shape[0],1,self.kv_lora_rank).to(dtype=dst.dtype)
        torch_npu.npu_scatter_nd_update_(dst,loc.reshape(-1,1),values)
        # Zero-width rope has no physical writes or ambiguous -1 reshape.
