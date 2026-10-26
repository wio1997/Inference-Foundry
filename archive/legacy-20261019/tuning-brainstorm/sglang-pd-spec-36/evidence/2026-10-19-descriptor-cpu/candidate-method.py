
def range_metadata(src_indices, dst_indices):
    if src_indices.size == 0 or dst_indices.size == 0:
        return []
    if src_indices.size != dst_indices.size:
        raise ValueError("group_concurrent_contiguous requires equal-length src/dst index arrays, "
                         f"got {src_indices.size} and {dst_indices.size}")
    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), brk))
    ends = np.concatenate((brk, np.array([src_indices.size], dtype=np.int64)))
    return [(int(src_indices[start]), int(dst_indices[start]), int(end-start))
            for start, end in zip(starts, ends)]

def candidate_sender(self, mooncake_session_id: str, prefill_kv_indices: npt.NDArray[np.int32], dst_kv_ptrs: list[int], dst_kv_indices: npt.NDArray[np.int32], executor: concurrent.futures.ThreadPoolExecutor, dst_layer_ids: Optional[List[int]]=None, dst_device_kv_indices: Optional[npt.NDArray[np.int32]]=None, dst_kv_item_len: Optional[int]=None, dst_attn_tp_size: Optional[int]=None):
    if dst_device_kv_indices is not None:
        raise NotImplementedError('Ascend PD transfer does not support HiSparse destination device KV indices')
    self._validate_envelope_kv_layout(dst_kv_ptrs, dst_kv_item_len, dst_attn_tp_size)
    if self.is_hybrid_mla_backend and self.pp_size > 1:
        return self._send_kvcache_generic(mooncake_session_id=mooncake_session_id, src_data_ptrs=self.kv_args.kv_data_ptrs, dst_data_ptrs=dst_kv_ptrs, item_lens=self.kv_args.kv_item_lens, prefill_data_indices=prefill_kv_indices, dst_data_indices=dst_kv_indices, executor=executor, src_layer_ids=self.kv_args.kv_layer_ids, dst_layer_ids=dst_layer_ids)
    transfer_ranges = range_metadata(prefill_kv_indices, dst_kv_indices)
    if self.pp_size > 1:
        if self.is_mla_backend:
            src_kv_ptrs, sliced_dst_kv_ptrs, layers_current_pp_stage = self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            layers_params = [(src_kv_ptrs[layer_id], sliced_dst_kv_ptrs[layer_id], self.kv_args.kv_item_lens[layer_id]) for layer_id in range(layers_current_pp_stage)]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            layers_params = [(src_k_ptrs[layer_id], dst_k_ptrs[layer_id], self.kv_args.kv_item_lens[layer_id]) for layer_id in range(layers_current_pp_stage)] + [(src_v_ptrs[layer_id], dst_v_ptrs[layer_id], self.kv_args.kv_item_lens[layers_current_pp_stage + layer_id]) for layer_id in range(layers_current_pp_stage)]
    else:
        num_layers = len(self.kv_args.kv_data_ptrs)
        layers_params = [(self.kv_args.kv_data_ptrs[layer_id], dst_kv_ptrs[layer_id], self.kv_args.kv_item_lens[layer_id]) for layer_id in range(num_layers)]

    def set_transfer_blocks(src_ptr, dst_ptr, item_len):
        transfer_blocks = []
        for src_start, dst_start, run_len in transfer_ranges:
            src_addr = src_ptr + src_start * item_len
            dst_addr = dst_ptr + dst_start * item_len
            length = item_len * run_len
            transfer_blocks.append((src_addr, dst_addr, length))
        return transfer_blocks

    def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
        transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
        transfer_blocks = []
        for src_ptr, dst_ptr, item_len in layers_params:
            transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
        return self._transfer_data(mooncake_session_id, transfer_blocks)
    if self.enable_custom_mem_pool:
        futures = [executor.submit(process_layer, src_ptr, dst_ptr, item_len) for src_ptr, dst_ptr, item_len in layers_params]
        for future in concurrent.futures.as_completed(futures):
            status = future.result()
            if status != 0:
                for f in futures:
                    f.cancel()
                return status
    else:
        return process_layers(layers_params)
    return 0
