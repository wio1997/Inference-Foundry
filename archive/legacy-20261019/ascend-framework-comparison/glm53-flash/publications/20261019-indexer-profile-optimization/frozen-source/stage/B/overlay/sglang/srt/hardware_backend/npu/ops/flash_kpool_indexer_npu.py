# SPDX-License-Identifier: Apache-2.0
"""Bounded eager Flash KPool candidate; no B103 NPU/model qualification yet.

FP32 WK/norm/gates and BF16 query/head-weight ordering follow the pinned
vLLM-Ascend Flash backend. Licensed compression/score kernels are generated
separately. CPU metadata validation is diagnostic, not a performance path.
"""
import torch
import os
import torch.nn.functional as F

_PERFORMANCE = os.environ.get('FLASH_PD_PERF_MODE','0') == '1'
if _PERFORMANCE and any(os.environ.get(key) != '0' for key in
    ['FLASH_PD_COMPRESSION_ABI_AUDIT','FLASH_PD_SCORE_ABI_AUDIT','FLASH_SDMA_VERIFY_BYTES']):
    raise ValueError('Performance mode requires explicit CPU audit-off flags')


def forward_flash_kpool(indexer, x, q_lora, positions, forward_batch,
                       layer_id, layer_scatter_modes=None, dynamic_scale=None):
    from sglang.srt.model_executor.forward_context import get_attn_backend, get_token_to_kv_pool
    from sglang.srt.hardware_backend.npu.ops.flash_kpool_compression import glm5_next_kpool_tail_compress_and_write_cache_triton
    from sglang.srt.hardware_backend.npu.ops.flash_indexer_query_parallel import query_parallel_indexer
    from sglang.srt.runtime_context import get_parallel

    if not indexer.skip_rope or indexer.index_kpool != 4 or not indexer.index_kpool_compress:
        raise ValueError('Initial NPU indexer supports Flash NoPE KPool4 only')
    if indexer.head_dim != 128 or indexer.n_heads != 32 or indexer.index_topk != 2048:
        raise ValueError('Unexpected Flash indexer dimensions')
    if forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend_v2():
        raise ValueError('MTP requires separately qualified KPool tail rollback')
    if dynamic_scale is not None or not isinstance(q_lora,torch.Tensor):
        raise ValueError('Flash initial indexer expects unquantized q_lora activations')
    count=x.shape[0]
    if count==0:return torch.empty((0,2051),dtype=torch.int32,device=x.device)
    pool=get_token_to_kv_pool()
    if not getattr(pool.full_kv_pool,'_flash_npu_kpool',False):
        raise ValueError('Actual Flash NPU pool was not selected')
    backend=get_attn_backend().full_attn_backend
    block_tables=backend.forward_metadata.block_tables
    if block_tables is None:raise ValueError('Missing actual NPU page table')
    seq_lens=forward_batch.seq_lens.to(device=x.device,dtype=torch.int32)
    slots=forward_batch.req_pool_indices.to(device=x.device,dtype=torch.int32)
    if forward_batch.forward_mode.is_extend():
        query_lens=forward_batch.extend_seq_lens.to(device=x.device,dtype=torch.int32)
    else:
        query_lens=torch.ones_like(seq_lens)
    if _PERFORMANCE:
        cpu_lengths=forward_batch.seq_lens_cpu
        if cpu_lengths is None or getattr(cpu_lengths,'device',None) is None or cpu_lengths.device.type!='cpu':
            raise ValueError('Native CPU sequence mirror required; no device sync fallback')
        sequences=cpu_lengths.tolist()
        if len(sequences)!=seq_lens.numel() or not sequences:
            raise ValueError('Native CPU sequence mirror cardinality mismatch')
        lengths=request_slots=None
        index_cache,tail_key,tail_score=pool.get_npu_kpool_buffers(layer_id)
        # Fresh seq_len starts at position0. A completed first pool reads all
        # nonnegative inputs from this query; negative positions are masked.
        # The tail seeder writes each retained raw row before a later chunk can
        # read it. Old tail contents are never consumed for a fresh request.
    else:
        # Validate complete query boundaries before writing any persistent state.
        lengths=query_lens.detach().cpu().tolist()
        sequences=seq_lens.detach().cpu().tolist()
        request_slots=slots.detach().cpu().tolist()
        actual_positions=positions.detach().cpu().tolist()
        if sum(lengths)!=count or len(lengths)!=len(sequences) or len(lengths)!=len(request_slots):
            raise ValueError('Flash KPool actual-token/request boundaries mismatch')
        active_slots=[slot for slot,length in zip(request_slots,lengths) if length]
        if len(set(active_slots))!=len(active_slots):raise ValueError('Duplicate active request tail slots')
        offset=0
        for length,sequence,slot in zip(lengths,sequences,request_slots):
            if actual_positions[offset:offset+length]!=list(range(sequence-length,sequence)):
                raise ValueError('Flash KPool positions must be contiguous at the actual prefix')
            if length and not 0<=slot<pool.full_kv_pool.req_pool_size:
                raise ValueError('Invalid actual Flash request tail slot')
            offset+=length
        if block_tables.shape[0]!=len(lengths) or block_tables.dtype not in [torch.int32,torch.int64]:
            raise ValueError('Actual NPU block table does not match KPool requests')
        index_cache,tail_key,tail_score=pool.get_npu_kpool_buffers(layer_id)
        for length,sequence,slot in zip(lengths,sequences,request_slots):
            if length and sequence==length:
                tail_key[slot].zero_();tail_score[slot].zero_()

    hidden=x.float()
    if indexer.wk.weight.dtype not in [torch.bfloat16,torch.float32]:
        raise ValueError('Flash WK must retain its declared FLOAT checkpoint projection')
    projected=F.linear(hidden,indexer.wk.weight.float())
    norm_weight=indexer.k_norm.weight.float() if indexer.k_norm.weight is not None else None
    norm_bias=indexer.k_norm.bias.float() if indexer.k_norm.bias is not None else None
    key=F.layer_norm(projected,(128,),norm_weight,norm_bias,
                     getattr(indexer.k_norm,'eps',getattr(indexer.k_norm,'variance_epsilon',1e-6)))
    gate=F.linear(hidden,indexer.index_kpool_compress_gate.float())
    query=indexer.wq_b(q_lora)[0].view(count,32,128)
    raw_weights=F.linear(hidden,indexer.weights_proj.weight.float())
    weights=raw_weights.to(query.dtype)
    weights=weights*(indexer.softmax_scale*indexer.n_heads**-0.5)
    query_ends=torch.cumsum(query_lens,dim=0,dtype=torch.int32)
    token_slots=torch.repeat_interleave(slots,query_lens,output_size=count)
    tail_slots=token_slots*4+positions.to(dtype=torch.int32).remainder(4)
    locations=forward_batch.out_cache_loc
    if locations is None or locations.numel()!=count:raise ValueError('Missing actual Flash cache write slots')
    index_locations=torch.div(locations,4,rounding_mode='floor')
    from sglang.srt.hardware_backend.npu.ops.flash_kpool_model_audit_npu import before as _model_before, after as _model_after
    _model_check=None if _PERFORMANCE else _model_before(indexer,layer_id,hidden,q_lora,projected,raw_weights,key,gate,query,weights,index_cache,tail_key,tail_score,positions,slots,index_locations,tail_slots,query_ends)
    if os.environ.get('FLASH_B103_PERSISTENT_OWNED_STORAGE','0') == '1':
        if not getattr(pool.full_kv_pool,'_flash_persistent_owned',False):
            raise ValueError('Small-input path requires independent persistent pool storage')
        # Keep the proven fresh allocation boundary for transient inputs only.
        # Cache/tail tensors retain their native SDK registered addresses.
        values=(key,gate,indexer.index_kpool_compress_ape,positions,query_ends,
                seq_lens,tail_slots,slots.unsqueeze(1),index_locations)
        copies=tuple(torch.empty(tuple(t.shape),dtype=t.dtype,device=t.device,
            memory_format=torch.contiguous_format).copy_(t) for t in values)
        key,gate,ape,positions,query_ends,seq_lens,tail_slots,tail_table,index_locations=copies
        block_tables=torch.empty(tuple(block_tables.shape),dtype=block_tables.dtype,
            device=block_tables.device,memory_format=torch.contiguous_format).copy_(block_tables)
        if _PERFORMANCE or (os.environ.get('FLASH_PD_COMPRESSION_ABI_AUDIT')=='0' and os.environ.get('FLASH_PD_SCORE_ABI_AUDIT')=='0'):
            stream=torch.npu.current_stream(x.device)
            for transient in (*copies,block_tables):transient.record_stream(stream)
    else:
        ape=indexer.index_kpool_compress_ape
        tail_table=slots.unsqueeze(1)
    glm5_next_kpool_tail_compress_and_write_cache_triton(
        tail_key,tail_score,index_cache,key,gate,ape,
        positions,query_ends,seq_lens,tail_slots,tail_table,index_locations,4,
        audit_context=None if _PERFORMANCE else {'layer_id':int(layer_id),'sequences':sequences,
            'query_lengths':lengths,'request_slots':request_slots})
    if not _PERFORMANCE:_model_after(_model_check,index_cache,tail_key,tail_score)
    pool_seq_lens=torch.div(seq_lens,4,rounding_mode='floor')
    parallel=get_parallel()
    output=query_parallel_indexer(query,index_cache,weights,query_ends,
        pool_seq_lens,block_tables,positions,index_topk=2048,index_kpool=4,
        max_pool_seq_len=max(sequences)//4,pack_tail=True,
        group=parallel.attn_tp_group,rank=parallel.attn_tp_rank,world_size=parallel.attn_tp_size,
        prefill=forward_batch.forward_mode.is_extend(),
        audit_context=None if _PERFORMANCE else {'layer_id':int(layer_id),'sequences':sequences,
            'query_lengths':lengths,'request_slots':request_slots})
    return output.squeeze(1)
