"""Narrow TP8/24-row model integration. Imported only by a pinned overlay.

This module is not installed by the native comparator. Service admission must
cover captured padded replays and the original real cache consumers.
"""
import torch
import torch.distributed as dist
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context

_audit = None


def _observer():
    global _audit
    if _audit is None:
        from owner_diagnostics import Observer
        _audit = Observer()
    return _audit


def owner_entry(attn, positions, hidden):
    impl = attn.v41_impl
    return (type(impl).__name__ == 'AscendDSAV41CPImpl'
            and type(impl).__module__ == 'vllm_ascend.attention.context_parallel.dsa_v41_cp'
            and tuple(hidden.shape) == (3, 5120)
            and hidden.dtype == torch.bfloat16
            and positions.shape[0] == 24
            and get_tp_group().world_size == 8)


def eligible(impl, attn, hidden, output, metadata_by_prefix):
    """Only host metadata/configuration; no rank-local tensor predicates.

    Tensor shape guards are fixed at capture. A captured 24-row graph can
    replay padded requests; Python cannot turn this path off during replay.
    Those real metadata/buffer states remain a required integration test.
    """
    if metadata_by_prefix is None:
        return False
    swa = metadata_by_prefix[impl.swa_prefix]
    global_swa = getattr(swa, 'global_metadata', None)
    if global_swa is None or tuple(hidden.shape) != (3, 5120) or tuple(output.shape) != (24, 5120):
        return False
    if (global_swa.num_actual_tokens != 24 or global_swa.num_input_tokens != 24
            or global_swa.num_prefill_tokens != 0 or global_swa.num_prefills != 0):
        return False
    if (not impl.multistream_dsv4_dsa_overlap or getattr(attn, 'packed_cache_ops', None) is not None
            or getattr(attn, 'uses_a5_packed_cache', False)):
        return False
    projector = attn.dsa_attn.dsa_attn.impl
    for wrapper in (projector.cv_wq_a, projector.cv_wkv):
        if wrapper._has_communication or wrapper._is_w8a8_dynamic or wrapper._is_mxfp8_dynamic:
            return False
        if wrapper.linear.weight.dtype != torch.bfloat16:
            return False
    if impl.role.is_kv_source:
        if impl.role.compress_ratio not in (1, 2):
            return False
        expected_dtype = torch.bfloat16 if impl.role.compress_ratio == 1 else torch.float32
        if attn.compressor.wkv.weight.dtype != expected_dtype:
            return False
        if impl.role.compress_ratio == 2 and attn.compressor.wgate.weight.dtype != torch.float32:
            return False
    group = get_tp_group()
    assert group.world_size == 8
    assert swa.cp_token_range == (3*group.rank_in_group, 3*(group.rank_in_group+1), 3, 24)
    assert swa.num_actual_tokens == 3
    assert hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
    return True


def prepare(impl, attn, local, metadata, metadata_by_prefix):
    from vllm_ascend.attention.dsa_v41 import dsv4_dsa_overlap_stream

    global_metadata = impl._global_layer_metadata(metadata_by_prefix)
    cos, sin = metadata.rope(attn.rotary_emb.layername, 3)
    global_cos, global_sin = global_metadata.rope(attn.rotary_emb.layername, 24)
    # The pinned CP builder aliases local Q RoPE to the same owner rows of the
    # global KV table. Do not substitute query positions from another batch.
    rank = get_tp_group().rank_in_group
    kv_cos, kv_sin = global_cos[rank*3:rank*3+3], global_sin[rank*3:rank*3+3]
    v1 = attn.dsa_attn.dsa_attn.impl
    wq_a, wkv, wq_b = v1.cv_wq_a, v1.cv_wkv, v1.cv_wq_b
    main = torch.npu.current_stream()
    aux = dsv4_dsa_overlap_stream()
    q_quant, q_scale = wq_a.quantize(local)
    q_a = wq_a.matmul(q_quant, q_scale, bias=attn.wq_a.bias)
    qa_done = main.record_event()
    with torch.npu.stream(aux):
        aux.wait_event(qa_done)
        kv_quant, kv_scale = wkv.quantize(local)
        kv = wkv.matmul(kv_quant, kv_scale, bias=attn.wkv.bias)
        kv_done = aux.record_event()
    qr = attn.q_norm(q_a)
    q_quant, q_scale = wq_b.quantize(qr)
    vector_ready = main.record_event()
    main.wait_event(kv_done)
    with torch.npu.stream(aux):
        aux.wait_event(vector_ready)
        kv = attn.kv_norm(kv).view(3, 1, attn.head_dim)
        impl._apply_rotary(attn, kv, kv_cos, kv_sin)
        pieces = [kv.view(3, 512).view(torch.int16)]
        if impl.role.is_kv_source:
            if impl.role.compress_ratio == 1:
                pieces.append(attn.compressor(local).view(torch.int16))
            else:
                fp32 = local.float()
                pieces.extend((attn.compressor.wkv(fp32).view(torch.int16),
                               attn.compressor.wgate(fp32).view(torch.int16)))
        packet = torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]
        received = torch.empty((24, packet.shape[-1]), dtype=torch.int16, device=local.device)
        work = dist.all_gather_into_tensor(received, packet, group=get_tp_group().device_group, async_op=True)
    q = wq_b.matmul(q_quant, q_scale, bias=attn.wq_b.bias).unflatten(-1, (-1, attn.head_dim))
    projected_sources = None
    with torch.npu.stream(aux):
        work.wait()
        kv = received[:, :512].contiguous().view(torch.bfloat16)
        if impl.role.is_kv_source:
            if impl.role.compress_ratio == 1:
                projected_sources = (received[:, 512:1024].contiguous().view(torch.bfloat16),)
            else:
                projected_sources = (received[:, 512:1536].contiguous().view(torch.float32),
                                     received[:, 1536:2560].contiguous().view(torch.float32))
        _observer().check(impl, attn, local, kv, projected_sources, global_metadata, global_cos, global_sin)
        impl._write_swa_cache(attn, global_metadata.swa, kv)
    main.wait_stream(aux)
    impl._apply_rotary(attn, q, cos, sin)
    prepared = impl._prepare_indexer_inputs(attn, local, qr, cos, sin, metadata)
    if impl.role.is_kv_source:
        impl._write_compressed_source(attn, local, global_metadata.positions[:24], global_cos, global_sin,
                                      global_metadata, prepared_indexer=prepared, projected_sources=projected_sources)
    else:
        impl._quantize_indexer_query(attn, prepared, metadata)
    return q.to(local.dtype), qr, prepared


def forward(impl, attn, hidden, output):
    context = get_forward_context()
    metadata = impl._get_layer_metadata(context.attn_metadata)
    q, qr, prepared = prepare(impl, attn, hidden, metadata, context.attn_metadata)
    positions = metadata.positions[:3]
    cos, sin = metadata.rope(attn.rotary_emb.layername, 3)
    indices = impl._select_sparse_indices(attn, hidden, qr, positions, cos, sin, metadata, prepared)
    attention = impl._forward_attention(attn, q, metadata, indices)
    impl._apply_rotary(attn, attention, cos, sin, inverse=True)
    # The CP implementation is patched to use projected.shape[0] for the
    # output layout. Input ownership must not truncate the full TP result.
    impl._project_output(attn, attention, hidden, metadata, projected=output)
    return output
