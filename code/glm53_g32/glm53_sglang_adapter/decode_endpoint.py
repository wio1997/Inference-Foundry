"""Eager topk1/W4: distinguish actual request owners from TP padding."""
import copy

def active_decode_view(backend, forward_batch, positions):
    m = backend.forward_metadata
    active = getattr(m, '_glm53_active_decode_rows', None)
    if active is None:
        return forward_batch, positions, positions.numel()
    padded = getattr(m, '_glm53_padded_decode_rows')
    request_rows = m._glm53_active_request_rows
    if positions.numel() != padded:
        raise RuntimeError('Active decode witness does not match current token shape')
    view = copy.copy(forward_batch)
    view.batch_size = request_rows
    for name in ('seq_lens', 'seq_lens_cpu', 'req_pool_indices', 'req_pool_indices_cpu'):
        value = getattr(view, name, None)
        if value is not None:
            setattr(view, name, value[:request_rows])
    view.out_cache_loc = view.out_cache_loc[:active]
    if view.forward_mode.is_draft_extend_v2():
        view.spec_info = copy.copy(view.spec_info)
        view.spec_info.num_accept_tokens = view.spec_info.num_accept_tokens[:request_rows]
    return view, positions[:active], active

def _refresh_eager_decode_endpoint_g17(backend, forward_batch, positions):
    if not getattr(backend, 'glm53_nope_mtp_optin', False):
        return
    mode = forward_batch.forward_mode
    decode = mode.is_decode_or_idle()
    verify = mode.is_target_verify()
    extend = mode.is_draft_extend_v2()
    if not ((decode or verify or extend) and forward_batch.spec_info is not None):
        return
    if backend.graph_mode:
        raise RuntimeError('Eager endpoint repair does not enable graph execution')
    if bool(backend.is_draft_worker) != (not verify):
        raise RuntimeError('Speculative endpoint role does not match native forward mode')
    import torch
    padded = forward_batch.seq_lens.numel()
    active = getattr(forward_batch, '_original_batch_size', None)
    original_tokens = getattr(forward_batch, '_original_num_tokens', None)
    if active is None:
        active = padded
    if type(active) is not int or not 0 < active <= padded:
        raise RuntimeError('Invalid native original request count')
    width = 1 if decode else 4
    if not decode and (getattr(forward_batch.spec_info, 'num_tokens_per_req', None) != 4
                       or (verify and forward_batch.spec_info.draft_token_num != 4)):
        raise RuntimeError('Native rectangular W4 speculative geometry required')
    if positions.ndim != 1 or positions.numel() != padded * width or forward_batch.batch_size != padded:
        raise RuntimeError('Physical token shape disagrees with native request width')
    if getattr(forward_batch, 'num_padding', 0) != 0:
        raise RuntimeError('Foreign graph padding is outside eager TP-prefix scope')
    if active < padded and original_tokens != active * width:
        raise RuntimeError('Native MLP-sync padding requires matching original token/request witnesses')
    for name in ('req_pool_indices', 'out_cache_loc'):
        value = getattr(forward_batch, name, None)
        if value is None or value.ndim != 1 or value.numel() != (padded * width if name == 'out_cache_loc' else padded):
            raise RuntimeError('Missing physical request/token mapping: ' + name)
    reqids = forward_batch.req_pool_indices[:active]
    if torch.unique(reqids).numel() != active or not bool((reqids >= 0).all().item()):
        raise RuntimeError('Active request owners must be distinct native nonnegative rows')
    metadata = backend.forward_metadata
    if metadata is None:
        raise RuntimeError('Actual native forward metadata is absent')
    expected = torch.arange(width, (active + 1) * width, width, device=positions.device, dtype=torch.int32)
    qends = metadata.actual_seq_lengths_q
    if qends is None:
        metadata.actual_seq_lengths_q = expected
    else:
        if (not isinstance(qends, torch.Tensor) or qends.ndim != 1
            or qends.dtype != torch.int32 or qends.device != positions.device):
            raise RuntimeError('Existing NoPE TND endpoints type/rank/dtype/device invalid')
        if qends.numel() == active and torch.equal(qends, expected):
            pass
        elif (not decode and active < padded and qends.numel() == padded
              and torch.equal(qends, torch.arange(width,(padded+1)*width,width,device=positions.device,dtype=torch.int32))):
            # EagerRunner replans target verify AFTER native TP padding. Its
            # canonical physical endpoints are valid, but dummy requests must
            # be removed at the NoPE boundary using the independent witnesses.
            metadata.actual_seq_lengths_q = qends[:active]
        else:
            raise RuntimeError('Existing NoPE TND endpoints are not canonical native ACTIVE/physical-W4 geometry')
    table = metadata.block_tables
    if table is None or table.ndim != 2 or table.shape[0] not in (active, padded):
        raise RuntimeError('Native block table does not cover active request rows')
    metadata.block_tables = table[:active]
    if decode:
        endpoints = positions[:active].to(device=backend.device, dtype=torch.int32) + 1
    else:
        seq = forward_batch.seq_lens[:active].to(dtype=torch.int32)
        actual_positions = positions[:active*width]
        expected_positions = ((seq[:,None] if verify else seq[:,None]-width)
                              + torch.arange(width,device=positions.device)).reshape(-1)
        if not torch.equal(actual_positions.to(expected_positions.dtype),expected_positions):
            raise RuntimeError('Active W4 positions disagree with native device sequence lengths')
        if extend:
            accepted = forward_batch.spec_info.num_accept_tokens
            if (accepted is None or accepted.ndim != 1 or accepted.numel() not in (active,padded)
                or not bool(((accepted[:active]>=1)&(accepted[:active]<=width)).all().item())):
                raise RuntimeError('Native active accepted counts must remain in 1..W')
        endpoints = seq + width if verify else seq
    metadata.actual_seq_lengths_kv = endpoints.to(device=backend.device,dtype=torch.int32)
    metadata._glm53_active_decode_rows = active * width
    metadata._glm53_padded_decode_rows = padded * width
    metadata._glm53_active_request_rows = active


def call_active_sparse_consumer(consumer, metadata, **kwargs):
    """Keep attention's physical output shape while excluding dummy requests."""
    active = getattr(metadata, '_glm53_active_decode_rows', None)
    if active is None:
        return consumer(**kwargs)
    query = kwargs['query']
    padded = metadata._glm53_padded_decode_rows
    if query.shape[0] != padded:
        raise RuntimeError('Sparse query shape disagrees with active decode witness')
    if kwargs['actual_seq_lengths_query'].numel() != metadata._glm53_active_request_rows or kwargs['actual_seq_lengths_kv'].numel() != metadata._glm53_active_request_rows:
        raise RuntimeError('Sparse metadata request axes disagree')
    kwargs['query'] = query[:active]
    kwargs['sparse_indices'] = kwargs['sparse_indices'][:active]
    if kwargs.get('query_rope') is not None:
        kwargs['query_rope'] = kwargs['query_rope'][:active]
    result = consumer(**kwargs)
    if active == padded:
        return result
    output = query.new_zeros((padded, *result[0].shape[1:]))
    output[:active].copy_(result[0])
    return (output, *result[1:])


"""Refresh each actual eager draft forward, including skip-topk layers."""
def _refresh_graph_decode_endpoint_g15(backend, forward_batch, positions):
    if not getattr(backend, 'glm53_nope_mtp_optin', False):
        return
    mode = forward_batch.forward_mode
    if not (mode.is_decode_or_idle() and forward_batch.spec_info is not None):
        return
    if backend.graph_mode:
        from .mtp_graph_bucket import enabled
        if not enabled():raise RuntimeError('Graph endpoint requires explicit candidate opt-in')
        backend.glm53_mtp_graph_bucket.refresh_endpoint(positions)
        return
    if not backend.is_draft_worker:
        raise RuntimeError('Speculative decode endpoint requires actual draft backend')
    import torch
    count = forward_batch.seq_lens.numel()
    if positions.ndim != 1 or positions.numel() != count:
        raise RuntimeError('Unpadded topk1 draft decode requires one actual position per request')
    metadata = backend.forward_metadata
    if metadata.actual_seq_lengths_q is None or metadata.actual_seq_lengths_q.numel() != count:
        raise RuntimeError('Native one-query/request metadata missing')
    # Called from actual prepare_nope(positions=...), AFTER prior step add_(1).
    # Device positions are authoritative; the CPU mirror may be published ahead.
    metadata.actual_seq_lengths_kv = positions.to(device=backend.device, dtype=torch.int32) + 1


def refresh_decode_endpoint(backend, forward_batch, positions):
    if backend.graph_mode:
        return _refresh_graph_decode_endpoint_g15(backend, forward_batch, positions)
    return _refresh_eager_decode_endpoint_g17(backend, forward_batch, positions)
