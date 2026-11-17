"""G48 opt-in eager SparseMLA TP4 head/query exchange; no cache mutation.

Derived from the frozen actual ratio2 component probe. Kernel/operator runtime
validation remains root's gate; this module never creates a process group.
"""
import os

ENABLED = os.environ.get('SGLANG_NPU_SMLA_QUERY_ROW_SHARD', '0') == '1'
# Ratio1 must be explicitly admitted after its separate component acceptance.
RATIOS = (1, 2) if os.environ.get('SGLANG_G48_RATIO1_ACCEPTED', '0') == '1' else (2,)


def eligible(*, enabled, ordinary, graph, rows, batch, max_q, end, tp, cp,
             cp_active, sp_active, tbo, replay, target, heads, dim, topk,
             ori_mask, left, right, explicit_ori):
    return bool(enabled and target and ordinary and not graph and rows == 8192
                and batch == 1 and max_q == rows and end >= rows and tp == 4
                and cp == 1 and not cp_active and not sp_active and not tbo
                and not replay and heads == 16 and dim == 512 and topk == 512
                and ori_mask == 4 and left == 127 and right == 0 and not explicit_ori)


class Workspace:
    """One backend-owned 384MiB scratch, reused on one execution stream.

    Output is freshly allocated 128MiB, never a scratch alias. Native kernel
    output adds another128MiB transient. No per-layer/slot cache is retained.
    Same-stream queue order protects scratch reuse; stream changes explicitly join
    the previous stream before scratch reuse (no additional collective).
    """
    def __init__(self, device):
        import torch
        self.stream = torch.npu.current_stream(device)
        self.qrecv = torch.empty((4, 2048, 16, 512), device=device, dtype=torch.bfloat16)
        self.qwide = torch.empty((2048, 64, 512), device=device, dtype=torch.bfloat16)
        self.ysend = torch.empty_like(self.qrecv)
        self.bytes = sum(t.numel() * t.element_size() for t in (self.qrecv, self.qwide, self.ysend))
        self.busy = False

    def acquire(self, q):
        import torch
        if self.busy or q.device != self.qwide.device:
            raise RuntimeError('G48 prepared workspace reentry/device contract violated')
        current = torch.npu.current_stream(q.device)
        if current != self.stream:
            current.wait_stream(self.stream)
            self.stream = current
        self.busy = True


def prepare(bindings, backend, fb, fm, *, rows, max_q, end):
    """Prepare once/forward/ratio OUTSIDE model forward; global metadata intact."""
    if not ENABLED:
        return None
    from sglang.srt.runtime_context import get_parallel, get_forward
    from sglang.srt.layers.cp.utils import is_cp_active
    import torch
    p = get_parallel()
    if not eligible(enabled=ENABLED, ordinary=fb.forward_mode.is_extend_without_speculative(),
                    graph=backend.graph_mode, rows=rows, batch=int(fb.batch_size),
                    max_q=max_q, end=end, tp=p.attn_tp_size, cp=p.attn_cp_size,
                    cp_active=is_cp_active(fb), sp_active=get_forward().sp_active,
                    tbo=bool(getattr(fb, 'tbo_children', None)),
                    replay=bool(getattr(fb, 'encoder_swa_replay', False)),
                    target=backend._v41_enabled, heads=backend._dsv4_q_head_num,
                    dim=backend._dsv4_head_dim, topk=int(backend._v41_cfg.index_topk),
                    ori_mask=fm.ori_mask_mode, left=fm.ori_win_left,
                    right=fm.ori_win_right, explicit_ori=fm.ori_sparse_indices is not None):
        return None
    group = p.attn_tp_group.device_group
    import torch.distributed as dist
    # Group rank is the canonical contiguous head-band rank. Do not silently
    # accept a coordinator/group ordering mismatch or create a replacement.
    if dist.get_world_size(group) != 4 or dist.get_rank(group) != p.attn_tp_rank:
        raise RuntimeError('G48 attention TP group/rank contract changed')
    device = fb.positions.device
    if backend._g48_workspace is None:
        backend._g48_workspace = Workspace(device)
    rank = p.attn_tp_rank
    local_end = end - 8192 + 2048 * (rank + 1)
    cu = torch.tensor([0, 2048], dtype=torch.int32, device=device)
    seq = torch.tensor([local_end], dtype=torch.int32, device=device)
    plans = {}
    for ratio in RATIOS:
        cseq = seq // ratio
        residual = seq % ratio if ratio == 2 else None
        metadata = bindings.sparse_metadata(64, 1, 512, cu_seqlens_q=cu,
            seqused_ori_kv=seq, seqused_cmp_kv=cseq, cmp_residual_kv=residual,
            batch_size=1, max_seqlen_q=2048, max_seqlen_ori_kv=local_end,
            max_seqlen_cmp_kv=local_end // ratio, ori_topk=0, ori_topk_length=None,
            cmp_topk=512, cmp_ratio=ratio, ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=127, ori_win_right=0, layout_q='TND', layout_kv='PA_BBND',
            has_ori_kv=True, has_cmp_kv=True)
        plans[ratio] = dict(cu=cu, seq=seq, cseq=cseq, residual=residual,
                            metadata=metadata)
    return dict(plans=plans, rank=rank, end=end, local_end=local_end,
                group=group, workspace=backend._g48_workspace,
                global_rank=p.tp_rank, fb=fb)


def run(bindings, context, q, full_sink, args):
    """Return restored owned [8192,16,512], or None for original path."""
    import torch
    ratio = args['cmp_ratio']
    if context is None or ratio not in context['plans']:
        return None
    ws = context['workspace']
    if (tuple(q.shape) != (8192, 16, 512) or q.dtype != torch.bfloat16
            or full_sink is None
            or tuple(full_sink.shape) != (64,) or full_sink.dtype != torch.float32
            or full_sink.device != q.device
            or args['ori_kv'].dtype != torch.bfloat16
            or args['cmp_kv'].dtype != torch.bfloat16
            or args['cmp_sparse_indices'].shape != (8192, 1, 512)
            or args['cmp_sparse_indices'].dtype != torch.int32
            or args['ori_sparse_indices'] is not None
            or args['ori_topk_length'] is not None):
        raise RuntimeError('G48 prepared collective operand contract violated; rank-local fallback forbidden')
    # Contiguity may differ without changing shape/dtype/math: normalize it
    # instead of letting one peer bypass a collective. No new collective here.
    q = q.contiguous()
    full_sink = full_sink.contiguous()
    import torch.distributed as dist
    start = context['rank'] * 2048
    local = context['plans'][ratio]
    call = dict(args)
    call.update(cmp_sparse_indices=args['cmp_sparse_indices'][start:start+2048],
                cu_seqlens_q=local['cu'], seqused_ori_kv=local['seq'],
                seqused_cmp_kv=local['cseq'], cmp_residual_kv=local['residual'],
                metadata=local['metadata'], sinks=full_sink)
    ws.acquire(q)
    try:
        dist.all_to_all_single(ws.qrecv.view(8192, 16, 512), q, group=context['group'])
        ws.qwide.view(2048, 4, 16, 512).copy_(ws.qrecv.permute(1, 0, 2, 3))
        wide, _ = bindings.sparse(ws.qwide, **call)
        ws.ysend.copy_(wide.reshape(2048, 4, 16, 512).permute(1, 0, 2, 3))
        restored = torch.empty_like(q)
        dist.all_to_all_single(restored, ws.ysend.view(8192, 16, 512), group=context['group'])
        return restored
    finally:
        ws.busy = False
