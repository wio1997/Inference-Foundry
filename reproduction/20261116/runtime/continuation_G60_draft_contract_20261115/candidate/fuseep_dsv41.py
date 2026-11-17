"""Isolated 1017/W8A8 mode2 candidate; consumes SGLang's trained routing."""
from pathlib import Path
import hashlib
import os

import torch
from sglang.srt.runtime_context import get_exec, get_parallel

_loaded = False
_group_names = {}
_PROVIDER_SHA256 = '5e7735b6b84626596a5eb7ac6e4b2a028fa227ee8f267f435ca6cd5f4928fbea'


def _load_binding():
    global _loaded
    if _loaded:
        return
    provider = Path(os.environ['SGLANG_DSV41_FUSED_MOE_OPAPI']).resolve()
    library = Path(os.environ['SGLANG_DSV41_FUSED_MOE_LIBRARY']).resolve()
    if hashlib.sha256(provider.read_bytes()).hexdigest() != _PROVIDER_SHA256:
        raise RuntimeError('Fused MoE provider differs from the pinned INT8 ABI')
    torch.ops.load_library(str(library))
    if Path(torch.ops.sglang_dsv41_fused.provider_path()).resolve() != provider:
        raise RuntimeError('Fused MoE resolved a different operator provider')
    _loaded = True


def forward_fuseep_dsv41(layer, hidden_states, topk_output):
    """Return unscaled routed output at the existing FuseEP boundary.

    SGLang retains shared-expert execution, output scaling and model merge.
    Existing mode2 checkpoint postprocessing owns NZ weights and INT64 scales.
    """
    if layer.moe_runner_config.swiglu_limit != 10.0:
        raise ValueError('This isolated candidate requires the 1017 raw clamp10')
    if get_exec().moe.fuseep_mode != 2:
        raise ValueError("The 1017 candidate requires FuseEP mode2")
    routing_contract = (layer.num_experts, layer.top_k)
    if (hidden_states.shape[0] > 2048
            or routing_contract not in {(384, 6), (128, 3)}
            or topk_output.topk_ids.shape[-1] != layer.top_k):
        raise ValueError(
            f"Unexpected 1017 target/draft routing: E/K={routing_contract}, "
            f"rows={hidden_states.shape[0]}, observedK={topk_output.topk_ids.shape[-1]}")
    _load_binding()
    group = get_parallel().moe_ep_group.device_group
    group_key = (id(group), hidden_states.device.index)
    if group_key not in _group_names:
        backend = group._get_backend(torch.device('npu'))
        _group_names[group_key] = backend.get_hccl_comm_name(group.rank())
    capacity = int(os.environ['SGLANG_DSV41_FUSED_MOE_MAX_RECV_TOKENS'])
    if capacity <= 0:
        raise ValueError('An explicit positive received-row capacity is required')
    ids = topk_output.topk_ids.to(torch.int32)
    probs = topk_output.topk_weights.to(torch.float32)
    # Standard NPU top-k represents padded rows by zero routing weights.
    # An entirely zero-weight row contributes exactly zero routed output.
    active = probs.ne(0).any(dim=-1)
    ids = torch.where(active.unsqueeze(-1), ids, 0)
    out = torch.zeros_like(hidden_states)
    counts = torch.empty((layer.w13_weight.shape[0],),
                         dtype=torch.int32, device=hidden_states.device)
    torch.ops.sglang_dsv41_fused.dispatch(
        hidden_states, [layer.w13_weight], [layer.w2_weight], ids,
        [layer.w13_weight_scale], [layer.w2_weight_scale], probs,
        _group_names[group_key], capacity, 10.0, group.size(), active,
        out, counts,
    )
    return out
