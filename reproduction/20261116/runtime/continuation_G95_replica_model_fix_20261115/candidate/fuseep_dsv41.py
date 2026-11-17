"""1017/W8A8 mode2 physical400 candidate; consumes SGLang's trained routing."""
from pathlib import Path
import hashlib
import os

import torch
from sglang.srt.runtime_context import get_exec, get_parallel

_G95_CAPTURE_DIR=os.environ.get('SGLANG_G95_ROUTE_CAPTURE_DIR','')
if _G95_CAPTURE_DIR:
    import route_histogram_observer as _g94_observer
    print('G95_ROUTE_OBSERVER_LOADED pid=%d observer_sha256=%s' % (os.getpid(),hashlib.sha256(Path(_g94_observer.__file__).read_bytes()).hexdigest()),flush=True)

if os.environ.get('SGLANG_G95_MAPPING_AUDIT_PLAN'):
    print('G95_FUSEEP_LOADED pid=%d adapter_sha256=%s source=%s' % (os.getpid(), hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), __file__), flush=True)

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


def received_capacity(global_limit, source_rows, topk, world_size, global_num_tokens):
    """Bound all incoming routes using SG's current DP-wide CPU metadata.

    DP counts cover full unsharded rows; each EP rank owns at most that DP's
    rows. max(DP rows)*EP*topk therefore remains safe for uneven DP lengths.
    Before a ForwardBatch exists, retain the original configured ceiling.
    """
    if global_limit <= 0 or source_rows < 0 or topk <= 0 or world_size != 16:
        raise ValueError('Invalid explicit capacity contract')
    if global_num_tokens is None:
        return global_limit
    if len(global_num_tokens) != 4 or any(not isinstance(n, int) or n < 0 for n in global_num_tokens):
        raise ValueError('Expected current four-DP CPU token counts')
    max_dp_rows = max(global_num_tokens)
    if max_dp_rows == 0:
        return global_limit
    if source_rows > max_dp_rows:
        raise ValueError(f'Source rows {source_rows} exceed DP metadata ceiling {max_dp_rows}')
    return min(global_limit, max_dp_rows * world_size * topk)


def exact_prefill_enabled(global_num_tokens, is_extend_in_batch, geometry):
    # The decision uses DP-global CPU metadata, never local source length.
    # Target/draft graphs use small global buckets and bypass this collective.
    return (geometry == (400, 6) and is_extend_in_batch
            and global_num_tokens is not None and max(global_num_tokens) > 128)


_histogram_groups = {}


def histogram_group(group):
    """A bounded scratch communicator for the 1600-byte physical-expert histogram."""
    import torch.distributed as dist
    import torch_npu
    key = id(group)
    if key not in _histogram_groups:
        ranks = dist.get_process_group_ranks(group)
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = {"hccl_buffer_size": 16,
                               "group_name": "sglang_prefill_route_histogram"}
        small = dist.new_group(ranks=ranks, backend="hccl", pg_options=options)
        _histogram_groups[key] = small
    return _histogram_groups[key]


def exact_received_capacity(ids, active, group, total_experts, global_limit, layer_id=None, global_tokens=None):
    """Count all active expert routes; return the largest exact owner total.

    One tiny EP histogram reduction plus one CPU scalar read is intentional
    for eager large-prefill only. No route/token truncation or approximation.
    """
    import torch.distributed as dist
    if total_experts % group.size() or ids.ndim != 2 or active.shape != ids.shape[:1]:
        raise ValueError('Invalid expert owner/count geometry')
    hist = torch.zeros(total_experts, dtype=torch.int32, device=ids.device)
    votes = active.unsqueeze(-1).expand_as(ids).reshape(-1).to(torch.int32)
    hist.scatter_add_(0, ids.reshape(-1).long(), votes)
    dist.all_reduce(hist, op=dist.ReduceOp.SUM, group=histogram_group(group))
    incoming = hist.reshape(group.size(), -1).sum(dim=1)
    capacity = max(1, int(incoming.max().item()))
    if capacity > global_limit:
        raise ValueError(f'Actual incoming routes {capacity} exceed configured ceiling {global_limit}')
    if _G95_CAPTURE_DIR:
        _g94_observer.observe(hist,capacity,layer_id,global_tokens,group)
    return capacity


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
            or routing_contract not in {(400, 6), (128, 3)}
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
    from sglang.srt.layers.dp_attention import get_dp_global_num_tokens
    capacity = received_capacity(capacity, hidden_states.shape[0], layer.top_k,
                                 group.size(), get_dp_global_num_tokens())
    ids = topk_output.topk_ids.to(torch.int32)
    probs = topk_output.topk_weights.to(torch.float32)
    # Standard NPU top-k represents padded rows by zero routing weights.
    # An entirely zero-weight row contributes exactly zero routed output.
    active = probs.ne(0).any(dim=-1)
    ids = torch.where(active.unsqueeze(-1), ids, 0)
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch
    global_tokens = get_dp_global_num_tokens()
    if global_tokens is not None and max(global_tokens) > 128:
        if exact_prefill_enabled(global_tokens, get_is_extend_in_batch(), routing_contract):
            capacity = exact_received_capacity(ids, active, group, layer.num_experts, capacity,
                                               layer_id=layer.layer_id, global_tokens=global_tokens)
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
