"""Scoped, diagnostic-only snapshot of a single pause transition.

This does not certify all physical Target/Draft KV content or a fixed-W0
continuation. Call it only at the natural first-completion parking boundary,
with an all-device fence before each capture. No model step is run here.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from copy import deepcopy
from typing import Any

import torch


STATE_FIELDS = {
    "block_table", "num_computed_tokens", "last_sampled_tokens",
    "draft_tokens", "target_input_ids", "target_positions",
    "target_query_start_loc", "target_seq_lens", "target_slot_mapping",
    "target_logits_indices", "accepted_tokens", "num_sampled",
    "num_rejected", "emitted_token_count", "active_mask", "cycle_index",
}
SERVING_FIELDS = (
    "generation", "request_ids", "initial_output_counts", "remaining",
    "_progress_baseline", "_initial_positions", "_parked",
    "_last_decision_progress", "_cycles",
)
HOST_MIRROR_NAMES = (
    "query_start_loc_cpu", "seq_lens_cpu", "_seq_lens_cpu",
    "seq_lens_cpu_upper_bound", "num_computed_tokens_cpu",
    "_num_computed_tokens_cpu",
)
PROPOSER_BUFFER_NAMES = (
    "_dspark_draft_buffer", "_dspark_seed_buffer", "hidden_states",
    "_dflash_hidden_states", "positions", "_slot_mapping_buffer",
)
PROPOSER_DICT_NAMES = (
    "_per_group_block_tables", "_per_group_slot_mappings",
    "_per_group_block_table_buffers", "_per_group_query_slot_mapping_buffers",
    "_per_group_context_slot_mapping_buffers",
)


def _identity(value: torch.Tensor) -> tuple[Any, ...]:
    return (str(value.device), int(value.untyped_storage().data_ptr()),
            int(value.data_ptr()), tuple(value.shape), tuple(value.stride()),
            int(value.storage_offset()), str(value.dtype))


def _collect_tensors(prefix: str, value: Any, into: dict[str, torch.Tensor],
                     *, strict_leaves: bool = False) -> None:
    if torch.is_tensor(value):
        if prefix in into:
            raise RuntimeError(f"duplicate tensor path: {prefix}")
        into[prefix] = value
    elif isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda row: str(row[0])):
            _collect_tensors(f"{prefix}.{key}", item, into,
                             strict_leaves=strict_leaves)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_tensors(f"{prefix}.{index}", item, into,
                             strict_leaves=strict_leaves)
    elif strict_leaves:
        raise RuntimeError(f"non-tensor required leaf: {prefix}")


def capture(serving: Any, *, max_cloned_bytes: int = 1 << 30) -> dict[str, Any]:
    """Capture enumerated state after a caller-issued all-device fence."""
    runtime = serving.runtime
    state = runtime.state
    handoff = runtime.proposer
    common = handoff.common_attn_metadata
    if not is_dataclass(state):
        raise RuntimeError("FixedDecodeState is not a dataclass")
    actual = {field.name for field in fields(state)}
    if actual != STATE_FIELDS or set(vars(state)) != STATE_FIELDS:
        raise RuntimeError(f"FixedDecodeState field drift: {sorted(actual ^ STATE_FIELDS)}")
    if runtime._schedule_pending is not False:
        raise RuntimeError("scheduled metadata pending at pause boundary")
    if handoff._host_copy_pending is not False:
        raise RuntimeError("Host count copy pending at pause boundary")
    if serving._cycles <= 0 or serving._last_decision_progress is None:
        raise RuntimeError("no committed first-completion boundary")
    if not any(serving._parked) or all(serving._parked):
        raise RuntimeError("boundary must have completed and unfinished slots")
    if getattr(handoff, "_host_count_copy", None) is None:
        raise RuntimeError("Host count buffer absent")
    common_vars = vars(common)
    if (common_vars.get("query_start_loc_cpu") is None
            or common_vars.get("_seq_lens_cpu") is None):
        raise RuntimeError("required Host mirror absent")

    tensors: dict[str, torch.Tensor] = {}
    scalars: dict[str, Any] = {}
    for name in sorted(STATE_FIELDS):
        value = getattr(state, name)
        if torch.is_tensor(value):
            tensors[f"state.{name}"] = value
        else:
            scalars[f"state.{name}"] = deepcopy(value)
    for name in SERVING_FIELDS:
        scalars[f"serving.{name}"] = deepcopy(getattr(serving, name))
    tensors["serving.token_history_prefix"] = serving._token_history[:serving._cycles]
    tensors["serving.count_history_prefix"] = serving._count_history[:serving._cycles]
    tensors["handoff.host_count_copy"] = handoff._host_count_copy
    tensors["handoff.committed_emitted_count"] = handoff._committed_emitted_count
    scalars["handoff.host_copy_pending"] = handoff._host_copy_pending
    scalars["runtime.schedule_pending"] = runtime._schedule_pending
    scalars["runtime.schedule_lifetime"] = deepcopy(runtime.schedule_lifetime())

    for name in HOST_MIRROR_NAMES:
        # Some public names are lazy properties; inspect only materialized
        # backing fields and reject a present non-tensor mirror.
        value = common_vars.get(name)
        if value is not None and not torch.is_tensor(value):
            raise RuntimeError(f"Host mirror {name} is not a tensor")
    # Capture every direct common tensor, including fields introduced by a
    # version change. Keep scalar metadata that can affect later execution.
    opaque_common: list[str] = []
    for name, value in common_vars.items():
        if torch.is_tensor(value):
            tensors[f"common.{name}"] = value
        elif isinstance(value, (str, int, float, bool, type(None))):
            scalars[f"common.{name}"] = value
        else:
            opaque_common.append(name)
    for name in PROPOSER_BUFFER_NAMES:
        value = getattr(handoff.proposer, name, None)
        if not torch.is_tensor(value):
            raise RuntimeError(f"required proposer buffer missing: {name}")
        _collect_tensors(f"proposer.{name}", value, tensors)
    draft_gids = {group.kv_cache_group_id
                  for group in handoff.proposer.draft_attn_groups}
    if not draft_gids:
        raise RuntimeError("draft attention groups missing")
    group_keys: dict[str, set[Any]] = {}
    for name in PROPOSER_DICT_NAMES:
        value = getattr(handoff.proposer, name, None)
        if not isinstance(value, dict) or not value:
            raise RuntimeError(f"required proposer dictionary missing: {name}")
        group_keys[name] = set(value)
        _collect_tensors(f"proposer.{name}", value, tensors,
                         strict_leaves=True)
    input_names = PROPOSER_DICT_NAMES[:2]
    draft_names = PROPOSER_DICT_NAMES[2:]
    if group_keys[input_names[0]] != group_keys[input_names[1]]:
        raise RuntimeError("proposer input group-key mismatch")
    if any(group_keys[name] != draft_gids for name in draft_names):
        raise RuntimeError("proposer draft buffer group-key mismatch")
    if not draft_gids <= group_keys[input_names[0]]:
        raise RuntimeError("draft group absent from input maps")
    context_slots = getattr(handoff.proposer, "_context_slot_mapping_buffers", None)
    if not isinstance(context_slots, (list, tuple)) or not context_slots:
        raise RuntimeError("required proposer context slot mappings missing")
    if len(context_slots) != len(handoff.proposer._layer_group_idx):
        raise RuntimeError("context slot mapping layer count mismatch")
    _collect_tensors("proposer.context_slot_mapping_buffers",
                     context_slots, tensors, strict_leaves=True)
    _collect_tensors("handoff.spec_metadata",
                     vars(handoff._spec_metadata), tensors)

    target_owner = getattr(runtime.target.binding.forward, "__self__", None)
    assets = getattr(target_owner, "assets", None)
    if assets is None or not assets.caches:
        raise RuntimeError("Target RuntimeAssets registry unavailable")
    cache_names = [cache.name for cache in assets.caches]
    mutable_names = [mutable.name for mutable in assets._mutable_tensors]
    if len(cache_names) != len(set(cache_names)) or len(mutable_names) != len(set(mutable_names)):
        raise RuntimeError("duplicate Target asset registry name")
    target_cache_identity = {
        cache.name: _identity(cache.tensor) for cache in assets.caches
    }
    target_mutable_identity = {
        mutable.name: _identity(mutable.tensor)
        for mutable in assets._mutable_tensors
    }
    identities = {name: _identity(value) for name, value in tensors.items()}
    total_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    if total_bytes > max_cloned_bytes:
        raise RuntimeError(f"probe clone budget exceeded: {total_bytes}")
    snapshots = {name: value.detach().clone() for name, value in tensors.items()}
    # Clone enqueues on NPU; complete it before the caller enters the pause
    # transition, otherwise the "before" image can race later side-stream work.
    if any(value.device.type == "npu" for value in tensors.values()):
        torch.npu.synchronize()
    return {
        "tensor_identities": identities,
        "tensor_values": snapshots,
        "scalars": scalars,
        "target_cache_identity": target_cache_identity,
        "target_mutable_identity": target_mutable_identity,
        "opaque_common_fields": sorted(opaque_common),
        "cloned_bytes": total_bytes,
        "full_target_kv_content_certified": False,
        "draft_kv_content_certified": False,
    }


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on any changed enumerated field or coverage drift."""
    mismatches: list[str] = []
    for name in ("tensor_identities", "scalars", "target_cache_identity",
                 "target_mutable_identity", "opaque_common_fields"):
        if before[name] != after[name]:
            mismatches.append(name)
    old, new = before["tensor_values"], after["tensor_values"]
    if set(old) != set(new):
        mismatches.append("tensor_names")
    for name in sorted(set(old) & set(new)):
        # Value equality is fail-closed for NaNs and does not certify bitwise
        # identity of signed zero; the report is scoped accordingly.
        if not torch.equal(old[name], new[name]):
            mismatches.append(f"tensor_value:{name}")
    if mismatches:
        raise AssertionError("pause changed enumerated state: " + ", ".join(mismatches))
    return {
        "captured_fields_unchanged": True,
        "opaque_common_fields": before["opaque_common_fields"],
        "complete_common_metadata_certified": False,
        "captured_tensor_names": sorted(old),
        "bitwise_tensor_identity_certified": False,
        "tensor_count": len(old),
        "cloned_bytes": before["cloned_bytes"],
        "target_cache_count": len(before["target_cache_identity"]),
        "full_target_kv_content_certified": False,
        "draft_kv_content_certified": False,
        "fixed_W0_certified": False,
        "numeric_bound_update": False,
    }
