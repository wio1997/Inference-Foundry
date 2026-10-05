"""Experimental SFA DCP metadata extent contract; activated only by task worker.

The native tensor operations and block/position formula remain unchanged.
Request indices cover the query extent; the allocated model input can include
padding. Slots outside the active query extent remain PAD_SLOT_ID (-1).
"""
import functools
import hashlib
import inspect
import json
import os

NATIVE_MODULE_SHA = "17d80412c16270ac0da8a188bdd6d6b7d8b108fbbf92752a6ddf965898085e57"
NATIVE_METHOD_SHA = "e92239ab4adb3d0af74ea46a836f9fab0c5c319347c8282f0917d734e7ad6084"


def query_extent(common):
    """Read the CPU query boundary supplied by the native metadata builder."""
    count = common.num_reqs
    if common.query_start_loc_cpu.device.type != "cpu":
        raise ValueError("query_start_loc_cpu must be on CPU")
    boundaries = common.query_start_loc_cpu[:count + 1].tolist()
    if (len(boundaries) != count + 1 or not boundaries or boundaries[0] != 0
            or any(a > b for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError("invalid request query boundaries")
    extent = int(boundaries[-1])
    if not 0 <= extent <= common.num_input_tokens:
        raise ValueError("query extent exceeds allocated model input")
    if not 0 <= common.num_actual_tokens <= common.num_input_tokens:
        raise ValueError("invalid actual token count")
    return extent


def build_slot_mapping(builder, common, block_table):
    import torch
    num_reqs = common.num_reqs
    num_input_tokens = common.num_input_tokens
    extent = query_extent(common)
    num_actual_tokens = min(common.num_actual_tokens, num_input_tokens, extent)
    local_cols = block_table.shape[1] // builder.dcp_size
    _, _, slots = builder._ensure_replicated_view_buffers(
        num_reqs, num_input_tokens, local_cols)
    slots.fill_(-1)
    if num_actual_tokens == 0:
        return slots
    query_lens = (common.query_start_loc[1:num_reqs + 1]
                  - common.query_start_loc[:num_reqs])
    req_indices = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.int32, device=builder.device),
        query_lens.to(device=builder.device),
        output_size=extent)[:num_actual_tokens]
    # Keep the native block and position arithmetic and tensor operations.
    num_actual_tokens = min(num_actual_tokens, req_indices.shape[0])
    req_indices = req_indices[:num_actual_tokens]
    positions = common.positions[:num_actual_tokens].to(
        device=builder.device, dtype=torch.int32)
    logical_block_idx = positions // builder.replicated_view_block_size
    block_offsets = positions % builder.replicated_view_block_size
    indices = req_indices * block_table.shape[1] + logical_block_idx
    blocks = block_table.flatten()[indices]
    slots[:num_actual_tokens] = blocks * builder.replicated_view_block_size + block_offsets
    return slots


def install():
    """Explicit task opt-in; reject a different installed native implementation."""
    from pathlib import Path
    from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadataBuilder
    cls = AscendSFADCPMetadataBuilder
    if getattr(cls, "_glm_query_extent_contract_installed", False):
        return
    original = cls._build_slot_mapping_replicated_view
    if inspect.unwrap(original) is not original:
        raise RuntimeError("unrecognized existing SFA metadata wrapper")
    native_file = Path(inspect.getsourcefile(original))
    if hashlib.sha256(native_file.read_bytes()).hexdigest() != NATIVE_MODULE_SHA:
        raise RuntimeError("installed SFA source identity changed")
    if hashlib.sha256(inspect.getsource(original).encode()).hexdigest() != NATIVE_METHOD_SHA:
        raise RuntimeError("installed SFA method identity changed")

    @functools.wraps(original)
    def contracted(builder, common, block_table):
        extent = query_extent(common)
        if extent == common.num_input_tokens and common.num_actual_tokens <= extent:
            return original(builder, common, block_table)
        result = build_slot_mapping(builder, common, block_table)
        print("GLM_SFA_QUERY_EXTENT_CORRECTED " + json.dumps(dict(
            pid=os.getpid(), num_reqs=common.num_reqs,
            query_extent=extent, model_input_tokens=common.num_input_tokens,
            declared_actual_tokens=common.num_actual_tokens,
            mapped_actual_tokens=min(common.num_actual_tokens, extent),
            padding_slot_id=-1, native_tensor_operations_preserved=True)), flush=True)
        return result

    cls._build_slot_mapping_replicated_view = contracted
    cls._glm_query_extent_contract_installed = True
    print("GLM_SFA_QUERY_EXTENT_CONTRACT_INSTALLED " + json.dumps(dict(
        pid=os.getpid(), native_module_sha=NATIVE_MODULE_SHA,
        native_method_sha=NATIVE_METHOD_SHA, metadata_only=True,
        operator_implementation_changes=0)), flush=True)
