# SPDX-License-Identifier: Apache-2.0
"""Backend payload/count interfaces for the original DeepInfer EPLB.

This module does not choose a balancing policy or its cadence. The original
ExpertLoadBalancer owns those decisions and calls W8A8ExpertShuffle instead
of its generic weight transport. Quantized bytes and graph addresses survive
the transfer; only one layer is staged at a time.
"""
import time
import torch


PAYLOAD_FIELDS = ('first', 'second', 'first_scale', 'second_scale',
                  'fused_first_scale', 'fused_second_scale')


def payload_fields(bank):
    fields = tuple(name for name in PAYLOAD_FIELDS if hasattr(bank, name))
    if fields[:4] != PAYLOAD_FIELDS[:4]:
        raise ValueError('EPLB requires both expert weights and base scales')
    if ('fused_first_scale' in fields) != ('fused_second_scale' in fields):
        raise ValueError('fused scale representations must migrate together')
    return fields


def storage_identity(bank):
    return tuple((name, id(getattr(bank, name)), getattr(bank, name).data_ptr(),
                  tuple(getattr(bank, name).shape), getattr(bank, name).dtype)
                 for name in payload_fields(bank))


def layer_layout(bank):
    count = bank.count
    if type(count) is not int or count < 1:
        raise ValueError('invalid local physical expert count')
    fields = payload_fields(bank)
    if any(getattr(bank, name).numel() % count for name in fields):
        raise ValueError('every payload must have an integral per-expert span')
    if bank.first.dtype != torch.int8 or bank.second.dtype != torch.int8:
        raise ValueError('this payload adapter preserves INT8 expert weights')
    if any(getattr(bank, name).device != bank.first.device for name in fields):
        raise ValueError('bank payloads must share the expert device')
    return (count, tuple((name, getattr(bank, name).numel() // count,
                         getattr(bank, name).dtype) for name in fields))


def stage_payload(bank):
    """Private ND communication tensors; never alias live graph parameters."""
    layer_layout(bank)
    staged = []
    for name in payload_fields(bank):
        value = getattr(bank, name).detach()
        if name in ('first', 'second') and value.device.type == 'npu':
            import torch_npu
            value = torch_npu.npu_format_cast(value, 2)
        staged.append(value.reshape(bank.count, -1).contiguous().clone())
    return staged


@torch.inference_mode()
def publish_payload(bank, staged):
    """Copy into the existing NZ/base/fused-scale storages, without requantizing."""
    fields = payload_fields(bank)
    if len(fields) != len(staged):
        raise ValueError('incomplete W8A8 expert payload')
    identity = storage_identity(bank)
    for name, value in zip(fields, staged):
        target = getattr(bank, name)
        if (value.shape != (bank.count, target.numel() // bank.count)
                or value.dtype != target.dtype or value.device != target.device):
            raise ValueError('W8A8 payload layout changed during migration')
    for name, value in zip(fields, staged):
        # NPU copy_ handles the ND source / existing NZ destination. It must
        # keep the destination object and storage used by every captured graph.
        getattr(bank, name).copy_(value.reshape_as(getattr(bank, name)))
    if storage_identity(bank) != identity:
        raise RuntimeError('expert migration replaced a graph-owned storage')


class W8A8ExpertShuffle:
    """Adapt only weight layout around the unchanged official P2P algorithm."""
    def __init__(self, banks, shuffle=None):
        self.banks = tuple(banks)
        if not self.banks:
            raise ValueError('empty expert layout group')
        layout = layer_layout(self.banks[0])
        if any(layer_layout(bank) != layout for bank in self.banks):
            raise ValueError('use the original homogeneous-layout grouping')
        self._shuffle = shuffle
        self.records = []

    @torch.inference_mode()
    def __call__(self, origin, expert_weights, current, ep_group):
        if (len(expert_weights) != len(self.banks) or origin.shape != current.shape
                or origin.shape[0] != len(self.banks)
                or origin.shape[1] != ep_group.size() * self.banks[0].count):
            raise ValueError('original EPLB map and W8A8 group differ')
        shuffle = self._shuffle
        if shuffle is None:
            from deepinfer.model_executor.eplb.shuffle import expert_model_parallel_shuffle_inplace
            shuffle = expert_model_parallel_shuffle_inplace
        if self.banks[0].first.device.type == "npu":
            # No other graph stream may still read a bank being migrated.
            torch.npu.synchronize()
        started = time.perf_counter()
        staged_bytes = 0
        for index, bank in enumerate(self.banks):
            identity = storage_identity(bank)
            staged = stage_payload(bank)
            staged_bytes = max(staged_bytes, sum(x.numel() * x.element_size() for x in staged))
            # The official function itself owns receive buffers and sender /
            # receiver pairing. A single-layer call bounds extra device memory.
            shuffle(origin[index:index + 1], [staged], current[index:index + 1], ep_group)
            publish_payload(bank, staged)
            if storage_identity(bank) != identity:
                raise RuntimeError('expert bank identity changed')
            del staged
        self.records.append(dict(layers=len(self.banks),peak_staged_payload_bytes=staged_bytes,
                                 host_seconds=time.perf_counter() - started))


def accumulate_expert_counts(destination, counts, *, cumulative=False):
    """Convert the dispatch ABI to original cumulative per-step receive stats.

    Fused FFN outputs independent counts per local expert. Non-fused dispatch
    with group_list_type=0 returns an inclusive prefix sum instead. Padding
    exclusion is supplied by the actual dispatch's active mask, not inferred
    here from a second routing implementation.
    """
    if (destination.dtype != torch.int32 or destination.ndim != 1
            or counts.shape != destination.shape
            or counts.dtype not in (torch.int32, torch.int64)
            or counts.device != destination.device):
        raise ValueError('invalid local expert count ABI')
    values = counts
    if cumulative:
        values = counts - torch.cat((counts.new_zeros(1), counts[:-1]))
    destination.add_(values.to(destination.dtype))


def map_logical_routes(logical_ids, logical_to_physical, logical_count, ep_rank):
    """Use the existing TileKernels row/rank seed with runtime table width."""
    if (logical_ids.ndim != 2 or logical_ids.dtype not in (torch.int32, torch.int64)
            or logical_to_physical.ndim != 2 or logical_count.ndim != 1
            or logical_to_physical.shape[0] != logical_count.numel()
            or any(x.device != logical_ids.device for x in (logical_to_physical, logical_count))
            or type(ep_rank) is not int or ep_rank < 0):
        raise ValueError('invalid original logical/physical routing ABI')
    # Router top-k supplies valid logical IDs; avoid CPU sync in graph capture.
    logical = logical_ids.reshape(-1).long()
    counts = logical_count.index_select(0, logical).long()
    seeds = (torch.arange(logical_ids.shape[0], dtype=torch.int64,
                          device=logical_ids.device) * 23333 + ep_rank)
    offsets = seeds.repeat_interleave(logical_ids.shape[1]).remainder(counts)
    mapped = logical_to_physical.reshape(-1).index_select(
        0, logical * logical_to_physical.shape[1] + offsets)
    return mapped.reshape_as(logical_ids).to(logical_ids.dtype)
