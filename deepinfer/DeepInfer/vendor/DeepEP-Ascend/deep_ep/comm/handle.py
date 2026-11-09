import time
from typing import Any, Optional, Tuple

import torch.distributed as dist

# noinspection PyUnresolvedReferences
import deep_ep._C as _C

from ..utils.envs import get_hccl_group_name


class HCCLCommHandle:
    """Own a DeepEP HCCL communicator or borrow one from a PyTorch group."""

    def __init__(self, hccl_comm: str, managed: bool, group: Optional[dist.ProcessGroup] = None):
        self.hccl_comm = hccl_comm
        self.managed = managed
        self.group = group
        self.destroy = _C.destroy_hccl_comm

    def __del__(self):
        if self.managed:
            self.destroy(self.hccl_comm)

    def get(self) -> str:
        """Return the HCCL communicator name consumed by the native runtime."""
        return self.hccl_comm


_storage = dict()


def get_hccl_comm_handle(group: dist.ProcessGroup, force_new_comm: bool = False) -> HCCLCommHandle:
    """Get a cached HCCL communicator, or create an independent managed one."""
    if not force_new_comm:
        if group not in _storage:
            _storage[group] = HCCLCommHandle(get_hccl_group_name(group, group.rank()), False, group)
        return _storage[group]
    unique_ids = [None] * group.size()
    dist.all_gather_object(unique_ids, _C.get_local_hccl_unique_id(), group)
    handle = HCCLCommHandle(_C.create_hccl_comm(unique_ids[0], group.size(), group.rank()), True)
    _storage[time.time_ns()] = handle
    return handle


def get_physical_domain_size(group_or_buffer: Any) -> Tuple[int, int]:
    """Return (scale-out, direct UBMEM) domain sizes for a group or buffer."""
    context = getattr(group_or_buffer, 'context', None)
    if context is not None:
        return context.get_physical_domain_size()
    return _C.get_physical_domain_size(get_hccl_comm_handle(group_or_buffer).get())


def get_logical_domain_size(group_or_buffer: Any,
                            allow_hybrid_mode: Optional[bool] = None) -> Tuple[int, int]:
    """Return the logical domain sizes supported by the Ascend backend."""
    context = getattr(group_or_buffer, 'context', None)
    if context is not None:
        return context.get_logical_domain_size()
    return _C.get_logical_domain_size(
        get_hccl_comm_handle(group_or_buffer).get(), True if allow_hybrid_mode is None else allow_hybrid_mode)


def destroy_all_managed_hccl_comm() -> None:
    """Release cached handles; live buffers retain their communicators."""
    _storage.clear()
