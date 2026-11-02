# SPDX-License-Identifier: Apache-2.0
"""Explicitly owned reusable host cache for selected-expert bank loads.

This opt-in session is not installed into the text runner. It owns one bounded
CPU payload cache across sequential bank leases. The borrower owns every bank
and must synchronize and release any NPU allocations before leaving its lease.
The cache bound excludes transient reader/assembly buffers, returned copies,
native/NZ allocations and allocator overhead. This is not production EP.
"""
from contextlib import contextmanager
from threading import RLock
from .rank_expert_cache import RankExpertHostCache
from .cached_expert_reader import CachedRankExpertReader
from .expert_bank import load_selected_expert_bank


class CachedExpertBankSession:
    """One reader/rank and one bounded cache, explicitly closed by its owner."""
    def __init__(self, reader, *, rank, world_size, cache_bytes, max_assembled_bytes):
        self._cache = RankExpertHostCache(reader, rank=rank, world_size=world_size,
                                          max_bytes=cache_bytes)
        try:
            self._proxy = CachedRankExpertReader(reader, self._cache,
                                                max_assembled_bytes=max_assembled_bytes)
        except BaseException:
            self._cache.close()
            raise
        self._rank = rank
        self._world_size = world_size
        self._lock = RLock()
        self._active = False
        self._entered = False

    def __enter__(self):
        with self._lock:
            if self._cache.stats()['closed'] or self._entered:
                raise RuntimeError('session already entered or closed')
            self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def stats(self):
        with self._lock:
            result = self._cache.stats()
            result['active_lease'] = self._active
            return result

    def validate_binding(self, reader, *, rank, world_size, cache_bytes, max_assembled_bytes):
        """Reject a closed session or one belonging to another execution path."""
        with self._lock:
            if self._cache.stats()['closed']:
                raise RuntimeError('session is closed')
            if (self._cache.reader is not reader or self._rank != rank or
                self._world_size != world_size or self._cache.max_bytes != cache_bytes or
                self._proxy.max_assembled_bytes != max_assembled_bytes):
                raise ValueError('cache session binding differs from ranked execution')

    def close(self):
        with self._lock:
            if self._active:
                raise RuntimeError('cannot close while a bank lease is active')
            self._cache.close()

    @contextmanager
    def bank(self, layer_id, global_expert_ids, device, *, max_experts=8,
             include_shared=True, max_checkpoint_bytes=512*1024**2,
             max_resident_bytes=512*1024**2):
        with self._lock:
            if self._cache.stats()['closed']:
                raise RuntimeError('session is closed')
            if self._active:
                raise RuntimeError('only one bank lease is supported')
            if type(include_shared) is not bool or (include_shared and self._rank != 0):
                raise ValueError('shared expert may be loaded only on rank zero')
            self._active = True
        try:
            value = load_selected_expert_bank(
                self._proxy, layer_id, global_expert_ids, device,
                max_experts=max_experts, include_shared=include_shared,
                max_checkpoint_bytes=max_checkpoint_bytes,
                max_resident_bytes=max_resident_bytes)
            yield value
        finally:
            with self._lock:
                self._active = False


@contextmanager
def session_selected_expert_bank(session, reader, layer_id, global_expert_ids, device, *,
                                 rank, world_size, cache_bytes, max_assembled_bytes,
                                 max_experts=8, include_shared=True,
                                 max_checkpoint_bytes=512*1024**2,
                                 max_resident_bytes=512*1024**2):
    """Borrow a sequential bank lease; session lifetime belongs to the caller."""
    if not isinstance(session, CachedExpertBankSession):
        raise ValueError('explicit reusable expert bank session required')
    session.validate_binding(reader, rank=rank, world_size=world_size,
                             cache_bytes=cache_bytes, max_assembled_bytes=max_assembled_bytes)
    with session.bank(layer_id, global_expert_ids, device, max_experts=max_experts,
                      include_shared=include_shared, max_checkpoint_bytes=max_checkpoint_bytes,
                      max_resident_bytes=max_resident_bytes) as bank:
        yield bank, session
