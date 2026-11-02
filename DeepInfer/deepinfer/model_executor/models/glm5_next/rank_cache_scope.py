# SPDX-License-Identifier: Apache-2.0
"""Own an optional rank-local bounded host-cache session for one model run.

The model caller must finish/synchronize all bank work before scope exit. The
session is opt-in; its logical CPU tensor bound excludes assembly, returned
copies, native/NZ allocations and allocator overhead. This is not production EP.
"""
from contextlib import contextmanager
from .cached_expert_session import CachedExpertBankSession


@contextmanager
def optional_rank_cache_session(reader, *, enabled, rank, world_size,
                                cache_bytes, max_assembled_bytes):
    if type(enabled) is not bool:
        raise ValueError('explicit boolean cache-session opt-in required')
    if not enabled:
        yield None
        return
    with CachedExpertBankSession(reader, rank=rank, world_size=world_size,
                                 cache_bytes=cache_bytes,
                                 max_assembled_bytes=max_assembled_bytes) as session:
        yield session
