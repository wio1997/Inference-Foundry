# SPDX-License-Identifier: Apache-2.0
"""Opt-in native selected-bank construction using a bounded host cache.

The returned bank has the existing selected-expert/global-ID contract. This
is not expert parallel dispatch. The context owns only the CPU cache; callers
own the returned bank and must release it after synchronizing NPU work.
Cache and assembled-payload limits exclude temporary reader buffers, returned
copies, native/NZ allocations and allocator overhead.
"""
from contextlib import contextmanager
from .cached_expert_reader import CachedRankExpertReader
from .rank_expert_cache import RankExpertHostCache
from .expert_bank import load_selected_expert_bank

@contextmanager
def cached_selected_expert_bank(reader,layer_id,global_expert_ids,device,*,rank,world_size,
                                cache_bytes,max_assembled_bytes,max_experts=8,include_shared=True,
                                max_checkpoint_bytes=512*1024**2,max_resident_bytes=512*1024**2):
    cache=RankExpertHostCache(reader,rank=rank,world_size=world_size,max_bytes=cache_bytes)
    try:
        proxy=CachedRankExpertReader(reader,cache,max_assembled_bytes=max_assembled_bytes)
        bank=load_selected_expert_bank(proxy,layer_id,global_expert_ids,device,
                                      max_experts=max_experts,include_shared=include_shared,
                                      max_checkpoint_bytes=max_checkpoint_bytes,
                                      max_resident_bytes=max_resident_bytes)
        yield bank,cache
    finally:
        cache.close()
