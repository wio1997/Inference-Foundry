"""Own a real SDK prefix-cache lease before native token-budget admission.

This is a scheduler bridge, not a new prefix cache or a model implementation.
No physical slots or device-copy queues are allocated at admission. Existing
SDK allocate_slots adopts the cached blocks on the first positive-token step.
"""
from dataclasses import dataclass


def verify_prefix_sources():
    import hashlib
    import inspect
    from pathlib import Path
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    for cls, expected in (
        (BlockPool, '34c2fd67148cecb7763ff49ad694a57073759b0fa67572a4270dc995b4acf8fc'),
        (KVCacheManager, '58800535eda854580d84e1025bf32fe01b2a43ce0bba501b4f3d125e4f2c8afa'),
    ):
        if hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest() != expected:
            raise RuntimeError('native prefix admission requires the pinned SDK pool contract')


@dataclass
class CachedPrefixLease:
    blocks: object
    tokens: int
    boundary: int
    pool: object
    pins: tuple
    released: bool = False

    @classmethod
    def acquire(cls, cache, blocks, tokens, boundary, prompt_length, block_size):
        if (type(tokens) is not int or not 0 < tokens < prompt_length
                or tokens % block_size or len(blocks.blocks) != cache.num_kv_cache_groups):
            raise ValueError('only a real aligned, nonterminal SDK prefix can be admitted')
        # Keep duplicate/null references: each touch must have its matching free.
        pins = tuple(block for group in blocks.blocks for block in group)
        if (not pins or any(block.pool is not None and block.pool is not cache.block_pool
                            for block in pins)):
            raise ValueError('prefix lease requires resident blocks in the SDK owner pool')
        cache.block_pool.touch(pins)
        return cls(blocks, tokens, boundary, cache.block_pool, pins)

    def release(self):
        if self.released:
            return False
        self.pool.free_blocks(reversed(self.pins))
        self.released = True
        return True
