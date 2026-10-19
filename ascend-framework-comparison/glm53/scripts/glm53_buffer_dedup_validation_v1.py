"""Validate confirmed L3 dedup beliefs on use; no key pinning or payload changes."""
import functools
import hashlib
import json
import os
from pathlib import Path

NATIVE_SHA = '3871533fdab985747c76963385a9cab79bbfa58ecee52827d49ba812f8f685e6'
BACKEND = 'sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store'


def install(module):
    assert os.environ.get('GLM53_BUFFER_DEDUP_VALIDATE') == '1'
    assert os.environ.get('GLM53_NODE') in ('34', '35'), 'P-only cache change'
    assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() == NATIVE_SHA
    cls = module.BufferModePipeline
    original = cls.enqueue_backup_intent
    assert not getattr(original, '_glm53_dedup_validation', False)
    assert original.__module__ == module.__name__

    @functools.wraps(original)
    def enqueue_backup_intent(self, node_id):
        cache = self._cache
        # Preserve native no-storage and pending-intent behavior. Nothing
        # confirmed is probed until native write completion established it.
        if cache.enable_storage and node_id not in self.inflight_backup_node_ids:
            snapshot = cache.tree_core.snapshot_buffer_backup(
                node_id, cache.hicache_storage_pass_prefix_keys
            )
            if snapshot is not None and snapshot.hash_values:
                hashes = snapshot.hash_values
                beliefs = cache.storage_existence_cache
                if beliefs.covers_all(module.PoolName.KV, hashes):
                    backend = cache.cache_controller.storage_backend
                    assert type(backend).__module__ == BACKEND
                    assert type(backend).__name__ == 'NpuMemcacheStore'
                    assert cache.host_memory_mode == 'buffer_only'
                    assert not cache.tree_core.has_swa_host_pool
                    assert len(cache.tree_core.components) == 1, 'FULL plus KV-derived sidecars only'
                    error = None
                    try:
                        # Native npu_memcache expands each page to its actual
                        # K/V/index component keys and returns the first miss.
                        local_hits = backend.batch_exists(hashes)
                        if type(local_hits) is not int or not 0 <= local_hits <= len(hashes):
                            raise ValueError('Invalid native page existence result')
                    except Exception as caught:
                        # A metadata error cannot establish existence. Revoke
                        # beliefs and let the original native backup path run.
                        local_hits = 0
                        error = type(caught).__name__
                    value = module.torch.tensor(local_hits, dtype=module.torch.int)
                    cache._all_reduce_attn_groups(value, module.torch.distributed.ReduceOp.MIN)
                    hits = int(value.item())
                    assert 0 <= hits <= local_hits <= len(hashes)
                    if hits < len(hashes):
                        beliefs.invalidate_beyond(module.PoolName.KV, hashes, keep_pages=hits)
                    stats = getattr(self, '_glm53_dedup_stats', None)
                    if stats is None:
                        stats = self._glm53_dedup_stats = {'checks': 0, 'pages_checked': 0, 'beliefs_revoked': 0}
                    stats['checks'] += 1
                    stats['pages_checked'] += len(hashes)
                    stats['beliefs_revoked'] += len(hashes) - hits
                    if stats['checks'] == 1 or hits < len(hashes) or stats['checks'] % 128 == 0:
                        print('GLM53_BUFFER_DEDUP_VALIDATED=' + json.dumps({'project_id': 'glm53', 'pid': os.getpid(), 'node': os.environ['GLM53_NODE'], 'pages': len(hashes), 'native_local_hits': local_hits, 'rank_min_hits': hits, 'metadata_error_type': error, **stats}), flush=True)
        # The original implementation still owns admission, pending coverage,
        # snapshots, source locks, D2H/SDK writes and asynchronous acknowledgments.
        return original(self, node_id)

    enqueue_backup_intent._glm53_dedup_validation = True
    cls.enqueue_backup_intent = enqueue_backup_intent
    return {'native_pipeline_sha256': NATIVE_SHA, 'changed_methods': ['BufferModePipeline.enqueue_backup_intent'], 'mechanism': 'Native existence validation before confirmed local storage dedup skip; rank-min revocation, original backup fallback'}
