"""Pure stubs plus exact native method ASTs; no framework/SDK/device imports."""
import __future__
import ast
from collections import deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import tempfile


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Beliefs:
    def __init__(self, keys):
        self.keys = set(keys)
        self.invalidations = []

    def covers_all(self, pool, hashes, extra_cover=None):
        return set(hashes) <= self.keys | set(extra_cover or {})

    def invalidate_beyond(self, pool, hashes, keep_pages):
        self.invalidations.append(keep_pages)
        self.keys.difference_update(hashes[keep_pages:])


def native_method(source, class_name, method_name, module_name, context):
    cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    tree = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    context.update(__name__=module_name)
    exec(compile(tree, '<exact-native-AST>', 'exec', flags=__future__.annotations.compiler_flag), context)
    return context[method_name]


def main():
    script_root = Path(__file__).resolve().parent
    source_root = Path(sys.argv[1]).resolve()
    source_path = source_root / 'srt__mem_cache__buffer_mode__pipeline.py'
    source = source_path.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == '3871533fdab985747c76963385a9cab79bbfa58ecee52827d49ba812f8f685e6'
    helper_path = script_root / 'glm53_buffer_dedup_validation_v1.py'
    assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == '78ec7134ff11ead6f30504e80aa23a0b5cf92f4c7149ac2f57dab4c756e4e08d'
    spec = importlib.util.spec_from_file_location('dedup_opt_fixture', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    torch = types.SimpleNamespace(tensor=lambda value, dtype: Scalar(value), int='CPU-int', distributed=types.SimpleNamespace(ReduceOp=types.SimpleNamespace(MIN='MIN')))
    pool = types.SimpleNamespace(KV='KV')
    module = types.ModuleType('fixture_exact_native_pipeline')
    module.__file__ = str(source_path)
    context = {'PoolName': pool, 'BufferBackupState': lambda **kwargs: types.SimpleNamespace(**kwargs), '_UnifiedBackupIntent': lambda **kwargs: types.SimpleNamespace(**kwargs), 'logger': types.SimpleNamespace(error=lambda *args: None)}
    original = native_method(source, 'BufferModePipeline', 'enqueue_backup_intent', module.__name__, context)
    module.BufferModePipeline = type('BufferModePipeline', (), {'enqueue_backup_intent': original})
    module.torch, module.PoolName = torch, pool
    previous = {key: os.environ.get(key) for key in ('GLM53_BUFFER_DEDUP_VALIDATE', 'GLM53_NODE')}
    os.environ.update(GLM53_BUFFER_DEDUP_VALIDATE='1', GLM53_NODE='34')
    helper.install(module)
    checks = []
    hashes = ['h0', 'h1', 'h2', 'h3']

    def make(local=4, reduced=None, beliefs=None, inflight=False, cover=(), enabled=True, snapshot=True, parent=True, oversize=False, backlog=False):
        obj = module.BufferModePipeline()
        obj.inflight_backup_node_ids = {7} if inflight else set()
        obj.inflight_backup_hashes = dict.fromkeys(cover, 1)
        obj.pending_write_queue = deque()
        obj.write_backlog_tokens_ = 1000 if backlog else 0
        obj.write_backlog_cap = 1000
        obj._backlog_cap_hits = 0
        obj.drops = []
        obj._log_backup_dropped = obj.drops.append
        obj._backup_parent_covered = lambda state: parent
        obj._backup_oversize = lambda *args: oversize
        value = types.SimpleNamespace(node_id=7, parent_node_id=0, parent_is_root=True, parent_last_hash=None, hash_values=list(hashes)) if snapshot else None
        tree = types.SimpleNamespace(snapshot_buffer_backup=lambda *args: value, has_swa_host_pool=False, components=(object(),))
        Backend = type('NpuMemcacheStore', (), {'__module__': helper.BACKEND})
        backend = Backend()
        backend.calls = []
        def exists(keys):
            backend.calls.append(list(keys))
            if isinstance(local, Exception):
                raise local
            return local
        backend.batch_exists = exists
        sync = []
        def reduce(value, op):
            assert op == 'MIN'
            sync.append(value.item())
            if reduced is not None:
                value.value = reduced
        cache = types.SimpleNamespace(enable_storage=enabled, tree_core=tree, hicache_storage_pass_prefix_keys=False, storage_existence_cache=Beliefs(hashes if beliefs is None else beliefs), cache_controller=types.SimpleNamespace(storage_backend=backend), host_memory_mode='buffer_only', _all_reduce_attn_groups=reduce, page_size=128)
        obj._cache = cache
        return obj, backend, sync

    scenarios = [
        ('confirmed_full', {}, 0, 1, []),
        ('stale_zero', {'local': 0}, 1, 1, [0]),
        ('stale_partial', {'local': 2}, 1, 1, [2]),
        ('remote_rank_shorter', {'local': 4, 'reduced': 1}, 1, 1, [1]),
        ('metadata_error', {'local': RuntimeError('mock unavailable')}, 1, 1, [0]),
        ('negative_api', {'local': -1}, 1, 1, [0]),
        ('oversize_api', {'local': 5}, 1, 1, [0]),
        ('float_api', {'local': 2.0}, 1, 1, [0]),
        ('bool_api', {'local': True}, 1, 1, [0]),
        ('unconfirmed', {'beliefs': ['h0']}, 1, 0, []),
        ('mixed_pending_cover', {'beliefs': ['h0', 'h1'], 'cover': ['h2', 'h3']}, 0, 0, []),
        ('node_pending', {'inflight': True}, 0, 0, []),
        ('no_storage', {'enabled': False}, 0, 0, []),
        ('no_snapshot', {'snapshot': False}, 0, 0, []),
        ('native_oversize_reject', {'local': 0, 'oversize': True}, 0, 1, [0]),
        ('native_parent_reject', {'local': 0, 'parent': False}, 0, 1, [0]),
        ('native_backlog_reject', {'local': 0, 'backlog': True}, 0, 1, [0]),
    ]
    for name, arguments, queue, calls, invalidations in scenarios:
        obj, backend, sync = make(**arguments)
        module.BufferModePipeline.enqueue_backup_intent(obj, 7)
        assert len(obj.pending_write_queue) == queue, name
        assert len(backend.calls) == len(sync) == calls, name
        assert obj._cache.storage_existence_cache.invalidations == invalidations, name
        if queue:
            assert obj.pending_write_queue[0].snapshot.node_id == 7
            assert obj.pending_write_queue[0].snapshot.hash_values == hashes
            assert obj.write_backlog_tokens_ == len(hashes) * 128
            assert obj.inflight_backup_node_ids == {7}
        assert hashes == ['h0', 'h1', 'h2', 'h3']
        checks.append(name)
    # The exact unmodified native method skips stale, confirmed beliefs.
    reference, _, _ = make(local=0)
    original(reference, 7)
    assert not reference.pending_write_queue
    # Consensus: 32 workers across four independent attention groups agree on
    # the first missing page even when native API observations differ by rank.
    for rank in range(32):
        group_min = rank // 8 % 4
        obj, _, sync = make(local=group_min + rank % (5 - group_min), reduced=group_min)
        module.BufferModePipeline.enqueue_backup_intent(obj, 7)
        assert obj._cache.storage_existence_cache.invalidations == [group_min]
        assert len(obj.pending_write_queue) == 1 and len(sync) == 1
    # Exercise the exact native backend key expansion, including the index
    # component: a missing sidecar must never count as a complete KV page.
    backend_source = (source_root / 'srt__mem_cache__storage__npu_memcache__npu_memcache_store.py').read_text()
    assert hashlib.sha256(backend_source.encode()).hexdigest() == 'ceda6ee4a7515e2bbf1177aa18eea6949ee9ebf80f5bcb0041c0ff360f83a1c5'
    exists = native_method(backend_source, 'NpuMemcacheStore', 'batch_exists', 'fixture_native_backend', {})
    backend = types.SimpleNamespace(mem_pool_host=types.SimpleNamespace(kv_buffer=object()), is_mla_backend=True, mla_suffix='', _tag_keys=lambda keys: ['tag:' + key for key in keys], _mla_uses_kv_split=lambda: True, _mla_fp8_packed_kv=lambda: False, _mla_has_index_k=lambda: True, _mla_has_index_scale=lambda: False, _mla_key_multiplier=lambda: 3)
    requested = []
    for missing, expected in ((None, 4), ('tag:h1__index_k', 1), ('tag:h0__k', 0), ('tag:h3__v', 3)):
        def component_exists(keys):
            requested[:] = keys
            return [0 if key == missing else 1 for key in keys]
        backend._batch_exist = component_exists
        assert exists(backend, hashes) == expected
        assert requested == [f'tag:{key}__{component}' for key in hashes for component in ('k', 'v', 'index_k')]
    rejected = []
    for kind in ('wrong_backend', 'wrong_mode', 'SWA', 'multiple_components'):
        obj, _, _ = make()
        if kind == 'wrong_backend':
            obj._cache.cache_controller.storage_backend = object()
        elif kind == 'wrong_mode':
            obj._cache.host_memory_mode = 'cache'
        elif kind == 'SWA':
            obj._cache.tree_core.has_swa_host_pool = True
        else:
            obj._cache.tree_core.components = (object(), object())
        try:
            module.BufferModePipeline.enqueue_backup_intent(obj, 7)
        except AssertionError:
            rejected.append(kind)
        else:
            raise AssertionError(kind)
    for kind in ('duplicate_install', 'wrong_role', 'disabled'):
        os.environ['GLM53_NODE'] = '36' if kind == 'wrong_role' else '34'
        os.environ['GLM53_BUFFER_DEDUP_VALIDATE'] = '0' if kind == 'disabled' else '1'
        try:
            helper.install(module)
        except AssertionError:
            rejected.append(kind)
        else:
            raise AssertionError(kind)
    os.environ.update(GLM53_BUFFER_DEDUP_VALIDATE='1', GLM53_NODE='34')
    with tempfile.TemporaryDirectory() as temporary:
        wrong_source = Path(temporary) / 'pipeline.py'
        wrong_source.write_text(source + '\n# source-guard negative fixture\n')
        previous_path = module.__file__
        module.__file__ = str(wrong_source)
        try:
            helper.install(module)
        except AssertionError:
            rejected.append('source_hash_mismatch')
        else:
            raise AssertionError('source_hash_mismatch')
        finally:
            module.__file__ = previous_path
    hook_path = script_root / 'glm53_buffer_dedup_import_hook_v1.py'
    assert hashlib.sha256(hook_path.read_bytes()).hexdigest() == 'c942570973771dde5db4a326c17eb5f18c2aa50bedbb1a56b60aeabbbe915595'
    hook_spec = importlib.util.spec_from_file_location('dedup_hook_fixture', hook_path)
    hook = importlib.util.module_from_spec(hook_spec)
    hook_spec.loader.exec_module(hook)
    target = types.ModuleType('fixture_lazy_loader_native_pipeline')
    target.__file__ = str(source_path)
    target.torch, target.PoolName = torch, pool
    class NativeLoader:
        def exec_module(self, loaded):
            method = native_method(source, 'BufferModePipeline', 'enqueue_backup_intent', loaded.__name__, dict(context))
            loaded.BufferModePipeline = type('BufferModePipeline', (), {'enqueue_backup_intent': method})
    hook.Loader(NativeLoader()).exec_module(target)
    assert target.BufferModePipeline.enqueue_backup_intent._glm53_dedup_validation
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    assert not any(name.startswith(('torch', 'sglang', 'memcache_hybrid', 'memfabric_hybrid')) for name in sys.modules)
    result = {'project_id': 'glm53', 'pure_stub_fixture_pass': True, 'original_native_pipeline_sha256': helper.NATIVE_SHA, 'helper_sha256': hashlib.sha256(helper_path.read_bytes()).hexdigest(), 'scenario_checks': checks, 'rank_consensus_cases': 32, 'native_component_existence_cases': 4, 'negative_cases': rejected, 'lazy_loader_install_checked': True, 'SDK_init_calls': 0, 'model_requests': 0, 'NPU_devices': False, 'native_model_qualification_still_required': True}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
    print('GLM53_BUFFER_DEDUP_CPU_FIXTURE=' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
