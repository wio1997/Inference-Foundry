"""Local preparation only: restore native full-batch MC API and native telemetry.

Actual deployment requires a separate frozen preflight/startup launcher. Official
SDK1.3 MetaService lease_ttl_ms=60000 must accompany this source; current live
configuration is untouched. No performance claim follows from this preparation.
"""
import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'config/overlays/vllm-sdma-v4'
OUT=ROOT/'config/overlays/vllm-sdma-v5'
CONFIG=ROOT/'config/vllm-correctness-v5'
assert not OUT.exists() and not CONFIG.exists()
assert hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest()=='deb21c5494942d0d352c2c0bb2d51aa5b726f315374566b8bbd3982db305bfe6'
shutil.copytree(OLD,OUT)
shutil.copytree(ROOT/'config/vllm-correctness-v4',CONFIG)

helper='''"""Observe one unchanged full-batch native SDK read, without retry or slicing."""
import hashlib,json,logging,time
from glm53_persistent_audit import persist
logger=logging.getLogger('vllm.glm53_memcache_native')
def native_get_into_layers(store,keys,addresses,sizes,direction):
    if not keys or len(keys)!=len(addresses) or len(keys)!=len(sizes):
        raise ValueError('Aligned nonempty native MemCache key/address/size rows required')
    for pointers,lengths in zip(addresses,sizes):
        if not pointers or len(pointers)!=len(lengths) or any(type(n) is not int or n<=0 for n in lengths):
            raise ValueError('Aligned positive native layer sizes required')
    begin=time.monotonic_ns()
    result=store.batch_get_into_layers(keys,addresses,sizes,direction)
    event={'event':'native_get_batch_complete','begin_monotonic_ns':begin,'terminal_monotonic_ns':time.monotonic_ns(),
           'first_key_index':0,'key_count':len(keys),'total_key_count':len(keys),
           'key_list_sha256':hashlib.sha256(json.dumps(keys,separators=(',',':')).encode()).hexdigest(),
           'payload_bytes_requested':sum(sum(row) for row in sizes),'direction':direction,'native_results':result,
           'batch_limit_blocks':None,'native_call_count':1,'retry_attempts':0,'fallback':False}
    persist('MC_GET',event)
    logger.info('GLM53_MC_GET='+json.dumps(event,sort_keys=True))
    if result is None or len(result)!=len(keys):
        raise RuntimeError('Native full-batch MemCache result shape mismatch; no retry')
    if any(int(value)!=0 for value in result):
        raise RuntimeError('Native full-batch MemCache read failed; no retry/fallback')
    return result
'''
ast.parse(helper)
(OUT/'python/glm53_memcache_native.py').write_text(helper)
backend=OUT/'distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py'
s=backend.read_text()
s=s.replace('from glm53_memcache_batch import batched_get_into_layers','from glm53_memcache_native import native_get_into_layers')
old='''        self._glm53_get_batch_blocks = int((extra_config or {}).get("glm53_get_batch_blocks", 32))
        if not 1 <= self._glm53_get_batch_blocks <= 128:
            raise ValueError("glm53_get_batch_blocks must be in 1..128")
'''
assert old in s
s=s.replace(old,'').replace('res = batched_get_into_layers(','res = native_get_into_layers(')
s=s.replace('self.store, key, addr, size, MmcDirect.COPY_G2L.value, self._glm53_get_batch_blocks','self.store, key, addr, size, MmcDirect.COPY_G2L.value')
assert 'glm53_get_batch_blocks' not in s and 'batched_get_into_layers' not in s
ast.parse(s);backend.write_text(s)
(OUT/'python/glm53_memcache_batch.py').unlink()
manifest=json.loads((OUT/'manifest.json').read_text())
manifest['parent_manifest_sha256']=hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest()
for entry in manifest['files']:
    entry['patched_sha256']=hashlib.sha256((OUT/entry['local']).read_bytes()).hexdigest()
    if entry['local']=='distributed/kv_transfer/kv_pool/ascend_store/backend/memcache_backend.py':
        original=ROOT/'runs/20261017T0300-framework-source/code-30/11-memcache_backend.py'
        assert hashlib.sha256(original.read_bytes()).hexdigest()==entry['original_sha256']
        patch=''.join(difflib.unified_diff(original.read_text().splitlines(keepends=True),
                                         backend.read_text().splitlines(keepends=True),
                                         fromfile=entry['local'],tofile=entry['local']))
        (OUT/entry['patch']).write_text(patch)
        entry['patch_sha256']=hashlib.sha256(patch.encode()).hexdigest()
manifest['helpers']=[entry for entry in manifest['helpers'] if entry['local']!='python/glm53_memcache_batch.py']
manifest['helpers'].append({'local':'python/glm53_memcache_native.py','sha256':hashlib.sha256(helper.encode()).hexdigest()})
manifest['status']='PREPARED_NOT_DEPLOYED'
manifest['changes_from_v4']=['Restore single full native SDK batch_get_into_layers call; observe status/bytes without slicing/retry/fallback.','Use documented SDK1.3 MetaService lease_ttl_ms=60000 in a separately frozen deployment; not a live mutation.']
manifest['required_meta_service_lease_ttl_ms']=60000
manifest['documentation']='runs/20261017T1340-official-memcache-docs-review/index.json'
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

for node in [30,31,43,45]:
    p=CONFIG/f'node-{node}.json';cfg=json.loads(p.read_text());args=cfg['launch_args']
    cfg['phase']='NATIVE_MC_LEASE_AND_PROFILE_OBSERVABILITY_QUALIFICATION_NOT_BASELINE'
    cfg['source_overlay']='config/overlays/vllm-sdma-v5'
    for connector in cfg['kv_transfer']['kv_connector_extra_config']['connectors']:
        connector.get('kv_connector_extra_config',{}).pop('glm53_get_batch_blocks',None)
    args[args.index('--kv-transfer-config')+1]=json.dumps(cfg['kv_transfer'])
    assert '--cudagraph-metrics' not in args and '--profiler-config' not in args
    profiler={'profiler':'torch','torch_profiler_dir':'/run/glm53/profile',
              'ignore_frontend':True,'delay_iterations':2,'max_iterations':8,
              'torch_profiler_with_stack':False,'torch_profiler_with_memory':False,
              'torch_profiler_record_shapes':False,'capture_torch_profiler':False}
    args+=['--cudagraph-metrics','--profiler-config',json.dumps(profiler)]
    cfg['required_meta_service_lease_ttl_ms']=60000
    cfg['profiler_intent']={'native_config':profiler,'activation':'Separate bounded /start_profile and /stop_profile run after baseline; never during formal baseline.','native_max_iteration_boundary':'WorkerProfiler uses > max_iterations; up to9 counted steps for configured8.'}
    p.write_text(json.dumps(cfg,indent=2)+'\n')
print('prepared_only',hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest())
