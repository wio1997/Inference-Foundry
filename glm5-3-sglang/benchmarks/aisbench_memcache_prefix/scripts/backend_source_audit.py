"""Verify native L3 counters and separately prove the P-side backend identity."""
from bench_profile import PROFILE
import json
from pathlib import Path

def verify_backend_source(root, label, native):
    assert isinstance(native, dict), 'Missing native cache-layer counters'
    assert native.get('storage') == 74368, ('Expected actual storage74368', native)
    assert native.get('device') == 0 and native.get('host') == 0, ('L1/L2 were flushed', native)
    name = native.get('storage_backend')
    assert name in (None, 'npu_memcache', 'NpuMemcacheStore'), ('Unexpected native backend', name)
    p = Path(root)/'logs'/f'cache-admin-after-warm-{label}.json'
    flush = json.loads(p.read_text())
    assert flush['operation'] == 'flush_l1_l2'
    info = flush['heads']['prefill']['server_info']
    info = info.get('server_args') or info
    assert info['enable_hierarchical_cache'] and info['hicache_storage_backend'] == 'npu_memcache'
    cfg = info['hicache_storage_backend_extra_config']
    cfg = json.loads(cfg) if isinstance(cfg, str) else cfg
    assert 'host_io_mode' not in cfg
    assert cfg['meta_service_url'] == PROFILE['meta_service_url']
    assert cfg['extra_backend_tag'] == PROFILE['extra_backend_tag']
    storage_label = label if label.startswith('cc') else 'cc' + label
    storage = json.loads((Path(root)/'logs'/f'prefix-storage-{storage_label}.json').read_text())
    assert storage['phase'] == 'STORAGE_PREFIX_AUDIT_PASS' and not storage['pending_rows']
    return dict(native_storage_tokens=native['storage'], native_backend_label=name,
        audited_prefill_backend='npu_memcache', evidence=str(p),
        provenance='Native D storage counter transferred from P; backend identity independently verified from actual P server_info, dedicated L3 object presence and idle flush. D without local L3 does not emit the backend name. Native fields are preserved, never fabricated.')
