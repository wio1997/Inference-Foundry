"""Local cluster configuration; checked-in examples contain no site addresses."""
import fcntl
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

EXAMPLE = {
    'prefill_url': 'http://P_LEADER:31400',
    'decode_url': 'http://D_LEADER:31401',
    'router_url': 'http://ROUTER_HOST:31480',
    'meta_service_url': 'tcp://META_HOST:25340',
    'extra_backend_tag': 'REPLACE_WITH_DEDICATED_CACHE_TAG',
    'model_path': '/model',
    'model_name': 'glm-5.3-pd',
    'benchmark_cwd': '/work/benchmark',
    'lock_roots': ['/work/aisbench-memcache-lock'],
    'p_audit_root': '/work/glm53-memcache-audit',
}
PROFILE_FILE = os.environ.get('GLM_BENCH_PROFILE')
PROFILE = dict(EXAMPLE)
if PROFILE_FILE:
    PROFILE.update(json.loads(Path(PROFILE_FILE).read_text()))

LEADERS = [
    ('prefill', PROFILE['prefill_url'], 64, 4, 64),
    ('decode', PROFILE['decode_url'], 32, 16, 32),
]

def validate_run_profile():
    assert PROFILE_FILE, 'Set GLM_BENCH_PROFILE to your local profile JSON'
    for key in ['prefill_url', 'decode_url', 'router_url', 'meta_service_url']:
        url = urlsplit(PROFILE[key])
        assert url.scheme in ('http', 'https', 'tcp') and url.hostname and url.port
        assert url.hostname.upper() not in {'P_LEADER', 'D_LEADER', 'ROUTER_HOST', 'META_HOST'}, key
        assert not url.username and not url.password, 'Keep credentials outside the profile'
    assert PROFILE['extra_backend_tag'] != EXAMPLE['extra_backend_tag']
    assert PROFILE['lock_roots'], 'Share lock directories with every load generator using this service'

def benchmark_locks(root):
    """Retain existing benchmark.lock interoperability, deduplicating paths."""
    result = []
    for folder in dict.fromkeys([*PROFILE['lock_roots'], str(root)]):
        directory = Path(folder)
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / 'benchmark.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result.append(handle)
    return result
