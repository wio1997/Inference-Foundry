#!/usr/bin/env python3
"""Snapshot stage-visible exact benchmark files and verify saved source manifest SHA; no NPU calls.

Only the owned stage container and listed task files are read. A missing container/file,
image mismatch, saved-manifest hash mismatch or unexpected resolved path stops the query.
Container mount destination/type/read-only state are recorded (never env vars,
credentials, serials or host mount source paths). Successful source snapshots
are written only under this task's agents/bounds/benchmark-client-snapshot/.
"""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SSH = '/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINERS = {
    'stage': 'wio-vllm-v41-repro169-stage-20261028',
}
EXPECTED_IMAGE = 'sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
FILES = {
    'prefix_api': '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/benchmark-20261028/client-packages/ais_bench_prefix_cache/models/vllm_prefix_cache_api.py',
    'vllm_api': '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/benchmark-20261028/client-packages/ais_bench/benchmark/models/api_models/vllm_custom_api.py',
    'perf_calc': '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/benchmark-20261028/client-packages/ais_bench/benchmark/calculators/default_perf_metric_calculator.py',
    'prefix_inferencer': '/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/benchmark-20261028/client-packages/ais_bench_prefix_cache/openicl/icl_inferencer/prefix_cache_gen_inferencer.py',
}
EXPECTED_HASH = {
    'prefix_api': 'cb0424af9e1ad9dd694aff5e7a0249a31cff40d77ff37b5d8caadc23d04a266e',
    'vllm_api': '8170e1ca344340a4919e05fb61df7bfdcd8d7f7b7cca9d427924d793c89da086',
    'perf_calc': 'd7411d3ba02ab530ecd733b48c49f067ab6c92e1179d2054316487012c51cdd6',
    'prefix_inferencer': 'adab7bf8b7b4042e454b7867863a87a254a9c429b74fca36eb6459e61ec05a0d',
}
DEST = HERE / 'benchmark-client-snapshot'
STATUS = HERE / 'benchmark-client-snapshot-status.json'
REMOTE_READER = r'''import base64,hashlib,json,os,sys
paths=json.loads(sys.argv[1]); result={}
for name,path in paths.items():
    try:
        resolved=os.path.realpath(path)
        with open(path,'rb') as f: data=f.read(1000001)
        if len(data)>1000000: raise ValueError('FILE_TOO_LARGE')
        result[name]={'status':'OK','realpath':resolved,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode('ascii')}
    except FileNotFoundError: result[name]={'status':'MISSING'}
    except PermissionError: result[name]={'status':'PERMISSION_DENIED'}
    except Exception as e: result[name]={'status':'ERROR','type':type(e).__name__}
print(json.dumps(result))'''

def remote(args):
    r = subprocess.run([SSH, shlex.join(args)], capture_output=True, text=True, timeout=90)
    if r.returncode:
        raise RuntimeError('REMOTE_COMMAND_FAILED exit=' + str(r.returncode))
    return r.stdout

def save_failure(code, detail):
    STATUS.write_text(json.dumps({'status': code, 'detail': detail, 'source_snapshots_written': False}, indent=2) + '\n')
    print(json.dumps({'status': code, 'detail': detail}))
    sys.exit(2)

if DEST.exists() or STATUS.exists():
    print(json.dumps({'status': 'STOP_OUTPUT_ALREADY_EXISTS'}))
    sys.exit(2)

try:
    meta = {}
    content = {}
    for label, name in CONTAINERS.items():
        parts = remote(['docker', 'inspect', '--format', '{{.Id}} {{.Image}} {{.State.Running}}', name]).split()
        if len(parts) != 3:
            save_failure('STOP_INVALID_CONTAINER_INSPECT', label)
        info = {'Id': parts[0], 'Image': parts[1], 'Running': parts[2] == 'true'}
        if not info['Running'] or info['Image'] != EXPECTED_IMAGE:
            save_failure('STOP_CONTAINER_IMAGE_OR_STATE', label)
        mounts_raw = json.loads(remote(['docker', 'inspect', '--format', '{{json .Mounts}}', name]))
        mounts = [{'destination': m.get('Destination'), 'type': m.get('Type'), 'rw': m.get('RW')}
                  for m in mounts_raw if any(
                      path == m.get('Destination') or path.startswith(str(m.get('Destination', '')) + '/')
                      or str(m.get('Destination', '')).startswith(path + '/') for path in FILES.values())]
        meta[label] = {'container': name, 'id': info.get('Id'), 'image_id': info.get('Image'),
                       'running': True, 'mounts_covering_targets': mounts}
        content[label] = json.loads(remote(['docker', 'exec', name, 'python3', '-c', REMOTE_READER,
                                           json.dumps(FILES, separators=(',', ':'))]))
    for key, expected in FILES.items():
        entries = [content[label].get(key, {}) for label in CONTAINERS]
        if any(x.get('status') != 'OK' for x in entries):
            save_failure('STOP_MISSING_OR_UNREADABLE_FILE', key)
        if any(x.get('realpath') != expected for x in entries):
            save_failure('STOP_UNEXPECTED_RESOLVED_PATH', key)
        if entries[0].get('sha256') != EXPECTED_HASH[key]:
            save_failure('STOP_SOURCE_MISMATCH', key)
        for e in entries:
            raw = base64.b64decode(e['base64'], validate=True)
            if len(raw) != e['bytes'] or hashlib.sha256(raw).hexdigest() != e['sha256']:
                save_failure('STOP_TRANSFER_HASH_MISMATCH', key)
except Exception as exc:
    save_failure('STOP_QUERY_ERROR', type(exc).__name__ + ': ' + str(exc)[:160])

DEST.mkdir()
snapshots = {}
for key in FILES:
    item = content['stage'][key]
    out = DEST / (key + '.py')
    out.write_bytes(base64.b64decode(item['base64']))
    snapshots[key] = {'path': str(out), 'sha256': item['sha256'], 'bytes': item['bytes'],
                      'container_path': FILES[key], 'matches_saved_client_source_manifest': True}
result = {'status': 'OK', 'containers': meta, 'snapshots': snapshots,
          'comparison_to_public_commit_content': 'NOT_CHECKED', 'no_npu_calls': True}
STATUS.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'status': 'OK', 'status_file': str(STATUS), 'snapshots': snapshots}, ensure_ascii=False))
