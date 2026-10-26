#!/usr/bin/env python3
"""Retrieve fixed public Git objects; never substitute a moving branch."""
import concurrent.futures
import datetime
import hashlib
import json
import pathlib
import subprocess
import urllib.request

R = pathlib.Path(__file__).resolve().parents[1]
O = R / 'runs/L027-R004-public-source'
O.mkdir(parents=True, exist_ok=False)
spec = json.loads((R / 'specs/L027-public-source.json').read_text())
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
def api(resource):
    p = subprocess.run(['gh', 'api', 'repos/sgl-project/sglang/' + resource], capture_output=True, text=True, timeout=40)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)
commit = api('git/commits/' + spec['commit'])
assert commit['sha'] == spec['commit'] and commit['tree']['sha'] == spec['tree']
tree = api('git/trees/' + spec['tree'] + '?recursive=1')
assert not tree.get('truncated')
blobs = {x['path']: x['sha'] for x in tree['tree'] if x['type'] == 'blob'}
def get(name):
    assert name in blobs and '..' not in name and name.startswith('python/sglang/srt/')
    url = 'https://raw.githubusercontent.com/sgl-project/sglang/' + spec['commit'] + '/' + name
    with urllib.request.urlopen(url, timeout=40) as response:
        data = response.read()
    digest = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert digest == blobs[name], name
    return name, data, {'url': url, 'git_blob_sha1': digest, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    fetched = list(pool.map(get, spec['paths']))
files = {}
for name, data, item in fetched:
    q = R / 'sources/installed' / name
    item['local_snapshot_preexisted'] = q.exists()
    if q.exists():
        assert q.read_bytes() == data, 'Fixed commit versus local snapshot drift: ' + name
        item['matches_existing_installed_snapshot'] = True
    else:
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(data)
        item['matches_existing_installed_snapshot'] = None
    item['provenance'] = 'PUBLIC_FIXED_COMMIT_GIT_BLOB_VERIFIED'
    files[name] = item
result = {'status': 'PASS', 'commit': spec['commit'], 'tree': spec['tree'], 'files': files,
          'new_local_files': sum(not x['local_snapshot_preexisted'] for x in files.values()),
          'remote_installed_tree_revalidated': False, 'native_executed': False,
          'started_utc': started, 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'files': len(files), 'new_local_files': result['new_local_files']}))
