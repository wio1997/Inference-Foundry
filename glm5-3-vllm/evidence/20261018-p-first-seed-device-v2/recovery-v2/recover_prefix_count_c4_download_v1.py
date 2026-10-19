"""Recover the already completed immutable collector; no inference rerun."""
from pathlib import Path
import datetime, gzip, hashlib, json, shlex, subprocess

r = Path(__file__).resolve().parent
dest = next((r/'prefix-count-c4-v1').glob('*/transport.json')).parent
t = json.loads((dest/'transport.json').read_text()); manifest = json.loads(t['stdout'])
assert t['rc'] == 0 and manifest['returncode'] == 0 and not manifest['timed_out']
remote = manifest['remote_root']; assert remote.endswith('/node33-prefix_count_c4_v1-3cd48a945cbd')
ssh = ['ssh', '-S', '/tmp/glm53-active-20261019.sock', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-p', '60011', 'root@61.241.77.34']
host = 'kubectl exec -i -n default sglang-tuning-access-33-20261013 -- nsenter -t 1 -m -- '
rows = []
for name in ['stdout.json', 'stderr.txt', 'context.json', 'result.json', 'script.py', 'artifacts_manifest.json', 'sdma-log-evidence.txt', 'metrics.json']:
    p = dest/name
    q = subprocess.run(ssh+[host+'gzip -c -- '+shlex.quote(remote+'/'+name)], capture_output=True, timeout=120)
    assert q.returncode == 0, q.stderr.decode(errors='replace')
    data = gzip.decompress(q.stdout); digest = hashlib.sha256(data).hexdigest()
    if name in manifest:
        assert len(data) == manifest[name]['bytes'] and digest == manifest[name]['sha256']
    if name == 'script.py': assert digest == manifest['script_sha256']
    if name == 'result.json': assert json.loads(data) == manifest
    if p.exists(): assert p.read_bytes() == data
    else: p.write_bytes(data)
    rows.append(dict(name=name, bytes=len(data), sha256=digest, compressed_bytes=len(q.stdout)))
    print(json.dumps(rows[-1]), flush=True)
receipt = dict(completed=True, observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), original_run_returncode=0, original_duration_s=manifest['duration_s'], downloads=rows, scope='Read-only compressed download of the same completed remote originals after stock30s cat timeout. All available original SHA/byte metadata verified. No request, rerun, source edit or device operation.')
(r/'prefix-count-c4-download-recovery-v1.json').write_text(json.dumps(receipt, indent=2)+'\n')
