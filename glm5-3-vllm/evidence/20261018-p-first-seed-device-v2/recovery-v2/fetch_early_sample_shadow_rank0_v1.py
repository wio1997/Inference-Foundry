"""Fetch just completed analysis files first; full all16 backup is separate."""
from pathlib import Path, PurePosixPath
import hashlib, json, shlex, subprocess, tarfile

r = Path(__file__).resolve().parent
run = next((r/'early-sample-shadow-c4-export-v1').glob('*/result.json')).parent
result = json.loads((run/'result.json').read_text()); remote = result['remote_root']
manifest = json.loads((run/'artifacts_manifest.json').read_text())
selected = [z for z in manifest if '/rank0/' in z['path'] or z['path'].endswith(('/metadata.json', '/raw-manifest.json', '/profile-result.json', '/export-result.json'))]
assert len(selected) >= 7 and all(not PurePosixPath(z['path']).is_absolute() and '..' not in PurePosixPath(z['path']).parts and z['path'].startswith('artifacts/') for z in selected)
dest = Path('/Users/wio/work/glm53-execution-evidence-backup/20261019-early-sample-shadow-c4'); dest.mkdir(exist_ok=False)
archive = dest/'analysis-inputs.tar.gz'; partial = dest/'analysis-inputs.partial'
ssh = ['ssh', '-S', '/tmp/glm53-active-20261019.sock', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-p', '60011', 'root@61.241.77.34']
command = 'kubectl exec -i -n default sglang-tuning-access-33-20261013 -- nsenter -t 1 -m -- tar -czf - -C '+shlex.quote(remote)+' -- '+' '.join(shlex.quote(z['path']) for z in selected)
with partial.open('xb') as f: q = subprocess.run(ssh+[command], stdout=f, stderr=subprocess.PIPE, timeout=180)
assert q.returncode == 0, q.stderr.decode(errors='replace')
partial.rename(archive)
with tarfile.open(archive, 'r:gz') as tar:
    assert {z.name for z in tar.getmembers()} == {z['path'] for z in selected}
    for entry in selected:
        member = tar.getmember(entry['path']); assert member.isfile()
        data = tar.extractfile(member).read(); assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
        p = dest/entry['path']; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
receipt = dict(completed=True, local_root=str(dest), archive=str(archive), archive_bytes=archive.stat().st_size,
               archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=selected,
               scope='Read-only selected rank0 traces/CSV/matrix plus metadata and all16 raw SHA manifests from completed CPU export, checked against original collector artifact SHA. Full all16 raw archives remain a separate required backup; no inference rerun.')
(r/'early-sample-shadow-c4-rank0-backup-v1.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(completed=True, files=len(selected), bytes=receipt['archive_bytes'], sha256=receipt['archive_sha256'], local_root=str(dest))))
