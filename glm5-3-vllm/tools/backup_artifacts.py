#!/usr/bin/env python3
"""Archive only this run's retained artifacts after device execution; download and verify."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--control', required=True)
    p.add_argument('--backup-root', type=Path, required=True)
    a = p.parse_args()
    result = json.loads((a.run/'result.json').read_text())
    root = result['remote_root']
    assert root.startswith('/data/wio/glm5-3-vllm-execution/20261017-inventory/node')
    node = int(a.run.name[4:6])
    assert node in (32, 33)
    pod = {32: 'sglang-node-access-32', 33: 'sglang-tuning-access-33-20261013'}[node]
    ssh = ['ssh', '-S', a.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-p', '60011', 'root@61.241.77.34']
    host = 'kubectl exec -i -n default '+pod+' -- nsenter -t 1 -m -- '
    archive = root + '/artifacts.tar.gz'
    code = '''import hashlib,json,pathlib,subprocess
r=pathlib.Path(ROOT);p=r/'artifacts.tar.gz'
if not p.exists(): subprocess.run(['tar','-czf',str(p),'-C',str(r),'artifacts'],check=True)
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(json.dumps({'sha256':h.hexdigest(),'bytes':p.stat().st_size}))
'''.replace('ROOT', repr(root))
    metadata = json.loads(subprocess.check_output(ssh+[host+'python3 -'], input=code.encode(), timeout=180))
    a.backup_root.mkdir(parents=True, exist_ok=True)
    dest = a.backup_root/(a.run.name+'.tar.gz')
    if not dest.exists():
        partial = dest.with_suffix('.partial')
        with partial.open('xb') as f:
            subprocess.run(ssh+[host+'cat '+shlex.quote(archive)], stdout=f, check=True, timeout=180)
        partial.rename(dest)
    h = hashlib.sha256()
    with dest.open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):
            h.update(b)
    assert h.hexdigest() == metadata['sha256'] and dest.stat().st_size == metadata['bytes']
    receipt = dict(metadata, remote_archive=archive, local_archive=str(dest.resolve()),
                   verified_client_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (a.run/'artifacts-backup.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
