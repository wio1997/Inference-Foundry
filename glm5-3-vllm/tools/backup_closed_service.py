#!/usr/bin/env python3
"""Back up only the two completed task service runs after verifying they are stopped."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

SERVICES = {
    'capacity': ('sglang-tuning-access-33-20261013', '/data/wio/glm5-3-vllm-execution/20261017-capacity/node33-tp16-w8a8c8-v1'),
    'meta': ('sglang-node-access-32', '/data/wio/glm5-3-vllm-execution/20261017-memcache/meta32-v1'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service', choices=SERVICES, required=True)
    p.add_argument('--control', required=True)
    p.add_argument('--backup-root', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    pod, root = SERVICES[a.service]
    ssh = ['ssh', '-S', a.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-p', '60011', 'root@61.241.77.34']
    host = 'kubectl exec -i -n default '+pod+' -- nsenter -t 1 -m -- '
    code = '''import hashlib,json,pathlib,subprocess
r=pathlib.Path(ROOT)
launch=json.loads((r/'launch.json').read_text())
info=json.loads(subprocess.check_output(['docker','inspect',launch['container_id']]))[0]
assert not info['State']['Running']
assert info['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
assert info['Name'].startswith('/glm53-')
p=r/'closed-service.tar.gz'
if not p.exists():
 names=sorted(x.name for x in r.iterdir() if x.name!=p.name and not x.is_symlink())
 subprocess.run(['tar','-czf',str(p),'-C',str(r)]+names,check=True)
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
print(json.dumps({'remote_archive':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size,'container_state':info['State']}))
'''.replace('ROOT', repr(root))
    report = json.loads(subprocess.check_output(ssh+[host+'python3 -'], input=code.encode(), timeout=120))
    a.backup_root.mkdir(parents=True, exist_ok=True)
    dest = a.backup_root/(a.service+'-closed-service.tar.gz')
    if not dest.exists():
        with dest.open('xb') as f:
            subprocess.run(ssh+[host+'cat '+shlex.quote(report['remote_archive'])], stdout=f, check=True, timeout=120)
    assert dest.stat().st_size == report['bytes']
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == report['sha256']
    report.update(local_archive=str(dest.resolve()), download_verified=True)
    a.receipt.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
