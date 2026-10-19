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
    p.add_argument('--chunk-size-mib', type=int, choices=[0, 8], default=0)
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
r=pathlib.Path(ROOT);p=r/'artifacts.tar.gz';chunk_bytes=CHUNK_BYTES_TOKEN
if not p.exists():
 tmp=p.with_suffix('.partial');assert not tmp.exists()
 subprocess.run(['tar','-czf',str(tmp),'-C',str(r),'artifacts'],check=True);tmp.rename(p)
h=hashlib.sha256()
chunks=[]
with p.open('rb') as f:
 for b in iter(lambda:f.read(chunk_bytes or 1048576),b''):
  h.update(b)
  if chunk_bytes:chunks.append({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
result={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
if chunk_bytes:result['chunks']=chunks
print(json.dumps(result))
'''.replace('ROOT', repr(root)).replace('CHUNK_BYTES_TOKEN', str(a.chunk_size_mib * 1024**2))
    metadata = json.loads(subprocess.check_output(ssh+[host+'python3 -'], input=code.encode(), timeout=180))
    a.backup_root.mkdir(parents=True, exist_ok=True)
    dest = a.backup_root/(a.run.name+'.tar.gz')
    if not dest.exists():
        partial = dest.with_suffix('.partial')
        with partial.open('xb') as f:
            if a.chunk_size_mib:
                for index, chunk in enumerate(metadata['chunks']):
                    code = ('from pathlib import Path;import sys\nwith Path(' + repr(archive)
                            + ").open('rb') as f:f.seek(" + str(index * a.chunk_size_mib * 1024**2)
                            + ');sys.stdout.buffer.write(f.read(' + str(chunk['bytes']) + '))\n')
                    data = subprocess.check_output(ssh+[host+'python3 -'], input=code.encode(), timeout=60)
                    assert len(data) == chunk['bytes'] and hashlib.sha256(data).hexdigest() == chunk['sha256']
                    f.write(data)
            else:
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
    print(json.dumps({k:v for k,v in receipt.items() if k != 'chunks'}))


if __name__ == '__main__':
    main()
