#!/usr/bin/env python3
"""Back up this task's completed model profiles and verify the received archive."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', type=int, choices=(32, 33), required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--evidence-run', type=Path, required=True)
    args = parser.parse_args()
    revision = 7 if args.node == 32 else 6
    root = f'/data/wio/glm5-3-vllm-execution/20261017-memcache/pd-node{args.node}-fixed-v{revision}'
    pod = {32: 'sglang-node-access-32', 33: 'sglang-tuning-access-33-20261013'}[args.node]
    ssh = ['ssh', '-S', args.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-p', '60011', 'root@61.241.77.34']
    host = 'kubectl exec -i -n default '+pod+' -- nsenter -t 1 -m -- '
    code = '''import hashlib,json,pathlib,shutil,tarfile
r=pathlib.Path(ROOT);archive=r/'pd-functional-profile-v1.tar.gz';receipt=archive.with_suffix('.receipt.json')
assert (r/'pd-functional-evidence-v2/analysis.json').is_file()
if not receipt.exists():
 assert not archive.exists(),'Unknown previous archive; inspect instead of overwriting'
 assert shutil.disk_usage(r).free>8*1024**3
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for name in ('profile','profile-raw-preserved','pd-functional-evidence-v1','pd-functional-evidence-v2','offline-analysis-v1','offline-analysis-v2','plan.json','launch.json','context.json'):
   tar.add(r/name,arcname=name)
  if r.name.startswith('pd-node32-'):tar.add(r.parent/'pd-functional-v1',arcname='pd-functional-v1')
 h=hashlib.sha256()
 with archive.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 receipt.write_text(json.dumps({'remote_archive':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest()})+'\\n')
print(receipt.read_text())
'''.replace('ROOT', repr(root))
    remote = json.loads(subprocess.check_output(ssh+[host+'python3 -'], input=code.encode(), timeout=600))
    args.backup_root.mkdir(parents=True, exist_ok=True)
    dest = args.backup_root/f'node{args.node}-full-model-pd-profile-v1.tar.gz'
    if not dest.exists():
        temporary = dest.with_suffix('.partial')
        with temporary.open('xb') as stream:
            subprocess.run(ssh+[host+'cat '+shlex.quote(remote['remote_archive'])], stdout=stream,
                           check=True, timeout=600)
        temporary.rename(dest)
    h = hashlib.sha256()
    with dest.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    assert dest.stat().st_size == remote['bytes'] and h.hexdigest() == remote['sha256']
    receipt = dict(remote, local_archive=str(dest.resolve()),
                   verified_client_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.evidence_run/'profile-backup.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
