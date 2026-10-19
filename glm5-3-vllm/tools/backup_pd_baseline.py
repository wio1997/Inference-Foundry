#!/usr/bin/env python3
"""Archive completed bounded PD tests and stopped samplers, then verify on Mac."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', type=int, choices=(32, 33), required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--base', choices=('pd-baseline','official-pd'), default='pd-baseline')
    parser.add_argument('--include-dir', action='append', default=[],
                        help='Completed category/run-id to retain in an incremental snapshot')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-z0-9-]+', args.snapshot)
    assert all(re.fullmatch(r'(tests|consumer-audits|resource-samples)/[a-zA-Z0-9-]+', value)
               for value in args.include_dir)
    pod = {32: 'sglang-node-access-32', 33: 'sglang-tuning-access-33-20261013'}[args.node]
    ssh = ['ssh', '-S', args.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-p', '60011', 'root@61.241.77.34']
    host = 'kubectl exec -i -n default '+pod+' -- nsenter -t 1 -m -- '
    code = '''import hashlib,json,pathlib,tarfile
r=pathlib.Path(BASE_ROOT)
included=INCLUDED
selected=[]
for category in ['tests','consumer-audits','resource-samples']:
 parent=r/category
 if not parent.exists(): continue
 for child in sorted(parent.iterdir()):
  if not child.is_dir() or child.is_symlink(): continue
  if included and str(child.relative_to(r)) not in included: continue
  assert (child/'result.json').is_file(), 'Run not completed: '+str(child)
  json.loads((child/'result.json').read_text())
  selected.append(child)
assert selected
assert not included or set(included)=={str(d.relative_to(r)) for d in selected}
files=[f for d in selected for f in d.rglob('*') if f.is_file()]
assert not any(f.is_symlink() for f in files)
assert sum(f.stat().st_size for f in files)<67108864, 'Unexpected archive size; inspect scope'
p=r/ARCHIVE_NAME
assert not p.exists(), 'Use existing receipt; do not overwrite an archive'
with tarfile.open(p,'w:gz') as archive:
 for d in selected: archive.add(d,arcname=str(d.relative_to(r)),recursive=True)
h=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps({'remote_archive':str(p),'bytes':p.stat().st_size,'sha256':h,
 'completed_directories':[str(d.relative_to(r)) for d in selected], 'source_files':len(files)}))
'''.replace('ARCHIVE_NAME', repr('completed-tests-'+args.snapshot+'.tar.gz')).replace('INCLUDED', repr(args.include_dir)).replace('BASE_ROOT',repr('/data/wio/glm5-3-vllm-execution/20261017-'+args.base))
    report = json.loads(subprocess.check_output(ssh+[host+'python3 -'], input=code.encode(), timeout=120))
    args.backup_root.mkdir(parents=True, exist_ok=True)
    target = args.backup_root/f'node{args.node}-completed-pd-tests-{args.snapshot}.tar.gz'
    with target.open('xb') as stream:
        subprocess.run(ssh+[host+'cat '+shlex.quote(report['remote_archive'])], stdout=stream, check=True, timeout=180)
    assert target.stat().st_size == report['bytes']
    assert hashlib.sha256(target.read_bytes()).hexdigest() == report['sha256']
    report.update(local_archive=str(target.resolve()), download_verified=True)
    args.receipt.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
