#!/usr/bin/env python3
"""Back up closed request/profile artifacts of the running official services."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--node',type=int,choices=(32,33),required=True)
    parser.add_argument('--control',required=True);parser.add_argument('--backup-root',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args()
    ssh=['ssh','-S',args.control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
    pod={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}[args.node]
    host='kubectl exec -i -n default '+pod+' -- nsenter -t 1 -m -- '
    code='''import hashlib,json,pathlib,tarfile,time
node=NODE
base=pathlib.Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
role='prefill' if node==32 else 'decode';model=base/f'{role}-node{node}-v1'
selected=[]
for name in ['profile-raw-preserved','plan.json','launch.json','context.json','setup.json','manifest.json','payload.py','start_capacity.py']:
 p=model/name
 if p.exists():selected.append(p)
for name in ['tests','proxy-v1']:
 p=base/name
 if p.exists():selected.append(p)
accepted=f'node{node}-profile-exported-v'+('1' if node==32 else '2')
for child in sorted((base/'consumer-audits').iterdir()):
 if not child.is_dir():continue
 for p in child.iterdir():
  if p.is_file() or (p.name=='parsed' and child.name==accepted):selected.append(p)
assert (base/'consumer-audits'/f'node{node}-profile-export-v1/result.json').is_file()
files=[p for x in selected for p in (x.rglob('*') if x.is_dir() else [x]) if p.is_file()]
assert not any(p.is_symlink() for p in files)
assert sum(p.stat().st_size for p in files)<8*1024**3
archive=base/'official-pd-evidence-v1.tar.gz';receipt=base/'official-pd-evidence-v1.receipt.json'
if not receipt.exists():
 assert not archive.exists(),'Unknown prior archive; do not overwrite'
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for p in selected:tar.add(p,arcname=str(p.relative_to(base)))
 h=hashlib.sha256()
 with archive.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 receipt.write_text(json.dumps({'remote_archive':str(archive),'bytes':archive.stat().st_size,'sha256':h.hexdigest(),'files':len(files),'raw_selected_bytes':sum(p.stat().st_size for p in files),'excluded_live_ascend_logs':True})+'\\n')
print(receipt.read_text())
'''.replace('NODE',str(args.node))
    report=json.loads(subprocess.check_output(ssh+[host+'python3 -'],input=code.encode(),timeout=360))
    print(json.dumps({'node':args.node,'remote_archive_ready':True,'bytes':report['bytes'],'sha256':report['sha256']}),flush=True)
    args.backup_root.mkdir(parents=True,exist_ok=True)
    target=args.backup_root/f'node{args.node}-official-pd-evidence-v1.tar.gz'
    if target.exists():
        old_hash=hashlib.sha256(target.read_bytes()).hexdigest()
        if target.stat().st_size!=report['bytes'] or old_hash!=report['sha256']:
            rejected=target.with_suffix('.truncated-'+old_hash[:12])
            assert not rejected.exists();target.rename(rejected)
            args.receipt.with_suffix('.transport-failure.json').write_text(json.dumps({
                'local_rejected_file':str(rejected),'bytes':rejected.stat().st_size,'sha256':old_hash,
                'expected_bytes':report['bytes'],'expected_sha256':report['sha256'],
                'reason':'kubectl exec stdout ended early despite rc0; retain failed download and recover the existing immutable archive only.'},indent=2)+'\n')
    if not target.exists():
        partial=target.with_suffix('.partial')
        writer="import pathlib,sys,time\nwith pathlib.Path("+repr(report['remote_archive'])+").open('rb') as stream:\n for block in iter(lambda:stream.read(262144),b''):sys.stdout.buffer.write(block)\nsys.stdout.buffer.flush()\ntime.sleep(8)\n"
        with partial.open('xb') as stream:
            subprocess.run(ssh+[host+'python3 -'],input=writer.encode(),stdout=stream,check=True,timeout=360)
        partial.rename(target)
    h=hashlib.sha256()
    with target.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    assert target.stat().st_size==report['bytes'] and h.hexdigest()==report['sha256']
    report.update(local_archive=str(target.resolve()),download_verified=True)
    args.receipt.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
