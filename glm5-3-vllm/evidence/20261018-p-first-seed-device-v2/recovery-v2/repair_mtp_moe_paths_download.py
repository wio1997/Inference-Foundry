"""Recover immutable collected JSON through a chunk-verified compressed archive."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

r=Path(__file__).resolve().parent
run=next((r/'mtp-moe-graph-paths-v2').glob('*/transport.json')).parent
meta=json.loads(json.loads((run/'transport.json').read_text())['stdout'])
assert meta['returncode']==0 and meta['remote_root'].endswith(run.name)
names=['stdout.json','stderr.txt','context.json','result.json','script.py','artifacts_manifest.json','sdma-log-evidence.txt','metrics.json']
ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34','kubectl exec -i -n default sglang-tuning-access-33-20261013 -- nsenter -t 1 -m -- python3 -']
code="""from pathlib import Path
import hashlib,json,tarfile
r=Path(REMOTE_COLLECTOR_ROOT);meta=json.loads((r/'result.json').read_text());names=SELECTED_FILES
for name in names:
 if name in meta:assert (r/name).stat().st_size==meta[name]['bytes'] and hashlib.sha256((r/name).read_bytes()).hexdigest()==meta[name]['sha256']
p=r/'collector-files-v1.tar.gz'
if not p.exists():
 with tarfile.open(p,'w:gz',compresslevel=3) as t:
  for name in names:t.add(r/name,arcname=name,recursive=False)
chunks=[];h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):chunks.append({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});h.update(b)
print(json.dumps({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'chunks':chunks}))
""".replace('REMOTE_COLLECTOR_ROOT',repr(meta['remote_root'])).replace('SELECTED_FILES',repr(names))
archive_meta=json.loads(subprocess.check_output(ssh,input=code.encode(),timeout=120))
backup=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/mtp-moe-paths-v2')
backup.mkdir(parents=True,exist_ok=True);dest=backup/(run.name+'.tar.gz')
assert not dest.exists()
with dest.open('xb') as out:
    for i,chunk in enumerate(archive_meta['chunks']):
        code="from pathlib import Path;import sys\nwith Path("+repr(archive_meta['path'])+").open('rb') as f:f.seek("+str(i*8*1024**2)+");sys.stdout.buffer.write(f.read("+str(chunk['bytes'])+"))\n"
        b=subprocess.check_output(ssh,input=code.encode(),timeout=40)
        assert len(b)==chunk['bytes'] and hashlib.sha256(b).hexdigest()==chunk['sha256']
        out.write(b)
assert dest.stat().st_size==archive_meta['bytes'] and hashlib.sha256(dest.read_bytes()).hexdigest()==archive_meta['sha256']
bad=[]
with tarfile.open(dest,'r:gz') as t:
    for member in t.getmembers():
        assert member.name in names and member.isfile()
        b=t.extractfile(member).read();p=run/member.name
        if member.name in meta:assert len(b)==meta[member.name]['bytes'] and hashlib.sha256(b).hexdigest()==meta[member.name]['sha256']
        if p.exists() and p.read_bytes()!=b:
            old=backup/(p.name+'.unverified-truncated');assert not old.exists();p.rename(old)
            bad.append({'path':str(old),'bytes':old.stat().st_size,'sha256':hashlib.sha256(old.read_bytes()).hexdigest()})
        p.write_bytes(b)
assert hashlib.sha256((run/'script.py').read_bytes()).hexdigest()==meta['script_sha256']
receipt={'remote_archive':archive_meta['path'],'local_archive':str(dest),'bytes':archive_meta['bytes'],'sha256':archive_meta['sha256'],'chunks_verified':len(archive_meta['chunks']),'original_unverified':bad,'all_collector_files_verified':True,'hardware_requests_repeated':0,'scope':'Collector cat returned mismatched bytes; preserve truncated copy, recover same immutable result through compressed archive and per-chunk plus whole/file SHA. No GPU rerun.'}
(r/'mtp-moe-graph-paths-download-repair-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
