"""Copy reviewed records only to the existing user-owned 169 main task directory."""
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import uuid

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER='wio-vllm-v41-repro169-stage-20261028'
files=['STATE.json','RESEARCH-HANDOFF.md','RESEARCH-INDEX.md','RESEARCH-CATALOG.json','ENVIRONMENT-SNAPSHOT-20261030.json','agents/bounds/hardware-source-and-shared-bandwidth-20261028.md','agents/fusion/INDEXER-HELPER-REVIEW.md','agents/fusion/SP-GATHER-QKV-PROJECTION-BOUNDARY.md']
files += [str(p.relative_to(ROOT)) for p in (ROOT/'publication-draft').iterdir() if p.is_file() and p.suffix in ('.md','.json','.py')]
files += ['diagnostic-20261029/'+n for n in ('sync_phase_records.py','launch_official_profile.py','official_profile_controller.py','official_profile_client.py','poll_official_profile.py','official-launch-latest.json','official-status-latest.json','official-profile-zcode.task.txt','official-profile-zcode-output.json','single_group_projection_diagnostic.py','single-group-projection-diagnostic.json')]
files += [str(p.relative_to(ROOT)) for p in (ROOT/'diagnostic-20261029/official-analysis').iterdir() if p.is_file() and p.suffix in ('.md','.json','.py','.txt','.stdout','.stderr')]
for task in ('fusion-coverage','layer0-diagnosis','theoretical-ceiling','ideal-scheduler-model'):
    files += ['task-loop-runs/'+task+'/'+n for n in ('index.json','status.md','resume-pack.json')]
catalog_path=ROOT/'RESEARCH-CATALOG.json'
if catalog_path.is_file():
    files += [entry['relative_path'] for entry in json.loads(catalog_path.read_text())['indexed_files']]
files=sorted(set(files))
manifest={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files}
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as tar:
    for n in files:
        data=(ROOT/n).read_bytes();info=tarfile.TarInfo(n);info.size=len(data);info.mode=0o600
        tar.addfile(info,io.BytesIO(data))
payload=base64.b64encode(buf.getvalue()).decode('ascii')
name='.record-sync-'+uuid.uuid4().hex+'.b64'
code=r'''import base64,hashlib,io,json,os,pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1]).resolve(strict=True)
name=sys.argv[2]
assert name.startswith('.record-sync-') and name.endswith('.b64') and '/' not in name
p=root/name
mode=sys.argv[3]
if mode=='begin':
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
elif mode=='append':
 assert p.is_file() and not p.is_symlink()
 with p.open('a',encoding='ascii') as f:f.write(sys.argv[4])
elif mode=='finish':
 expected=json.loads(sys.argv[4]);verified=[];items={}
 blob=base64.b64decode(p.read_text(),validate=True)
 with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
  for m in tar.getmembers():
   assert m.isfile() and m.name in expected and m.name not in items
   target=root/m.name
   assert target.resolve().is_relative_to(root)
   data=tar.extractfile(m).read()
   assert hashlib.sha256(data).hexdigest()==expected[m.name]
   items[m.name]=data
 assert set(items)==set(expected)
 for n,data in items.items():
  target=root/n;target.parent.mkdir(parents=True,exist_ok=True)
  assert target.resolve().is_relative_to(root)
  tmp=target.with_name('.'+target.name+'.record-sync.tmp');tmp.write_bytes(data);tmp.replace(target)
  assert hashlib.sha256(target.read_bytes()).hexdigest()==expected[n]
  verified.append(n)
 p.unlink()
 print(json.dumps({'sha256_verified':True,'verified_file_count':len(verified)}))
else:raise ValueError(mode)
'''
def call(mode,arg=''):
    argv=['docker','exec',CONTAINER,'python3','-c',code,REMOTE,name,mode,arg]
    r=subprocess.run([SSH,shlex.join(argv)],capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError(mode+' failed: '+r.stderr[-2000:])
    return r.stdout
call('begin')
chunks=0
for offset in range(0,len(payload),48000):
    call('append',payload[offset:offset+48000]);chunks+=1
receipt=json.loads(call('finish',json.dumps(manifest,separators=(',',':'))))
assert receipt['sha256_verified'] and receipt['verified_file_count']==len(files)
out={'status':'SYNCED_AND_SHA256_VERIFIED','container':CONTAINER,'remote_root':REMOTE,'files':manifest,'receipt':receipt,'chunks':chunks,'device_work':False,'service_changes':False}
(ROOT/'diagnostic-20261029/phase-sync-receipt.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':out['status'],'verified_file_count':len(files),'chunks':chunks}))
