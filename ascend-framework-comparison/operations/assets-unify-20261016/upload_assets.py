#!/usr/bin/env python3
"""Resumable local upload of one source snapshot and pinned B103; no service deployment."""
from pathlib import Path
import subprocess,shlex,json,fcntl,os,time,hashlib
A=Path(__file__).resolve().parent;R='/data/wio/ascend-framework-comparison/assets/unify-20261016';IMG='/data/wio/images'
SSH=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-o','BatchMode=yes','-o','ConnectTimeout=20','root@61.241.77.34']
B103=A.parent/'stable-migration-b103-20261016/sglang-a3-b103-arm64-verified.tar';SHA='9c6e36ec870dedf51d28e3ce7310c9204bf3a525beff0d92ff78fe4947722b92';SIZE=17106358784

def state(phase,**kw):
 p=A/'UPLOAD_PROGRESS.json.tmp';p.write_text(json.dumps({'phase':phase,'pid':os.getpid(),'epoch':time.time(),'models_started':False,**kw},indent=2)+'\n');p.replace(A/'UPLOAD_PROGRESS.json')
def remote(code,t=45):
 p=subprocess.run(SSH+[shlex.join(['python3','-c',code])],capture_output=True,text=True,timeout=t);assert p.returncode==0,p.stderr[-1200:];return p.stdout

def send(local,remote_partial,label):
 for attempt in range(2):
  state(label,local_archive=str(local),expected_bytes=local.stat().st_size,attempt=attempt+1,remote_partial=remote_partial)
  with (A/'rsync-upload.log').open('ab',buffering=0) as log:
   p=subprocess.run(['rsync','--partial','--inplace','--progress','--timeout=300','-e',shlex.join(SSH[:-1]),str(local),SSH[-1]+':'+remote_partial],stdout=log,stderr=log)
  if p.returncode==0:return
  if attempt==0:time.sleep(30)
 raise RuntimeError('Upload failed; existing partial preserved')

def finalize(partial,final,size,digest,marker=None):
 code='''from pathlib import Path
import hashlib,json,os
partial=Path(PART);final=Path(FINAL);size=SIZE;expected=DIGEST
p=final if final.exists() else partial
assert p.is_file() and not p.is_symlink() and p.stat().st_size==size
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert h.hexdigest()==expected,'Remote SHA256 mismatch; preserved partial'
if p==partial:partial.rename(final)
MARKER
print(json.dumps({'path':str(final),'bytes':size,'sha256':expected}))
'''.replace('PART',repr(partial)).replace('FINAL',repr(final)).replace('SIZE',str(size)).replace('DIGEST',repr(digest)).replace('MARKER',marker or '')
 return remote(code,t=1200)

lock=os.open(A/'upload.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 assert B103.is_file() and B103.stat().st_size==SIZE
 remote('from pathlib import Path\nPath('+repr(IMG+'/.incoming')+').mkdir(parents=True,exist_ok=True)')
 code=json.load(open(A/'CODE_ARTIFACT.json'));send(A/code['name'],R+'/'+code['name']+'.partial','UPLOADING_SOURCE_SNAPSHOT');finalize(R+'/'+code['name']+'.partial',R+'/'+code['name'],code['bytes'],code['sha256'])
 name='sglang-a3-b103-arm64-verified.tar';send(B103,IMG+'/.incoming/'+name+'.partial','UPLOADING_B103')
 state('VERIFYING_REMOTE_B103',expected_sha256=SHA)
 meta={'archive_name':name,'bytes':SIZE,'sha256':SHA,'image_id':'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263','repo_tags':json.load(open(A/'B103_TAR_IDENTITY.json'))['repo_tags']}
 marker='m=Path('+repr(IMG+'/B103_READY.json')+');tmp=m.with_name(m.name+".tmp");tmp.write_text(json.dumps('+repr(meta)+',indent=2)+"\\n");tmp.replace(m)'
 proof=finalize(IMG+'/.incoming/'+name+'.partial',IMG+'/'+name,SIZE,SHA,marker);state('B103_SOURCE_UPLOADED_AND_SHA_VERIFIED',proof=json.loads(proof),model_started=False)
except Exception as e:state('FAILED_PARTIALS_PRESERVED',error=str(e));raise
