#!/usr/bin/env python3
"""Stage the already-verified B103 SDK bundle in private directories only."""
import hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0400-stage-sdk13-vllm'
ARCHIVE=b.ROOT/'runs/20261017T0350-meta13-overlay/artifacts/meta-sdk13.tar.gz'
SHA='f858a88e60e33f2c954bd9920b56b96193e2be53749fa16e83316d461cb731b3'
DEST='/data/wio/inference-foundry/glm53/overlays/sdk13-20261017'
HOST=r'''
import sys,pathlib,hashlib,tarfile,io,json,socket,subprocess
assert socket.gethostname()==sys.argv[1]
image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
r=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15);assert r.returncode==0 and r.stdout.strip()==image
dest=pathlib.Path(sys.argv[2]);expected=sys.argv[3]
assert not dest.exists(), 'No overwrite/replay'
data=sys.stdin.buffer.read(120000001);assert len(data)<=120000000 and hashlib.sha256(data).hexdigest()==expected
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
 manifest=json.load(tar.extractfile('_glm53_overlay_manifest.json'))
 assert manifest['source_image']=='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
 for item in manifest['files']:
  rel=pathlib.PurePosixPath(item['path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in manifest['packages']
  member=tar.getmember(item['path']);assert member.isfile()
  payload=tar.extractfile(member).read();assert len(payload)==item['bytes'] and hashlib.sha256(payload).hexdigest()==item['sha256']
 dest.mkdir(parents=True)
 for item in manifest['files']:
  member=tar.getmember(item['path']);out=dest/item['path'];out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(tar.extractfile(member).read());out.chmod(0o755 if member.mode&0o111 else 0o644)
 (dest/'_glm53_overlay_manifest.json').write_text(json.dumps(manifest,indent=2))
(dest/'archive.sha256').write_text(expected+'\n')
print(json.dumps({'project_id':'glm53','host':socket.gethostname(),'dest':str(dest),'files':len(manifest['files']),'archive_sha256':expected,'bytes':len(data),'system_sdk_changed':False}))
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==SHA
 results={}
 for n in (30,31,43,45):
  pod,hostname=b.NODES[n]
  m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.'+str(n)
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',HOST,hostname,DEST,SHA])
  with ARCHIVE.open('rb') as f:r=subprocess.run(b.SSH+[command],stdin=f,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
  record={'rc':r.returncode,'stdout':r.stdout.decode(errors='replace'),'stderr':r.stderr.decode(errors='replace')};(RUN/('node-%s.json'%n)).write_text(json.dumps(record,indent=2));results[n]=r.returncode;print(n,r.returncode,flush=True)
  assert r.returncode==0, 'Stage failure; stop without replay'
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'archive_sha256':SHA,'dest':DEST,'nodes':results,'system_sdk_changed':False,'services_started':0,'model_requests':0},indent=2))
if __name__=='__main__':main()
