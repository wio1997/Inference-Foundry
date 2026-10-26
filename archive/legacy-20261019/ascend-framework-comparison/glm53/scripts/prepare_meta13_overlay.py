#!/usr/bin/env python3
"""Extract existing B103 SDK assets and stage a private CPU Meta overlay on30."""
import hashlib,importlib.util,json,shlex,subprocess,tarfile
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0350-meta13-overlay'
DEST='/data/wio/inference-foundry/glm53/overlays/meta-sdk13-20261017'
IMAGE='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
CODE=r'''
import pathlib,tarfile,sys,json,hashlib,io,importlib.metadata as md
root=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages')
names=['memfabric_hybrid','memcache_hybrid']
for package in list(names):
 assert md.version(package)=='1.3.0'
 d=md.distribution(package);names.append(pathlib.Path(d._path).name)
 if (root/(package+'.libs')).exists():names.append(package+'.libs')
files=[]
for name in names:
 for path in sorted((root/name).rglob('*')):
  if '__pycache__' in path.parts or not path.is_file():continue
  assert not path.is_symlink(), 'SDK symlink needs review: '+str(path)
  files.append(path)
assert sum(p.stat().st_size for p in files)<180000000
manifest={'source_image':'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263','packages':names,'files':[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for path in files:tar.add(path,arcname=str(path.relative_to(root)),recursive=False)
 data=json.dumps(manifest,sort_keys=True).encode();info=tarfile.TarInfo('_glm53_overlay_manifest.json');info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
'''

def identity(n):
 pod,hostname=b.NODES[n]
 r=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert r['rc']==0 and r['stdout']=='172.16.12.'+str(n)
 return pod,hostname

def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 artifacts=RUN/'artifacts';artifacts.mkdir()
 pod,hostname=identity(34)
 args=['docker','run','--rm','--pull=never','--read-only','--network=none','--pids-limit=32','--memory=1g','--cpus=2','--label','project_id=glm53','--name','glm53-meta13-extract-20261017-34','--entrypoint','timeout',IMAGE,'90','python3','-c',CODE]
 hc='import socket,subprocess,sys\nassert socket.gethostname()=='+repr(hostname)+'\nr=subprocess.run('+repr(args)+',stdout=sys.stdout.buffer,stderr=sys.stderr.buffer,timeout=110)\nsys.exit(r.returncode)\n'
 cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 archive=artifacts/'meta-sdk13.tar.gz'
 with archive.open('xb') as out:
  r=subprocess.run(b.SSH+[cmd],input=hc.encode(),stdout=out,stderr=subprocess.PIPE,timeout=140)
 (RUN/'extract-stderr.txt').write_bytes(r.stderr)
 assert r.returncode==0, 'Extraction failed; preserve stderr'
 assert archive.stat().st_size<120000000
 sha=hashlib.sha256(archive.read_bytes()).hexdigest()
 # Independently validate every archived file against its internal manifest.
 with tarfile.open(archive) as tar:
  manifest=json.load(tar.extractfile('_glm53_overlay_manifest.json'))
  for item in manifest['files']:
   data=tar.extractfile(item['path']).read();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
 (RUN/'overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 pod,hostname=identity(30)
 upload=r'''
import sys,pathlib,hashlib,tarfile,io,json,socket
assert socket.gethostname()==sys.argv[1]
dest=pathlib.Path(sys.argv[2]);expected=sys.argv[3]
assert not dest.exists(), 'Do not overwrite any existing overlay'
data=sys.stdin.buffer.read(120000001);assert len(data)<=120000000 and hashlib.sha256(data).hexdigest()==expected
dest.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(data)) as tar:
 manifest=json.load(tar.extractfile('_glm53_overlay_manifest.json'))
 for item in manifest['files']:
  rel=pathlib.PurePosixPath(item['path']);assert not rel.is_absolute() and '..' not in rel.parts
  assert rel.parts[0] in manifest['packages']
  member=tar.getmember(item['path']);assert member.isfile()
  payload=tar.extractfile(member).read();assert hashlib.sha256(payload).hexdigest()==item['sha256']
  out=dest.joinpath(*rel.parts);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(payload);out.chmod(0o755 if member.mode&0o111 else 0o644)
 (dest/'_glm53_overlay_manifest.json').write_text(json.dumps(manifest,indent=2))
(dest/'archive.sha256').write_text(expected+'\n')
print(json.dumps({'project_id':'glm53','dest':str(dest),'files':len(manifest['files']),'archive_sha256':expected,'bytes':len(data)}))
'''
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',upload,hostname,DEST,sha])
 with archive.open('rb') as inp:r=subprocess.run(b.SSH+[command],stdin=inp,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
 (RUN/'stage-30.json').write_text(json.dumps({'rc':r.returncode,'stdout':r.stdout.decode(errors='replace'),'stderr':r.stderr.decode(errors='replace')},indent=2))
 assert r.returncode==0
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'archive_sha256':sha,'files':len(manifest['files']),'source_image':IMAGE,'dest_node':30,'dest':DEST,'installed_system_files_changed':False,'model_requests':0,'npu_compute_requests':0},indent=2))
 print('overlay staged',sha,flush=True)
if __name__=='__main__':main()
