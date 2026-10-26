#!/usr/bin/env python3
"""Import only the hash-verified task image archive on owned node44; no tag replacements."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R011-image-import';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,os,pathlib,subprocess,sys,tarfile
j=json.load(sys.stdin);root=pathlib.Path('/data/wio/sglang-mech36-20261018/L039-R009-transfer');image='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263';archive=root/'image-c5.tar'
assert json.loads((root.parent/'owner.json').read_text())['task']=='sglang-pd-spec-36'
progress=json.loads((root/'progress.json').read_text());file=progress['completed']['image-c5.tar'];assert file['sha256']==file['source_sha256']==j['image_sha256'] and archive.stat().st_size==file['bytes']
digest=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):digest.update(b)
assert digest.hexdigest()==j['image_sha256']
with tarfile.open(archive,'r:') as tar:
 f=tar.extractfile('manifest.json');manifest=json.load(f);assert len(manifest)==1 and not manifest[0].get('RepoTags'),'Do not modify any existing image tags'
 config=tar.extractfile(manifest[0]['Config']).read();assert hashlib.sha256(config).hexdigest()==image.split(':')[1]
def run(a,t=30):
 p=subprocess.run(a,capture_output=True,text=True,timeout=t);return p
docker_root=pathlib.Path(run(['docker','info','--format','{{.DockerRootDir}}']).stdout.strip());stat=os.statvfs(docker_root);assert stat.f_bavail*stat.f_frsize>2*file['bytes']+10*1024**3
p=run(['docker','image','inspect','--format','{{json .Id}}',image]);v={'image':image,'source_archive_sha256':j['image_sha256'],'archive_bytes':file['bytes'],'tags_replaced':False,'model_loaded':False,'device_compute_executed':False,'docker_root':str(docker_root),'available_before_bytes':stat.f_bavail*stat.f_frsize}
if p.returncode==0:assert json.loads(p.stdout)==image;v['import']='ALREADY_PRESENT_SAME_ID'
else:
 p=run(['docker','load','--input',str(archive)],240);v['load_exit_code']=p.returncode;v['load_stdout']=p.stdout;v['load_stderr']=p.stderr;assert p.returncode==0,p.stderr;v['import']='IMPORTED'
p=run(['docker','image','inspect','--format','{{json .}}',image]);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['Id']==image
v['verified_image']={k:x.get(k) for k in ['Id','Architecture','Size','RepoTags']};v['status']='PASS';print(json.dumps(v))
'''
JUMP=r'''
import json,subprocess,sys
j=json.load(sys.stdin);p=subprocess.run(['kubectl','get','pod','-n','default','sglang-mech36-prefill-44-20261019','-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['metadata']['uid']=='41367624-b523-4682-8be5-778f104ca373' and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['status']['phase']=='Running'
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-container-access-44-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(j),capture_output=True,text=True,timeout=330);sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
launch=json.loads((R/'runs/L039-R009-transfer-launch/remote-result.json').read_text());image_sha=launch['source']['manifest']['files']['image-c5.tar']['sha256'];assert image_sha
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST,'image_sha256':image_sha}),capture_output=True,text=True,timeout=360)
(O/'result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'image':launch['source']['manifest']['image'],'device_compute_executed':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['result.json','stderr.log']}};(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
