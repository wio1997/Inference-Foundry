#!/usr/bin/env python3
"""Resume only the verified local B103 import; no transfers, containers or model requests."""
from pathlib import Path
import json,os,fcntl,hashlib,subprocess,time,tarfile,shutil
R=Path('/data/wio/ascend-framework-comparison/assets/unify-20261016');J=json.loads(__import__('sys').stdin.read());C=json.load(open(R/'config.json'));n=C['node'];Q=R/'import-repair-20261017';ID='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263';ALIAS='sglang-ascend:cann9.1.0-a3-B103-verified';SIZE=17106358784;SHA='9c6e36ec870dedf51d28e3ce7310c9204bf3a525beff0d92ff78fe4947722b92';ARCHIVE=Path('/data/wio/images/sglang-a3-b103-arm64-verified.tar')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(phase,**kw):
 v={'phase':phase,'node':n,'pid':os.getpid(),'epoch':time.time(),'models_started':False,'model_requests':0,**kw};t=Q/'PROGRESS.json.tmp';t.write_text(json.dumps(v,indent=2)+'\n');t.replace(Q/'PROGRESS.json');return v
def inspect(ref):
 p=subprocess.run(['docker','image','inspect',ref],capture_output=True,text=True,timeout=30)
 return json.loads(p.stdout)[0] if p.returncode==0 else None
def tags(value):
 assert value is None or isinstance(value,list),'Invalid RepoTags type'
 out=[] if value is None else value
 assert all(isinstance(x,str) and x for x in out)
 return out
assert n in [34,35,36,41,46,47] and C==J['nodes'][str(n)] and os.uname().nodename==C['hostname']
lock=os.open(R/'.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.load(open(R/'STATUS.json'));assert old['phase']=='FAILED_PARTIALS_PRESERVED' and old['error']=="'NoneType' object is not iterable"
p=Path('/proc/'+str(old['pid'])+'/cmdline');assert not p.exists() or str(R/'node_assets.py').encode() not in p.read_bytes()
assert not Q.exists(),'Existing import repair: do not duplicate';Q.mkdir();(Q/'PREVIOUS_STATUS.json').write_text(json.dumps(old,indent=2)+'\n')
try:
 save('VERIFYING_EXISTING_ARCHIVE_AND_WEIGHTS')
 assert ARCHIVE.is_file() and not ARCHIVE.is_symlink() and ARCHIVE.stat().st_size==SIZE and sha(ARCHIVE)==SHA
 with tarfile.open(ARCHIVE,'r:') as t:
  m=json.load(t.extractfile('manifest.json'));assert len(m)==1;x=m[0];conf=t.extractfile(x['Config']).read();assert 'sha256:'+hashlib.sha256(conf).hexdigest()==ID;cfg=json.loads(conf);assert cfg['architecture']=='arm64' and cfg['os']=='linux' and len(cfg['rootfs']['diff_ids'])==len(x['Layers'])==21;archive_tags=tags(x.get('RepoTags'))
 source=Path(C['original_weight']);canon=Path('/data/wio/models/GLM-5.3-w8a8');assert canon.resolve()==source.resolve();assert sha(source/'config.json')==C['original_config_sha256']
 filemap={str(f.relative_to(source)):f.stat().st_size for f in source.rglob('*') if f.is_file() and len(f.relative_to(source).parts)<=3};assert len(filemap)==193 and sum(filemap.values())==764240829290 and hashlib.sha256(json.dumps(filemap,sort_keys=True).encode()).hexdigest()==C['original_manifest_sha256']
 fs=json.loads(subprocess.check_output(['findmnt','-J','-T',str(canon),'-o','SOURCE,FSTYPE,TARGET'],text=True))['filesystems'][0];assert fs['fstype']=='ext4' and fs['source'].startswith('/dev/nvme')
 original={'path':str(canon),'resolved':str(canon.resolve()),'local_nvme':True,'files':193,'bytes':764240829290,'verification':'File names/sizes and config SHA; no fresh original-weight payload rehash.'};flash=None
 if C['flash']:
  fm=J['flash'];froot=Path('/data/wio/models/GLM-5.3-Flash-w8a8');actual={str(f.relative_to(froot)):f.stat().st_size for f in froot.rglob('*') if f.is_file()};assert froot.is_dir() and not froot.is_symlink() and actual=={k:v['bytes'] for k,v in fm['files'].items()};assert sha(froot/'config.json')==fm['files']['config.json']['sha256'];fs=json.loads(subprocess.check_output(['findmnt','-J','-T',str(froot),'-o','SOURCE,FSTYPE,TARGET'],text=True))['filesystems'][0];assert fs['fstype']=='ext4' and fs['source'].startswith('/dev/nvme');flash={'path':str(froot),'files':97,'bytes':fm['bytes'],'all_file_payload_sha256_verified':True,'source_manifest_sha256':fm['manifest_sha256'],'payload_sha_checked_by':'Original reviewed copy worker before final-directory rename and before failed image-tag guard; this repair rechecks maps/config/localdisk, not333GB payload.'}
 current=inspect(ID)
 for ref in archive_tags+[ALIAS]:
  existing=inspect(ref);assert existing is None or existing['Id']==ID,'Existing tag conflict; do not overwrite'
 docker_root=subprocess.check_output(['docker','info','--format','{{.DockerRootDir}}'],text=True).strip();assert docker_root=='/docker_data/data'
 if current is None:
  assert shutil.disk_usage(docker_root).free>=35*1024**3
  save('IMPORTING_VERIFIED_UNTAGGED_B103',archive=str(ARCHIVE),archive_sha256=SHA,repo_tags_normalized=archive_tags)
  with (Q/'docker-load.log').open('xb') as log:subprocess.run(['docker','load','-i',str(ARCHIVE)],stdout=log,stderr=log,check=True,timeout=1800)
 current=inspect(ID);assert current and current['Id']==ID and current['Architecture']=='arm64' and current['Os']=='linux'
 subprocess.run(['docker','image','tag',ID,ALIAS],check=True,capture_output=True,timeout=30);assert inspect(ALIAS)['Id']==ID
 image={'archive':str(ARCHIVE),'archive_sha256':SHA,'image_id':ID,'platform':'linux/arm64','imported':True,'local_alias':ALIAS,'archive_repo_tags':archive_tags}
 v=save('ROLE_RESOURCES_READY_NO_MODEL_STARTED',original_weight=original,flash_weight=flash,images={'sglang':image},docker_root=docker_root,role=C['role'],canonical_root='/data/wio',reserve=False)
 t=R/'STATUS.json.import-repair.tmp';t.write_text(json.dumps(v,indent=2)+'\n');t.replace(R/'STATUS.json');print(json.dumps(v))
except Exception as e:save('FAILED_IMPORT_REPAIR_PRESERVED',error=str(e));raise
