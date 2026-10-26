#!/usr/bin/env python3
"""Prepare only local files/symlinks and pinned Docker images; never start containers/models."""
from pathlib import Path
import json,os,fcntl,hashlib,subprocess,time,urllib.request,urllib.error,shutil,tarfile
from urllib.parse import quote
R=Path(__file__).resolve().parent;C=json.load(open(R/'config.json'));BASE=Path('/data/wio');PROJECT=BASE/'ascend-framework-comparison';IMAGES=BASE/'images';MODELS=BASE/'models'

def save(phase,**kw):
 d={'phase':phase,'node':C['node'],'pid':os.getpid(),'epoch':time.time(),'model_requests':0,'models_started':False,**kw};p=R/'STATUS.json.tmp';p.write_text(json.dumps(d,indent=2)+'\n');p.replace(R/'STATUS.json')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def run(a,t=30):
 p=subprocess.run(a,capture_output=True,text=True,timeout=t)
 assert p.returncode==0,p.stderr[-1500:];return p.stdout

def manifest(key):
 deadline=time.monotonic()+86400
 while time.monotonic()<deadline:
  try:
   with urllib.request.urlopen(C['mirror']+'/manifest.json',timeout=20) as f:j=json.load(f)
   assert j['run_id']==C['run_id']
   if j.get(key,{}).get('ready'):return j[key]
  except (urllib.error.URLError,TimeoutError):pass
  save('WAITING_FOR_'+key.upper());time.sleep(60)
 raise RuntimeError('Asset source timeout, partial files preserved')
def download(url,target,meta):
 target=Path(target);partial=target.with_name(target.name+'.partial');expected=meta['bytes'];digest=meta['sha256'];target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  assert target.is_file() and not target.is_symlink() and target.stat().st_size==expected and sha(target)==digest,'Existing file identity mismatch';return
 for attempt in range(3):
  start=partial.stat().st_size if partial.exists() else 0
  assert start<=expected
  if start<expected:
   headers={'Range':'bytes='+str(start)+'-'} if start else {}
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
     if start:assert response.status==206 and response.headers['Content-Range'].startswith('bytes '+str(start)+'-')
     else:assert response.status==200
     with partial.open('ab' if start else 'wb') as out:
      while True:
       b=response.read(4*1024**2)
       if not b:break
       out.write(b)
      out.flush();os.fsync(out.fileno())
   except (urllib.error.URLError,TimeoutError,ConnectionError,OSError):
    if attempt==2:raise
    time.sleep(15);continue
  assert partial.stat().st_size==expected,'Incomplete source response'
  assert sha(partial)==digest,'Downloaded SHA256 mismatch; preserved partial'
  partial.rename(target);return
 raise RuntimeError('Copy incomplete')

def inspect(id):
 p=subprocess.run(['docker','image','inspect',id],capture_output=True,text=True)
 return json.loads(p.stdout)[0] if p.returncode==0 else None

def image(key):
 m=manifest(key);target=IMAGES/m['archive_name'];save('COPYING_'+key.upper(),archive=str(target),bytes=m['bytes'])
 download(C['mirror']+'/images/'+m['archive_name'],target,m)
 current=inspect(m['image_id'])
 if current is None:
  # Never overwrite a previously different tag via docker load.
  for tag in m['repo_tags']:
   old=inspect(tag);assert old is None or old['Id']==m['image_id'],'Existing tag conflicts with fixed image'
  docker_root=Path(run(['docker','info','--format','{{.DockerRootDir}}']).strip());assert shutil.disk_usage(docker_root).free>=35*1024**3,'Docker store space insufficient'
  save('IMPORTING_'+key.upper(),archive=str(target))
  with (R/(key+'-docker-load.log')).open('ab') as log:
   subprocess.run(['docker','load','-i',str(target)],stdout=log,stderr=log,check=True,timeout=3600)
 current=inspect(m['image_id']);assert current and current['Id']==m['image_id'] and current['Architecture']=='arm64' and current['Os']=='linux'
 return {'archive':str(target),'archive_sha256':m['sha256'],'image_id':current['Id'],'platform':'linux/arm64','imported':True}

lock=os.open(R/'.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 assert os.uname().nodename==C['hostname'] and C['node'] not in [38,42]
 for p in [PROJECT,IMAGES,MODELS]:
  p.mkdir(parents=True,exist_ok=True);assert p.resolve().is_relative_to(BASE)
 for d in ['config','source','logs','runs','assets']:(PROJECT/d).mkdir(exist_ok=True)
 fs_model=json.loads(run(['findmnt','-J','-T',str(MODELS),'-o','SOURCE,FSTYPE,TARGET']))['filesystems'][0];assert fs_model['fstype']=='ext4' and fs_model['source'].startswith('/dev/nvme')
 code=manifest('code');archive=R/code['name'];download(C['mirror']+'/code/'+code['name'],archive,code)
 snapshot=PROJECT/'source'/('snapshot-'+code['commit'][:8]);assert not snapshot.exists();snapshot.mkdir()
 with tarfile.open(archive,'r:') as tar:
  for member in tar:
   path=snapshot/member.name;assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts and (member.isdir() or member.isfile()),'Unsafe source archive entry'
   if member.isdir():path.mkdir(parents=True,exist_ok=True)
   else:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(member) as src,path.open('xb') as out:shutil.copyfileobj(src,out)
    path.chmod(member.mode & 0o777)
 (snapshot/'SOURCE_COMMIT.txt').write_text(code['commit']+'\n')
 source=Path(C['original_weight']);assert source.is_dir();cfg=json.load(open(source/'config.json'));assert cfg['model_type']=='glm_moe_dsa'
 assert sha(source/'config.json')==C['original_config_sha256']
 files={str(f.relative_to(source)):f.stat().st_size for f in source.rglob('*') if f.is_file() and len(f.relative_to(source).parts)<=3}
 assert len(files)==193 and sum(files.values())==764240829290
 assert hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()==C['original_manifest_sha256']
 fs=json.loads(run(['findmnt','-J','-T',str(source),'-o','SOURCE,FSTYPE,TARGET']))['filesystems'][0];assert fs['fstype']=='ext4' and fs['source'].startswith('/dev/nvme')
 canonical=MODELS/'GLM-5.3-w8a8'
 if os.path.lexists(canonical):assert canonical.resolve()==source.resolve()
 else:canonical.symlink_to(source,target_is_directory=True)
 original={'path':str(canonical),'resolved':str(canonical.resolve()),'local_nvme':True,'files':193,'bytes':764240829290,'verification':'File names/sizes and config SHA; no fresh payload rehash of original weights.'}
 save('CANONICAL_ORIGINAL_WEIGHT_READY',original_weight=original)
 flash=None;images={}
 if C['flash']:
  m=manifest('flash');target=MODELS/'GLM-5.3-Flash-w8a8';stage=MODELS/('.incoming-flash-'+C['run_id']);assert not target.is_symlink()
  if target.exists():stage=target
  else:stage.mkdir(exist_ok=True)
  assert shutil.disk_usage(MODELS).free>=m['bytes']+30*1024**3 if stage!=target else True
  for k,v in sorted(m['files'].items()):
   assert not Path(k).is_absolute() and '..' not in Path(k).parts
   save('COPYING_FLASH',file=k,expected_total_bytes=m['bytes'],destination=str(stage))
   download(C['mirror']+'/flash/'+quote(k,safe='/'),stage/k,v)
  actual={str(f.relative_to(stage)):f.stat().st_size for f in stage.rglob('*') if f.is_file()};assert actual=={k:v['bytes'] for k,v in m['files'].items()}
  if stage!=target:stage.rename(target)
  flash={'path':str(target),'files':len(m['files']),'bytes':m['bytes'],'all_file_payload_sha256_verified':True,'source_manifest_sha256':m['manifest_sha256']}
 if C['framework']=='vllm':images['vllm']=image('vllm')
 elif C['framework']=='sglang':images['sglang']=image('sglang')
 save('ROLE_RESOURCES_READY_NO_MODEL_STARTED',original_weight=original,flash_weight=flash,images=images,docker_root=run(['docker','info','--format','{{.DockerRootDir}}']).strip(),role=C['role'],canonical_root='/data/wio',reserve=C['framework']=='reserve')
except Exception as e:save('FAILED_PARTIALS_PRESERVED',error=str(e));raise
