#!/usr/bin/env python3
"""One finite private LAN file-distribution task; no inference, containers, SSH configuration or NFS changes."""
from pathlib import Path
import os,json,hashlib,fcntl,threading,time,subprocess,shlex,concurrent.futures,tarfile,re
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import unquote
R=Path(__file__).resolve().parent;P=json.load(open(R/'PLAN.json'));DATA=Path('/data/wio');PORT=28423;RUN=P['run_id'];M={'run_id':RUN,'code':{'ready':False},'flash':{'ready':False},'vllm':{'ready':False},'sglang':{'ready':False}};MU=threading.Lock();ROOT=DATA/'ascend-framework-comparison';IMAGES=DATA/'images';FLASH=Path(P['flash_source']['path']);FILEMAP={'/flash/'+k:FLASH/k for k in P['flash_source']['files']};FILEMAP['/code/'+P['code_artifact']['name']]=R/P['code_artifact']['name']

def atomic(p,d):
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def status(phase,**kw):atomic(R/'STATUS.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time(),'models_started':False,**kw})
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def publish(k,value):
 with MU:M[k]=value;atomic(R/'MANIFEST.json',M)

class Handler(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):self.serve(True)
 def do_HEAD(self):self.serve(False)
 def serve(self,body):
  if self.client_address[0] not in {'172.16.12.'+str(n) for n in P['nodes']}|{'172.16.12.1'}:self.send_error(403);return
  name=unquote(self.path)
  if name=='/manifest.json':path=R/'MANIFEST.json'
  else:path=FILEMAP.get(name)
  if path is None or not path.is_file():self.send_error(404);return
  size=path.stat().st_size;start=0;end=size-1;rg=self.headers.get('Range')
  if rg:
   m=re.fullmatch(r'bytes=(\d+)-(\d*)',rg)
   if not m:self.send_error(416);return
   start=int(m[1]);end=min(int(m[2]) if m[2] else end,end)
   if start>=size or start>end:self.send_error(416);return
  self.send_response(206 if rg else 200);self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
  if rg:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
  self.end_headers()
  if body:
   try:
    with path.open('rb') as f:
     f.seek(start);left=end-start+1
     while left:
      b=f.read(min(left,4*1024**2))
      if not b:break
      self.wfile.write(b);left-=len(b)
   except (BrokenPipeError,ConnectionResetError):pass

NODE_BOOT='''import sys,json,os,subprocess,hashlib
from pathlib import Path
j=json.load(sys.stdin);root=Path(j['task_root']);assert os.uname().nodename==j['config']['hostname'];assert not root.exists(),'Task already exists; do not duplicate'
root.mkdir(parents=True);(root/'node_assets.py').write_text(j['source']);assert hashlib.sha256((root/'node_assets.py').read_bytes()).hexdigest()==j['source_sha256'];(root/'config.json').write_text(json.dumps(j['config'],indent=2)+'\\n')
with (root/'worker.log').open('wb') as log:p=subprocess.Popen(['python3','-u',str(root/'node_assets.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
(root/'LAUNCH.json').write_text(json.dumps({'pid':p.pid,'node':j['config']['node']})+'\\n');print(json.dumps({'pid':p.pid,'node':j['config']['node']}))
'''
def start(n):
 cfg=P['nodes'][n];args=['kubectl','exec','-i','-n','default',cfg['pod'],'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',NODE_BOOT]
 payload={'task_root':str(R),'config':cfg,'source':(R/'node_assets.py').read_text(),'source_sha256':P['node_worker_sha256']}
 p=subprocess.run(args,input=json.dumps(payload),capture_output=True,text=True,timeout=45);assert p.returncode==0,p.stderr[-1500:];return json.loads(p.stdout)
def flash_hash():
 checkpoint=R/'FLASH_SHA256.json';m=json.load(open(checkpoint));assert m['bytes']==332970562139 and set(m['files'])==set(P['flash_source']['files'])
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==P['repair_flash_checkpoint_sha256']
 for name,item in m['files'].items():
  p=FLASH/name;assert p.stat().st_size==item['bytes'] and p.stat().st_mtime_ns<=checkpoint.stat().st_mtime_ns
 assert hashlib.sha256(json.dumps(m['files'],sort_keys=True).encode()).hexdigest()==m['manifest_sha256']
 publish('flash',m);return
 hashes={};total=0
 for name,size in sorted(P['flash_source']['files'].items()):
  path=FLASH/name;before=path.stat();assert before.st_size==size and not path.is_symlink();value=digest(path);after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns);assert name not in P['flash_source']['expected_small_sha256'] or value==P['flash_source']['expected_small_sha256'][name],'Flash metadata changed after review';hashes[name]={'bytes':size,'sha256':value};total+=size
  atomic(R/'FLASH_HASH_PROGRESS.json',{'files':len(hashes),'bytes':total,'epoch':time.time()})
 body=json.dumps(hashes,sort_keys=True).encode();m={'ready':True,'files':hashes,'bytes':total,'manifest_sha256':hashlib.sha256(body).hexdigest(),'source':str(FLASH)};atomic(R/'FLASH_SHA256.json',m);publish('flash',m)
def vllm_hash():
 name='vllm-ascend-v0.30.0rc1-a3-arm64.tar';path=IMAGES/name;partial=IMAGES/(name+'.reexport.partial');ref='quay.io/ascend/vllm-ascend:v0.30.0rc1-a3-arm64'
 observed=json.loads(subprocess.check_output(['docker','image','inspect',ref],text=True))[0];assert observed['Id']==P['vllm_image_id'] and observed['Architecture']=='arm64'
 if not path.exists():
  assert not partial.exists(),'Preserve unknown earlier export partial'
  atomic(R/'VLLM_EXPORT_PROGRESS.json',{'phase':'EXPORTING_PINNED_EXISTING_DOCKER_IMAGE','image_id':P['vllm_image_id'],'target':str(path),'epoch':time.time()})
  with (R/'vllm-docker-save.log').open('ab') as log:subprocess.run(['docker','save','-o',str(partial),ref],stdout=log,stderr=log,check=True,timeout=3600)
  work=partial
 else:
  assert not path.is_symlink();work=path
 with tarfile.open(work,'r:*') as tar:
  manifest=json.load(tar.extractfile('manifest.json'));assert len(manifest)==1;config=tar.extractfile(manifest[0]['Config']).read();assert 'sha256:'+hashlib.sha256(config).hexdigest()==P['vllm_image_id']
  for layer in manifest[0]['Layers']:
   assert layer.startswith('blobs/sha256/');h=hashlib.sha256()
   with tar.extractfile(layer) as f:
    for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
   assert h.hexdigest()==layer.rsplit('/',1)[1], 'Reexport layer SHA mismatch: '+layer
 if work==partial:partial.rename(path)
 m={'ready':True,'archive_name':name,'bytes':path.stat().st_size,'sha256':digest(path),'image_id':P['vllm_image_id'],'repo_tags':manifest[0].get('RepoTags') or [],'origin':'Fresh docker save of pinned existing source image; old archive preserved after failed layer check.'};FILEMAP['/images/'+name]=path;publish('vllm',m);atomic(R/'VLLM_ARTIFACT.json',m)
def check_node(n):
 cfg=P['nodes'][n];code='from pathlib import Path;print(Path('+repr(str(R/'STATUS.json'))+').read_text())'
 p=subprocess.run(['kubectl','exec','-n','default',cfg['pod'],'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',code],capture_output=True,text=True,timeout=20)
 if p.returncode:return {'phase':'STATUS_UNAVAILABLE','error':p.stderr[-500:]}
 return json.loads(p.stdout)

lock=os.open(R/'.source.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert os.uname().nodename=='A3-5U-Atlas900RCKA3-1'
try:
 for d in [IMAGES,ROOT/'config',ROOT/'source',ROOT/'logs',ROOT/'runs']:d.mkdir(parents=True,exist_ok=True)
 assert (R/P['code_artifact']['name']).is_file();assert digest(R/P['code_artifact']['name'])==P['code_artifact']['sha256'];M['code']={'ready':True,**P['code_artifact']};atomic(R/'MANIFEST.json',M);server=ThreadingHTTPServer(('172.16.12.1',PORT),Handler);server.daemon_threads=True;thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 launched=json.load(open(R/'NODE_LAUNCHES.json'))
 assert {x['node'] for x in launched}=={int(n) for n in P['nodes']}
 atomic(R/'NODE_RESUME_EXISTING.json',{'existing_node_workers_preserved':launched,'epoch':time.time()});status('SOURCE_HASH_AND_NODE_STAGING_ACTIVE',node_launches=launched)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as prep:
  futures=[prep.submit(flash_hash),prep.submit(vllm_hash)];deadline=time.monotonic()+86400
  while time.monotonic()<deadline:
   for f in futures:
    if f.done():f.result()
   codefile=R/P['code_artifact']['name']
   if codefile.exists() and not M['code']['ready']:
    assert codefile.stat().st_size==P['code_artifact']['bytes'] and digest(codefile)==P['code_artifact']['sha256'];publish('code',{'ready':True,**P['code_artifact']})
   ready=IMAGES/'B103_READY.json'
   if ready.exists() and not M['sglang']['ready']:
    b=json.load(open(ready));assert b['image_id']==P['sglang_image_id'] and b['sha256']==P['sglang_archive_sha256'];path=IMAGES/b['archive_name'];assert path.is_file() and path.stat().st_size==b['bytes'];FILEMAP['/images/'+b['archive_name']]=path;publish('sglang',{'ready':True,**b})
   rows={n:check_node(n) for n in P['nodes']};atomic(R/'NODE_STATES.json',rows);ph=[r['phase'] for r in rows.values()]
   if all(p in ['ROLE_RESOURCES_READY_NO_MODEL_STARTED','FAILED_PARTIALS_PRESERVED'] for p in ph):
    state='ALL_ROLE_RESOURCES_READY_NO_MODEL_STARTED' if all(p=='ROLE_RESOURCES_READY_NO_MODEL_STARTED' for p in ph) else 'PARTIAL_FAILURE_PRESERVED'
    status(state,nodes=rows);atomic(R/'RESULT.json',{'phase':state,'nodes':rows,'models_started':False,'epoch':time.time()});break
   status('RESOURCE_DISTRIBUTION_ACTIVE',nodes=rows,source_ready={k:M[k]['ready'] for k in ['flash','vllm','sglang']});time.sleep(60)
  else:raise RuntimeError('Resource pipeline24h deadline')
 server.shutdown();server.server_close()
except Exception as e:status('FAILED_SOURCE_PRESERVE_NODE_STATES',error=str(e));raise
