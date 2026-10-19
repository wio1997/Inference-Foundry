#!/usr/bin/env python3
"""Obtain blob-verified original gateway and an isolated official Rust toolchain."""
import datetime,hashlib,json,os,pathlib,platform,subprocess,tarfile,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L033-R005-setup-v3';O.mkdir(parents=True,exist_ok=False)
D=R/'.deps/L033-gateway-v3';D.mkdir(parents=True,exist_ok=False)
commit='ce0a8fea5f4245839d174b2b8c55692be1810a7f';tree=json.loads((R/'notes/L030-public-tree.json').read_text());assert not tree['truncated']
expected={x['path']:x for x in tree['tree'] if x['type']=='blob' and x['path'].startswith('sgl-model-gateway/')}
result={'status':'RUNNING','commit':commit,'original_source_mutated':False,'native_executed':False,'steps':[]}
def save(): (O/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def run(args,name,env=None,timeout=600):
 with (O/(name+'.stdout.log')).open('xb') as so,(O/(name+'.stderr.log')).open('xb') as se:
  c=subprocess.run(args,stdout=so,stderr=se,env=env,timeout=timeout)
 result['steps'].append({'step':name,'exit_code':c.returncode});save();assert c.returncode==0,name
save()
try:
 import base64,concurrent.futures,time
 def acquire(item):
  name,info=item;local=R/'sources/installed'/name;data=None;last=None
  if local.is_file():
   raw=local.read_bytes()
   if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==info['sha']:data=raw
  if data is None:
   for attempt in range(3):
    try:
     c=subprocess.run(['gh','api','repos/sgl-project/sglang/git/blobs/'+info['sha']],capture_output=True,text=True,timeout=35)
     assert c.returncode==0,c.stderr
     j=json.loads(c.stdout);assert j['sha']==info['sha'];data=base64.b64decode(j['content']);break
    except Exception as e:last=str(e);time.sleep(1+attempt)
  if data is None:return name,None,last
  sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest();assert sha==info['sha'],name
  p=D/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  return name,{'git_blob_sha1':sha,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)},None
 files={};errors={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  for name,item,error in pool.map(acquire,expected.items()):
   if item:files[name]=item
   else:errors[name]=error
 (O/'source-index.json').write_text(json.dumps({'commit':commit,'tree_sha':tree['sha'],'acquisition':'GitHub Git blobs API + verified prior source snapshot; max8 concurrent, <=3 attempts/resource','files':files,'errors':errors},indent=2)+'\n')
 assert set(files)==set(expected),(len(files),len(expected),list(errors))
 # Preserve narrow canonical snapshot used by the controller review.
 for name in ['sgl-model-gateway/src/policies/tree.rs','sgl-model-gateway/src/policies/utils.rs']:
  q=R/'sources/installed'/name;data=(D/name).read_bytes()
  if q.exists():assert q.read_bytes()==data
  else:q.write_bytes(data)
 result['source_files']=len(files);save()
 assert platform.machine()=='arm64'
 url='https://static.rust-lang.org/rustup/archive/1.28.2/aarch64-apple-darwin/rustup-init';init=D/'rustup-init';digest=D/'rustup-init.sha256'
 for u,p,n in [(url,init,'rustup-download'),(url+'.sha256',digest,'rustup-sha-download')]:run(['curl','--fail','--location','--retry','3','--connect-timeout','25','--max-time','180',u,'--output',str(p)],n,timeout=220)
 h=hashlib.sha256(init.read_bytes()).hexdigest();assert h==digest.read_text().split()[0];result['rustup']={'url':url,'version':'1.28.2','sha256':h};init.chmod(0o755)
 env=os.environ.copy();env.update({'CARGO_HOME':str(D/'cargo-home'),'RUSTUP_HOME':str(D/'rustup-home'),'RUSTUP_TOOLCHAIN':'1.90.0'})
 run([str(init),'-y','--no-modify-path','--profile','minimal','--default-toolchain','1.90.0','--component','clippy'],'rustup-install',env,timeout=600)
 versions={}
 for n in ['rustc','cargo']:
  p=D/'cargo-home/bin'/n;c=subprocess.run([str(p),'-vV'],env=env,capture_output=True,text=True,timeout=30);assert c.returncode==0;versions[n]=c.stdout
 result.update({'status':'PASS','toolchain':'1.90.0','toolchain_reason':'repository declares channel1.90; resolved exact1.90.0, minimal plusclippy','versions':versions,'cargo_home':str(D/'cargo-home'),'rustup_home':str(D/'rustup-home'),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});save();print(json.dumps({'status':'PASS','source_files':len(files),'toolchain':'1.90.0'}))
except Exception as e:
 result.update({'status':'ERROR','error_type':type(e).__name__,'error':str(e),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});save();raise
