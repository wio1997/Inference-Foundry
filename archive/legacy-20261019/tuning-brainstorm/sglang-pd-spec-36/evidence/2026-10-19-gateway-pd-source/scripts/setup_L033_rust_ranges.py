#!/usr/bin/env python3
"""Install official hash-verified Rust components in a task-owned prefix using bounded HTTP ranges."""
import concurrent.futures,datetime,hashlib,json,pathlib,platform,re,subprocess,tarfile,tomllib
R=pathlib.Path(__file__).resolve().parents[1];D=R/'.deps/L033-rust-direct';D.mkdir(parents=True,exist_ok=False);O=R/'runs/L033-R006-rust-direct';O.mkdir(parents=True,exist_ok=False);prefix=D/'toolchain';result={'status':'RUNNING','version':'1.90.0','target':'aarch64-apple-darwin','native_executed':False,'components':[]}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(): (O/'result.json').write_text(json.dumps(result,indent=2)+'\n')
def command(args):
 c=subprocess.run(args,capture_output=True,text=True,timeout=100);assert c.returncode==0,(c.returncode,c.stderr[-1000:]);return c
save();assert platform.machine()=='arm64'
def download(url,name):
 assert url.startswith('https://static.rust-lang.org/')
 h=command(['curl','--silent','--show-error','--head','--fail','--location','--connect-timeout','15','--max-time','60',url]).stdout.lower();size=int(re.findall(r'content-length:\s*(\d+)',h)[-1]);parts=D/(name+'.parts');parts.mkdir(exist_ok=False);chunk=512*1024
 def part(pair):
  a,b=pair;p=parts/f'{a:012d}.part';hp=parts/f'{a:012d}.headers';last=None
  for attempt in range(2):
   try:
    command(['curl','--silent','--show-error','--fail','--location','--connect-timeout','15','--max-time','90','--range',f'{a}-{b}','--max-filesize',str(b-a+1),'--dump-header',str(hp),'--output',str(p),url]);headers=hp.read_text().lower();assert re.search(r'content-range:\s*bytes\s+'+str(a)+'-'+str(b)+'/'+str(size),headers);assert p.stat().st_size==b-a+1;return p
   except Exception as e:last=e
  raise RuntimeError(str(last))
 ranges=[(a,min(a+chunk-1,size-1)) for a in range(0,size,chunk)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:ps=list(pool.map(part,ranges))
 dst=D/name
 with dst.open('xb') as f:
  for p in ps:f.write(p.read_bytes())
 assert dst.stat().st_size==size
 return dst,{'url':url,'bytes':size,'sha256':sha(dst),'range_parts':len(ps),'max_concurrent':16,'all_content_ranges_validated':True}
try:
 url='https://static.rust-lang.org/dist/channel-rust-1.90.0.toml';manifest,info=download(url,'channel-rust-1.90.0.toml');h=command(['curl','--silent','--show-error','--fail','--location','--connect-timeout','15','--max-time','60',url+'.sha256']).stdout.split()[0];assert sha(manifest)==h;result['release_manifest']=info;save();m=tomllib.loads(manifest.read_text());assert m['date']=='2025-09-18'
 for name in ['rustc','rust-std','cargo','clippy-preview']:
  x=m['pkg'][name]['target']['aarch64-apple-darwin'];assert x['available'];archive,metadata=download(x['xz_url'],name+'.tar.xz');assert sha(archive)==x['xz_hash'];metadata['component']=name;metadata['official_manifest_sha256']=x['xz_hash'];result['components'].append(metadata);save()
  target=D/(name+'-unpacked');target.mkdir()
  with tarfile.open(archive,'r:xz') as tf:tf.extractall(target,filter='data')
  installers=list(target.glob('*/install.sh'));assert len(installers)==1
  with (O/(name+'-install.stdout.log')).open('xb') as so,(O/(name+'-install.stderr.log')).open('xb') as se:
   c=subprocess.run(['sh',str(installers[0]),'--prefix='+str(prefix),'--disable-ldconfig'],stdout=so,stderr=se,timeout=120)
  assert c.returncode==0,name
 versions={}
 for n in ['rustc','cargo']:
  c=subprocess.run([str(prefix/'bin'/n),'-vV'],capture_output=True,text=True,timeout=30);assert c.returncode==0;versions[n]=c.stdout
 result.update({'status':'PASS','versions':versions,'prefix':str(prefix),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});save();print(json.dumps({'status':'PASS','components':len(result['components']),'version':'1.90.0'}))
except Exception as e:
 result.update({'status':'ERROR','error_type':type(e).__name__,'error':str(e),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});save();raise
