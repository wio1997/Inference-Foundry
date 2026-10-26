"""Read-only native request audit and failed metadata-helper diagnostics, no SDK/model calls."""
import concurrent.futures,hashlib,importlib.util,json,re
from pathlib import Path
P=Path(__file__).resolve()
def load(name,file,sha):
 path=P.with_name(file);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('pool','start_vllm_shared_pool_v4.py','33869c715fc71e7fdcd312e500f08733da7722a0e870a723726d19cd6bd6becf')
identity=load('identity','glm53_vllm_shared_model_identity_v3.py','24a93a46cfc19946e4e52cc2d255e92ea552c127b560d12ee1fea7108a18fb89')
STREAM_PATH=P.with_name('glm53_stream_native_logs_v2.py');STREAM_SHA='88d1854ec78200cb5b6661dd97e40b447bb1f9dfc768deba6f65ba0831c1be47'
assert hashlib.sha256(STREAM_PATH.read_bytes()).hexdigest()==STREAM_SHA
STREAM=STREAM_PATH.read_text();RUN=base.b.ROOT/'runs/20261017T1783-vllm-shared-long-native-v2'
HOST=r'''
import io,sys,urllib.request,time,hashlib,gzip,socket,re,pathlib,subprocess,json
checked=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert checked.returncode==0
container=json.loads(checked.stdout)[0];assert container['Id']==CID and container['Name']=='/'+NAME and container['Image']==IMAGE and container['State']['Running'] and container['Config']['Labels']['project_id']=='glm53' and container['Config']['Labels']['run_id']=='20261017T1766-vllm-shared-pool-v4'
shared_root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1766-vllm-shared-pool-v4');supervisor=json.loads((shared_root/'supervisor-status.json').read_text());provider=json.loads((shared_root/'provider-status.json').read_text())
identity_scope=IDENTITY_ENV
try:
 exec(compile(IDENTITY,'glm53_vllm_shared_model_identity_v3.py','exec'),identity_scope);model_identity=identity_scope['model_identity']
except Exception as error:model_identity={'error':repr(error),'qualification':'Live model identity failed; still preserve actual native model logs and states'}
archive=pathlib.Path(ARCHIVE);assert not archive.exists(),'No checkpoint overwrite/replay';archive.mkdir(parents=True)
scope={};exec(compile(STREAM,'glm53_stream_native_logs_v2.py','exec'),scope)
out={'project_id':'glm53','node':NODE,'container_id':CID,'actual_model_identity':model_identity,'container_state':container['State'],'supervisor':supervisor,'provider_phase':provider['phase'],'monotonic_ns':time.monotonic_ns(),'model_requests':0,'SDK_init_calls':0,'logs':{},'persistent_audit':{}}
for stream in ['stdout','stderr']:
 path=shared_root/('model.'+stream+'.txt');assert path.is_file()
 with path.open('rb') as file:out['logs'][stream]=scope['summarize_log_stream'](file,archive/('model.'+stream+'.gz'))
def cmd(args,timeout=30):
 p=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return {'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
out['npu']=cmd(['npu-smi','info']);assert out['npu']['rc']==0
rows=[]
for line in out['npu']['stdout'].split('Process id',1)[-1].splitlines():
 cols=[v.strip() for v in line.split('|')]
 if len(cols)>=5 and len(cols[1].split())==2 and all(v.isdigit() for v in cols[1].split()) and cols[2].isdigit():rows.append({'card':int(cols[1].split()[0]),'chip':int(cols[1].split()[1]),'host_pid':int(cols[2])})
assert rows
for row in rows:
 proc=pathlib.Path('/proc')/str(row['host_pid']);assert CID in (proc/'cgroup').read_text();row['namespace_pid']=int(next(line.split()[-1] for line in (proc/'status').read_text().splitlines() if line.startswith('NSpid:')))
out['owned_npu_process_rows']=rows;out['top']=cmd(['docker','top',CID,'-eo','pid,ppid,stat,comm,etime']);out['loadavg']=pathlib.Path('/proc/loadavg').read_text();out['meminfo']=pathlib.Path('/proc/meminfo').read_text()
for relative,item in FILES.items():assert hashlib.sha256((pathlib.Path(STAGED)/relative).read_bytes()).hexdigest()==item['sha256']
for endpoint in ['/health','/v1/models','/metrics']:
 try:
  request=urllib.request.Request('http://172.16.12.'+str(NODE)+':'+str(PORT)+endpoint)
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open(request,timeout=10) as response:raw=response.read(8*1024**2+1);assert len(raw)<=8*1024**2
  dest=archive/(endpoint.replace('/','-')+'.gz');dest.write_bytes(gzip.compress(raw,mtime=0))
  out[endpoint]={'status':response.status,'body':raw.decode(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
 except Exception as error:out[endpoint]={'error':repr(error)}
paths=sorted((shared_root/'audit').glob('events-*.jsonl'));assert len(paths)<=256;total=0
for path in paths:
 raw=path.read_bytes();total+=len(raw);assert len(raw)<=8*1024**2 and total<=64*1024**2
 assert not raw or raw.endswith(b'\n'),'Incomplete native audit row; preserve without silent truncation'
 out['persistent_audit'][path.name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'rows':[json.loads(line) for line in raw.splitlines()]}
if NODE==30:
 info=json.loads(cmd(['docker','inspect',META])['stdout'])[0];assert info['Id']==META and info['Image']==IMAGE and info['Config']['Labels']['project_id']=='glm53' and info['Config']['Labels']['run_id']=='20261017T1766-vllm-shared-pool-v4';out['meta_state']=info['State']
profile=shared_root/'profile';out['profiler_artifacts']=[{'path':str(p.relative_to(shared_root)),'bytes':p.stat().st_size} for p in profile.rglob('*') if p.is_file()] if profile.exists() else []

out['native_SDK_small_file_snapshots']=[]
for sdk_cid in ([CID,META] if NODE==30 else [CID]):
 result=cmd(['docker','exec',sdk_cid,'python3','-c',SDK_STATS_SOURCE],60)
 item={'container_id':sdk_cid,'rc':result['rc'],'stderr':result['stderr']}
 if result['rc']==0:item['files']=json.loads(result['stdout'])
 else:item['stdout']=result['stdout']
 out['native_SDK_small_file_snapshots'].append(item)
if NODE==43:
 out['long_client_artifacts']={};client=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1779-vllm-shared-long-semantic-v2');total=0
 for path in sorted(client.iterdir()):
  if not path.is_file() or not (path.name in ['client-stderr.txt','semantic-result.json','semantic-phase-status.json'] or path.name.startswith(('metadata-','metrics-'))):continue
  raw=path.read_bytes();total+=len(raw);assert len(raw)<=8*1024**2 and total<=32*1024**2
  out['long_client_artifacts'][path.name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}

print(json.dumps(out))
'''
def capture(node,capacity):
 item=capacity['providers'][str(node)];code=HOST
 values={'CID':item['container_id'],'NAME':item['name'],'NODE':node,'IMAGE':base.IMAGE,'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/20261017T1783-vllm-shared-long-native-v2','STREAM':STREAM,'SDK_STATS_SOURCE':"import pathlib,json,hashlib\nout=[]\npaths=sorted(pathlib.Path('/var/log/memfabric_hybrid').glob('ptracer_*.dat'))\nmeta=pathlib.Path('/var/log/memcache_hybrid/logs/mmc-meta-audit.log')\nif meta.is_file():paths.append(meta)\nassert len(paths)<=256\nfor path in paths:\n if path.is_symlink() or not path.is_file():continue\n with path.open('rb') as file:raw=file.read(2097153)\n assert len(raw)<=2097152,'SDKptracer snapshot perfile capexceeded; no truncation'\n out.append({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'complete_line_ending':not raw or raw.endswith(b'\\n'),'text':raw.decode(errors='replace')})\nassert sum(item['bytes'] for item in out)<=64*1024**2\nprint(json.dumps(out))\n",'IDENTITY':identity.HOST,'IDENTITY_ENV':{'CID':item['container_id'],'NAME':item['name'],'IMAGE':base.IMAGE,'NODE':node},'FILES':base.FILES,'STAGED':base.STAGED,'PORT':30300 if node in [30,31] else 30400,'META':json.loads(capacity['meta']['stdout'])['container_id']}
 pattern=r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b';code=re.sub(pattern,lambda m:repr(values[m.group(0)]),code)
 __import__('ast').parse(code);return base.invoke(node,code,240)
def main():
 assert (base.b.ROOT/'runs/20261017T1779-vllm-shared-long-semantic-v2/receipt.json').is_file()
 start=json.loads((base.b.ROOT/'runs/20261017T1768-vllm-shared-model-start-v2/receipt.json').read_text());assert len(start['starts'])==4 and all(x==0 for x in start['starts'].values())
 path=base.b.ROOT/'results/vllm-shared-pool-capacity-v4.json';assert hashlib.sha256(path.read_bytes()).hexdigest()=='a57d238ac0d4c6b7b4c6f70df83665c284dc3a47dbbed123769796ca5e8d67bd';capacity=json.loads(path.read_text())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'model_requests':0,'SDK_init_calls':0}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(capture,n,capacity):n for n in [30,31,43,45]}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f]
    try:r=f.result()
    except Exception as error:r={'error':repr(error)}
    (RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][str(n)]=r.get('rc',r.get('error'));print(n,receipt['nodes'][str(n)],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
