#!/usr/bin/env python3
"""Single read-only owned-container startup checkpoint, no inference."""
import concurrent.futures
import hashlib
import importlib.util
import json
import shlex
import re
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T1747-sglang-shared-long-native-audit-v1'
IDS={34: '10d14e531385739498eacfb08d75d6cc3b370cf78f74af14e9fae6736bbb744e', 35: 'eb04fad0928296da105ae2a0d90b16ef273550e17facebba56eab88ba6aa795d', 36: '0d02ac7770dab23c14a55fe18b0dc8f7d5999b455926403b91227b9d95bcf20d', 41: 'f77216e73f51473383da9f6c7fe46e6ba40338db890ed0db656d18a032225fc6'}
META='b850ebfa32eeb44b21ea9a0cd44cee80e68a962df6276b5b77169da03289ea08'
SDK_STATS_SOURCE="import pathlib,json,hashlib\nout=[]\npaths=sorted(pathlib.Path('/var/log/memfabric_hybrid').glob('ptracer_*.dat'))\nmeta=pathlib.Path('/var/log/memcache_hybrid/logs/mmc-meta-audit.log')\nif meta.is_file():paths.append(meta)\nassert len(paths)<=256\nfor path in paths:\n if path.is_symlink() or not path.is_file():continue\n with path.open('rb') as file:raw=file.read(2097153)\n assert len(raw)<=2097152,'SDKptracer snapshot perfile capexceeded; no truncation'\n out.append({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'complete_line_ending':not raw or raw.endswith(b'\\n'),'text':raw.decode(errors='replace')})\nassert sum(item['bytes'] for item in out)<=64*1024**2\nprint(json.dumps(out))\n"
STREAM_SOURCE=(P.parent/'glm53_stream_native_logs_v1.py').read_text()
assert hashlib.sha256(STREAM_SOURCE.encode()).hexdigest()=='7a9a5f212a89d8bc51e65204e59df6d0e94e6f0e53e46eb0f9013d7b76caa108'
HOST=r'''
import socket,subprocess,json,sys,pathlib,urllib.request,time,hashlib,gzip
assert socket.gethostname()==EXPECTED
exec(compile(STREAM_SOURCE,'glm53_native_stream_logs','exec'))
def cmd(args,timeout=20):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def filtered_logs(cid):
 import io
 archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1747-sglang-shared-long-native-audit-v1');archive.mkdir(parents=True,exist_ok=True)
 result={'rc':0,'capture':'Full streaming rawarchive/SHA, boundedlineandtotalcaps; no payload truncation'}
 raw=cmd(['docker','logs',cid],60) if cid==META else None
 if raw is not None:result['rc']=raw['rc']
 for stream in ['stdout','stderr']:
  if raw is not None:source=io.BytesIO(raw[stream].encode())
  else:source=(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1720-sglang-shared-pool-v2')/('model.'+stream+'.txt')).open('rb')
  with source:summary=summarize_log_stream(source,archive/(cid+'.'+stream+'.gz'))
  for key,value in summary.items():result[stream+('_' if key!='tail' else '')+(key if key!='tail' else '')]=value
 return result

def inspect(cid,name):
 raw=cmd(['docker','inspect',cid]);assert raw['rc']==0
 data=json.loads(raw['stdout'])[0]
 assert data['Id']==cid and data['Name']=='/'+name and data['Config']['Labels']['project_id']=='glm53'
 assert data['Config']['Labels']['run_id']=='20261017T1720-sglang-shared-pool-v2'
 assert data['Image']=='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
 return {'id':cid,'name':name,'image':data['Image'],'state':data['State'],'selected_env':{k:v for e in data['Config']['Env'] for k,_,v in [e.partition('=')] if k in ['DEEP_USE_MODE','ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE','ASCEND_GLOBAL_RESOURCE_CONFIG','ASCEND_LOCAL_COMM_RES','PYTHONHASHSEED','MMC_LOCAL_CONFIG_PATH','MMC_META_CONFIG_PATH','SGLANG_HICACHE_HOST_MEM_BACKEND']},
         'logs':filtered_logs(cid),
         'top':cmd(['docker','top',cid,'-eo','pid,ppid,stat,comm,etime'])}
out={'node':NODE,'monotonic_ns':time.monotonic_ns(),'model':inspect(CID,NAME),
     'npu':cmd(['npu-smi','info'],30),'load':pathlib.Path('/proc/loadavg').read_text(),
     'meminfo':pathlib.Path('/proc/meminfo').read_text(),
     'listeners':cmd(['ss','-lntp'])}
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1720-sglang-shared-pool-v2')
out['supervisor_status']=json.loads((root/'supervisor-status.json').read_text());out['provider_status']=json.loads((root/'provider-status.json').read_text())
port=30500 if NODE in (34,35) else 30600
for endpoint in ['/get_server_info','/metrics']:
 try:
  with urllib.request.urlopen(f'http://172.16.12.{NODE}:{port}'+endpoint,timeout=10) as response:
   raw=response.read(8000001);assert len(raw)<=8000000,'Fullresponsecap exceeded'
   archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1747-sglang-shared-long-native-audit-v1');path=archive/(str(NODE)+endpoint.replace('/','-')+'.gz')
   with path.open('xb') as file:
    with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as gz:gz.write(raw)
   out[endpoint]={'status':response.status,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'archive':str(path)}
   if endpoint=='/metrics':out[endpoint]['body']=raw.decode()
   else:
    data=json.loads(raw);selected=[]
    def visit(obj,prefix=''):
     if isinstance(obj,dict):
      for key,value in obj.items():
       if key in ['max_total_num_tokens','max_running_requests','speculative_num_steps','speculative_num_draft_tokens','dp_size','tp_size','ep_size','kv_cache_dtype','deepep_mode','mem_fraction_static','num_running_reqs','num_waiting_reqs','hicache_host_memory_mode','hicache_size','enable_hierarchical_cache','hicache_storage_backend']:selected.append({'path':prefix+key,'value':value})
       if isinstance(value,(dict,list)):visit(value,prefix+key+'.')
     elif isinstance(obj,list):
      for i,value in enumerate(obj):visit(value,prefix+str(i)+'.')
    visit(data);out[endpoint]['selected']=selected;out[endpoint]['top_level_keys']=list(data)
 except Exception as e:out[endpoint]={'error':str(e)}
if NODE==34:out['meta']=inspect(META,'glm53-sglang-shared-v2-meta-20261017-34')

out['persistent_audit']={}
for path in sorted(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1720-sglang-shared-pool-v2/audit').glob('sdma-*.jsonl')):
 data=path.read_bytes();assert len(data)<2000000
 out['persistent_audit'][path.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'rows':[json.loads(line) for line in data.splitlines()]}
if NODE==43:
 out['functional_client_progress']={}
 for name in ['semantic-progress.json','semantic-result.json','semantic-phase-status.json']:
  path=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1435-sglang-long-semantic-v1')/name
  if path.is_file():
   raw=path.read_bytes();out['functional_client_progress'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'data':json.loads(raw)}
if NODE==34:
 out['Meta_HTTP_readonly_diagnostics']={}
 direct=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 for endpoint in ['/health','/metrics/summary','/metrics/ptracer']:
  try:
   with direct.open('http://172.16.12.34:29782'+endpoint,timeout=3) as response:
    body=response.read(2097153);assert len(body)<=2097152,'Metaendpointfullcapexceeded'
    out['Meta_HTTP_readonly_diagnostics'][endpoint]={'status':response.status,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'text':body.decode() if len(body)<=65536 else None}
  except Exception as error:out['Meta_HTTP_readonly_diagnostics'][endpoint]={'error':repr(error)}
out['native_SDK_small_file_snapshots']=[]
for sdk_cid in ([CID,META] if NODE==34 else [CID]):
 r=cmd(['docker','exec',sdk_cid,'python3','-c',SDK_STATS_SOURCE],60)
 item={'container_id':sdk_cid,'rc':r['rc'],'stderr':r['stderr']}
 if r['rc']==0:item['files']=json.loads(r['stdout'])
 else:item['stdout']=r['stdout']
 out['native_SDK_small_file_snapshots'].append(item)
print(json.dumps(out))
'''


def inspect(node):
 pod,hostname=b.NODES[node]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code=HOST
 for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-sglang-shared-v2-20261017-{node}'),('META',META),('STREAM_SOURCE',STREAM_SOURCE),('SDK_STATS_SOURCE',SDK_STATS_SOURCE)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],code,timeout=150)


def main():
 long=json.loads((b.ROOT/'runs/20261017T1740-sglang-shared-long-semantic-v4/receipt.json').read_text());assert long['rc']==0 and all(value==0 for value in long['model_precheck'].values())
 result=json.loads((b.ROOT/'runs/20261017T1725-sglang-shared-model-start-v2/receipt.json').read_text())
 assert len(result['starts'])==4 and all(x==0 for x in result['starts'].values())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
   futures={pool.submit(inspect,n):n for n in IDS}
   for f in concurrent.futures.as_completed(futures):
    n=futures[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2));receipt['nodes'][n]=r.get('rc',r.get('error'));print(n,receipt['nodes'][n],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
