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
RUN=b.ROOT/'runs/20261017T2032-sglang-buffer-dedup-startup-stable-v1'
capacity=json.loads((b.ROOT/'results/sglang-buffer-dedup-pool-capacity-v1.json').read_text());assert capacity['pass']
IDS={int(n):v['container_id'] for n,v in capacity['providers'].items()}
META=json.loads(capacity['meta']['stdout'])['container_id']
HOST=r'''
import socket,subprocess,json,sys,pathlib,urllib.request,time,hashlib,gzip
assert socket.gethostname()==EXPECTED
def cmd(args,timeout=20):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def filtered_logs(cid):
 if cid==META:raw=cmd(['docker','logs',cid],60)
 else:
  parent=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T2028-sglang-buffer-dedup-v1');raw={'rc':0}
  for stream in ['stdout','stderr']:
   p=parent/('model.'+stream+'.txt')
   if p.exists():
    assert p.stat().st_size<=100000000,'Fullmodel log exceeded checkpoint budget; preserve without truncation'
    raw[stream]=p.read_text(errors='replace')
   else:raw[stream]='MODEL_LOG_NOT_YET_CREATED'
 archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T2032-sglang-buffer-dedup-startup-stable-v1')
 archive.mkdir(parents=True,exist_ok=True)
 result={'rc':raw['rc']}
 for stream in ('stdout','stderr'):
  text=raw[stream]
  path=archive/(cid+'.'+stream+'.gz')
  with path.open('xb') as file:
   with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as gz:gz.write(text.encode())
  result[stream+'_archive']=str(path)
  result[stream+'_sha256']=hashlib.sha256(text.encode()).hexdigest();result[stream+'_bytes']=len(text.encode())
  selected=[line for line in text.splitlines() if line.startswith(('(Worker','(EngineCore','(APIServer')) or any(k in line for k in ('GLM53_','ERROR','Traceback','Exception','Error:'))]
  result[stream]='\n'.join(text.splitlines()[-180:]);result[stream+'_selected_lines']=len(selected)
  result[stream+'_audit_lines']=[line for line in text.splitlines() if 'GLM53_SGLANG_SDMA=' in line or 'GLM53_SGLANG_ENTRY=' in line]
  result[stream+'_error_lines']=[line for line in text.splitlines() if any(k in line for k in ['ERROR','Traceback','Exception','Error:'])]
  result[stream+'_metadata_markers']=[line for line in text.splitlines() if 'GLM53_DEEPEP_METADATA_' in line or 'GLM53_METADATA_HOOK_FATAL' in line]
  result[stream+'_buffer_dedup_markers']=[line for line in text.splitlines() if 'GLM53_BUFFER_DEDUP_' in line]
  result[stream+'_capacity_lines']=[line for line in text.splitlines() if any(key in line for key in ('Load weight','Load weights','KV Cache','KV cache','memory pool','available_gpu_mem','mem_fraction_static','max_total_num_tokens','Attention TP','DP','hicache'))]
  import re
  sdk_lines=[line for line in text.splitlines() if 'hybm_data_op_sdma.cpp' in line]
  copies=[re.search(r'CopyG2GAsync.*length:(\d+) st:(\d+)',line) for line in sdk_lines]
  copies=[match for match in copies if match]
  result[stream+'_cache_diagnostic_lines']=[line for line in text.splitlines() if any(k in line.lower() for k in ['backup','prefetch','host pool','storage hit','batch_put','batch_get'])][-1000:]
  result[stream+'_native_sdma']={'source_lines':len(sdk_lines),'copy_calls':len(copies),
      'copy_descriptor_bytes':sum(int(match.group(1)) for match in copies),
      'copy_status_values':sorted({int(match.group(2)) for match in copies}),
      'physical_link_bytes':None,'last_lines':sdk_lines[-10:]}

 return result
def inspect(cid,name):
 raw=cmd(['docker','inspect',cid]);assert raw['rc']==0
 data=json.loads(raw['stdout'])[0]
 assert data['Id']==cid and data['Name']=='/'+name and data['Config']['Labels']['project_id']=='glm53'
 assert data['Config']['Labels']['run_id']=='20261017T2028-sglang-buffer-dedup-v1'
 assert data['Image']=='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
 return {'id':cid,'name':name,'image':data['Image'],'state':data['State'],'selected_env':{k:v for e in data['Config']['Env'] for k,_,v in [e.partition('=')] if k in ['DEEP_USE_MODE','ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE','ASCEND_GLOBAL_RESOURCE_CONFIG','ASCEND_LOCAL_COMM_RES','PYTHONHASHSEED','MMC_LOCAL_CONFIG_PATH','MMC_META_CONFIG_PATH','SGLANG_HICACHE_HOST_MEM_BACKEND']},
         'logs':filtered_logs(cid),
         'top':cmd(['docker','top',cid,'-eo','pid,ppid,stat,comm,etime'])}
# Wait only for native startup completion, with a fixed bound; no inference calls.
shared=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T2028-sglang-buffer-dedup-v1')
deadline=time.monotonic()+1800
while True:
 state=json.loads((shared/'supervisor-status.json').read_text())
 assert state['phase']=='MODEL_RUNNING' and 'model_exit_code' not in state,state
 warm=False
 for stream in ['stdout','stderr']:
  p=shared/('model.'+stream+'.txt')
  if p.exists():
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1048576));tail=f.read().decode(errors='replace')
   if 'Disaggregation warmup requests completed for all '+('4' if NODE==34 else '16')+' DP ranks' in tail:warm=True
 if NODE in [35,41]:warm=True
 try:
  with urllib.request.urlopen('http://172.16.12.'+str(NODE)+':'+('30500' if NODE in [34,35] else '30600')+'/health',timeout=5) as response:healthy=response.status==200
 except Exception:healthy=False
 gauges=[];registered=False
 if healthy:
  try:
   with urllib.request.urlopen('http://172.16.12.'+str(NODE)+':'+('30500' if NODE in [34,35] else '30600')+'/metrics',timeout=5) as response:metric=response.read(8388609);assert len(metric)<=8388608
   gauges=[float(line.rsplit(' ',1)[1]) for line in metric.decode().splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]
   events=[]
   for auditfile in (shared/'audit').glob('sdma-*.jsonl'):
    raw=auditfile.read_bytes();assert len(raw)<=16777216 and (not raw or raw.endswith(b'\n'));events.extend(json.loads(line) for line in raw.splitlines())
   registered=len({x['pid'] for x in events if x['event']=='initialize_complete'})==16 and len({x['pid'] for x in events if x['event']=='model_state_registered'})==16
  except Exception:registered=False
 if warm and healthy and gauges and not any(gauges) and registered:break
 if time.monotonic()>deadline:raise TimeoutError('Native startup bounded1800s; preserve model/logs, no stop/retry')
 time.sleep(5)
out={'node':NODE,'monotonic_ns':time.monotonic_ns(),'model':inspect(CID,NAME),
     'npu':cmd(['npu-smi','info'],30),'load':pathlib.Path('/proc/loadavg').read_text(),
     'meminfo':pathlib.Path('/proc/meminfo').read_text(),
     'listeners':cmd(['ss','-lntp'])}
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T2028-sglang-buffer-dedup-v1')
out['supervisor_status']=json.loads((root/'supervisor-status.json').read_text());out['provider_status']=json.loads((root/'provider-status.json').read_text())
port=30500 if NODE in (34,35) else 30600
for endpoint in ['/get_server_info','/metrics']:
 try:
  with urllib.request.urlopen(f'http://172.16.12.{NODE}:{port}'+endpoint,timeout=10) as response:
   raw=response.read(8000001);assert len(raw)<=8000000,'Fullresponsecap exceeded'
   archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T2032-sglang-buffer-dedup-startup-stable-v1');path=archive/(str(NODE)+endpoint.replace('/','-')+'.gz')
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
if NODE==34:out['meta']=inspect(META,'glm53-sglang-buffer-dedup-v1-meta-20261017-34')

out['persistent_audit']={}
for path in sorted(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T2028-sglang-buffer-dedup-v1/audit').glob('sdma-*.jsonl')):
 data=path.read_bytes();assert len(data)<2000000
 out['persistent_audit'][path.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'rows':[json.loads(line) for line in data.splitlines()]}
if NODE==43:
 out['functional_client_progress']={}
 for name in ['semantic-progress.json','semantic-result.json','semantic-phase-status.json']:
  path=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1435-sglang-long-semantic-v1')/name
  if path.is_file():
   raw=path.read_bytes();out['functional_client_progress'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'data':json.loads(raw)}
print(json.dumps(out))
'''


def inspect(node):
 pod,hostname=b.NODES[node]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code=HOST
 for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-sglang-buffer-dedup-v1-20261017-{node}'),('META',META)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],code,timeout=2000)


def main():
 result=json.loads((b.ROOT/'runs/20261017T2030-sglang-buffer-dedup-model-start-v1/receipt.json').read_text())
 assert len(result['starts'])==4 and all(x==0 for x in result['starts'].values())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   futures={pool.submit(inspect,n):n for n in IDS}
   for f in concurrent.futures.as_completed(futures):
    n=futures[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2));receipt['nodes'][n]=r.get('rc',r.get('error'));print(n,receipt['nodes'][n],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
