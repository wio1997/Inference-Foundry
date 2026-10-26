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
RUN=b.ROOT/'runs/20261017T1410-sglang-v4-matrix-state-v1'
IDS={34: 'b91ac6ca6fffa3caa5ab08869997b0590f53846db48723991b9189cd7d248aeb', 35: 'b3cae6b8479be6db5a2bd001eade675330f01b238176292abdca172fec93c529', 36: 'd0b3515f3cb028e15aae787631bf807b63f394dbc57fd5ae782023a676588915', 41: '3cb55401fe1bc03e21709048285d4d77439cf2e228640009a8957bfb2b6e885c'}
META='8a06e693e2343d9ae39202c75fae12e86995f816c6a899a3934c238089adc48b'
HOST=r'''
import socket,subprocess,json,sys,pathlib,urllib.request,time,hashlib,gzip
assert socket.gethostname()==EXPECTED
def cmd(args,timeout=20):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def filtered_logs(cid):
 raw=cmd(['docker','logs',cid],60)
 archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1410-sglang-v4-matrix-state-v1')
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
  result[stream+'_capacity_lines']=[line for line in text.splitlines() if any(key in line for key in ('Load weight','Load weights','KV Cache','KV cache','memory pool','available_gpu_mem','mem_fraction_static','max_total_num_tokens','Attention TP','DP','hicache'))]
  import re
  sdk_lines=[line for line in text.splitlines() if 'hybm_data_op_sdma.cpp' in line]
  copies=[re.search(r'CopyG2GAsync.*length:(\d+) st:(\d+)',line) for line in sdk_lines]
  copies=[match for match in copies if match]
  result[stream+'_native_sdma']={'source_lines':len(sdk_lines),'copy_calls':len(copies),
      'copy_descriptor_bytes':sum(int(match.group(1)) for match in copies),
      'copy_status_values':sorted({int(match.group(2)) for match in copies}),
      'physical_link_bytes':None,'last_lines':sdk_lines[-10:]}

 return result
def inspect(cid,name):
 raw=cmd(['docker','inspect',cid]);assert raw['rc']==0
 data=json.loads(raw['stdout'])[0]
 assert data['Id']==cid and data['Name']=='/'+name and data['Config']['Labels']['project_id']=='glm53'
 assert data['Image']=='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
 return {'id':cid,'name':name,'image':data['Image'],'state':data['State'],'selected_env':{k:v for e in data['Config']['Env'] for k,_,v in [e.partition('=')] if k in ['DEEP_USE_MODE','ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE','ASCEND_GLOBAL_RESOURCE_CONFIG','ASCEND_LOCAL_COMM_RES','PYTHONHASHSEED','MMC_LOCAL_CONFIG_PATH','MMC_META_CONFIG_PATH','SGLANG_HICACHE_HOST_MEM_BACKEND']},
         'logs':filtered_logs(cid),
         'top':cmd(['docker','top',cid,'-eo','pid,ppid,stat,comm,etime'])}
out={'node':NODE,'monotonic_ns':time.monotonic_ns(),'model':inspect(CID,NAME),
     'npu':cmd(['npu-smi','info'],30),'load':pathlib.Path('/proc/loadavg').read_text(),
     'meminfo':pathlib.Path('/proc/meminfo').read_text(),
     'listeners':cmd(['ss','-lntp'])}
port=30500 if NODE in (34,35) else 30600
for endpoint in ['/get_server_info','/metrics']:
 try:
  with urllib.request.urlopen(f'http://172.16.12.{NODE}:{port}'+endpoint,timeout=2) as response:
   raw=response.read(8000001);assert len(raw)<=8000000,'Fullresponsecap exceeded'
   archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1410-sglang-v4-matrix-state-v1');path=archive/(str(NODE)+endpoint.replace('/','-')+'.gz')
   with path.open('xb') as file:
    with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as gz:gz.write(raw)
   out[endpoint]={'status':response.status,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'archive':str(path)}
   if endpoint=='/metrics':out[endpoint]['body']=raw.decode()
   else:
    data=json.loads(raw);selected=[]
    def visit(obj,prefix=''):
     if isinstance(obj,dict):
      for key,value in obj.items():
       if key in ['max_total_num_tokens','max_running_requests','speculative_num_steps','speculative_num_draft_tokens','dp_size','tp_size','ep_size','kv_cache_dtype','deepep_mode','mem_fraction_static','num_running_reqs','num_waiting_reqs']:selected.append({'path':prefix+key,'value':value})
       if isinstance(value,(dict,list)):visit(value,prefix+key+'.')
     elif isinstance(obj,list):
      for i,value in enumerate(obj):visit(value,prefix+str(i)+'.')
    visit(data);out[endpoint]['selected']=selected;out[endpoint]['top_level_keys']=list(data)
 except Exception as e:out[endpoint]={'error':str(e)}
if NODE==34:out['meta']=inspect(META,'glm53-sglang-correctness-v4-meta-20261017-34')

out['persistent_audit']={}
for path in sorted(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1320-sglang-correctness-v4/audit').glob('sdma-*.jsonl')):
 data=path.read_bytes();assert len(data)<2000000
 out['persistent_audit'][path.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'rows':[json.loads(line) for line in data.splitlines()]}
if NODE==34:
 out['functional_client_progress']={}
 for name in ['progress.json','result.json']:
  path=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1405-sglang-pd-matrix-v3')/name
  if path.is_file():
   raw=path.read_bytes();out['functional_client_progress'][name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'data':json.loads(raw)}
print(json.dumps(out))
'''


def inspect(node):
 pod,hostname=b.NODES[node]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code=HOST
 for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-sglang-correctness-v4-20261017-{node}'),('META',META)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],code,timeout=150)


def main():
 result=json.loads((b.ROOT/'runs/20261017T1320-sglang-correctness-v4/receipt.json').read_text())
 assert result['meta']==0 and all(x==0 for x in result['models'].values())
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
