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
RUN=b.ROOT/'runs/20261017T1160-sglang-v2-dispatch-failure-v1'
IDS={34: '533648568fe9df4b604819bee901c11762ac3ec36a34cc62cf151f955701ee39', 35: '4063f601648836976da5030069fb30d4c236ea86faecb28b5e0fbc4226a05b7c', 36: 'd541c9d9f2208dfa680b27c3a83612f6667b4628c24f3c28082d08eba3b46948', 41: '0177c625748cf89f2c589a1fc30e7efb3cb646f76a9a0dc0d0556f8bf3cf7982'}
META='af297e9e530d2eb012a24ca2090101078b2a0c5b305133da61e0cfa0b5f540e8'
HOST=r'''
import socket,subprocess,json,sys,pathlib,urllib.request,time,hashlib,gzip
assert socket.gethostname()==EXPECTED
def cmd(args,timeout=20):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def filtered_logs(cid):
 raw=cmd(['docker','logs',cid],60)
 archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1160-sglang-v2-dispatch-failure-v1')
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
 return {'id':cid,'name':name,'image':data['Image'],'state':data['State'],
         'logs':filtered_logs(cid),
         'top':cmd(['docker','top',cid,'-eo','pid,ppid,stat,comm,etime'])}
out={'node':NODE,'monotonic_ns':time.monotonic_ns(),'model':inspect(CID,NAME),
     'npu':cmd(['npu-smi','info'],30),'load':pathlib.Path('/proc/loadavg').read_text(),
     'meminfo':pathlib.Path('/proc/meminfo').read_text(),
     'listeners':cmd(['ss','-lntp'])}
port=30500 if NODE in (34,35) else 30600
for endpoint in ['/health','/get_server_info']:
 try:
  with urllib.request.urlopen(f'http://172.16.12.{NODE}:{port}'+endpoint,timeout=2) as response:
   out[endpoint]={'status':response.status,'body':response.read(65536).decode()}
 except Exception as e:out[endpoint]={'error':str(e)}
if NODE==34:out['meta']=inspect(META,'glm53-sglang-correctness-v2-meta-20261017-34')

if NODE==41:
 import io,tarfile,datetime
 raw=subprocess.run(['docker','cp',CID+':/root/ascend/log/debug/plog/.','-'],capture_output=True,timeout=60)
 out['cann_plog']={'docker_cp_rc':raw.returncode,'stderr':raw.stderr.decode(errors='replace')[-2000:],'files':[]}
 if raw.returncode==0:
  archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1160-sglang-v2-dispatch-failure-v1')/'D41-cann-plog.tar.gz'
  with archive.open('xb') as file:
   with gzip.GzipFile(fileobj=file,mode='wb',mtime=0) as gz:gz.write(raw.stdout)
  out['cann_plog'].update(archive=str(archive),tar_sha256=hashlib.sha256(raw.stdout).hexdigest(),tar_bytes=len(raw.stdout))
  with tarfile.open(fileobj=io.BytesIO(raw.stdout),mode='r:') as tar:
   for member in tar.getmembers():
    if not member.isfile() or member.size>64000000:continue
    data=tar.extractfile(member).read();lines=data.decode(errors='replace').splitlines()
    hits=[i for i,line in enumerate(lines) if any(key in line for key in ('MoeLowLatencyDispatchV2','Tiling','tiling','expertNum','epWorldSize'))]
    indices=sorted({j for i in hits for j in range(max(0,i-3),min(len(lines),i+6))})
    if indices:out['cann_plog']['files'].append({'name':member.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'mtime':member.mtime,'selected_lines':[[i+1,lines[i]] for i in indices[-160:]]})
out['persistent_audit']={}
for path in sorted(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1110-sglang-correctness-v2/audit').glob('sdma-*.jsonl')):
 data=path.read_bytes();assert len(data)<2000000
 out['persistent_audit'][path.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'rows':[json.loads(line) for line in data.splitlines()]}
print(json.dumps(out))
'''


def inspect(node):
 pod,hostname=b.NODES[node]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
 code=HOST
 for key,value in [('EXPECTED',hostname),('NODE',node),('CID',IDS[node]),('NAME',f'glm53-sglang-correctness-v2-20261017-{node}'),('META',META)]:
  code=re.sub(r'\b'+re.escape(key)+r'\b',lambda _:repr(value),code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return b.invoke([command],code,timeout=150)


def main():
 result=json.loads((b.ROOT/'runs/20261017T1110-sglang-correctness-v2/receipt.json').read_text())
 assert result['meta']==0 and all(x==0 for x in result['models'].values())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   futures={pool.submit(inspect,n):n for n in IDS}
   for f in concurrent.futures.as_completed(futures):
    n=futures[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (RUN/f'node-{n}.json').write_text(json.dumps(r,indent=2));receipt['nodes'][n]=r.get('rc',r.get('error'));print(n,receipt['nodes'][n],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
