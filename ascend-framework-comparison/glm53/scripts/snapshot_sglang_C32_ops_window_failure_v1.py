"""One-time read-only native failure snapshot; no model/SDK calls or service control."""
import ast,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261018T-L08-window-failure-snapshot-v1'
p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
sp=importlib.util.spec_from_file_location('base',p);base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base)
p=ROOT/'config/C32-ops-window-runtime-v2/runtime.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='19f1fa64ce0d55390488ff6f6da236779dcba4086e0a7740256dd56def5d8e67'
RUNTIME=json.loads(p.read_text())
HOST=r'''import pathlib,subprocess,socket,json,hashlib,gzip,re,time,urllib.request
assert socket.gethostname()==HOSTNAME
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30);return {'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
d=json.loads(cmd(['docker','inspect',CID])['stdout'])[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED and not d['HostConfig']['Privileged']
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs')/SHARED;dest=root.parent/RID;dest.mkdir()
out={'node':NODE,'container_id':CID,'state':d['State'],'monotonic_ns':time.monotonic_ns(),'model_requests':0,'SDK_init_calls':0,'service_control_calls':0}
for name in ['supervisor-status.json','provider-status.json']:out[name]=json.loads((root/name).read_text())
out['logs']={}
for stream in ['stdout','stderr']:
 p=root/('model.'+stream+'.txt');raw=p.read_bytes();assert len(raw)<=256*1024**2
 q=dest/('model.'+stream+'.gz')
 with q.open('xb') as f:
  with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
 lines=raw.decode(errors='replace').splitlines();selected=set()
 for i,line in enumerate(lines):
  if re.search(r'ERROR|Traceback|Exception|RuntimeError|error code|Error Code|Inner Error|NEEDED|SMALL|HCCL_BUFFSIZE|what\(\)|Fatal|Aicore|AICore',line):selected.update(range(max(0,i-4),min(len(lines),i+8)))
 contexts=[{'line':i+1,'text':lines[i]} for i in sorted(selected)];assert len(json.dumps(contexts))<=4*1024**2
 out['logs'][stream]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'archive':str(q),'archive_sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'error_contexts':contexts,'Ops_markers':[l for l in lines if 'GLM53_C32_OPS_' in l],'capacity':[l for l in lines if 'max_total_num_tokens=' in l or 'KV Cache' in l]}
out['npu']=cmd(['npu-smi','info']);out['top']=cmd(['docker','top',CID,'-eo','pid,ppid,stat,comm,etime'])
for endpoint in ['health','metrics']:
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open('http://172.16.12.'+str(NODE)+':'+('30500' if NODE in [34,35] else '30600')+'/'+endpoint,timeout=10) as res:
   raw=res.read(16*1024**2+1);assert len(raw)<=16*1024**2;out[endpoint]={'status':res.status,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'idle_gauges':[l for l in raw.decode().splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]}
 except Exception as exc:out[endpoint]={'error':repr(exc)}
print(json.dumps(out))
'''
def render(n):
 item=RUNTIME['runtime_containers'][str(n)];values={'CID':item['container_id'],'NAME':item['name'],'IMAGE':RUNTIME['model_image'],'HOSTNAME':base.NODES[n][1],'SHARED':'20261018T-L08-sglang-C32-ops-window-v2','RID':RID,'NODE':n}
 code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),HOST);ast.parse(code);return code
def main():
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'nodes':{},'max_SSH':1,'model_requests':0,'service_control_calls':0}
 try:
  for n in [36,41,34,35]:
   pod,_=base.NODES[n];mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);(run/f'mapping-{n}.json').write_text(json.dumps(mapped,indent=2)+'\n');assert mapped['rc']==0 and mapped['stdout']=='172.16.12.'+str(n)
   command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);result=base.invoke([command],render(n),timeout=180);(run/f'node-{n}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes'][str(n)]=result['rc'];print(n,result['rc'],flush=True);assert result['rc']==0
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
