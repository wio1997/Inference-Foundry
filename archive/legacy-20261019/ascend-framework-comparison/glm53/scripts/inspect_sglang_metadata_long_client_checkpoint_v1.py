"""One bounded CPU43-only long-client checkpoint; no model/SDK/API calls."""
import ast,hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1];RUN=ROOT/'runs/20261017T1967-sglang-metadata-long-client-checkpoint-v1'
dep=P.parent/'inventory_readonly.py';assert hashlib.sha256(dep.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
spec=importlib.util.spec_from_file_location('base',dep);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
HOST=r'''import socket,pathlib,json,hashlib,subprocess,time
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
rid='20261017T1964-sglang-metadata-long-v1';root=pathlib.Path('/data/wio/inference-foundry/glm53/runs')/rid
cid=(root/'client.cid').read_text().strip();assert len(cid)==64
r=subprocess.run(['docker','inspect',cid],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0]
assert d['Id']==cid and d['Name']=='/glm53-sglang-metadata-long-v1-20261017-43' and d['Image']=='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==rid and not d['HostConfig']['Privileged']
out={'project_id':'glm53','node':43,'client_container_id':cid,'client_state':d['State'],'model_requests':0,'SDK_init_calls':0,'snapshots':{},'monotonic_ns':time.monotonic_ns()}
for name in ['semantic-progress.json','semantic-result.json','semantic-phase-status.json','metadata-gate-0.json','metadata-gate-1.json']:
 p=root/name
 if not p.exists():out['snapshots'][name]={'exists':False};continue
 raw=p.read_bytes();assert len(raw)<=32*1024**2;item={'exists':True,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 try:obj=json.loads(raw)
 except json.JSONDecodeError:out['snapshots'][name]=item|{'inflight_write_observed':True};continue
 if 'cases' in obj:
  item['cases']=[{k:row.get(k) for k in ['semantic_case','case','request_id','required_input','required_output','count_pass','pd_lifecycle_pass','semantic_pass']} for row in obj['cases']]
  item['gateway']=[{k:row.get(k) for k in ['logical_request_id','status','prefill_http_status','decode_http_status','prefill_dispatch_monotonic_ns','decode_dispatch_monotonic_ns','prefill_terminal_monotonic_ns','decode_terminal_monotonic_ns']} for row in obj['gateway']]
  item['semantic_pass']=obj.get('semantic_pass')
 elif name.startswith('metadata-gate-'):
  item.update(rc=obj.get('rc'),passed=obj.get('pass'),known_component_keys=obj.get('metadata',{}).get('known_component_keys'),existing_component_keys=obj.get('metadata',{}).get('last_existing_component_keys'))
 else:item['phase']=obj
 out['snapshots'][name]=item
err=root/'client-stderr.txt';raw=err.read_bytes();assert len(raw)<=16*1024**2
out['client_stderr']={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'error_lines':[line for line in raw.decode(errors='replace').splitlines() if any(k in line for k in ['Traceback','ERROR','Error:','AssertionError'])][-30:]}
print(json.dumps(out))
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod=b.NODES[43][0];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.43'
 ast.parse(HOST)
 r=b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],HOST,timeout=90)
 (RUN/'node-43.json').write_text(json.dumps(r,indent=2)+'\n');(RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','rc':r['rc'],'launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0},indent=2)+'\n');assert r['rc']==0
if __name__=='__main__':main()
