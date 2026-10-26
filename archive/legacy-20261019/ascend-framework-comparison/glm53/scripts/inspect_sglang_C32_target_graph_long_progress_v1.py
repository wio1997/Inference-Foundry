"""Single read-only semantic-client progress checkpoint; no inference or SDK init."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('inventory',p);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
HOST=r'''import socket,pathlib,json,hashlib,subprocess,time
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-target-graph-long-v3')
out={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'monotonic_ns':time.monotonic_ns(),'files':{}}
for name in ['semantic-progress.json','semantic-result.json','semantic-phase-status.json','metadata-gate-0.json','metadata-gate-1.json']:
 p=root/name
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<=16*1024**2;x=json.loads(raw);v={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
  if name.startswith('semantic-') and 'cases' in x:v.update(cases_completed=len(x['cases']),cases=[{k:c.get(k) for k in ['semantic_case','case','required_input','required_output','count_pass','pd_lifecycle_pass','semantic_pass']} for c in x['cases']],semantic_pass=x.get('semantic_pass'),attempts=x.get('external_backend_HTTP_attempts'),successful_backends=x.get('successful_backend_inferences'))
  elif name=='semantic-phase-status.json':v.update(client_rc=x.get('client_rc'),error=x.get('error'),gate_pass={k:g.get('pass') for k,g in x.get('gates',{}).items()})
  else:v.update(rc=x.get('rc'),passed=x.get('pass'))
  out['files'][name]=v
print(json.dumps(out))
'''
def main():
 run=ROOT/'runs/20261018T-L10-long-progress-v1';run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write('once')
 dispatch=json.loads((ROOT/'runs/20261018T-L10-sglang-target-graph-long-v3/dispatch.json').read_text());assert all(x==0 for x in dispatch['model_precheck'].values()) and len(dispatch['model_precheck'])==4
 pod=b.NODES[43][0];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.43'
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 result=b.invoke([command],HOST,timeout=45);(run/'node-43.json').write_text(json.dumps(result,indent=2)+'\n');assert result['rc']==0;print(result['stdout'])
if __name__=='__main__':main()
