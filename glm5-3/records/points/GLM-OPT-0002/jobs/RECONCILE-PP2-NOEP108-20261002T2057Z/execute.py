from pathlib import Path
import json,ast,sys,subprocess,shlex,hashlib,urllib.request
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc,same_process
j=Path(__file__).parent;p=j.parents[1];r=p/'runs/GLM-RUN-0108'
s=json.loads((r/'state.json').read_text());assert s['status']=='failed'and not same_process(s['owner'])
spec=json.loads((r/'controller_spec.json').read_text())
for x in spec['stages'][0]['sources']:assert Path(x['path']).read_bytes()==Path(x['snapshot']).read_bytes()and hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
tree=ast.parse((r/'deploy.py').read_text());codes=[n.value.value for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='code'for t in n.targets)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str)]
code=next(c for c in codes if 'API_alive' in c and 'unknown=' in c)
owners=json.loads((r/'startup_model_identities.json').read_text());out={}
for key,o in owners.items():
 args=['python3','-c',code]
 if o['host']=='167':args=['ssh','-o','BatchMode=yes','root@172.16.10.167',shlex.join(args)]
 z=subprocess.run(args,input=json.dumps(o).encode(),capture_output=True,timeout=100);(j/(key+'.stdout')).write_bytes(z.stdout);(j/(key+'.stderr')).write_bytes(z.stderr);z.check_returncode();v=json.loads(z.stdout);assert v['API_alive']and v['API_owner']==o and len(v['npu_worker_pids'])==16
 out[key]=v
assert owners['D0']==json.loads((r.parent/'GLM-RUN-0089/adopted_model_identities.json').read_text())['D0']
http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with http.open('http://172.16.10.166:9081/metrics',timeout=10)as res:b=res.read()
(j/'P.metrics').write_bytes(b)
import re
assert all(float(x)==0 for x in re.findall(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)',b.decode(),re.M))
atomic_json(j/'native_member_identities.json',out)
reduction=dict(at=utc(),state=s,source_pins=len(spec['stages'][0]['sources']),owners=owners,native_members=out,P89same=True,models=0,requests=0,signals=0,limits=['Read-only exact108 failedAPI ancestry NPU16/boot-start/argv snapshot; no cleanup or readyD claim','P89 same API and idle; no new inference'])
atomic_json(j/'reduction.json',reduction);b=(j/'reduction.json').read_bytes()
atomic_json(j/'result.json',dict(schema_version=1,job_id=j.name,status='completed',summary='Read-only failedD108/P89 exactAPI ancestry/NPU16 each snapshot; P89idle, no signals or inference',execution=dict(inner_exit_code=0,acceptance='passed',processes=[]),findings=[],evidence=[dict(id='reduction',path=str(j/'reduction.json'),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator='exact108startupowners/ancestry/NPU/boot-start')],unknowns=reduction['limits'],decision_request=None,next_check_at=None))
print(json.dumps(dict(NPU_workers={k:len(v['npu_worker_pids'])for k,v in out.items()},signals=0)))
