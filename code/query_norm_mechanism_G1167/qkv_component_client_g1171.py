from pathlib import Path
import datetime,hashlib,json,os,signal,time,urllib.request
R=Path('/task/client_ds41_g1122');O=R/'qkv_component_g1171';O.mkdir(exist_ok=False);start=time.monotonic()
def save(name,v):(O/name).write_text(json.dumps(v,indent=2)+'\n')
def state(phase,**more):save('state.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),phase=phase,elapsed_s=time.monotonic()-start,**more))
def alarm(*_):raise TimeoutError('240s oneprefix operand observation bound')
signal.signal(signal.SIGALRM,alarm);signal.alarm(240)
body=json.loads(Path('/task/client_ds41_g1122/same_prefix_layers_g1149/SG_request.json').read_text());save('request.json',body);state('request_inflight')
try:
 q=urllib.request.Request('http://<ROLE_HOST>:19094/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=220) as f:b=f.read();assert f.status==200
 (O/'response.json').write_bytes(b);v=json.loads(b);assert v['meta_info']['prompt_tokens']==30 and v['meta_info']['completion_tokens']==1
 save('result.json',{'response':v,'response_SHA256':hashlib.sha256(b).hexdigest(),'scope':'Oneexactmath1 firsttoken request, twoexistingobserverpathsarmed; numericaldiagnostic only, noquality/performanceacceptance'});state('complete_requires_parent_review')
except BaseException as e:state('failed_requires_review',error=repr(e));raise
