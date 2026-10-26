"""One read-only D36 snapshot explaining completed profile control HTTP500."""
import ast,hashlib,importlib.util,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/20261017T1936-sglang-D-profile-stop-diagnostic-v1'
def load(name,path,sha):
 p=ROOT/path;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('base','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
identity=load('identity','scripts/glm53_shared_model_identity_v1.py','e89e2a6a45878328ea0772849bf629a76a177abbfb58fb27988c36c8cf6a6ec4')
codec=load('codec','scripts/glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
HOST=r'''
import pathlib,json,time,urllib.request
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1720-sglang-shared-pool-v2');prof=root/'profile/20261017T1920-sglang-formal-C16-D-profile-v1';archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1936-sglang-D-profile-stop-diagnostic-v1');assert not archive.exists();archive.mkdir(parents=True)
out={'project_id':'glm53','node':36,'model_requests':0,'SDK_init_calls':0,'new_profile_control_requests':0,'model_identity':model_identity,'profile_exists':prof.is_dir(),'profile_file_manifest':[],'profile_log_markers':[],'model_supervisor':json.loads((root/'supervisor-status.json').read_text())}
if prof.is_dir():
 for path in sorted(prof.rglob('*')):
  if path.is_file():
   assert not path.is_symlink() and len(out['profile_file_manifest'])<8192
   out['profile_file_manifest'].append({'relative':str(path.relative_to(prof)),'bytes':path.stat().st_size})
for stream in ['stdout','stderr']:
 with (root/('model.'+stream+'.txt')).open(errors='replace') as file:
  for line in file:
   if any(x in line for x in ['Profiling starts','Profiling done','Stop profiling','Profiling is not in progress','Failed to stop','Failed to start','Traceback','ERROR','Error in ASGI','RuntimeError','ValueError']):
    assert len(out['profile_log_markers'])<2048;out['profile_log_markers'].append({'stream':stream,'line':line.rstrip()[:6000]})
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for endpoint in ['/health','/metrics']:
 with opener.open('http://172.16.12.36:30600'+endpoint,timeout=10) as response:raw=response.read(8*1024**2+1);assert len(raw)<=8*1024**2
 out[endpoint]={'status':response.status,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 if endpoint=='/metrics':out[endpoint]['occupancy_lines']=[line for line in raw.decode().splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]
'''
def main():
 prior=ROOT/'runs/20261017T1920-sglang-formal-C16-D-profile-v1';receipt=json.loads((prior/'D-profile-control-receipt.json').read_text());assert receipt['rc']==0;control=json.loads(receipt['stdout']);assert control['profile_start_attempts']==control['profile_stop_attempts']==1 and control['stop_response']['status']==500
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 node=36;pod,host=base.NODES[node];mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.36'
 env={'CID':'0d02ac7770dab23c14a55fe18b0dc8f7d5999b455926403b91227b9d95bcf20d','NAME':'glm53-sglang-shared-v2-20261017-36','IMAGE':'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263','NODE':36}
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+''.join(k+'='+repr(v)+'\n' for k,v in env.items())+identity.HOST+HOST+codec.ENCODER;ast.parse(code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);result=codec.decode(base.invoke([command],code,timeout=120));(RUN/'node-36.json').write_text(json.dumps(result,indent=2)+'\n');print('node36',result['rc'],flush=True)
if __name__=='__main__':main()
