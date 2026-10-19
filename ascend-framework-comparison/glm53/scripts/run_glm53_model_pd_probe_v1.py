#!/usr/bin/env python3
"""One bounded first real-model probe on existing exact owned services."""
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0520-model-pd-probe-v1'
PAYLOAD=(P.parent/'probe_glm53_model_pd_payload.py').read_text()
PAYLOAD_SHA='a1e05aafe53ce26cf6294aa26ad0f3969e914a4b403e28c2054509c9a5669d09'
assert hashlib.sha256(PAYLOAD.encode()).hexdigest()==PAYLOAD_SHA
CID='a522315f51028f930cb140058884763ca4c6ed0a5a9d369cb837039a9129209a'
CODE=r'''
import pathlib,socket,subprocess,json,sys,urllib.request,time
assert socket.gethostname()==EXPECTED
raw=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=15)
assert raw.returncode==0;data=json.loads(raw.stdout)[0]
assert data['Id']==CID and data['Name']=='/glm53-vllm-correctness-v1-20261017-30'
assert data['Config']['Labels']['project_id']=='glm53' and data['State']['Running']
assert data['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
health={}
for node in (30,31,43,45):
 port=30300 if node in (30,31) else 30400
 with urllib.request.urlopen(f'http://172.16.12.{node}:{port}/health',timeout=3) as response:
  health[node]=response.status;assert response.status==200
print('GLM53_HEALTH_GATE='+json.dumps(health),flush=True)
logger_code='import pathlib,json,hashlib;p=pathlib.Path("/vllm-workspace/vllm/vllm/logger.py");d=p.read_bytes();print(json.dumps({"sha256":hashlib.sha256(d).hexdigest(),"text":d.decode()}))'
source=subprocess.run(['docker','exec',CID,'python3','-c',logger_code],capture_output=True,text=True,timeout=20)
print('GLM53_LOGGER_SOURCE='+json.dumps({'rc':source.returncode,'stdout':source.stdout,'stderr':source.stderr}),flush=True)
try:
 result=subprocess.run(['docker','exec','-i',CID,'python3','-'],input=PAYLOAD,text=True,capture_output=True,timeout=780)
 print(result.stdout,end='');print(result.stderr,file=sys.stderr,end='');rc=result.returncode
except subprocess.TimeoutExpired as e:
 for raw,stream in [(e.stdout,sys.stdout),(e.stderr,sys.stderr)]:
  if raw:print(raw.decode(errors='replace') if isinstance(raw,bytes) else raw,file=stream,end='')
 print(str(e),file=sys.stderr);rc=124
print('GLM53_PROBE_PROCESS_RC='+str(rc),flush=True)
sys.exit(rc)
'''


def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,hostname=b.NODES[30]
 m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert m['rc']==0 and m['stdout']=='172.16.12.30'
 code=CODE.replace('EXPECTED',repr(hostname)).replace('CID',repr(CID)).replace('PAYLOAD',repr(PAYLOAD))
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 result=b.invoke([command],code,timeout=850)
 (RUN/'client-30.json').write_text(json.dumps(result,indent=2)+'\n')
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),
          'payload_sha256':PAYLOAD_SHA,'rc':result['rc'],'client_result':None,
          'qualification':'OUTPUT_PROBE_ONLY_NOT_PD_STATE_OR_PERFORMANCE_ACCEPTANCE'}
 for line in result['stdout'].splitlines():
  if line.startswith('GLM53_CLIENT_RESULT='):receipt['client_result']=json.loads(line.split('=',1)[1])
 (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('probe',result['rc'],flush=True)


if __name__=='__main__':main()
