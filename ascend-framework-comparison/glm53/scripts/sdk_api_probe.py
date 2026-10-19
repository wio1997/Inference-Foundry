#!/usr/bin/env python3
"""CPU-only dynamic API probe with host driver mounted strictly read-only."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0215-sdk-api'
CODE=r'''
import os,json,pathlib,sys,traceback
# Disable import-time provisioning using the package's documented absent-home gate.
# No device is exposed; host driver and image root are read-only.
os.environ.pop('ASCEND_HOME_PATH',None)
out={}
try:
 import memfabric_hybrid as mf
 out['module']=mf.__file__
 for name in ['TransferEngine','TransDataOpType','TransferOpcode','create_config_store','set_conf_store_tls','get_last_err_msg']:
  obj=getattr(mf,name,None)
  out[name]={'exists':obj is not None,'doc':getattr(obj,'__doc__',None),'members':{}}
  if name in ['TransferEngine','TransDataOpType','TransferOpcode'] and obj is not None:
   for attr in dir(obj):
    if attr.startswith('_'):continue
    x=getattr(obj,attr);out[name]['members'][attr]={'repr':repr(x),'doc':getattr(x,'__doc__',None)}
 out['maps']=[x.strip() for x in pathlib.Path('/proc/self/maps').read_text().splitlines() if any(t in x.lower() for t in ['libmf_','libhcom','libascend_hal','_pymf_transfer'])]
except Exception:out['error']=traceback.format_exc()
print('GLM53_API_JSON='+json.dumps(out),flush=True)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 for n,image in [(30,'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'),(34,'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263')]:
  pod,hostname=b.NODES[n]
  mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
  assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(n)
  args=['docker','run','--rm','--pull=never','--read-only','--network=none','--pids-limit=64','--memory=1g','--cpus=1','--label','project_id=glm53','--name','glm53-sdk-api-20261017-'+str(n),'-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/cann-9.1.0/lib64','--entrypoint','python3',image,'-c',CODE]
  hc='import subprocess,socket,sys\nassert socket.gethostname()=='+repr(hostname)+'\nr=subprocess.run('+repr(args)+',text=True,capture_output=True,timeout=60)\nprint(r.stdout,end="");print(r.stderr,file=sys.stderr,end="");sys.exit(r.returncode)\n'
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
  r=b.invoke([command],hc,timeout=90);(RUN/('api-%s.json'%n)).write_text(json.dumps(r,indent=2));print(n,r['rc'],flush=True)
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'npu_compute_requests':0,'no_devices':True,'host_driver_readonly':True},indent=2))
if __name__=='__main__':main()
