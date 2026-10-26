#!/usr/bin/env python3
"""Stage diagnostics for rc139; one die, no model or network transfer."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0225-runtime-smoke'
CODE=r'''
import faulthandler,os,json,sys,pathlib
faulthandler.enable(all_threads=True)
def mark(x,**kw): print('GLM53_STAGE='+json.dumps({'stage':x,**kw}),flush=True)
mark('before_torch',env={k:os.environ.get(k) for k in ['ASCEND_HOME_PATH','ASCEND_OPP_PATH','LD_LIBRARY_PATH','PATH','ASCEND_RT_VISIBLE_DEVICES']},devices=[str(x) for x in pathlib.Path('/dev').glob('*') if any(t in x.name for t in ['davinci','devmm','hdc'])])
import torch
mark('torch_imported',version=torch.__version__)
import torch_npu
mark('torch_npu_imported',version=torch_npu.__version__)
mark('before_set_device')
torch.npu.set_device(0)
mark('device_set',count=torch.npu.device_count(),name=torch.npu.get_device_name(0))
t=torch.arange(1024,dtype=torch.float32).npu()
torch.npu.synchronize()
assert float(t.sum().cpu())==523776.0
mark('tensor_correct',bytes=t.numel()*t.element_size())
# Only remove provisioning-home immediately before package import, after NPU runtime initialization.
home=os.environ.pop('ASCEND_HOME_PATH',None)
mark('before_memfabric')
import memfabric_hybrid as mf
if home:os.environ['ASCEND_HOME_PATH']=home
mark('memfabric_imported',sdma=mf.TransferEngine.TransDataOpType.SDMA.value)
mark('runtime_pass')
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 n=30;pod,hostname=b.NODES[n]
 mapping=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 assert mapping['rc']==0 and mapping['stdout']=='172.16.12.30'
 args=['docker','run','--rm','--pull=never','--network=none','--shm-size=1g','--pids-limit=256','--memory=8g','--cpus=4','--ulimit','memlock=-1','--label','project_id=glm53','--name','glm53-runtime-smoke-20261017-30']
 for d in ['/dev/davinci0','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:args+=['--device',d]
 args+=['-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','ASCEND_RT_VISIBLE_DEVICES=0','--entrypoint','timeout','sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83','60','python3','-c',CODE]
 hc='''import socket,subprocess,sys,json,pathlib
assert socket.gethostname()==HOSTNAME
before=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
assert before.returncode==0 and before.stdout.count('No running processes')==8
lock=pathlib.Path('/data/wio/inference-foundry/glm53/locks/runtime-smoke-20261017.once')
with lock.open('x') as f:f.write('glm53 one-shot stage probe')
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=90)
print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='')
after=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
print('GLM53_AFTER='+json.dumps({'rc':after.returncode,'npu':after.stdout}),flush=True)
sys.exit(r.returncode)
'''.replace('HOSTNAME',repr(hostname)).replace('ARGS',repr(args))
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 r=b.invoke([command],hc,timeout=150);(RUN/'node-30.json').write_text(json.dumps(r,indent=2));print(r['rc'],flush=True)
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0,'NPU_tensor_bytes':4096},indent=2))
if __name__=='__main__':main()
