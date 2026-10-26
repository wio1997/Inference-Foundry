#!/usr/bin/env python3
"""Once-only private source overlay staging and exact-image config preflight."""
import base64
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

P = Path(__file__).resolve()
prior = P.parent / 'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest() == 'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
spec = importlib.util.spec_from_file_location('base', prior)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
RUN = b.ROOT / 'runs/20261017T0420-vllm-overlay-preflight-v1'
DEST = '/data/wio/inference-foundry/glm53/overlays/vllm-sdma-v1-20261017'
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
MANIFEST_SHA = '7a9cfac5d139cdecb32cc024632ad515a7b862c2e3e2b16e191f9dffef4af1de'
manifest_data = (b.ROOT / 'config/vllm-overlay-stage-manifest-v1.json').read_bytes()
assert hashlib.sha256(manifest_data).hexdigest() == MANIFEST_SHA
FILES = json.loads(manifest_data)
for entry in FILES.values():
    assert hashlib.sha256((b.ROOT / entry['local']).read_bytes()).hexdigest() == entry['sha256']


def stage(node):
    pod, hostname = b.NODES[node]
    mapping = b.invoke(['kubectl get pod -n default ' + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"], timeout=30)
    assert mapping['rc'] == 0 and mapping['stdout'] == '172.16.12.' + str(node)
    bundle = {name: {**entry, 'base64': base64.b64encode((b.ROOT / entry['local']).read_bytes()).decode()}
              for name, entry in FILES.items()}
    host_code = '''import pathlib,hashlib,socket,subprocess,json,base64,sys
assert socket.gethostname()==EXPECTED
result=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,timeout=15)
assert result.returncode==0 and result.stdout.strip()==IMAGE
dest=pathlib.Path(DEST);assert not dest.exists(),'No overwrite/replay'
bundle=json.loads(sys.stdin.read())
for name,entry in bundle.items():
 rel=pathlib.PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
 data=base64.b64decode(entry['base64']);assert hashlib.sha256(data).hexdigest()==entry['sha256']
dest.mkdir(parents=True)
for name,entry in bundle.items():
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(entry['base64']))
print(json.dumps({'dest':str(dest),'host':socket.gethostname(),'files':len(bundle),'staged':True}))
'''.replace('EXPECTED', repr(hostname)).replace('IMAGE', repr(IMAGE)).replace('DEST', repr(DEST))
    command = shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',host_code])
    return b.invoke([command], json.dumps(bundle), timeout=90)


def preflight():
    node = 30; pod, hostname = b.NODES[node]
    mapping = b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
    assert mapping['rc']==0 and mapping['stdout']=='172.16.12.30'
    source = json.loads((b.ROOT/'config/overlays/vllm-sdma-v1/manifest.json').read_text())
    args = ['docker','run','--rm','--pull=never','--network=none','--shm-size=1g','--memory=6g','--cpus=4','--pids-limit=512',
            '--label','project_id=glm53','--name','glm53-vllm-overlay-preflight-v1-20261017-30',
            '-v',DEST+'/source:/opt/glm53/source:ro','-v',DEST+'/config:/opt/glm53/config:ro',
            '-v',DEST+'/preflight.py:/opt/glm53/preflight.py:ro',
            '-v','/data/wio/inference-foundry/glm53/overlays/sdk13-20261017:/opt/glm53/sdk13:ro',
            '-v','/data/wio/models/GLM-5.3-w8a8:/data/wio/models/GLM-5.3-w8a8:ro',
            '-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
            '-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
            '-e','VLLM_HOST_IP=172.16.12.30','-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1']
    for entry in source['files']:
        args += ['-v',DEST+'/source/'+entry['local']+':'+entry['container_path']+':ro']
    for dev in [f'/dev/davinci{i}' for i in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:
        args += ['--device', dev]
    bootstrap = '''set -eu
export PYTHONPATH=/opt/glm53/source/python:/opt/glm53/sdk13:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}
export PYTHONDONTWRITEBYTECODE=1
export VLLM_CACHE_ROOT=/tmp/glm53-cache
exec timeout 180 python3 /opt/glm53/preflight.py
'''
    args += ['--entrypoint','/bin/bash',IMAGE,'-c',bootstrap]
    code = '''import socket,subprocess,sys,json
assert socket.gethostname()==EXPECTED
before=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
assert before.returncode==0 and before.stdout.count('No running processes')==8,'NPU occupied; no compute changes'
r=subprocess.run(ARGS,capture_output=True,text=True,timeout=200)
print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='')
after=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30)
print('GLM53_POSTCHECK_JSON='+json.dumps({'rc':after.returncode,'npu_after':after.stdout}),flush=True)
sys.exit(r.returncode)
'''.replace('EXPECTED',repr(hostname)).replace('ARGS',repr(args))
    command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
    return b.invoke([command],code,timeout=260)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),
             'manifest_sha256':MANIFEST_SHA,'nodes':{},'model_requests':0,'preflight_rc':None}
    try:
        for node in (30,31,43,45):
            result=stage(node);(RUN/f'node-{node}.json').write_text(json.dumps(result,indent=2))
            receipt['nodes'][node]=result['rc'];print('stage',node,result['rc'],flush=True)
            assert result['rc']==0,'Stage failed; preserve partial state, do not replay'
        result=preflight();(RUN/'preflight-30.json').write_text(json.dumps(result,indent=2))
        receipt['preflight_rc']=result['rc'];print('preflight',result['rc'],flush=True)
    finally:
        (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
