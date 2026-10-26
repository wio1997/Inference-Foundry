#!/usr/bin/env python3
"""Native config/graph selector validation only after owned idle shutdown."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'stage_vllm_graph_audit_v4.py';assert hashlib.sha256(prior.read_bytes()).hexdigest()=='b2329bcf0a6e1c93616ac43b44ce77beb5c35c9a9d15840b3d987c19276e3c3b'
s=importlib.util.spec_from_file_location('stage',prior);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
b=m.b;DEST=m.DEST;IMAGE=m.IMAGE
RUN=b.ROOT/'runs/20261017T1090-vllm-graph-preflight-v4'
def preflight():
    node = 30; pod, hostname = b.NODES[node]
    mapping = b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
    assert mapping['rc']==0 and mapping['stdout']=='172.16.12.30'
    source = json.loads((b.ROOT/'config/overlays/vllm-sdma-v4/manifest.json').read_text())
    args = ['docker','run','--rm','--pull=never','--network=none','--shm-size=1g','--memory=6g','--cpus=4','--pids-limit=512','--tmpfs','/run/glm53:rw,size=64m',
            '--label','project_id=glm53','--name','glm53-vllm-graph-preflight-v4-20261017-30',
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
 stopped=json.loads((b.ROOT/'runs/20261017T1060-stop-vllm-correctness-v3/receipt.json').read_text());assert stopped['meta_stop']==0 and all(stopped['stops'].get(str(n))==0 for n in (30,31,43,45))
 staged=json.loads((b.ROOT/'runs/20261017T1030-stage-vllm-graph-v4/receipt.json').read_text());assert all(v==0 for v in staged['nodes'].values())
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 r=preflight();(RUN/'preflight-30.json').write_text(json.dumps(r,indent=2)+'\n')
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'preflight_rc':r['rc'],'nodes':staged['nodes'],'model_requests':0},indent=2)+'\n');print('graph native preflight',r['rc'],flush=True)
if __name__=='__main__':main()
