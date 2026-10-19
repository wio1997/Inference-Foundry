"""Once-only accepted prefix staging; relay-local byte pipe, no SDK/model calls."""
import hashlib, importlib.util, json, shlex
from pathlib import Path

P=Path(__file__).resolve()
def load(name,file,sha):
    path=P.parent/file;assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('inventory','inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
transfer=load('prefix_transfer','glm53_prefix_transfer_v1.py','688bf6a58d263c0ba1b897e9815a8ed8c0b2acc93a66d6669d541a6057d448b5')
MANIFESTS={16:'dff4e3943c14f8d12c1964f0aad9ed419441e6b94274f40507d2fdd34f4e3ab9',32:'9bdba915c372eb2c23da018a18c2c9653ded9c5012d875db8b496effb5983b67',64:'c4a6222088e7a8d7b9fafd1666e58601f3a0cf5055350871b1cb6a60aa7145d3'}
RUNS={16:'20261017T1760-sglang-formal-C16-v1',32:'20261017T1770-sglang-formal-C32-v1',64:'20261017T1780-sglang-formal-C64-v1'}
RUN=base.ROOT/'runs/20261017T1742-stage-formal-prefixes-v1'

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as file:file.write(str(__import__('os').getpid()))
    result={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),
            'transfers':{},'model_requests':0,'SDK_init_calls':0,'new_containers':0}
    try:
        for C,manifest in MANIFESTS.items():
            for node in [43,34]:
                pod=base.NODES[node][0]
                checked=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
                assert checked['rc']==0 and checked['stdout']==f'172.16.12.{node}'
            command,destination=transfer.render(C,manifest,RUNS[C],base.NODES[43][0],base.NODES[34][0])
            response=base.invoke([command],timeout=180)
            (RUN/f'transfer-C{C}.json').write_text(json.dumps(response,indent=2)+'\n')
            result['transfers'][str(C)]={'rc':response['rc'],'destination':destination}
            assert response['rc']==0,response
            details=json.loads(response['stdout']);assert details['concurrency']==C and details['manifest_sha256']==manifest
            assert len(details['files'])==C+1
    finally:
        (RUN/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
