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
RUN = b.ROOT / 'runs/20261017T1030-stage-vllm-graph-v4'
DEST = '/data/wio/inference-foundry/glm53/overlays/vllm-sdma-v5-20261017'
IMAGE = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
MANIFEST_SHA = 'e82da89d4e5fcec74f6783e237a90640e04dd7a857e243fc5c07e4e193016d6f'
manifest_data = (b.ROOT / 'config/vllm-overlay-stage-manifest-v5.json').read_bytes()
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
        receipt['phase']='STAGING_ONLY_PREFLIGHT_DEFERRED_UNTIL_OWN_MODELS_IDLE_AND_STOPPED'
    finally:
        (RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
