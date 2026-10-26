"""Read exactly two existing, frozen SGLang traces; no model or new profile."""
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import shlex
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RID = '20261019T-L18-existing-SG-native-trace-pair-v1'
REVIEW = 'results/C32-L18-existing-trace-transfer-review-v1.json'
SOCKET = '/Users/wio/work/sglang-glm53-pd/ssh-control-l17-recovery-20261019'
PROFILE_ROOT = '/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v3/profile/20261019T-L13-sglang-C32-D-native-decode-two-step-profile-v1/'
HOST = r'''import pathlib,socket,hashlib,json,gzip,base64
assert socket.gethostname()==__HOSTNAME__
p=pathlib.Path(__PATH__)
root=pathlib.Path('/data/wio/inference-foundry/glm53')
assert p.resolve(strict=True).is_relative_to(root.resolve(strict=True))
cursor=p
while cursor!=root:
 assert not cursor.is_symlink()
 cursor=cursor.parent
before=p.stat();assert before.st_size==__BYTES__ and before.st_mtime_ns==__MTIME__
raw=p.read_bytes();assert len(raw)==__BYTES__ and hashlib.sha256(raw).hexdigest()==__SHA__
after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
packed=gzip.compress(raw,mtime=0)
print(json.dumps({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime_ns':after.st_mtime_ns,'packed_sha256':hashlib.sha256(packed).hexdigest(),'gzip_base64':base64.b64encode(packed).decode(),'model_requests':0,'new_profiles':0,'remote_writes':0}))
'''


def plans():
    inv_path=ROOT/'scripts/inventory_readonly.py'
    assert hashlib.sha256(inv_path.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
    spec=importlib.util.spec_from_file_location('l18_inventory',inv_path)
    inv=importlib.util.module_from_spec(spec);spec.loader.exec_module(inv)
    inv.SSH=list(inv.SSH);inv.SSH[2]=SOCKET
    out={}
    for node in [36,41]:
        old=json.loads((ROOT/f'runs/20261019T-L15-sglang-existing-D-timeline-source-v1/node-{node}.json').read_bytes())
        assert old['rc']==0
        old=json.loads(old['stdout'])['timeline']
        expected=old['file']['expected'];relative=old['relative']
        assert '..' not in Path(relative).parts and relative.endswith('/ASCEND_PROFILER_OUTPUT/trace_view.json')
        assert expected['bytes']<=24*1024**2
        code=HOST
        for k,v in {'HOSTNAME':inv.NODES[node][1],'PATH':PROFILE_ROOT+relative,'BYTES':expected['bytes'],'MTIME':expected['mtime_ns'],'SHA':expected['sha256']}.items():
            code=code.replace('__'+k+'__',repr(v))
        compile(code,'host','exec')
        out[node]={'code':code,'sha256':hashlib.sha256(code.encode()).hexdigest(),'expected':expected}
    return inv,out


def main():
    inv,plan=plans()
    review=json.loads((ROOT/REVIEW).read_bytes())
    assert review['collector_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert review['host_codes_sha256']=={str(n):v['sha256'] for n,v in plan.items()}
    assert Path(SOCKET).is_socket()
    run=ROOT/'runs'/RID;run.mkdir(exist_ok=False)
    (run/'execution-once.lock').write_text(str(os.getpid()))
    start=time.monotonic();receipt={'project_id':'glm53','run_id':RID,'model_requests':0,'new_profiles':0,'SDK_init_calls':0,'remote_writes':0,'max_concurrent_SSH':1,'nodes':{},'pass':False}
    try:
        for node,item in plan.items():
            assert time.monotonic()-start<150,'total collection time budget'
            pod=inv.NODES[node][0]
            mapping=inv.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=20)
            (run/f'mapping-{node}.json').write_text(json.dumps(mapping)+'\n')
            assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
            command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
            wire=inv.invoke([command],item['code'],timeout=60)
            (run/f'node-{node}-wire.json').write_text(json.dumps(wire)+'\n')
            receipt['nodes'][str(node)]={'rc':wire['rc']};assert wire['rc']==0
            packed=json.loads(wire['stdout']);data=base64.b64decode(packed['gzip_base64'],validate=True)
            assert hashlib.sha256(data).hexdigest()==packed['packed_sha256']
            raw=gzip.decompress(data);assert len(raw)==item['expected']['bytes'] and hashlib.sha256(raw).hexdigest()==item['expected']['sha256']
            (run/f'node-{node}-trace.json').write_bytes(raw)
            receipt['nodes'][str(node)].update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        receipt['pass']=True
    finally:
        receipt['elapsed_seconds']=time.monotonic()-start
        (run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
