"""One SDK payload fixture inside the owned P34 container; no new container/model."""
import ast, hashlib, importlib.util, json
from pathlib import Path

P = Path(__file__).resolve()
def load(name, file, sha):
    path = P.parent / file
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

base = load('shared_pool_v3', 'start_sglang_shared_pool_v3.py', '743e3522c116844697b343dd1c27226430f83fa7cc8970d37753faa8c8d00450')
guard = load('shared_guard', 'glm53_shared_container_precheck_v1.py', '2690417538c821cad345c92f748f24b5afcbf93b1eea5c821b0066289f6af532')
payload = P.parent / 'glm53_sdk_host_readback_fixture_v1.py'
assert hashlib.sha256(payload.read_bytes()).hexdigest() == 'f5bf88adf229d4524bc4949bf993a621e0a37e0fc80dc261c21afd51610f5aae'
cap_path = base.b.ROOT / 'results/sglang-shared-pool-capacity-v2.json'
assert hashlib.sha256(cap_path.read_bytes()).hexdigest() == '821b144afc3b1bef26832a744471b82a70bf6f7fe86afbe102efcbfb05260770'
capacity = json.loads(cap_path.read_text()); assert capacity['pass']
RUN = base.b.ROOT / 'runs/20261017T1722-sdk-host-readback-v1'
ACTION = r'''
r=subprocess.run(['docker','exec',CID,'python3','-c',PAYLOAD],capture_output=True,text=True,timeout=180)
assert r.returncode==0,(r.stdout,r.stderr)
data=json.loads(r.stdout.split('GLM53_SDK_HOST_READBACK=')[-1]);assert data['pass'] and data['close_rc']==0
after=check_npu()
assert json.loads((root/'supervisor-status.json').read_text())['model_started'] is False
print(json.dumps({'project_id':'glm53','container_id':CID,'payload_rc':r.returncode,'payload':data,'stderr':r.stderr,'npu_before':npu_before,'npu_after':after,'model_requests':0}))
'''

def main():
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'execution-once.lock').open('x') as stream: stream.write(str(__import__('os').getpid()))
    action = base.replace(ACTION, {'PAYLOAD': payload.read_text()})
    code = guard.render(34, action, capacity, base.FILES, base.STAGED); ast.parse(code)
    result = base.invoke(34, code, timeout=260)
    (RUN/'node-34.json').write_text(json.dumps(result, indent=2)+'\n')
    (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':result['rc'],'model_requests':0},indent=2)+'\n')
    assert result['rc']==0, result

if __name__ == '__main__':
    main()
