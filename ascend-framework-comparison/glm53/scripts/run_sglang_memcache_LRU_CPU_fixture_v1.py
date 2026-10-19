"""Once-only exact-image CPU SDK experiment in a separate network namespace."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[1]
RID = '20261017T2019-sglang-memcache-LRU-CPU-v1'
CORE = 'scripts/glm53_memcache_LRU_CPU_fixture_v1.py'
CORE_SHA = '22caec568f1bcdce2b1150ebeff7807df49c4acd8a97135a71a6f7505d5388db'


def load(name, relative, sha):
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load('inventory', 'scripts/inventory_readonly.py', 'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
codec = load('codec', 'scripts/glm53_native_output_transport_v1.py', 'ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
HOST = r'''import pathlib,subprocess,json,gzip,hashlib,socket
assert socket.gethostname()==HOSTNAME
def cmd(args,timeout=30):
 result=subprocess.run(args,capture_output=True,text=True,timeout=timeout)
 return {'rc':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
identity=cmd(['docker','inspect',CURRENT]);assert identity['rc']==0
production=json.loads(identity['stdout'])[0];assert production['Id']==CURRENT and production['Image']==IMAGE and production['State']['Running'] and production['Config']['Labels']['project_id']=='glm53' and production['Config']['Labels']['run_id']=='20261017T1954-sglang-metadata-v1'
archive=pathlib.Path(OUTDIR);assert not archive.exists();archive.mkdir(parents=True)
source=archive/'fixture.py';source.write_text(SOURCE);assert hashlib.sha256(source.read_bytes()).hexdigest()==SOURCE_SHA
assert cmd(['docker','inspect',NAME])['rc']!=0
args=['docker','run','--pull=never','--network=none','--cpus=2','--memory=2g','--pids-limit=512','--shm-size=256m','--ulimit','memlock=-1','--label','project_id=glm53','--label','run_id='+RID,'--name',NAME,'-v',str(archive)+':/run/glm53','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','--entrypoint','python3',IMAGE,'/run/glm53/fixture.py']
try:executed=cmd(args,240)
except subprocess.TimeoutExpired:
 stopped=cmd(['docker','stop','--time','15',NAME],25)
 executed={'rc':124,'error':'Isolated CPU fixture exceeded 240s','stop':stopped}
out={'project_id':'glm53','model_requests':0,'experiment':RID,'SDK_scope':'isolated host_shm CPU metadata/LRU experiment; not model SDMA qualification','execution':executed}
container=cmd(['docker','inspect',NAME]);assert container['rc']==0
info=json.loads(container['stdout'])[0];assert info['Image']==IMAGE and info['Config']['Labels']['project_id']=='glm53' and info['Config']['Labels']['run_id']==RID and not info['HostConfig']['Devices'] and not info['HostConfig'].get('DeviceRequests') and info['HostConfig']['NetworkMode']=='none'
assert not info['State']['Running'];out['isolated_container']={key:info[key] for key in ['Id','Name','Image','State']};out['no_NPU_devices']=True;out['network_mode']='none'
for stream in ['stdout','stderr']:
 raw=executed.get(stream,'').encode();assert len(raw)<=16*1024**2
 dest=archive/('fixture-'+stream+'.gz');dest.write_bytes(gzip.compress(raw,mtime=0));out['execution'][stream+'_archive']={'path':str(dest),'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()};out['execution'][stream]=raw[-12000:].decode(errors='replace')
result=archive/'result.json'
if result.is_file():out['fixture_result']=json.loads(result.read_text())
out['case_results']=[]
for mode in ['control','exist_touch']:
 path=archive/mode/'result.json'
 if path.is_file():out['case_results'].append(json.loads(path.read_text()))
after=cmd(['docker','inspect',CURRENT]);assert after['rc']==0
production_after=json.loads(after['stdout'])[0];assert production_after['Id']==CURRENT and production_after['State']['Running'] and production_after['State']['StartedAt']==production['State']['StartedAt'];out['production_container_unchanged']=True
'''


def render(source):
    values = {'HOSTNAME': base.NODES[34][1], 'CURRENT': 'dc23a47a7050f1ad3fb958846fe7bec789a87ec7775f4e715536a6602a6c68d3', 'IMAGE': 'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263', 'OUTDIR': '/data/wio/inference-foundry/glm53/runs/' + RID, 'SOURCE': source, 'SOURCE_SHA': CORE_SHA, 'RID': RID, 'NAME': 'glm53-LRU-CPU-v1-20261017-34'}
    code = re.sub(r'\b(?:' + '|'.join(values) + r')\b', lambda match: repr(values[match[0]]), HOST) + codec.ENCODER
    ast.parse(code)
    return code


def main():
    source = (ROOT / CORE).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == CORE_SHA
    ast.parse(source)
    prior = json.loads((ROOT / 'runs/20261017T2018-sglang-C64-native-cache-source-v1/receipt.json').read_text())
    assert prior['nodes'] == {'34': 0}
    run = ROOT / 'runs' / RID
    run.mkdir(exist_ok=True)
    with (run / 'execution-once.lock').open('x') as file:
        file.write(str(__import__('os').getpid()))
    receipt = {'project_id': 'glm53', 'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'core_sha256': CORE_SHA, 'model_requests': 0, 'nodes': {}}
    try:
        pod, _ = base.NODES[34]
        mapped = base.invoke(['kubectl get pod -n default ' + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"], timeout=30)
        assert mapped['rc'] == 0 and mapped['stdout'] == '172.16.12.34'
        command = shlex.join(['kubectl', 'exec', '-i', '-n', 'default', pod, '-c', 'hostctl', '--', 'nsenter', '-t', '1', '-m', '--', 'python3', '-'])
        wire = base.invoke([command], render(source), timeout=290)
        result = codec.decode(wire) if wire['rc'] == 0 else wire
        (run / 'node-34.json').write_text(json.dumps(result, indent=2) + '\n')
        receipt['nodes']['34'] = result['rc']
        print('ISOLATED_CPU_CAPTURE_RC=' + str(result['rc']), flush=True)
    finally:
        (run / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
