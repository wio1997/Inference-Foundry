"""Read existing raw profiler metadata and installed analysis API; no SDK import."""
import hashlib
import importlib.util
import json
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RID = '20261019T-L22-native-export-source-v1'
INNER = r'''import pathlib,json,hashlib,sys,sysconfig,importlib.util
root=pathlib.Path('/run/glm53/profile')
folders=sorted(root.glob('*_ascend_pt'));assert len(folders)==16
workers=[]
for folder in folders:
 infos=sorted(folder.glob('profiler_info*.json'));assert len(infos)==1
 metadata=folder/'profiler_metadata.json'
 row={'folder':folder.name,'info':json.loads(infos[0].read_bytes()),'info_sha256':hashlib.sha256(infos[0].read_bytes()).hexdigest(),'metadata':json.loads(metadata.read_bytes()) if metadata.exists() else None}
 row['raw_done_files']=len(list(folder.glob('PROF_*/**/*.done')))
 row['framework_files']=[{'name':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (folder/'FRAMEWORK').glob('*') if p.is_file()]
 workers.append(row)
out={'workers':workers,'source':{},'model_requests':0,'SDK_init_calls':0}
if NODE==43:
 sys.path.extend([sysconfig.get_path('purelib'),sysconfig.get_path('platlib')])
 spec=importlib.util.find_spec('torch_npu');assert spec and spec.origin
 package=pathlib.Path(spec.origin).parent
 out['torch_npu_package']=str(package)
 for relative in ['profiler/profiler.py','profiler/_profiler_interface.py','profiler/profiler_interface.py','profiler/analysis/profiler_analysis.py','profiler/analysis/prof_common_func/_cann_package_manager.py']:
  p=package/relative
  if not p.is_file():out['source'][relative]={'missing':True};continue
  raw=p.read_bytes();assert len(raw)<=1024**2
  out['source'][relative]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
print(json.dumps(out))
'''

def main():
    run = ROOT / 'runs' / RID
    run.mkdir(exist_ok=False)
    invpath = ROOT / 'scripts/inventory_readonly.py'
    assert hashlib.sha256(invpath.read_bytes()).hexdigest() == 'e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
    spec = importlib.util.spec_from_file_location('l22_export_inventory', invpath)
    inv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inv)
    inv.SSH = list(inv.SSH)
    inv.SSH[2] = '/Users/wio/work/sglang-glm53-pd/ssh-control-l20-foreground-20261019'
    cids = {43: '0b3266fcc1f72005a763fdbd5191cecb564d67b448508c2e1d42e965a2973779',
            45: 'c1f460d681f216677d48e3ca881c72bd3a325f10b19d2d83cba299d47dae8a99'}
    receipt = {'project_id': 'glm53', 'model_requests': 0, 'SDK_init_calls': 0,
               'remote_writes': 0, 'max_concurrent_SSH': 1, 'nodes': {}}
    for node, cid in cids.items():
        pod, host = inv.NODES[node]
        mapped = inv.invoke(['kubectl get pod -n default ' + shlex.quote(pod) + " -o jsonpath='{.status.hostIP}'"], timeout=15)
        (run / f'mapping-{node}.json').write_text(json.dumps(mapped) + '\n')
        assert mapped['rc'] == 0 and mapped['stdout'] == f'172.16.12.{node}'
        inner = f'NODE={node}\n' + INNER
        code = f'''import socket,subprocess,json,pathlib,hashlib
assert socket.gethostname()=={host!r}
cid={cid!r}
p=subprocess.run(['docker','inspect',cid,'--format','{{{{.Id}}}} {{{{.State.Running}}}} {{{{.Image}}}}'],text=True,capture_output=True,timeout=10)
assert p.returncode==0 and p.stdout.split()==[cid,'true','sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83']
p=subprocess.run(['docker','exec','-i',cid,'python3','-S','-'],input={inner!r},text=True,capture_output=True,timeout=30)
result={{'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'tails':{{}}}}
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1766-vllm-shared-pool-v4')
for name in ['model.stdout.txt','model.stderr.txt']:
 path=root/name
 if path.exists():
  size=path.stat().st_size
  with path.open('rb') as stream:stream.seek(max(0,size-128*1024));raw=stream.read(128*1024)
  result['tails'][name]={{'offset':max(0,size-128*1024),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode(errors='replace')}}
print(json.dumps(result))
'''
        command = shlex.join(['kubectl', 'exec', '-i', '-n', 'default', pod, '-c', 'hostctl', '--', 'nsenter', '-t', '1', '-m', '--', 'python3', '-'])
        wire = inv.invoke([command], code, timeout=60)
        (run / f'node-{node}-wire.json').write_text(json.dumps(wire) + '\n')
        receipt['nodes'][str(node)] = {'wire_rc': wire['rc']}
        if wire['rc'] == 0:
            value = json.loads(wire['stdout'])
            receipt['nodes'][str(node)]['inner_rc'] = value['rc']
            (run / f'node-{node}-host.json').write_text(json.dumps(value) + '\n')
            if value['rc'] == 0:
                data = json.loads(value['stdout'])
                (run / f'node-{node}-metadata-source.json').write_text(json.dumps(data, indent=2) + '\n')
                receipt['nodes'][str(node)]['workers'] = len(data['workers'])
                source = run / 'source'
                source.mkdir(exist_ok=True)
                for name, item in data['source'].items():
                    if item.get('text'):
                        raw = item['text'].encode()
                        assert hashlib.sha256(raw).hexdigest() == item['sha256']
                        (source / name.replace('/', '__')).write_bytes(raw)
    receipt['pass'] = all(x.get('wire_rc') == x.get('inner_rc') == 0 and x.get('workers') == 16 for x in receipt['nodes'].values())
    (run / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
