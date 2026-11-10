#!/usr/bin/env python3
"""Run one bounded private CPU probe in the owned image, never touch service."""
from pathlib import Path
from hashlib import sha256
import json,subprocess
ROOT=Path('audit/execution-search')
REMOTE=r'''
from pathlib import Path,PurePosixPath
from hashlib import sha256
import json,subprocess,sys,time
payload=json.loads(sys.argv[1]);c,=json.loads(subprocess.check_output(['docker','inspect','wio-vllm-v41-repro171-20261102']))
assert c['Id']=='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb' and c['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b' and c['State']['Running']
dest=PurePosixPath(payload['container_root']);assert str(dest).startswith('/data/tiankuan/wio/') and '..' not in dest.parts
mounts=[m for m in c['Mounts'] if m['Type']=='bind' and dest.is_relative_to(PurePosixPath(m['Destination']))]
mount=max(mounts,key=lambda m:len(m['Destination']));host=Path(mount['Source'])/str(dest.relative_to(PurePosixPath(mount['Destination'])))
assert host.resolve().is_relative_to('/data/tiankuan/wio') and not host.exists()
host.mkdir(parents=True)
for n in ['tmp','cache']: (host/n).mkdir()
probe=host/'probe.py';probe.write_text(payload['probe_source']);assert sha256(probe.read_bytes()).hexdigest()==payload['probe_sha256']
(host/'spec.json').write_text(json.dumps(payload['spec'],indent=2)+'\n')
env={'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TMPDIR':str(dest/'tmp'),'XDG_CACHE_HOME':str(dest/'cache')}
cmd=['docker','exec','-w',str(dest)]
for k,v in env.items():cmd.extend(['-e',k+'='+v])
cmd.extend([c['Id'],'python3','-B','-I','-S',str(dest/'probe.py'),str(dest)])
manifest=dict(container_id=c['Id'],image=c['Image'],host_root=str(host),container_root=str(dest),command=cmd,probe_sha256=payload['probe_sha256'],new_NPU_work=0,service_actions=0)
(host/'launch.json').write_text(json.dumps(manifest,indent=2)+'\n')
start=time.monotonic();p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
(host/'controller-process.json').write_text(json.dumps(dict(docker_exec_client_pid=p.pid,controller_pid=__import__('os').getpid(),started_monotonic=start))+'\n')
stdout,stderr=p.communicate()
(host/'stdout.txt').write_text(stdout);(host/'stderr.txt').write_text(stderr)
manifest.update(returncode=p.returncode,elapsed_s=time.monotonic()-start,stdout=stdout,stderr=stderr)
if (host/'started.json').exists():manifest['started']=json.loads((host/'started.json').read_text())
if (host/'result.json').exists():
 raw=(host/'result.json').read_bytes();manifest.update(result=json.loads(raw),result_sha256=sha256(raw).hexdigest())
(host/'receipt.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False))
'''
def main():
    guard=json.loads((ROOT/'common-frame-cpu-import-guard.json').read_text());text='\n'.join(line for e in guard['autoload_guard'] for line in e['lines'])
    assert 'if _is_device_backend_autoload_enabled():' in text and '_import_device_backends()' in text
    deps=json.loads((ROOT/'common-frame-key-dependencies.json').read_text());r=next(r for r in deps['sources'] if r['runtime_path'].endswith('/dsa_cp.py'))
    probe=Path('tools/probe_common_frame_torch_cpu.py');payload=dict(container_root='/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/cpu-frame-run-20261110T104319Z',probe_source=probe.read_text(),probe_sha256=sha256(probe.read_bytes()).hexdigest(),spec=dict(helper_source_path=r['runtime_path'],helper_source_sha256=r['runtime_sha256'],torch_init_sha256=guard['torch_init_sha256']))
    (ROOT/'common-frame-torch-cpu-launch.json').write_text(json.dumps(dict(container_root=payload['container_root'],probe_sha256=payload['probe_sha256'],spec=payload['spec'],scope='CPU correctness/26calllocalcost only; autoload disabled and forbidNPUlibraryguards. Privateowned files only.'),ensure_ascii=False,indent=2)+'\n')
    result=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(payload)],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    d=json.loads(result.stdout);d.update(launcher_sha256=sha256(Path(__file__).read_bytes()).hexdigest());(ROOT/'common-frame-torch-cpu.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(returncode=d['returncode'],host_root=d['host_root'],elapsed_s=d['elapsed_s'],result_status=d.get('result',{}).get('status'),stdout=d['stdout'],stderr=d['stderr']),ensure_ascii=False))
if __name__=='__main__':main()
