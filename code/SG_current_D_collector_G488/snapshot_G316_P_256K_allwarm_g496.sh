if [ "$(hostname)" = "<ROLE_NODE>" ]; then
python3 - <<'REMOTE_PY'
from pathlib import Path
import datetime,hashlib,json,socket,subprocess
assert socket.gethostname()=='<ROLE_NODE>'
root=Path('<TASK_ROOT>');base=root/'sg_pd_g316';h=json.loads((base/'actual_launch_handle.json').read_text())
assert str(base/'node_controller.py').encode() in Path('/proc',str(h['pid']),'cmdline').read_bytes()
assert h['pid']==936231
info=json.loads(subprocess.check_output(['docker','inspect','yxy-glm53-sg-p-g316']))[0];assert info['Id']=='0e8232575c7a4d9ad4065eda5d9682b9096876f030ddb1602c8c42dfdd30f5dc' and info['State']['Running']
assert any(m['Source']==str(base) and m['Destination']=='/task/sg_pd_g316' and m['RW'] for m in info['Mounts'])
roots=[base/'cache/triton',base/'cache/ascend',base/'cache/torch_extensions',base/'home/.cache/vllm',base/'home/.cache/torch',base/'tmp/torchinductor_root']
files={str(p.relative_to(base)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for r in roots for p in r.rglob('*') if p.is_file() and not p.name.endswith('.lock')}
baseline=json.loads((base/'P_cache_after_256K_warm1_g440.json').read_text());assert baseline['files']==files,'Active P compilation changed since first256warm';assert len(files)==636
d={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actual_P_controller_pid':h['pid'],'service_generation':316,'container_id':info['Id'],'files':files,'scope':'Parent invokes only after D actual terminal/allfive warm audit. Independent current-P 636files equal after firstwarm baseline; no additions/removals/metadata changes. Formal not submitted by this script.'}
p=base/'P_cache_after_256K_allwarm_g496.json';assert not p.exists();p.write_text(json.dumps(d,indent=2));print('G496_P_256K_ALLWARM_CACHE_JSON '+json.dumps(d))
REMOTE_PY
fi
