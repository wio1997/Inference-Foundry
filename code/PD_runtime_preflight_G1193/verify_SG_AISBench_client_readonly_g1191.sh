python3 - <<'PY'
from pathlib import Path
import datetime
import hashlib
import json
import os
import socket
import subprocess

host=socket.gethostname()
assert host=='<DECODE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
client=root/'client_ds41_g1122'
cpid=2760614
cid='d8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
assert str(client/'cpu_client_controller.py').encode() in Path('/proc',str(cpid),'cmdline').read_bytes().split(b'\0')
info=json.loads(subprocess.check_output(['docker','inspect',cid],timeout=12))[0]
assert info['State']['Running'] and info['State']['Pid']>0
assert info['Image']=='sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76'
assert info['Config']['Labels']['codex.controller_pid']==str(cpid)
cfg=info['HostConfig']
assert cfg['CpusetCpus']=='184-187' and cfg['ReadonlyRootfs'] and not cfg['Privileged'] and not cfg['Devices']
coord=Path('/PRIVATE_COORD_ROOT',f'{host}-cpu184-187')
owner=json.loads(coord.with_suffix('.owner.json').read_text())
assert owner['pid']==cpid and owner['state']=='claimed'
fds=[os.readlink(f) for f in Path('/proc',str(cpid),'fd').iterdir()]
assert str(coord.with_suffix('.lock')) in fds
prefix='/task/client_ds41_g1122'
mounts={m['Destination']:m for m in info['Mounts']}
assert mounts['/task']['Source']==str(root) and not mounts['/task']['RW']
assert mounts[prefix]['Source']==str(client) and mounts[prefix]['RW']
assert not mounts[prefix+'/datasets']['RW'] and not mounts['/models/ds41']['RW']

# Read installed Python files through the running container's filesystem.
# This avoids importing AISBench or occupying its CPU set during GSM8K.
container=Path('/proc',str(info['State']['Pid']),'root')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
sources={'base_api.py':'203a416444b972193274a13e1593328a288b6879a3d62e4f201bbb9adfa81755',
         'vllm_custom_api.py':'8170e1ca344340a4919e05fb61df7bfdcd8d7f7b7cca9d427924d793c89da086'}
for filename,expected in sources.items():
    path=container/'benchmark/ais_bench/benchmark/models/api_models'/filename
    assert sha(path)==expected,filename
assets=client/'benchmark_pipeline_SG_g1183'
manifest=json.loads((assets/'preparation_manifest.json').read_text())
for filename,digest in manifest['files'].items():
    assert sha(assets/filename)==digest,filename
profile_path=root/'client_ds41_g503/original_G83_dataset_manifest.json'
original=json.loads(profile_path.read_text())
profiles=original['profiles']
assert len(profiles)==24 and len({(x['phase'],x['input_tokens']) for x in profiles})==24
dataset=container/prefix.lstrip('/')/'datasets'
for item in profiles:
    path=dataset/Path(item['path']).name
    assert path.is_file() and path.stat().st_size>0,path
tokenizer=container/'models/ds41/tokenizer.json'
assert sha(tokenizer)==original['tokenizer_sha256']
print('G1191_AISBENCH_CLIENT_READONLY_JSON '+json.dumps({
    'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_controller_PID':cpid,'CPU_container_ID':cid,
    'active_container_and_fd_verified':True,'readonly_model_dataset_and_writable_result_mounts_verified':True,
    'AISBench_handler_SHA256':sources,'staged_files_SHA_verified':len(manifest['files']),
    'dataset_manifest_SHA256':sha(profile_path),'tokenizer_SHA256':original['tokenizer_sha256'],
    'dataset_profile_files_present':24,'total_requests_expected':sum(x['requests'] for x in profiles),
    'dataset_content_revalidation_done':False,'service_usage_validated':False,'model_requests_submitted':False}))
PY
