python3 - <<'PY'
from pathlib import Path
import ast,datetime,hashlib,json,socket,subprocess

assert socket.gethostname()=='<DECODE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
cid='d8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
container=json.loads(subprocess.check_output(['docker','inspect',cid],timeout=8))[0]
assert container['State']['Running'] and container['Config']['Labels']['codex.controller_pid']=='2760614'
assert container['HostConfig']['CpusetCpus']=='184-187' and not container['HostConfig']['Devices']
assert any(m['Destination']=='/task' and m['Source']==str(root) and not m['RW'] for m in container['Mounts'])
fs=Path('/proc',str(container['State']['Pid']),'root')
base=fs/'task/client_ds41_g503/profile_inputs_g772'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=base/'manifest.json'
assert sha(manifest)=='308d46f5ae05d7d30b8af9b4711b24582fe4b8a0c6fdc76f32473720f7acf4f4'
d=json.loads(manifest.read_text())
data=fs/d['dataset_path'].lstrip('/')
assert data.parent==base
assert sha(data)==d['dataset_sha256']=='5cf5dbc8174409a627c8a3439f4b07e0ca99d53ac19c187ed4988eff354d0c00'
rows=[json.loads(line) for line in data.read_bytes().splitlines()]
assert len(rows)==len(d['records'])==16
for seq,(r,m) in enumerate(zip(rows,d['records'])):
    assert m['data_id']==seq and m['input_tokens']==32768 and r['max_out_len']==1024
    assert hashlib.sha256(r['question'].encode()).hexdigest()==m['question_sha256']
assert d['new_prefixes_unique'] and d['original_reserved_prefixes_checked']==448
assert sha(fs/'models/ds41/tokenizer.json')==d['tokenizer_sha256']
config={}
for engine,item in d['configurations'].items():
    a=fs/item['path'].lstrip('/')
    assert a.parent==base and sha(a)==item['sha256']
    config[engine]=a.read_text()
norm=config['sglang'].replace('sglang-ds41flash-910b-pd','vllm-ds41flash-910b-pd').replace('host_port=18090','host_port=18100')
assert ast.dump(ast.parse(norm))==ast.dump(ast.parse(config['vllm']))
print('G1197_SG_MATCHED_PROFILE_INPUTS_JSON '+json.dumps({
    'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'current_container_ID':cid,
    'profile_inputs_manifest_SHA256':sha(manifest),'dataset_sha256':d['dataset_sha256'],
    'all16_prompt_SHA_verified':True,'profile_input_tokens':32768,'profile_output_tokens':1024,
    'tokenizer_sha256':d['tokenizer_sha256'],'framework_configuration_AST_match':True,
    'framework_configurations_SHA256':{k:v['sha256'] for k,v in d['configurations'].items()},
    'input_assets_only':True,'profile_started':False,'model_requests_submitted':False}))
PY
