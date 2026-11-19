python3 - <<'PY'
from pathlib import Path
import datetime,hashlib,json,socket,subprocess

assert socket.gethostname()=='<ROLE_HOSTNAME>'
root=Path('/PRIVATE_TASK_ROOT')
cid='d8593a995ef7192cfd5fd9e696d652e24b3d02d77253d2b96a3415a4097f9090'
cur=json.loads(subprocess.check_output(['docker','inspect',cid],timeout=10))[0]
assert cur['State']['Running'] and cur['Config']['Labels']['codex.controller_pid']=='2760614'
assert any(m['Destination']=='/task' and m['Source']==str(root) and not m['RW'] for m in cur['Mounts'])
fs=Path('/proc',str(cur['State']['Pid']),'root')
def sha(b):
    return hashlib.sha256(b).hexdigest()
assets=fs/'task/client_ds41_g503/long_quality_g703'
manifest=(assets/'manifest.json').read_bytes()
assert sha(manifest)=='2b2a82681901b3e19e0ed8b799b1e12085a9f7bd4f2caf90d30d62750e484b5d'
data=json.loads(manifest)
assert sha((fs/'models/ds41/tokenizer.json').read_bytes())==data['tokenizer_sha256']
assert data['benchmark_prefixes_checked']==448
rows=[]
for item in data['records']:
    length=item['input_tokens']
    assert item['request_file']==f'request_{length}.json' and item['no_reserved_prefix_collision']
    raw=(assets/item['request_file']).read_bytes()
    q=json.loads(raw)
    assert sha(raw)==item['request_sha256']
    assert q['model']=='ds41flash' and q['temperature']==0 and q['seed']==0
    assert q['max_tokens']==128 and not q['stream'] and not q['add_special_tokens']
    rows.append({'input_tokens':length,'request_sha256':sha(raw),'bytes':len(raw),'token_ids_sha256':item['token_ids_sha256']})
assert {r['input_tokens'] for r in rows}=={32768,65536,131072,262144}
print('G1203_SG_MATCHED_LONG_INPUTS_JSON '+json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'actual_CPU_container_ID':cid,'native_manifest_SHA256':sha(manifest),'records':rows,'same_tokenizer_sha256':data['tokenizer_sha256'],
    'actual_model_requests_submitted':False,'scope':'Exact native retrieval workload bytes visible in current CPU client. SG token and answer validation pending.'}))
PY
