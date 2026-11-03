from pathlib import Path
import hashlib,json,re,shlex,subprocess
T=Path(__file__).resolve().parents[2]
S=T/'agents/diagnosis/inventory_hccl_host_db.py'
inventory=T/'evidence/used-raw-source-sha256.json'
raw=inventory.read_bytes();source=json.loads(raw);items=[]
for entry in source['records']:
 p=entry['relative_path']
 m=re.match(r'dp0_pp0_tp([0-7])_dcp0_ep\1_rank\1_.*?/PROF_[^/]+/host/start_info$',p)
 if m:items.append({'rank':int(m[1]),'relative_path':p,'sha256':entry['sha256']})
if len(items)!=8 or sorted(i['rank'] for i in items)!=list(range(8)):raise RuntimeError('ambiguous input inventory')
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',S.read_text(),json.dumps(items)])],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('inventory failed: '+r.stderr)
x=json.loads(r.stdout);x['script_sha256']=hashlib.sha256(S.read_bytes()).hexdigest();x['source_inventory_sha256']=hashlib.sha256(raw).hexdigest()
out=T/'agents/diagnosis/hccl-host-db-inventory.json'
if out.exists():raise RuntimeError('output already exists')
out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'INVENTORY_COMPLETED','output':str(out),'ranks':len(x['ranks'])}))
