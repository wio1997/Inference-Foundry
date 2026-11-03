"""Read only source snapshots in the authorized image and find DSpark callers."""
from pathlib import Path
import base64,hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[1]
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
remote=r'''from pathlib import Path
import base64,hashlib,json
roots=[Path('/vllm-workspace/vllm-ascend/vllm_ascend'),Path('/vllm-workspace/vllm/vllm')]
needles=['precompute_and_store_context_kv','combine_hidden_states']
hits=[];skipped=[]
for root in roots:
 if not root.is_dir():skipped.append({'root':str(root),'reason':'not found'});continue
 for p in root.rglob('*.py'):
  if p.is_symlink() or not p.is_file():continue
  if p.stat().st_size>2*1024*1024:skipped.append({'path':str(p),'reason':'file exceeds 2MiB read bound'});continue
  b=p.read_bytes(); s=b.decode('utf-8',errors='replace');m=[(i+1,line.strip()) for i,line in enumerate(s.splitlines()) if any(n in line for n in needles)]
  if m:
   hits.append({'path':str(p),'size':len(b),'sha256':hashlib.sha256(b).hexdigest(),'matches':m,'base64':base64.b64encode(b).decode('ascii')})
assert len(hits)<=60 and sum(h['size'] for h in hits)<20*1024*1024
print(json.dumps({'roots':[str(r) for r in roots],'needles':needles,'hits':hits,'skipped':skipped}))
'''
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',remote])],capture_output=True,timeout=180)
assert r.returncode==0,r.stderr.decode();data=json.loads(r.stdout)
for h in data['hits']:
 b=base64.b64decode(h.pop('base64'));assert hashlib.sha256(b).hexdigest()==h['sha256']
 p=T/'sources'/h['path'].lstrip('/');p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==h['sha256'],str(p)
 else:p.write_bytes(b)
 h['snapshot_path']=str(p.relative_to(T))
doc={'status':'READ_ONLY_SOURCE_SCAN','roots':data['roots'],'needles':data['needles'],'hits':data['hits'],'skipped':data['skipped'],
     'source_identity':'Existing CPU stage container from the frozen image; later verify image/source commits before making provenance claims',
     'no_device_work':True}
assert len(json.dumps(doc).encode())<32768
p=T/'evidence/dspark-runner-source-scan.json';assert not p.exists();p.write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps(doc,indent=2))
