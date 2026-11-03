#!/usr/bin/env python3
"""Fixed read-only source discovery with staging-container/mount attestation."""
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_ep0_ep8_op_summary_inventory.py';OUT=T/'agents/diagnosis/ep0-ep8-op-summary-inventory.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';STAGE='42a5ed03e3a3014d64e47af56a0304a3e9724573bf8df0c98adc47866c587602';IMAGE='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b';ROOT='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
expect={'agents/diagnosis/query_ep0_ep8_op_summary_inventory.py':'ee77cd53e21579c4e3c208d4ee44ad24fe9f894d13e249011ea2f79d6d926e64','evidence/used-raw-source-sha256.json':'cd5bac3b56c51fa2d7ce13199523ff9f88b94c4edf4c74131d4aedbdf2fc7c2a','sources/primary/frozen-image-clock/torch_cann_file_parser.py':'fb47549af5aa1547461884d43150107eeca9131869959495bee8e5403fc9d4a3','sources/primary/frozen-image-clock/torch_path_manager.py':'5fbc58950394161f03bc8c7b5b4b21757d6d7b1bf3216a06e9a6596037212b39'}
for rel,h in expect.items():
 if hashlib.sha256((T/rel).read_bytes()).hexdigest()!=h:raise RuntimeError('fixed local source changed')
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output exists')
def remote(args,timeout=120):
 r=subprocess.run([SSH,shlex.join(args)],capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError('container inspection or bounded read failed: exit '+str(r.returncode))
 if len(r.stdout)>100000:raise RuntimeError('oversize output')
 return r.stdout
def inspect():
 cid,img,mounts=remote(['docker','inspect','--format','{{.Id}}|{{.Image}}|{{json .Mounts}}',STAGE]).strip().split('|',2)
 if cid!=STAGE or img!=IMAGE:raise RuntimeError('unexpected staging identity or image')
 mounts=json.loads(mounts)
 if not isinstance(mounts,list):raise RuntimeError('invalid mounts')
 matches=[m for m in mounts if m.get('Destination')=='/work']
 if len(matches)!=1 or matches[0].get('Type')!='bind' or matches[0].get('Source')!='/data/tiankuan/wio/main':raise RuntimeError('unexpected staging mount')
 for m in mounts:
  d=m.get('Destination','').rstrip('/') or '/'
  if d=='/' or (d!='/work' and (ROOT==d or ROOT.startswith(d+'/') or d.startswith(ROOT+'/'))):raise RuntimeError('unexpected trace overmount')
 return (cid,img,sorted(mounts,key=lambda m:json.dumps(m,sort_keys=True,separators=(',',':'))))
before=inspect();x=json.loads(remote(['docker','exec',STAGE,'python3','-c',S.read_text()],240));after=inspect()
if before!=after:raise RuntimeError('container mount metadata changed')
if [s['rank'] for s in x['samples']]!=['ep0','ep8'] or any(len(s['directories'])!=3 for s in x['samples']):raise RuntimeError('unexpected output coverage')
x['source_sha256']=expect;x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['staging_container_id']=STAGE;x['staging_image_id']=IMAGE;x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':'BOUNDED_INVENTORY_COMPLETE','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),'ranks':len(x['samples']),'matched_files':sum(len(d['matches']) for s in x['samples'] for d in s['directories'])}))
