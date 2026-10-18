from pathlib import Path
import hashlib,json,subprocess,time
r=Path('/Users/wio/work/Inference-Foundry-execution-plan/glm5-3-vllm/evidence/20261018-p-first-seed-device-v2/recovery-v2');b=Path('/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/recovery-v2/P-attention');run=next((r/'prefill-attention-export-rank0').glob('node32-*'));remote=json.loads((run/'result.json').read_text())['remote_root']+'/artifacts.tar.gz';assert remote.startswith('/data/wio/glm5-3-vllm-execution/20261017-inventory/node32-prefill_attention_export_rank0_')
ssh=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34','kubectl exec -i -n default sglang-node-access-32 -- nsenter -t 1 -m -- python3 -']
code="import hashlib,json,pathlib\np=pathlib.Path(PATH);rows=[];h=hashlib.sha256()\nwith p.open('rb') as f:\n while True:\n  b=f.read(8*1024**2)\n  if not b:break\n  h.update(b);rows.append({'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})\nprint(json.dumps({'bytes':p.stat().st_size,'sha256':h.hexdigest(),'chunks':rows}))\n".replace('PATH',repr(remote))
meta=json.loads(subprocess.check_output(ssh,input=code.encode(),timeout=60));(r/'prefill-attention-profile-archive-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');dest=b/(run.name+'.tar.gz')
if dest.exists():
 prior={'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
 assert prior['bytes']!=meta['bytes'] or prior['sha256']!=meta['sha256']
 bad=dest.with_suffix('.unverified-truncated');assert not bad.exists();dest.rename(bad);(r/'prefill-attention-profile-transfer-failure.json').write_text(json.dumps({'download':prior,'expected':{k:meta[k] for k in ['bytes','sha256']},'unverified_saved':str(bad),'method':'Single cat through kubectl exec returnedrc0 but was truncated; no success receipt accepted.'},indent=2)+'\n')
partial=dest.with_suffix('.verified.partial');offset=0;start=time.monotonic()
with partial.open('xb') as f:
 for i,c in enumerate(meta['chunks']):
  code="import pathlib,sys\np=pathlib.Path(PATH)\nwith p.open('rb') as f:\n f.seek(OFFSET);b=f.read(LENGTH)\nassert len(b)==LENGTH;sys.stdout.buffer.write(b)\n".replace('PATH',repr(remote)).replace('OFFSET',str(offset)).replace('LENGTH',str(c['bytes']))
  for attempt in range(3):
   data=subprocess.check_output(ssh,input=code.encode(),timeout=60)
   if len(data)==c['bytes'] and hashlib.sha256(data).hexdigest()==c['sha256']:break
  else:raise AssertionError(('chunk',i,len(data),c))
  f.write(data);f.flush();offset+=len(data)
  if i%8==0:print(json.dumps({'verified_chunks':i+1,'total':len(meta['chunks']),'bytes':offset}),flush=True)
assert partial.stat().st_size==meta['bytes'] and hashlib.sha256(partial.read_bytes()).hexdigest()==meta['sha256'];partial.rename(dest)
receipt={k:meta[k] for k in ['bytes','sha256']};receipt.update(remote_archive=remote,local_archive=str(dest),verified_chunks=len(meta['chunks']),duration_s=time.monotonic()-start,method='8MiB ordered chunks, eachlength/SHA andwholearchiveSHA verified');(r/'prefill-attention-profile-backup.json').write_text(json.dumps(receipt,indent=2)+'\n');(run/'artifacts-backup.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
