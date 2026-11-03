#!/usr/bin/env python3
"""Inspect static communicator and model source from read-only stage only."""
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,os,pathlib
root=pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/distributed');out=[]
for p in sorted(root.rglob('*.py')):
 if p.is_symlink() or p.stat().st_size>500000:continue
 b=p.read_bytes();lines=b.decode().splitlines();hits=[i for i,l in enumerate(lines) if 'def all_gather(' in l or 'def _all_gather_out_place(' in l]
 for i in hits:
  out.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'start_line':i+1,'lines':lines[i:min(i+75,len(lines))]})
p=pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py');b=p.read_bytes();lines=b.decode().splitlines()
print(json.dumps({'model':{'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'start_line':1195,'lines':lines[1194:1224]},'methods':out}))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True);data=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'engram-gather-impl-source.json';o.write_text(json.dumps(data,indent=2)+'\n');print('model_sha',data['model']['sha256'],'methods',len(data['methods']))
