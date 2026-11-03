#!/usr/bin/env python3
"""Fetch one static kernel source file from the read-only stage container."""
import base64,hashlib,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib
p=pathlib.Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/engram_lookup.py')
b=p.read_bytes()
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'source':b.decode('utf-8')}))'''
e=base64.b64encode(remote.encode()).decode()
cmd='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=60,check=True)
data=json.loads(p.stdout)
target=pathlib.Path(__file__).resolve().parent/'engram-lookup-source.json'
target.write_text(json.dumps(data,indent=2)+'\n')
print('source path',data['path'],'sha256',data['sha256'],'chars',len(data['source']))
