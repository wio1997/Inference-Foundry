#!/usr/bin/env python3
"""Read exact fixed-source line ranges, no device or service access."""
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib
specs=[('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py',3169,3242),('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/vl_model.py',225,250),('/vllm-workspace/vllm/vllm/distributed/device_communicators/base_device_communicator.py',230,275)]
out=[]
for n,a,b in specs:
 p=pathlib.Path(n);data=p.read_bytes();lines=data.decode().splitlines();out.append({'path':n,'sha256':hashlib.sha256(data).hexdigest(),'start_line':a,'end_line':b,'lines':lines[a-1:b]})
print(json.dumps(out))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'engram-producer-anchor-sources.json';o.write_text(json.dumps(v,indent=2)+'\n');print('static sources',len(v))
