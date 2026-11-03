#!/usr/bin/env python3
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib
p=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/triton/backends/ascend/driver.py');b=p.read_bytes();s=b.decode();lines=s.splitlines();hits=[i for i,l in enumerate(lines) if 'getPointer(' in l or 'ptr_info.dev_ptr =' in l];sections=[]
for i in hits:
 sections.append({'line':i+1,'lines':[{'line':j+1,'text':lines[j]} for j in range(max(i-4,0),min(len(lines),i+9))]})
print(json.dumps({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'sections':sections}))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
p=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=60,check=True);v=json.loads(p.stdout);dest=pathlib.Path(__file__).parent/'engram-ascend-pointer-detail.json';dest.write_text(json.dumps(v,indent=2)+'\n');print('sections',len(v['sections']),'sha',v['sha256'])
