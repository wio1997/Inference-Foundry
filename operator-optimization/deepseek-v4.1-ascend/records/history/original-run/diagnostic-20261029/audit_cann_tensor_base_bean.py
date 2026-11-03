#!/usr/bin/env python3
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib
r=pathlib.Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis');paths=['profiling_bean/ge/ge_tensor_base_bean.py','msparser/add_info/tensor_add_info_bean.py','msparser/data_struct_size_constant.py'];out=[]
for name in paths:
 p=r/name;b=p.read_bytes();s=b.decode().splitlines();
 if name.endswith('data_struct_size_constant.py'):s=s[:80]
 out.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'lines':s})
print(json.dumps(out))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'cann-tensor-base-bean.json';o.write_text(json.dumps(v,indent=2)+'\n');print('sources',len(v))
