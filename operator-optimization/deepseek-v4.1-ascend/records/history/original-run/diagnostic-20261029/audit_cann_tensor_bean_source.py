#!/usr/bin/env python3
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib,re
r=pathlib.Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
spec=['msparser/add_info/tensor_add_info_bean.py','msparser/data_struct_size_constant.py','msparser/interface/data_parser.py','msconfig/tables_config.py','msmodel/add_info/tensor_add_info_model.py'];out=[]
for rel in spec:
 p=r/rel;b=p.read_bytes();ls=b.decode().splitlines()
 if rel.endswith(('tensor_add_info_bean.py','tensor_add_info_model.py')):keep=set(range(len(ls)))
 else:
  pats={'msparser/data_struct_size_constant.py':'TENSOR_ADD_INFO','msparser/interface/data_parser.py':'def parse_bean_data|def check_magic_num|def group_aging_file','msconfig/tables_config.py':'TensorInfoV2Map'}[rel];ix=[i for i,l in enumerate(ls) if re.search(pats,l)];keep=set(j for i in ix for j in range(max(0,i-5),min(len(ls),i+80)))
 out.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'lines':[{'line':i+1,'text':ls[i]} for i in sorted(keep)]})
print(json.dumps(out))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 nice -n 15 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'cann-tensor-bean-source.json';o.write_text(json.dumps(v,indent=2)+'\n');print('sources',len(v))
