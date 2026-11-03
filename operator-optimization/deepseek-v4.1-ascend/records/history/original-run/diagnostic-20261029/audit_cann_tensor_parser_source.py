#!/usr/bin/env python3
"""Read CANN installed tensor parser/DTO and relevant schema definitions."""
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib,re
r=pathlib.Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')
spec=['msparser/add_info/tensor_add_info_parser.py','profiling_bean/db_dto/tensor_info_dto.py','common_func/file_name_manager.py','framework/file_dispatch.py','msconfig/tables_config.py','mscalculate/cann/cann_analysis_gear.py','mscalculate/cann/cann_event_generator.py','common_func/db_name_constant.py'];out=[]
for rel in spec:
 p=r/rel;b=p.read_bytes();ls=b.decode().splitlines()
 if rel.endswith(('tensor_add_info_parser.py','tensor_info_dto.py')):lines=[{'line':i+1,'text':l} for i,l in enumerate(ls)]
 else:
  ix=[i for i,l in enumerate(ls) if re.search('tensor_info|TensorInfo|TENSOR_INFO',l)];keep=set(j for i in ix for j in range(max(0,i-4),min(len(ls),i+7)));lines=[{'line':i+1,'text':ls[i]} for i in sorted(keep)]
 out.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'lines':lines})
print(json.dumps(out))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 nice -n 15 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'cann-tensor-parser-source.json';o.write_text(json.dumps(v,indent=2)+'\n');print('sources',len(v))
