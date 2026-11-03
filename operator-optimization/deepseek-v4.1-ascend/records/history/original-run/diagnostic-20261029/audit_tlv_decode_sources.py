#!/usr/bin/env python3
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib,re
r=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis');out=[]
for p in sorted(r.rglob('*.py')):
 if p.is_symlink() or p.stat().st_size>400000:continue
 b=p.read_bytes();ls=b.decode().splitlines()
 if not (p.name in ('_profiler_config.py','_event_tree_parser.py','_fwk_file_parser.py') or any('class TLVDecoder' in l for l in ls)):continue
 picks=[]
 if p.name=='_event_tree_parser.py':picks=[(1,70)]
 elif p.name=='_fwk_file_parser.py':picks=[(1,155)]
 elif p.name=='_profiler_config.py':picks=[(1,260)]
 else:picks=[(1,len(ls))]
 for a,z in picks:out.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'start':a,'lines':ls[a-1:z]})
print(json.dumps(out))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 nice -n 15 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'tlv-decode-sources.json';o.write_text(json.dumps(v,indent=2)+'\n');print('sources',len(v))
