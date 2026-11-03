#!/usr/bin/env python3
"""Read saved status/clock fields and source conversion methods, not tensor data."""
import base64,json,pathlib,shlex,subprocess
remote=r'''import hashlib,json,pathlib,re
root=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029')
status=json.loads((root/'status.json').read_text());paths=[]
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 name=p.name
 if name in ('torch.op_range','profiler_info_8.json','start.log','info.json') or name.endswith('tensor_info.slice_0'):
  entry={'relative_path':str(p.relative_to(root)),'bytes':p.stat().st_size}
  if name=='profiler_info_8.json':
   v=json.loads(p.read_text());entry['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();entry['start_info']=v.get('start_info');entry['end_info']=v.get('end_info');entry['config']={k:v.get('config',{}).get('common_config',{}).get(k) for k in ('record_shapes','profile_memory','with_stack')};entry['record_op_args']=v.get('config',{}).get('experimental_config',{}).get('record_op_args')
  paths.append(entry)
r=pathlib.Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis');sources=[]
for rel,pat in [('_profiler_config.py','def get_local_time'),('prof_parse/_cann_file_parser.py','def get_localtime_diff')]:
 p=r/rel;b=p.read_bytes();ls=b.decode().splitlines()
 for i,l in enumerate(ls):
  if pat in l:sources.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'start':i+1,'lines':ls[i:i+40]})
print(json.dumps({'status_keys':list(status.keys()),'db_entries':[{'path':v.get('path'),'sha256':v.get('sha256')} for v in status.get('dbs',[])],'file_entries':paths,'source_methods':sources}))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 nice -n 15 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=90,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'raw-profiler-clock-audit.json';o.write_text(json.dumps(v,indent=2)+'\n');print('clock files',len(v['file_entries']),'source methods',len(v['source_methods']))
