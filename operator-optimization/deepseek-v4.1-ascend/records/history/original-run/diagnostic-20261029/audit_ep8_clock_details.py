#!/usr/bin/env python3
import base64,json,pathlib,shlex,subprocess
remote=r'''import ast,json,pathlib
r=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/offline-ep8-20261029/dp1_pp0_tp0_dcp0_ep8_rank0_746_20261029050352206_ascend_pt');out=[]
for p in r.rglob('*'):
 if not p.is_file() or p.is_symlink():continue
 if p.name in ('info.json','host_start.log','start_info','start_info.json'):
  o={'path':str(p.relative_to(r))}
  try:
   if p.name=='info.json':
    x=json.loads(p.read_text());o['cpu_frequency']=[q.get('Frequency') for q in x.get('CPU',[]) if isinstance(q,dict)]
   elif p.name.startswith('start_info'):
    x=ast.literal_eval(p.read_text());o['keys']=list(x);o['selected']={k:v for k,v in x.items() if k in ('collectionTimeBegin','MonotonicTimeBegin','clockMonotonicRaw')}
   else:
    import configparser
    x=configparser.ConfigParser();x.read(p);o['host_selected']={k:x.get('Host',k,fallback=None) for k in ('clock_monotonic_raw','cntvct')}
  except Exception as e:o['error']=type(e).__name__
  out.append(o)
print(json.dumps(out))'''
e=base64.b64encode(remote.encode()).decode();cmd='docker exec wio-vllm-v41-repro169-stage-20261028 python3 -c '+shlex.quote("import base64;exec(base64.b64decode('"+e+"'))")
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=60,check=True);v=json.loads(r.stdout);o=pathlib.Path(__file__).parent/'ep8-clock-details.json';o.write_text(json.dumps(v,indent=2)+'\n');print('clock metadata files',len(v))
