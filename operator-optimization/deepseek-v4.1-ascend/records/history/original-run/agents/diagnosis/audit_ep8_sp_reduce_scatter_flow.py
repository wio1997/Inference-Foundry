#!/usr/bin/env python3
"""Read an existing EP8 trace via stage CPU and summarize HCCL flow anchors only."""
import base64,hashlib,json,pathlib,shlex,subprocess
root=pathlib.Path(__file__).resolve().parents[2]
anchor=root/'agents/diagnosis/tensormove-ep8-neighbors.json'
assert hashlib.sha256(anchor.read_bytes()).hexdigest()=='22c3cf7e9248ca2f11da46a2aac8856ac66165dde36db2c09248b6f04d71b14b'
d=json.loads(anchor.read_text());events=[e for e in d['host'] if e['name']=='HcclReduceScatter' and str(e['pid'])=='737' and str(e['tid'])=='737']
assert len(events)==1
host=events[0]
code=r'''from decimal import Decimal
from pathlib import Path
import hashlib,ijson,json,sys
p=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json')
h=hashlib.sha256()
with p.open('rb') as f:
 for block in iter(lambda:f.read(1048576),b''):h.update(block)
assert h.hexdigest()=='53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23'
lo=Decimal(sys.argv[1]);hi=lo+Decimal(sys.argv[2]); starts=[]
with p.open('rb') as f:
 for e in ijson.items(f,'item'):
  if e.get('ph')=='s' and str(e.get('pid'))=='737' and str(e.get('tid'))=='737' and e.get('ts') is not None and lo<=Decimal(str(e['ts']))<=hi:
   starts.append({k:e.get(k) for k in ('name','cat','id','pid','tid','ts')})
keys={(str(e['name']),str(e['id'])) for e in starts}; flows={key:{'s':0,'f':[]} for key in keys}
with p.open('rb') as f:
 for e in ijson.items(f,'item'):
  key=(str(e.get('name')),str(e.get('id')))
  if key not in keys:continue
  if e.get('ph')=='s':flows[key]['s']+=1
  if e.get('ph')=='f':flows[key]['f'].append({k:e.get(k) for k in ('pid','tid','ts')})
endpoints={(str(e['pid']),str(e['tid']),str(e['ts'])) for v in flows.values() for e in v['f']}
xs={key:[] for key in endpoints}
with p.open('rb') as f:
 for e in ijson.items(f,'item'):
  if e.get('ph')!='X':continue
  key=(str(e.get('pid')),str(e.get('tid')),str(e.get('ts')))
  if key in endpoints:xs[key].append({'name':e.get('name'),'dur':e.get('dur'),'cat':e.get('cat'),'args':{k:v for k,v in e.get('args',{}).items() if k in ('Task Id','Model Id','Physic Stream Id','Task Type')}})
result=[]
for a in starts:
 key=(str(a['name']),str(a['id']));row=flows[key]
 result.append({'name':a['name'],'cat':a['cat'],'start_delta_us':str(Decimal(str(a['ts']))-lo),'s_count_global':row['s'],'ends':[{'pid':e['pid'],'tid':e['tid'],'delta_from_copy_start_us':(None if str(e['pid'])=='737' else str(Decimal(str(e['ts']))-Decimal(sys.argv[3]))),'matching_X':xs[(str(e['pid']),str(e['tid']),str(e['ts']))]} for e in row['f']]})
print(json.dumps({'trace_sha256':h.hexdigest(),'anchor':'single HcclReduceScatter host scope','scope':'all flow start categories within this host event, full trace lookup, exact endpoint matching','starts':result},default=str,indent=2))'''
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code,str(host['ts']),str(host['dur']),str(d['dev_t'])])
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=150,check=True)
out=json.loads(r.stdout)
print(json.dumps(out,ensure_ascii=False,indent=2))
