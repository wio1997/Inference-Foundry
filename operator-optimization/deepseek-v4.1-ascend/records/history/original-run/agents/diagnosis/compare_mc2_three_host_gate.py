#!/usr/bin/env python3
"""Compare previously recorded gate/MC2 task and host launch in 3 cases; no NPU."""
import hashlib,json,shlex,subprocess
from pathlib import Path
from decimal import Decimal
root=Path(__file__).resolve().parents[2];src=root/'agents/diagnosis/mc2-db852-extremes.json';d=json.loads(src.read_text());targets=[]
for ep,layer in [(0,15),(3,15),(3,36)]:
 x=next(x for x in d['rows'] if x['ep']==ep and x['layer']==layer);targets.append({**x,'startNs':int(Decimal(x['start_us'])*1000)})
code=r'''import json,sqlite3,pathlib,sys
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces');out=[]
for x in json.loads(sys.argv[1]):
 p=list((base/x['rank']/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'));assert len(p)==1
 c=sqlite3.connect(p[0].as_uri()+'?mode=ro',uri=True);c.execute('pragma query_only=on');c.row_factory=sqlite3.Row
 params=(x['startNs'],int(x['stream']),int(x['task']),int(x['model']))
 m=c.execute('select * from TASK where startNs=? and streamId=? and taskId=? and modelId=?',params).fetchall();assert len(m)==1;(m,)=m
 g=c.execute("select t.*,s.value op from TASK t join COMPUTE_TASK_INFO i on i.globalTaskId=t.globalTaskId join STRING_IDS s on s.id=i.opType where t.deviceId=? and t.streamId=? and t.startNs<? and s.value='MoeGatingTopKHash' order by t.startNs desc limit 1",(m['deviceId'],m['streamId'],m['startNs'])).fetchall()
 if len(g)!=1:out.append({'target':x,'error':'missing or nonunique gating'});continue
 (g,)=g;record={'ep':x['ep'],'layer':x['layer'],'mc2_task':dict(m),'gate_task':dict(g),'gap_ns':m['startNs']-g['endNs'],'events':{}}
 for label,t in [('gate',g),('mc2',m)]:
  launches=c.execute("select a.*,s.value resolvedName from CANN_API a left join STRING_IDS s on s.id=a.name where a.connectionId=? and s.value='launch'",(t['connectionId'],)).fetchall();item={'launches':[dict(a) for a in launches]}
  if len(launches)==1:
   a=launches[0];args=(a['globalTid'],a['startNs'],a['endNs'])
   for table in ['CANN_API','PYTORCH_API']:
    item[table]=[dict(z) for z in c.execute('select a.*,s.value resolvedName from '+table+' a left join STRING_IDS s on s.id=a.name where a.globalTid=? and a.startNs<=? and a.endNs>=? order by a.startNs',args)]
  record['events'][label]=item
 out.append(record);c.close()
print(json.dumps(out))'''
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code,json.dumps(targets)])
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=50,check=True)
out={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'cases':json.loads(r.stdout)}
(root/'agents/diagnosis/mc2-three-host-gate.json').write_text(json.dumps(out,indent=2)+'\n')
for a in out['cases']:
 print('case',a['ep'],a['layer'],'gap ns',a['gap_ns'],'gate task',a['gate_task']['taskId'])
 for label,item in a['events'].items():
  print(label,'launch count',len(item['launches']))
  for table in ['CANN_API','PYTORCH_API']:
   for z in item.get(table,[]):
    if ('GatingTopKHash' in str(z['resolvedName']) or 'DispatchFFNCombine' in str(z['resolvedName'])):print(' ',table,z['resolvedName'],'duration ns',int(z['endNs'])-int(z['startNs']))
