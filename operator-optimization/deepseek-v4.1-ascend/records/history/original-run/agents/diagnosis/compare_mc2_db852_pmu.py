#!/usr/bin/env python3
"""Inspect four existing per-rank task PMU records via read-only SQLite."""
import hashlib,json,subprocess,shlex
from pathlib import Path
root=Path(__file__).resolve().parents[2]
src=root/'agents/diagnosis/mc2-db852-extremes.json'
d=json.loads(src.read_text()); targets=[x for x in d['rows'] if x['ep'] in (0,3)]
base='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces'
code=r'''import json,sqlite3,pathlib,sys
from decimal import Decimal
base,raw=sys.argv[1:];targets=json.loads(raw);res=[]
for x in targets:
 folder=pathlib.Path(base)/x['rank']/'ASCEND_PROFILER_OUTPUT';files=sorted(folder.glob('ascend_pytorch_profiler_*.db'))
 if len(files)!=1:res.append({'ep':x['ep'],'layer':x['layer'],'error':'db count '+str(len(files))});continue
 c=sqlite3.connect(files[0].as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON')
 start=int(Decimal(x['start_us'])*1000)
 rows=c.execute('select globalTaskId,taskId,streamId,modelId,startNs,endNs from TASK where taskId=? and streamId=? and modelId=? and startNs = ?',(int(x['task']),int(x['stream']),int(x['model']),start)).fetchall()
 row={'ep':x['ep'],'layer':x['layer'],'target':x,'db_name':files[0].name,'matches':rows}
 if len(rows)==1:
  gid=rows[0][0];row['global_id_task_count']=c.execute('select count(*) from TASK where globalTaskId=?',(gid,)).fetchone()[0]
  row['pmu']=c.execute('select s.value,p.value,p.timestampNs from TASK_PMU_INFO p left join STRING_IDS s on s.id=p.name where p.globalTaskId=? order by s.value,p.timestampNs',(gid,)).fetchall()
 c.close();res.append(row)
print(json.dumps(res))'''
cmd=shlex.join(['docker','exec','wio-vllm-v41-repro169-stage-20261028','python3','-c',code,base,json.dumps(targets)])
r=subprocess.run(['/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh',cmd],capture_output=True,text=True,timeout=50,check=True)
out={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'results':json.loads(r.stdout),'limits':['one capture, task/PMU mapping conditional on exporter semantics','no new device work']}
(root/'agents/diagnosis/mc2-db852-extremes-pmu.json').write_text(json.dumps(out,indent=2)+'\n')
for x in out['results']:
 print(x['ep'],x['layer'],'matches',len(x.get('matches',[])),'id_unique',x.get('global_id_task_count'),'pmu_records',len(x.get('pmu',[])),'error',x.get('error'))
 for y in x.get('pmu',[]):
  if y[0] in ('aic_total_cycles','aic_mac_ratio','aiv_mte2_ratio','aiv_mte3_ratio','aiv_vec_ratio'):print(' ',y)
