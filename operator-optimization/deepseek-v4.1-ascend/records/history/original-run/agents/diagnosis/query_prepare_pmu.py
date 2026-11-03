#!/usr/bin/env python3
"""Read existing profile DBs in the task container; inspect 32 selected preparation task counters."""
import csv,hashlib,json,pathlib,sqlite3
from decimal import Decimal
R=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
p=R/'agents/diagnosis/indexer-attention-event-identity.json';raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='8377661b150c2024a62622b5855c36c057c119d32493a8996a8f49c409d2badc'
src=json.loads(raw);out=[];dbhash={};csvhash={}
for ep in (3,8):
 ps=list((R/'diagnostic-20261029/runtime/diagnostic-traces').glob(f'*ep{ep}_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{ep}.db'));assert len(ps)==1
 db=ps[0];h=hashlib.sha256()
 with db.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 assert h.hexdigest()==src['db_sha256'][str(ep)];dbhash[str(ep)]=h.hexdigest()
 c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('pragma query_only=on')
 csvpath=db.parent/'kernel_details.csv';csvhash[str(ep)]=hashlib.sha256(csvpath.read_bytes()).hexdigest()
 with csvpath.open(newline='') as f: csvrows=list(csv.DictReader(f))
 for row in [x for x in src['pairs'] if x['ep']==ep]:
  t=[t for t in row['middle_tasks'] if t['info']['name']=='_prepare_indexer_indices_kernel'];assert len(t)==1;t=t[0]
  x=[dict(a) for a in c.execute('select * from TASK where globalTaskId=?',(t['global_task_id'],))];assert len(x)==1 and x[0]['taskId']==t['task_id'] and x[0]['streamId']==row['same_stream']
  info=[dict(a) for a in c.execute('select blockNum,mixBlockNum,gridDim,blockDim from COMPUTE_TASK_INFO where globalTaskId=?',(t['global_task_id'],))];assert len(info)==1
  pmu=[dict(a) for a in c.execute('select p.timestampNs,s.value as name,p.value from TASK_PMU_INFO p join STRING_IDS s on p.name=s.id where p.globalTaskId=?',(t['global_task_id'],))]
  names={a['name']:a['value'] for a in pmu};times=sorted({a['timestampNs'] for a in pmu})
  assert len(names)==len(pmu)
  cr=[r for r in csvrows if r.get('Stream ID')==str(row['same_stream']) and r.get('Task ID')==str(t['task_id'])]
  assert len(cr)==1 and int(cr[0]['aiv_total_cycles'])==names['aiv_total_cycles']
  for k in ('aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio'):
   assert abs(Decimal(cr[0][k])-Decimal(str(names[k])))<=Decimal('0.000501'),(ep,t['task_id'],k)
  out.append({'ep':ep,'layer':row['layer'],'scope_id':row['scope_id'],'task_id':t['task_id'],'global_task_id':t['global_task_id'],'task_duration_ns':t['duration_ns'],'compute_info':info[0],'pmu_rows':len(pmu),'pmu_timestamps_relative_to_task_end_ns':[n-x[0]['endNs'] for n in times],'csv_match':True,'pmu':{k:names.get(k) for k in ['aic_total_cycles','aiv_total_cycles','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio','aiv_total_time']}})
 c.close()
assert len(out)==32
print(json.dumps({'source_sha256':hashlib.sha256(raw).hexdigest(),'db_sha256':dbhash,'csv_sha256':csvhash,'samples':out,'limits':['Counters are exporter fields, not direct elapsed wait or device utilization; matching task IDs are local to this capture.','PMU timestamps and export formula need verification; values are not independent per-phase timelines.','Task count and selected samples do not establish coverage or runtime tensor identity.']},indent=2))
