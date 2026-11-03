#!/usr/bin/env python3
"""Read seven saved profiler DBs; summarize counters for 28 previously selected QLI tasks."""
import collections, hashlib, json, pathlib, sqlite3, statistics, time
R=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
p=R/'agents/diagnosis/qli-target-scope-distribution.json'; raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='61aa739e43197d4c0725d81aea30624728de0df9c7b2ffea772d5de53b0ca804'
src=json.loads(raw);typ='VllmAscendQuantLightningIndexerV2'
refs=[{'ep':x['ep'],'layer':x['layer'],'scope':x['scope_id'],'start':int(x['flows'][typ]),'task':int(x['task_ids'][typ])} for x in src['records'] if x['all_signatures_equal_ref'] and x['same_track'] and x['ordered_notoverlap']]
assert len(refs)==28
keys=['aic_total_cycles','aic_mac_ratio','aic_scalar_ratio','aic_mte2_ratio','aic_mte3_ratio','aiv_total_cycles','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio'];rows=[]
for ep in range(7):
 ps=list((R/'diagnostic-20261029/runtime/diagnostic-traces').glob(f'*_ep{ep}_rank*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_{ep}.db'));assert len(ps)==1
 c=sqlite3.connect(ps[0].as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');t=time.monotonic();c.set_progress_handler(lambda:int(time.monotonic()-t>20),100000)
 for ref in [x for x in refs if x['ep']==ep]:
  task=list(c.execute('SELECT globalTaskId,streamId,taskId,startNs,endNs FROM TASK WHERE taskId=? AND startNs=?',(ref['task'],ref['start'])));assert len(task)==1;task=task[0]
  gid_count=c.execute('SELECT count(*) FROM TASK WHERE globalTaskId=?',(task['globalTaskId'],)).fetchone()[0];assert gid_count==1
  pmu=list(c.execute('SELECT p.timestampNs,s.value name,p.value FROM TASK_PMU_INFO p JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTaskId=?',(task['globalTaskId'],)));counts=collections.Counter(x['name'] for x in pmu);assert all(v==1 for v in counts.values())
  vals={x['name']:x['value'] for x in pmu};times=sorted(set(x['timestampNs'] for x in pmu));rows.append({**ref,'gid':task['globalTaskId'],'stream':task['streamId'],'global_id_task_count':gid_count,'pmu_count':len(pmu),'pmu_timestamps_from_end_ns':[x-task['endNs'] for x in times],'metrics':{k:vals.get(k) for k in keys}})
 c.close()
summary={}
for k in keys:
 vs=[x['metrics'][k] for x in rows if isinstance(x['metrics'][k],(int,float))];summary[k]={'valid':len(vs),'missing':len(rows)-len(vs),'min':min(vs) if vs else None,'median':statistics.median(vs) if vs else None,'max':max(vs) if vs else None}
print(json.dumps({'selection_sha256':hashlib.sha256(raw).hexdigest(),'rows':rows,'summary':summary,'limits':['Seven DB files are not hashed in this targeted read; task start and id must uniquely match the previously hashed selection.','PMU timestamp and exporter provenance do not establish independent phase timing.','AIC/AIV and their pipes have separate denominators and can overlap; rates are not device utilization, wall time, or root causes.']},indent=2))
