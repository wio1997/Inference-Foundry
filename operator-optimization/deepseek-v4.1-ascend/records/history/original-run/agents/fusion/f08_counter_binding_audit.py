#!/usr/bin/env python3
"""Read-only two-task check on the frozen EP3 capture; run inside the stage container."""
import csv, hashlib, json, pathlib, sqlite3
from decimal import Decimal
root = pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt')
out = root/'ASCEND_PROFILER_OUTPUT'
dev = root/'PROF_000001_20261029023525664_00002128QFKFLBMH/device_3'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
info_path, sample_path = dev/'info.json.3', dev/'sample.json'
info = json.loads(info_path.read_text())['DeviceInfo'][0]
sample = json.loads(sample_path.read_text())
c = sqlite3.connect((out/'ascend_pytorch_profiler_3.db').as_uri()+'?mode=ro', uri=True); c.row_factory=sqlite3.Row
with (out/'kernel_details.csv').open(newline='') as f:
    reader=csv.DictReader(f)
    csvrows=[r for r in reader if r.get('Stream ID')=='35' and r.get('Task ID') in ('13226','13468')]
assert len(csvrows)==2
cases=[]
for task_id,gid in [(13226,2078),(13468,2942)]:
    task=[dict(x) for x in c.execute('select * from TASK where globalTaskId=?',(gid,))]
    assert len(task)==1 and task[0]['streamId']==35 and task[0]['taskId']==task_id
    pmu=[dict(x) for x in c.execute('select p.timestampNs,s.value as name,p.value from TASK_PMU_INFO p join STRING_IDS s on p.name=s.id where p.globalTaskId=?',(gid,))]
    ts=set(x['timestampNs'] for x in pmu); assert len(ts)==1 and len(pmu)==28
    m={x['name']:x['value'] for x in pmu}; assert len(m)==28
    row=next(x for x in csvrows if x['Task ID']==str(task_id))
    assert int(row['aiv_total_cycles'])==m['aiv_total_cycles']
    assert abs(Decimal(row['aiv_vec_ratio'])-Decimal(str(m['aiv_vec_ratio'])))<Decimal('0.000501')
    compute=[dict(x) for x in c.execute('select blockNum,mixBlockNum,taskType from COMPUTE_TASK_INFO where globalTaskId=?',(gid,))]; assert len(compute)==1
    cases.append(dict(task_id=task_id,globalTaskId=gid,task=task[0],compute=compute[0],csv_selected={k:row.get(k) for k in ('Duration(us)','aiv_total_cycles','aiv_vec_ratio','Accelerator Core','Block Num')},pmu_timestampNs=next(iter(ts)),pmu_timestamp_minus_task_end_ns=next(iter(ts))-task[0]['endNs'],pmu_metrics={k:m[k] for k in ('aiv_total_cycles','aiv_total_time','aiv_vec_time','aiv_vec_ratio','aiv_scalar_time','aiv_scalar_ratio','aiv_mte2_time','aiv_mte2_ratio','aiv_mte3_time','aiv_mte3_ratio')}))
result={'source_scope':'EP3 frozen profiler DB and kernel_details CSV; SQL mode=ro; two exact tasks only','profiler_version':'CANN 9.2.0 reported in capture', 'device_metadata':{k:info[k] for k in ('id','aiv_num','aiv_frequency','ai_core_num')},'capture_settings':{k:sample[k] for k in ('aiv_profiling_mode','aiv_metrics','pcSampling')},'small_input_sha256':{'device_info':sha(info_path),'device_sample':sha(sample_path)},'cases':cases,'limits':['Unique TASK and one PMU timestamp per globalTaskId in this DB; exporter timestamp semantics and universal matching rule unproved.','Numerical consistency with 48 x 1800 / 1000 does not prove counter calculation, frequency units, block/core occupancy or time units.','Time ratios describe PMU export; no basis to decompose task wall duration or claim waiting cause.']}
print(json.dumps(result,ensure_ascii=False,indent=2))
