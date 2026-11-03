#!/usr/bin/env python3
"""Read EP9 saved profile and list named MC2-opprof artifacts, no NPU calls."""
import csv,hashlib,json,os,pathlib,sqlite3
from decimal import Decimal as D
base=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces')
profile=base/'dp1_pp0_tp1_dcp0_ep9_rank1_752_20261029023525649_ascend_pt';out=profile/'ASCEND_PROFILER_OUTPUT';csvfile=out/'kernel_details.csv';db=out/'ascend_pytorch_profiler_9.db'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(csvfile)=='20af7cd5f9ca46fe2aeddc66a51cd30cea5048a374c87ba572f903bd1410cdf5'
with csvfile.open(newline='') as f:rows=[r for r in csv.DictReader(f) if r.get('Stream ID')=='47' and r.get('Task ID')=='46755' and r.get('Name')=='DispatchFFNCombine' and D(r['Start Time(us)'].strip())==D('1793241333127617.287')]
assert len(rows)==1;row=rows[0]
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
tasks=[dict(x) for x in c.execute('select * from TASK where streamId=47 and taskId=46755 and connectionId=118407 order by startNs')];assert len(tasks)==2
assert tasks[1]['globalTaskId']==4709 and tasks[1]['startNs']==1793241333127617287 and tasks[1]['endNs']==1793241333401182441
assert tasks[0]['startNs']<tasks[1]['startNs'] and tasks[0]['endNs']>tasks[1]['endNs']
per_task=[]
for task in tasks:
 gid=task['globalTaskId'];cmp=[dict(x) for x in c.execute('select * from COMPUTE_TASK_INFO where globalTaskId=?',(gid,))]
 pmu=[dict(x) for x in c.execute('select p.timestampNs,s.value as name,p.value from TASK_PMU_INFO p join STRING_IDS s on p.name=s.id where p.globalTaskId=?',(gid,))]
 per_task.append({'task':task,'compute_rows':len(cmp),'pmu_rows':len(pmu),'pmu_timestamps':sorted(set(x['timestampNs'] for x in pmu)),'selected_pmu':{x['name']:x['value'] for x in pmu if x['name'] in ('aic_total_cycles','aic_mac_ratio','aic_mte2_ratio','aiv_total_cycles','aiv_vec_ratio','aiv_mte2_ratio','aiv_scalar_ratio','aiv_mte3_ratio')}})
assert len(per_task[1]['selected_pmu'])==8 and per_task[1]['compute_rows']==1 and per_task[1]['pmu_rows']==28
counts={'files':0,'opprof_dirs':0,'exact_trace_json':0,'visualize_data_bin':0,'dump_workspace_bin':0,'time_stamp_core_csv':0};found=[];errors=[]
for d,sub,files in os.walk(base,followlinks=False,onerror=lambda e:errors.append(str(e))):
 counts['files']+=len(files)
 for v in sub:
  if v.startswith('OPPROF_'):counts['opprof_dirs']+=1;found.append(str((pathlib.Path(d)/v).relative_to(base)))
 for v in files:
  k={'trace.json':'exact_trace_json','visualize_data.bin':'visualize_data_bin','dump_workspace.bin':'dump_workspace_bin'}.get(v)
  if v.startswith('time_stamp_core_') and v.endswith('.csv'):k='time_stamp_core_csv'
  if k:counts[k]+=1;found.append(str((pathlib.Path(d)/v).relative_to(base)))
samples=list(profile.rglob('sample.json'));selected=[]
for f in samples:
 x=json.loads(f.read_text());selected.append({'path_relative':str(f.relative_to(profile)),'sha256':sha(f),'settings':{k:x.get(k) for k in ('pcSampling','aic_profiling_mode','aic_metrics','aiv_profiling_mode','aiv_metrics')}})
keys=('Name','Task ID','Stream ID','Start Time(us)','Duration(us)','aic_total_cycles','aic_mac_ratio','aic_mte2_ratio','aiv_total_cycles','aiv_vec_ratio','aiv_mte2_ratio','aiv_scalar_ratio','aiv_mte3_ratio')
result={'scope':'EP9 exact selected MC2 and diagnostic-traces filename-only inventory; existing output','csv_sha256':sha(csvfile),'db_current_sha256':sha(db),'db_tables':[x[0] for x in c.execute("select name from sqlite_master where type='table' order by name")],'task_rows':per_task,'csv_selected':{k:row.get(k) for k in keys},'device_sample':selected,'filename_inventory':{'root':'diagnostic-20261029/runtime/diagnostic-traces','exact_rules':'dir basename OPPROF_; file basename trace.json, visualize_data.bin, dump_workspace.bin or time_stamp_core_*.csv','counts':counts,'matches':found,'errors':errors},'limits':['exact filename search in fixed root, no other directories or raw slice decoding','distinct DB records are nested and share stream/task/connection, identity and nesting not proven to be distinct kernels or phases','selected PMU matches timestamp and globalTaskId in this database but exporter semantics and independent verification remain unknown','counters cannot establish internal waiting, transport, memory bytes, physical activity or online performance']}
print(json.dumps(result,indent=2,ensure_ascii=False))
