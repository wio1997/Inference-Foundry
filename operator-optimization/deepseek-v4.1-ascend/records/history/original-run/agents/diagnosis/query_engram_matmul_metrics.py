#!/usr/bin/env python3
"""Read exact EP8 six MatMul CSV rows and bounded candidate PMU records."""
import argparse,csv,hashlib,json,sqlite3
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);ap.add_argument('--db',required=True);a=ap.parse_args()
T=Path(a.task_root); wp=T/'captured-layer-attribution/module-call-metrics.json';ep=T/'agents/diagnosis/engram-boundary-controls.json';mp=T/'evidence/TRACE-EXPORT-SHA256.json'
windows=json.loads(wp.read_text());s=[r for r in json.loads(ep.read_text())['samples'] if '_ep8_' in r['rank'] and r['after_layer'] in (0,13)];assert len(s)==6
targets=[]
for r in s:
 x=[z for z in windows if z['rank']==r['rank'] and z['run_in_model']==r['run'] and z['layer']==r['after_layer']];assert len(x)==1
 v=[z for z in r['events'] if z['type']=='MatMulV2' and z['input_shapes']=='"3,6144;25600,6144"' and z['input_dtypes']=='DT_BF16;DT_BF16'];assert len(v)==1
 e=v[0];base=x[0]['window_end_ns'];targets.append({'rank':r['rank'],'run':r['run'],'after_layer':r['after_layer'],'stream':e['stream'],'model':e['model'],
   'start_ns':base+e['event_start_rel_ns'],'end_ns':base+e['event_end_rel_ns']})
assert len({x['rank'] for x in targets})==1;rank=targets[0]['rank'];csvpath=Path(a.trace_root)/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
expected={x['rank']:x['kernel_csv_sha256'] for x in json.loads(mp.read_text())['exports']};assert sha(csvpath)==expected[rank]
fields=['Name','Type','Model ID','Task ID','Stream ID','Start Time(us)','Duration(us)','Wait Time(us)','Accelerator Core','Block Num',
 'Input Shapes','Input Data Types','Output Shapes','Output Data Types','aicore_time(us)','aic_total_cycles','aic_mac_time(us)','aic_mac_ratio',
 'aic_scalar_time(us)','aic_scalar_ratio','aic_mte1_time(us)','aic_mte1_ratio','aic_mte2_time(us)','aic_mte2_ratio',
 'aic_fixpipe_time(us)','aic_fixpipe_ratio','aiv_time(us)','aiv_total_cycles','aiv_vec_time(us)','aiv_vec_ratio','cube_utilization(%)']
matched={}
with csvpath.open() as f:
 for z in csv.DictReader(f):
  if z['Type']!='MatMulV2' or z['Input Shapes']!='"3,6144;25600,6144"':continue
  start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'].strip())*1000)
  key=(z['Model ID'],z['Stream ID'],start,end)
  if any((t['model'],t['stream'],t['start_ns'],t['end_ns'])==key for t in targets):
   if key in matched:raise RuntimeError('duplicate matching CSV row')
   matched[key]={k:z[k] for k in fields}
assert len(matched)==6
path=Path(a.db).resolve(strict=True);dbsha=sha(path);assert dbsha=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
c=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
groups={};results=[]
for t in targets:
 tasks=c.execute('''SELECT globalTaskId,streamId,taskId,modelId,startNs,endNs FROM TASK WHERE streamId=? AND modelId=? AND CAST(startNs AS INTEGER)=? AND CAST(endNs AS INTEGER)=?''',
      (int(t['stream']),int(t['model']),t['start_ns'],t['end_ns'])).fetchall()
 assert len(tasks)==1;task=dict(tasks[0]);gid=task['globalTaskId']
 if str(gid) not in groups:
  vals=c.execute('''SELECT p.timestampNs,s.value AS counter_name,p.value FROM TASK_PMU_INFO p
   LEFT JOIN STRING_IDS s ON s.id=p.name WHERE p.globalTaskId=? AND s.value IN
   ('aic_total_cycles','aic_mac_ratio','aic_mte1_ratio','aic_mte2_ratio','aic_fixpipe_ratio','aiv_total_cycles','aiv_vec_ratio')
   ORDER BY CAST(p.timestampNs AS INTEGER),s.value LIMIT 501''',(gid,)).fetchall();assert len(vals)<=500
  groups[str(gid)]=[dict(x) for x in vals]
 candidates=sorted({int(v['timestampNs']) for v in groups[str(gid)]})
 nearest=min(candidates,key=lambda v:abs(v-t['end_ns'])) if candidates else None
 results.append({'target':t,'task':task,'csv_fields':matched[(t['model'],t['stream'],t['start_ns'],t['end_ns'])],
   'candidate_pmu_time_ns':nearest,'candidate_pmu_minus_task_end_ns':nearest-t['end_ns'] if nearest is not None else None,
   'candidate_count_within_1000ns':sum(abs(p-t['end_ns'])<=1000 for p in candidates)})
c.close()
out={'scope':'EP8 six exact target large BF16 MatMulV2 rows; raw PMU grouped by repeated globalTaskId',
 'kernel_sha256':sha(csvpath),'db_sha256':dbsha,'window_sha256':sha(wp),'interval_sha256':sha(ep),
 'samples':results,'pmu_rows_by_globalTaskId':groups,
 'limits':['Exact CSV timestamps/model/stream/shape identify an exported sample, not tensor identity or full resource work.',
  'PMU timestamp nearest TASK end is a candidate, not proven replay ID or clock-semantic rule; threshold 1000ns is an explicitly stated search window, not a clock-error guarantee.',
  'Ratios are reported pipeline proxies; do not infer chip utilization, causal wait, HBM saturation, exclusive time or speedup.',
  'Per-pipeline metrics may overlap; neither sum nor subtract them to calculate total wait.',
  'No device workload is launched.']}
print(json.dumps(out,ensure_ascii=False,indent=2,default=str))
