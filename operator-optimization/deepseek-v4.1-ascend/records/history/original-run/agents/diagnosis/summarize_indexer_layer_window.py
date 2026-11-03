#!/usr/bin/env python3
"""Summarize Indexer occupancy of matched per-rank source-attributed layer windows."""
from pathlib import Path
import hashlib,json,statistics
T=Path(__file__).resolve().parents[2]
P=T/'agents/diagnosis/indexer-attention-paired-runs.json'
W=T/'captured-layer-attribution/module-call-metrics.json'
F=T/'agents/fusion/f07-q-kv-streams.json'
EXPECTED={P:'d8d9a7e651a82616b5387557f4568dba96ed78b30da17cccf575da9b602de0d3',W:'d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a',F:'04f9f65cb91b5dbf76456b411923f1a8032e1d7897d93a66ca1aa7c49ab82ec5'}
for p,v in EXPECTED.items():
 if hashlib.sha256(p.read_bytes()).hexdigest()!=v:raise RuntimeError('input SHA mismatch')
x=json.loads(P.read_text());window_rows=json.loads(W.read_text());windows={}
for w in window_rows:
 if w['run_in_model'] not in (1,2,3) or w['layer'] not in (2,8,14,20,24,28,32,36):continue
 key=(w['rank'],w['layer'],w['run_in_model'])
 if key in windows:raise RuntimeError('duplicate window')
 windows[key]=w
if len(windows)!=384:raise RuntimeError('wrong window count')
selected={}
for r in x['samples']:
 for prefix,run in (('from',r['from_run']),('to',r['to_run'])):
  key=(r['rank'],r['layer'],run)
  obs={kind:{'start_ns':r[kind][prefix+'_start_ns'],'end_ns':r[kind][prefix+'_end_ns']} for kind in ('indexer','attention')}
  if key in selected and selected[key]!=obs:raise RuntimeError('inconsistent pair values')
  selected[key]=obs
if set(selected)!=set(windows):raise RuntimeError('different selected windows')
rows=[]
for key,w in sorted(windows.items()):
 rank,layer,run=key; a=selected[key]['indexer'];b=selected[key]['attention'];start=w['window_start_ns'];end=w['window_end_ns'];length=end-start
 if length<=0 or not (start<=a['start_ns']<a['end_ns']<=b['start_ns']<b['end_ns']<=end):raise RuntimeError('invalid task/window ordering')
 i=a['end_ns']-a['start_ns'];att=b['end_ns']-b['start_ns'];gap=b['start_ns']-a['end_ns']
 rows.append({'rank':rank,'layer':layer,'run':run,'model_id':w['model_id'],'main_stream_id':w['main_stream_id'],
  'window_start_ns':start,'window_end_ns':end,'window_duration_us':length/1000,'indexer':a,'attention':b,
  'indexer_duration_us':i/1000,'attention_duration_us':att/1000,'gap_duration_us':gap/1000,
  'indexer_fraction':i/length,'attention_fraction':att/length,'indexer_plus_attention_fraction':(i+att)/length})
def stats(v):return {'min':min(v),'median':statistics.median(v),'max':max(v)}
summary=[]
for layer in (2,8,14,20,24,28,32,36):
 for run in (1,2,3):
  v=[r for r in rows if r['layer']==layer and r['run']==run]
  if len(v)!=16 or len({r['rank'] for r in v})!=16:raise RuntimeError('incomplete group')
  summary.append({'layer':layer,'run':run,'samples':len(v),**{k:stats([r[k] for r in v]) for k in ('window_duration_us','indexer_duration_us','attention_duration_us','gap_duration_us','indexer_fraction','attention_fraction','indexer_plus_attention_fraction')}})
f07=json.loads(F.read_text()); selected_f07=[]
for v in f07['samples']:
 if v['layer'] not in (2,24):continue
 key=(f07['rank'],v['layer'],v['run']);w=windows[key];r=selected[key]
 if str(v['model'])!=str(w['model_id']) or str(v['main_stream'])!=str(w['main_stream_id']):raise RuntimeError('f07 provenance mismatch')
 start=r['indexer']['start_ns']-w['window_start_ns'];end=r['indexer']['end_ns']-w['window_start_ns']
 other=[e for e in v['events'] if e['stream']!=v['main_stream'] and e['start_rel_ns']<end and e['end_rel_ns']>start]
 selected_f07.append({'rank':f07['rank'],'layer':v['layer'],'run':v['run'],'selected_other_stream_intersections':len(other),'types':sorted({e['type'] for e in other})})
if len(selected_f07)!=6:raise RuntimeError('f07 scope incomplete')
print(json.dumps({'inputs_sha256':{str(p.relative_to(T)):v for p,v in EXPECTED.items()},'rows':rows,'summary':summary,
 'f07_selected_scope':selected_f07,'scope':'source-attributed same-rank model and stream per-layer windows; 16 ranks × 8 layers × 3 runs',
 'limits':['ratios describe same-main-stream window occupancy, not exposed critical path or end-to-end benefit',
 'F07 is only six EP8 windows with selected types, same model and fully window-contained tasks; no intersections do not exclude other operations or cross-boundary tasks',
 'other streams, host, other devices, and unrecorded tasks may overlap; no resource attribution',
 'different runs are not established as same request or workload; time boundaries rely on exporter'],
 'no_device_work':True},indent=2,ensure_ascii=False))
