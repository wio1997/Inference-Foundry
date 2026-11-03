#!/usr/bin/env python3
"""Recompute selected three-event spans for preexisting 4760-row F08 scopes."""
import hashlib,json,statistics
from decimal import Decimal as D
from pathlib import Path
R=Path(__file__).resolve().parents[2];src=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';sel=R/'agents/fusion/f08-4760-rank-overlap.json'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(sel.read_bytes()).hexdigest()=='ea892e7faf87ad87e010b3c40c5a4f5a5ecbc711ec06b7d2e9107b3e9cbcab5f'
w={(x['ep'],x['scope_id']):x for x in json.loads(sel.read_text())['records']};assert len(w)==320
out=[];seen=set()
for line in src.open():
 j=json.loads(line)
 if j.get('kind')!='rank':continue
 for scope in j['data']['layers']:
  key=scope['ep'],scope['layer_scope_id'];r=w.get(key)
  if not r:continue
  assert key not in seen;seen.add(key)
  ts=[t for g in scope['operator_groups'] for t in g['tasks']];by={t['flow_id']:t for t in ts};assert len(by)==len(ts)
  norm,rope,scatter=[by[f] for f in r['flow_ids']]
  assert [t['csv']['Type'] for t in (norm,rope,scatter)]==['RmsNorm','InplacePartialRotaryMul','ScatterNdUpdateSk']
  assert len({t['csv']['Stream ID'] for t in (norm,rope,scatter)})==1
  n,q,a=[D(t['device_start_us']) for t in (norm,rope,scatter)];ne=n+D(norm['trace_duration_us']);qe=q+D(rope['trace_duration_us']);ae=a+D(scatter['trace_duration_us']);assert n<=ne<=q<=qe<=a<=ae
  assert D(r['duration_us'])==ae-a
  out.append({'ep':r['ep'],'layer':r['layer'],'scope_id':r['scope_id'],'class':r['class'],'norm_to_rope_gap_us':str(q-ne),'rope_to_scatter_gap_us':str(a-qe),'three_event_span_us':str(ae-n),'scatter_duration_us':r['duration_us']})
assert seen==set(w)
def stat(v,k):
 x=[D(r[k]) for r in v];return [str(min(x)),str(statistics.median(x)),str(max(x))]
groups={}
for cl in ('short','long'):
 v=[r for r in out if r['class']==cl];groups[cl]={'count':len(v),**{k:stat(v,k) for k in ('norm_to_rope_gap_us','rope_to_scatter_gap_us','three_event_span_us')}}
assert len([r for r in out if r['class']=='short'])==41
assert D(groups['short']['three_event_span_us'][0])>D(groups['long']['three_event_span_us'][2])
examples=[r for r in out if r['ep'] in (0,3) and r['layer']==9];assert len(examples)==2
result={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256(sel.read_bytes()).hexdigest(),'groups':groups,'examples':examples,'records':out,
 'limits':['Three selected same-stream event endpoints only, not full layer, resource idleness, critical path, or request latency.','Sample-specific observation; no cross-device timestamp subtraction; no tensor identity or causality inferred.']}
p=R/'agents/fusion/f08-4760-gap-spans.json';p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'groups':groups,'examples':examples},indent=2));print('output_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
