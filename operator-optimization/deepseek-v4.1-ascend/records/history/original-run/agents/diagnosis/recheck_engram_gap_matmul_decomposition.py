import json,hashlib,statistics
from pathlib import Path
inputs={
 'agents/diagnosis/engram-boundary-controls.json':'b88c60f030b6994934aed5cfd37a50b5ab2a0e23060149a27c66415ea4d4e765',
 'agents/diagnosis/engram-pairwise-existing.json':'0487234fa2637a5c1c6add39bb54960b865f11b25644dedc9c3792fb15be9efc'}
data=[]
for name,expected in inputs.items():
 raw=Path(name).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError('input hash mismatch: '+name)
 data.append(json.loads(raw))
samples=data[0]['samples'];rows=data[1]['rows']
idx={}
for s in samples:
 k=(s['rank'],s['run'],s['after_layer'])
 if k in idx:raise ValueError('duplicate scope')
 idx[k]=s
if len(rows)!=48:raise ValueError('incomplete pairs')
seen=set();out=[]
for row in rows:
 k=(row['rank'],row['run'])
 if k in seen:raise ValueError('duplicate pair')
 seen.add(k);a=idx[k+(0,)];b=idx[k+(13,)]
 for s,suffix in [(a,'after0'),(b,'after13')]:
  es=[e for e in s['events'] if e['type']=='MatMulV2' and e.get('input_shapes')=='"3,6144;25600,6144"']
  if len(es)!=1:raise ValueError('non-unique matmul')
  e=es[0]
  if e['event_end_rel_ns']-e['event_start_rel_ns']!=row['duration_ns_'+suffix]:raise ValueError('duration mismatch')
  if e['event_start_rel_ns']!=e['clipped_start_rel_ns'] or e['event_end_rel_ns']!=e['clipped_end_rel_ns']:raise ValueError('event clipped')
 diff=a['gap_ns']-b['gap_ns'];mat=row['duration_difference_ns']
 if mat!=row['duration_ns_after0']-row['duration_ns_after13']:raise ValueError('pair mismatch')
 out.append({'run':row['run'],'gap_difference_ns':diff,'matmul_difference_ns':mat,'residual_ns':diff-mat})
summary=[]
for run in [1,2,3]:
 ss=[x for x in out if x['run']==run]
 if len(ss)!=16:raise ValueError('unexpected rank count')
 def stats(field):
  z=[v[field]/1000 for v in ss]
  return {'min':min(z),'median':statistics.median(z),'max':max(z)}
 summary.append({'run':run,'pairs':len(ss),'first_gap_longer':sum(x['gap_difference_ns']>0 for x in ss),'gap_diff_us':stats('gap_difference_ns'),'matmul_diff_us':stats('matmul_difference_ns'),'residual_us':stats('residual_ns')})
print(json.dumps({'scope':'48 same-rank/profile-round pairs; selected layer boundary windows, descriptive not causal','inputs_sha256':inputs,'event_identity':'same exported scope and shape-duration match, not tensor/source identity','summary':summary},indent=2))
