#!/usr/bin/env python3
"""Read existing selected F08 triples and captured scope tasks; no NPU use."""
import hashlib,json,statistics
from collections import Counter,defaultdict
from decimal import Decimal as D
from pathlib import Path
R=Path(__file__).resolve().parents[2]; p=R/'agents/fusion'; src=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
t=json.loads((p/'f08-actual-triples.json').read_text()); v=json.loads((p/'f08-scatter-variation.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(src)==t['source_sha256']==v['source_sha256']
want={(x['ep'],x['scope_id']):x for x in t['records'] if x['rows']==8180}; samples={(x['ep'],x['scope']):x for x in v['all_samples_sorted']};assert len(want)==len(samples)==640 and set(want)==set(samples)
rows=[]
for ln in src.open():
 x=json.loads(ln)
 if x.get('kind')!='rank':continue
 ep=x['data']['ep']
 for scope in x['data']['layers']:
  key=(ep,scope['layer_scope_id'])
  if key not in want:continue
  assert scope['trace_scope_count']==1 and scope['sidecar_thread_matches_trace']
  ts=[(g['operator_name'],task) for g in scope['operator_groups'] for task in g['tasks']]; match=[(n,y) for n,y in ts if y['flow_id']==want[key]['flow_ids'][2]];assert len(match)==1
  name,target=match[0]; csv=target['csv'];s=samples[key]; dur=D(target['trace_duration_us']);assert D(str(s['duration_us']))==dur
  assert target['device_start_us']==s['start_us'] and csv['Task ID']==s['task']
  a=D(target['device_start_us']);b=a+dur; intervals=[];same=[]; labels=Counter(); matmuls=[]
  for n,o in ts:
   if o is target:continue
   c=D(o['device_start_us']);d=c+D(o['trace_duration_us']);lo=max(a,c);hi=min(b,d)
   if 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3' in n: matmuls.append((n,c,d))
   if hi>lo:
    if str(o['device_track_tid'])==str(target['device_track_tid']):same.append(str(hi-lo))
    else:intervals.append((lo,hi));labels[n]+=1
  intervals.sort(); merged=[]
  for lo,hi in intervals:
   if not merged or lo>merged[-1][1]:merged.append([lo,hi])
   else:merged[-1][1]=max(merged[-1][1],hi)
  union=sum((hi-lo for lo,hi in merged),D(0)); assert union<=dur
  nearest=min(matmuls,key=lambda z: max(D(0),z[1]-b,a-z[2])) if matmuls else None
  nearest_relation=('overlap' if nearest[1]<b and nearest[2]>a else 'after' if nearest[1]>=b else 'before') if nearest else 'none'
  nearest_gap=str(max(D(0),nearest[1]-b,a-nearest[2])) if nearest else None
  rows.append({'nearest_quant_matmul_relation':nearest_relation,'nearest_quant_matmul_gap_us':nearest_gap,'quant_matmul_in_scope_n':len(matmuls),'ep':ep,'layer':s['layer'],'digest':s['digest'],'scope':s['scope'],'class':'short' if dur<50 else 'long' if dur>200 else 'middle','duration_us':str(dur),'aiv_cycles':int(s['aiv_cycles']),'stream':s['stream'],'model_id':csv['Model ID'],'scatter_name':name,'track_tid':str(target['device_track_tid']),'host_tid':str(target['host_tid']),'scope_tasks':len(ts),'trace_scope_count':scope['trace_scope_count'],'matched_task_count':scope['matched_task_count'],'other_track_overlap_union_us':str(union),'other_track_overlap_fraction':str(union/dur),'other_track_overlap_task_count':sum(labels.values()),'same_track_overlap_task_count':len(same),'overlap_names':dict(labels)})
assert len(rows)==640

def agg(xs):
 f=lambda x:Counter(str(r[x]) for r in xs)
 frac=[float(r['other_track_overlap_fraction']) for r in xs]; cycles=[r['aiv_cycles'] for r in xs]; dur=[float(r['duration_us']) for r in xs];names=Counter()
 for r in xs:names.update(r['overlap_names'])
 return {'n':len(xs),'eps':dict(f('ep')),'digests':dict(f('digest')),'streams':dict(f('stream')),'model_ids':dict(f('model_id')),'names':dict(f('scatter_name')),'tracks':dict(f('track_tid')),'host_tids':dict(f('host_tid')),'scope_tasks':dict(f('scope_tasks')),'trace_scope_count':dict(f('trace_scope_count')),'matched_task_count':dict(f('matched_task_count')),'duration_median_us':statistics.median(dur) if dur else None,'aiv_cycles_median':statistics.median(cycles) if cycles else None,'overlap_nonzero_n':sum(x>0 for x in frac),'overlap_fraction_min_median_max':[min(frac),statistics.median(frac),max(frac)] if frac else None,'same_track_overlap_n':sum(r['same_track_overlap_task_count']>0 for r in xs),'nearest_matmul_relations':dict(f('nearest_quant_matmul_relation')),'quant_matmul_in_scope_n':dict(f('quant_matmul_in_scope_n')), 'nearest_matmul_nonoverlap_gap_min_median_max': ([min(gaps),statistics.median(gaps),max(gaps)] if (gaps:=[float(r['nearest_quant_matmul_gap_us']) for r in xs if r['nearest_quant_matmul_relation'] in ('after','before')]) else None),'other_track_operator_counts':dict(names.most_common())}
out={'source_sha256':sha(src),'triples_sha256':sha(p/'f08-actual-triples.json'),'variation_sha256':sha(p/'f08-scatter-variation.json'),'scope':'Only other events in the extracted selected layer scope; no full-device coverage or cross-stream synchronization implied.','all':{c:agg([r for r in rows if r['class']==c]) for c in ('short','long')},'ep3_db852':{c:agg([r for r in rows if r['ep']==3 and r['digest'].startswith('db852') and r['class']==c]) for c in ('short','long')},'ep3_db852_rows':[r for r in rows if r['ep']==3 and r['digest'].startswith('db852')]}
(p/'f08-scope-overlap-audit.json').write_text(json.dumps(out,indent=2)+'\n')
for k in ('all','ep3_db852'):
 print(k)
 for c,x in out[k].items():print(c,json.dumps(x,ensure_ascii=False))
