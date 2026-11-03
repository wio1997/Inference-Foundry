#!/usr/bin/env python3
"""CPU-only conditional shape/order statistics on nine saved EP8 windows."""
from pathlib import Path
import hashlib,json,statistics
p=Path(__file__).resolve().parent;src=p/'f07-q-kv-streams.json';x=json.loads(src.read_text());out=[]
def time(a,b):return (b-a)/1000
def overlap(a,b):return max(0,min(a['end_rel_ns'],b['end_rel_ns'])-max(a['start_rel_ns'],b['start_rel_ns']))/1000
for w in x['samples']:
 ev=w['events'];m=str(w['main_stream']);model=str(w['model']);own=lambda e:e['model']==model
 qb=[e for e in ev if own(e) and e['stream']==m and e['type']=='QuantBatchMatmulV3' and e['input_shapes'].strip('"').startswith('3,1280;1024,80,16,32;')]
 sf=[e for e in ev if own(e) and e['stream']==m and e['type']=='SparseFlashMla']
 if len(sf)!=1:raise RuntimeError('attention shape or bracket ambiguous')
 rr=[e for e in ev if own(e) and e['stream']==m and e['type']=='InplacePartialRotaryMul' and e['input_shapes'].strip('"').startswith('3,1,64,512;') and e['end_rel_ns']<=sf[0]['start_rel_ns']]
 ss=[e for e in ev if own(e) and e['stream']!=m and e['type']=='ScatterNdUpdateSk' and e['input_shapes'].strip('"').endswith(';24,2;24,512')]
 vv=[e for e in ev if own(e) and e['stream']!=m and e['type']=='InplacePartialRotaryMul' and e['input_shapes'].strip('"').startswith('24,1,1,512;')]
 if list(map(len,(qb,sf,rr,ss,vv)))!=[1,1,1,1,1]:raise RuntimeError('source-shape candidates not unique in window')
 q,r,s,v=qb[0],rr[0],ss[0],vv[0]
 out.append({'run':w['run'],'layer':w['layer'],'main_stream':m,'other_stream':s['stream'],
  'qb_us':time(q['start_rel_ns'],q['end_rel_ns']),'pre_attention_rope_us':time(r['start_rel_ns'],r['end_rel_ns']),
  'scatter_tail_after_qb_us':time(q['end_rel_ns'],s['end_rel_ns']),'qb_rope_gap_us':time(q['end_rel_ns'],r['start_rel_ns']),
  'scatter_end_to_rope_start_us':time(s['end_rel_ns'],r['start_rel_ns']),'qb_scatter_overlap_us':overlap(q,s),'qb_aux_rope_overlap_us':overlap(q,v)})
assert len(out)==9
fields=('qb_us','pre_attention_rope_us','scatter_tail_after_qb_us','qb_rope_gap_us','scatter_end_to_rope_start_us','qb_scatter_overlap_us','qb_aux_rope_overlap_us')
stats={k:{'min':min(v[k] for v in out),'median':statistics.median(v[k] for v in out),'max':max(v[k] for v in out)} for k in fields}
result={'input_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'sample_count':len(out),'samples':out,'statistics_us':stats,
 'limitations':['Shape-and-order associations are candidates, not tensor identity, Q/KV ownership, or proven event dependencies.',
 'One rank, three selected layers and three internal sampling ordinals; do not generalize to all workload.',
 'Intervals may overlap on one device; observed overlap does not prove a proposed rearrangement can overlap without contention or speedup.']}
dst=p/'f07-q-kv-stream-summary.json';body=(json.dumps(result,indent=2,ensure_ascii=False)+'\n').encode()
if dst.exists() and dst.read_bytes()!=body:raise RuntimeError('different preexisting summary')
if not dst.exists():dst.write_bytes(body)
print(json.dumps({'sha256':hashlib.sha256(body).hexdigest(),'statistics_us':stats},indent=2))
