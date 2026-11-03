#!/usr/bin/env python3
"""CPU-only recheck of frozen layer-gap counters against known BF16 candidate samples."""
import csv,hashlib,json
from collections import Counter
from decimal import Decimal as D
from pathlib import Path
T=Path(__file__).resolve().parents[2]
files={
 'gap':('agents/diagnosis/engram-boundary-mainstream-counters.json','c5a915dfcaf615ba878d082a31f9820ec98e3878f7f5c0bc136d7fd7794953e7'),
 'matmul':('agents/diagnosis/16rank-engram-matmul-metrics.json','35ff1b007a471903324d96de5eeb67b80a13251a4c4dadd8619eead77c23db1d'),
 'draft_ranks':('eager-draft-mainstream-attribution-v2/module_metrics.csv','9e96d35b57fe4652da6e955e3a0f1318f8638b5094f42b12c9c2299c3b43fe23')}
for k,(p,s) in files.items():
 if hashlib.sha256((T/p).read_bytes()).hexdigest()!=s:raise RuntimeError('hash mismatch '+k)
gap=json.loads((T/files['gap'][0]).read_text()); mm=json.loads((T/files['matmul'][0]).read_text())
if gap['kernel_sha256_by_rank']!=mm['kernel_sha256_by_rank']:raise RuntimeError('CSV sources mismatch')
with (T/files['draft_ranks'][0]).open() as f:ranks={r['rank'] for r in csv.DictReader(f) if r['kind']=='EAGER_DRAFT_POOLED'}
if len(ranks)!=15:raise RuntimeError('draft rank mismatch')
ms={(s['target']['rank'],s['target']['run'],s['target']['after_layer']):s for s in mm['samples']}
if len(ms)!=96:raise RuntimeError('matmul sample scope mismatch')
selected=[z for z in gap['rows'] if z['rank'] in ranks]
if len(selected)!=135 or len({(z['rank'],z['run'],z['after_layer']) for z in selected})!=135:raise RuntimeError('interval scope')
result=[]
for layer in (0,12,13):
 rows=[z for z in selected if z['after_layer']==layer]
 if len(rows)!=45 or any(sum(z['type_counts'].values())!=z['positive']+z['skipped'] for z in rows):raise RuntimeError('row count')
 n=sum((D(z['num']) for z in rows),D(0));d=sum((D(z['den']) for z in rows),D(0))
 r={'after_layer':layer,'windows':len(rows),'numerator':str(n),'denominator':str(d),'weighted_mac_proxy':str(n/d) if d else None,'positive_rows':sum(z['positive'] for z in rows),'skipped_nonpositive_or_missing_rows':sum(z['skipped'] for z in rows),'zero_denominator_windows':sum(D(z['den'])==0 for z in rows),'crossing_rows_excluded':sum(z['crossing'] for z in rows),'positive_count_per_window':dict(Counter(z['positive'] for z in rows)),'all_type_rows':dict(sum((Counter(z['type_counts']) for z in rows),Counter()))}
 if layer!=12:
  bn=bd=D(0)
  for z in rows:
   m=ms[(z['rank'],z['run'],layer)]; target=m['target'];f=m['csv_fields']
   if target['model']!=z['model'] or target['stream']!=z['stream'] or not(z['start_ns']<=target['start_ns']<target['end_ns']<=z['end_ns']):raise RuntimeError('bad task-window binding')
   if f['Type']!='MatMulV2' or f['Input Shapes']!='"3,6144;25600,6144"' or f['Input Data Types']!='DT_BF16;DT_BF16':raise RuntimeError('bad task identity')
   dc=D(f['aic_total_cycles']);nc=dc*D(f['aic_mac_ratio'])
   if dc<=0 or dc>D(z['den']) or nc>D(z['num']):raise RuntimeError('bad component')
   bn+=nc;bd+=dc
  r['matched_bf16_candidate']={'num':str(bn),'den':str(bd),'proxy':str(bn/bd),'fraction_of_recorded_aic_den':str(bd/d)}
  r['all_other_tasks']={'num':str(n-bn),'den':str(d-bd),'proxy':str((n-bn)/(d-bd)) if d>bd else None}
 result.append(r)
print(json.dumps({'scope':'same 15 ranks as strict draft, 3 runs, three pre-existing layer boundary intervals each; not per-module measurements','inputs_sha256':{k:s for k,(_,s) in files.items()},'results':result,'limits':['Positive-cycle weighted MAC proxy is neither whole-NPU utilization nor time.','These intervals are inferred boundaries; they contain multiple exported types, not validated Engram modules.','Only main stream selected; nonpositive or missing cycles excluded and not treated as idle. Zero denominator is unknown.','The large shape-matched candidate has not been tied to a historical Engram weight tensor.','Sample groups have unequal work and not matched requests; not global ranking or speedup.','All-type counts do not identify which individual types had positive counters; all-other-tasks is a subtraction, not specific operator attribution.']},ensure_ascii=False,indent=2))
