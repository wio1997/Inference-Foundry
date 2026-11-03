#!/usr/bin/env python3
"""Audit per-name minimum versus unique most frequent physical signature (offline)."""
from collections import Counter,defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parents[2]
extractor=root/'agents/diagnosis/analyze_recapture_all_kernel_groups.py'
x=json.loads(subprocess.check_output(['python3',str(extractor),'--all-groups'],cwd=root))
assert x['source_sha256']=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert x['aic_eligible_group_count']==1596 and x['aiv_eligible_group_count']==3966

def sig(g):
 return json.dumps({'core':g['core'],'physical_shape_dtype_format':g['physical_shape_dtype_format'],'per_rank_task_count':g['per_rank_task_count']},sort_keys=True,separators=(',',':'))
def point(g):
 return {'layer':g['layer'],'ratio':g['weighted_displayed_ratio'],'signature':json.loads(sig(g))}
out={'source_sha256':x['source_sha256'],'extractor_sha256':sha256(extractor.read_bytes()).hexdigest(),'selection':'same engine, digest, raw name, csv type; require >=5 eligible eight-rank layer/signature groups and >=2 signatures; tied dominant signatures reported separately','engines':{}}
for engine in ('aic','aiv'):
 data=defaultdict(list)
 for g in x['all_'+engine+'_eligible_groups']:data[(g['digest'],g['operator_name'],g['csv_type'])].append(g)
 audit=[];counts=Counter()
 for key,gs in sorted(data.items()):
  if len(gs)<5:continue
  bysig=defaultdict(list)
  for g in gs:bysig[sig(g)].append(g)
  if len(bysig)<2:continue
  counts['eligible_multi_signature_operator_digest']+=1
  mx=max(map(len,bysig.values()));leaders=[k for k,v in bysig.items() if len(v)==mx]
  minimum=min(Decimal(g['weighted_displayed_ratio']) for g in gs)
  minrows=[g for g in gs if Decimal(g['weighted_displayed_ratio'])==minimum]
  if len(leaders)!=1:counts['tied_most_frequent_signature']+=1;continue
  counts['unique_most_frequent_signature']+=1
  dom=leaders[0];domrows=bysig[dom]
  outside=all(sig(g)!=dom for g in minrows)
  counts['all_minima_outside_unique_most_frequent_signature']+=outside
  dm=min(domrows,key=lambda g:(Decimal(g['weighted_displayed_ratio']),g['layer']))
  audit.append({'digest':key[0],'operator_name':key[1],'csv_type':key[2], 'groups':len(gs),'distinct_signatures':len(bysig),'dominant_signature_groups':len(domrows),'global_minimum_groups':len(minrows),'minimum_signature_groups':len(bysig[sig(minrows[0])]),'minimum_outside_dominant':outside,'minimum':point(minrows[0]),'dominant_minimum':point(dm)})
 out['engines'][engine]={'counts':dict(counts),'unique_dominant_groups':audit}
print(json.dumps(out,ensure_ascii=False,indent=2))
