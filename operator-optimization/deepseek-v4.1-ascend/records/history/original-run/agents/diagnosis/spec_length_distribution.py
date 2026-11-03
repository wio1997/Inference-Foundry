#!/usr/bin/env python3
"""Conditional reconstruction from saved position counters and current source semantics."""
import hashlib,json
from pathlib import Path
T=Path(__file__).resolve().parents[2]
paths=[T/'agents/diagnosis/spec-position-accounting.json',T/'agents/diagnosis/spec-counter-workspace-semantics.json']
x,y=[json.loads(p.read_text()) for p in paths]
src=[f for f in y['files'] if f['path'].endswith('/v1/spec_decode/metrics.py')]
assert len(src)==1
assert src[0]['sha256']=='35e0a709042471559937f1a1979a721af9b910476efaa8b6e29865bace985afe'
out=[]
for row in x['rows']:
 n=int(row['drafts']);a=list(map(int,row['accepted_per_position']));total=int(row['accepted'])
 assert n>=a[0] and all(a[i]>=a[i+1] for i in range(len(a)-1)) and sum(a)==total
 dist=[n-a[0]]+[a[i]-a[i+1] for i in range(len(a)-1)]+[a[-1]]
 assert sum(dist)==n and sum(i*c for i,c in enumerate(dist))==total
 out.append({'phase':row['phase'],'drafts':n,'accepted_counts_0_to_5':dist,'accepted_at_least_1_to_5':a,'accepted_at_least_1_to_5_over_all_drafts':[v/n for v in a]})
print(json.dumps({'scope':'Conditional on historical observation semantics matching current source and matched counter snapshots','source_sha256':src[0]['sha256'],'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'rows':out,'limitations':['Accepted counts are not individual requests or per-position success conditional on proposal.','No link to profile task windows or draft isolated latency.','Current workspace source is not a historical live-object snapshot.']},ensure_ascii=False,indent=2))
