#!/usr/bin/env python3
"""Compare saved v2/v5 main-stream per-type AIC proxy aggregates; CPU only."""
from pathlib import Path
from decimal import Decimal as D
import hashlib,json
T=Path(__file__).resolve().parents[2];A=T/'agents/diagnosis/draft-zero-attention-types.json';B=T/'agents/diagnosis/draft-target-operator-decomposition.json';V=T/'eager-draft-mainstream-attribution-v5/draft_call_metrics.json';S=T/'eager-draft-mainstream-attribution-v2/draft_call_metrics.json'
exp={A:'9a1a75cd23ebfc4a1001f2eabd92a06d794a976225254ca4d98745e7ed9919e5',B:'7384031bcfa37219d347f535ae5cf8cf7a099d4f89565867af7288de9d795197',V:'6473445dda048c157cf7e451bf959b973c5138a05e0272172a5ea98cfef21923',S:'0e6f0cb4d810fdd9f7ec99f7b9a7dbe2123a7db7e5a927115e721b8fd8982c48'}
for p,h in exp.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==h
zero=json.loads(A.read_text());one=json.loads(B.read_text()); v=json.loads(V.read_text()); strict=json.loads(S.read_text());key=lambda q:(q['rank'],q['start_ns'],q['end_ns'])
assert len(v)==288 and len(strict)==162 and len(zero['windows'])==126
assert {key(a) for a in strict}=={key(a) for a in v if a['attention_count_allstreams']==1}
assert {(a['rank'],a['start'],a['end']) for a in zero['windows']}=={key(a) for a in v if a['attention_count_allstreams']==0}
assert all(zero['csv_sha256_by_rank'][k]==h for k,h in one['csv_sha256_by_rank'].items())
def summarize(rows):
 den=sum(D(a['den']) for a in rows);num=sum(D(a['num']) for a in rows);m=next(a for a in rows if a['type']=='DispatchFFNCombine'); md=D(m['den']);mn=D(m['num']); assert den>md>0
 return {'rows':sum(a['rows'] for a in rows),'skipped':sum(a['skipped_no_positive_cycles'] for a in rows),'num':str(num),'den':str(den),'proxy_percent':str(100*num/den),'mc2_count':m['rows'],'mc2_den_percent':str(100*md/den),'mc2_num_percent':str(100*mn/num),'mc2_proxy_percent':str(100*mn/md),'other_proxy_percent':str(100*(num-mn)/(den-md))}
z=summarize(zero['operators']);s=summarize([a for a in one['operators'] if a['kind']=='draft']);assert z['mc2_count']==126 and s['mc2_count']==162
for label,group,windows in [('zero',z,zero['windows']),('strict',s,[a for a in one['windows'] if a['kind']=='draft'])]:
 assert D(group['num'])==sum((D(a['num']) for a in windows),D(0));assert D(group['den'])==sum((D(a['den']) for a in windows),D(0));assert group['skipped']==sum(a['skipped'] for a in windows)
print(json.dumps({'zero':z,'strict':s,'matched_rank_count':len(one['csv_sha256_by_rank']),'rank_scope':'zero 16 ranks, strict 15 ranks; unpaired windows and unverified workload equality','limits':['AIC total_cycles weighted exported MAC ratio: not elapsed time or entire NPU utilisation','other types and MC2 differ in workload and coverage; none establishes causation','duplicate or overlapping exported rows counted as previous method; unknown precision in CSV ratio']},indent=2))
