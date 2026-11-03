#!/usr/bin/env python3
"""Recheck saved main-stream MC2 decomposition; no device calls."""
import csv, hashlib, json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
T=Path(__file__).resolve().parents[2]
inputs={
 'draft_mc2':('draft-mc2-analysis/strict_draft_calls.csv','8c8b6539870fce5fd21a089c1eb3556fd70e2d458f37f87fb16ea24342b1c75f'),
 'draft_calls':('eager-draft-mainstream-attribution-v2/draft_call_metrics.json','0e6f0cb4d810fdd9f7ec99f7b9a7dbe2123a7db7e5a927115e721b8fd8982c48'),
 'metrics':('eager-draft-mainstream-attribution-v2/module_metrics.csv','9e96d35b57fe4652da6e955e3a0f1318f8638b5094f42b12c9c2299c3b43fe23'),
 'target_mc2':('agents/diagnosis/layer0-dp-type-counters.json','0482575955229199d9b2c25e1cff2590d82a27d8a91f89e0d8f3418e7be37fad'),
 'target_calls':('captured-layer-attribution/module-call-metrics.json','d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a')}
for k,(p,h) in inputs.items():
 if hashlib.sha256((T/p).read_bytes()).hexdigest()!=h:raise RuntimeError('HASH_MISMATCH '+k)
with (T/inputs['draft_mc2'][0]).open() as f: dm=list(csv.DictReader(f))
with (T/inputs['metrics'][0]).open() as f: metrics=list(csv.DictReader(f))
dc=json.loads((T/inputs['draft_calls'][0]).read_text()); tc=json.loads((T/inputs['target_calls'][0]).read_text()); tm=json.loads((T/inputs['target_mc2'][0]).read_text())['mc2_by_window']
dkeys={(x['rank'],x['start_ns'],x['end_ns']):x for x in dc}
if len(dkeys)!=162 or len(dm)!=162:raise RuntimeError('DRAFT_COVERAGE')
ranks={x['rank'] for x in dc}
if len(ranks)!=15:raise RuntimeError('RANK_COUNT')
seen=set();dn=Decimal(0);dd=Decimal(0);bn=Decimal(0);bd=Decimal(0);dt=Decimal(0);bt=Decimal(0)
for x in dm:
 key=(x['rank'],int(x['bracket_start_ns']),int(x['bracket_end_ns']))
 if key in seen or key not in dkeys:raise RuntimeError('DRAFT_WINDOW_IDENTITY')
 seen.add(key);ref=dkeys[key]
 if (Decimal(x['block_aic_num'])!=Decimal(ref['aic_mac_proxy_numerator']) or Decimal(x['block_aic_den'])!=Decimal(ref['aic_cycles_denominator']) or not key[1]<=int(x['mc2_start_ns'])<=int(x['mc2_end_ns'])<=key[2]):raise RuntimeError('DRAFT_COUNTER_OR_BOUNDARY')
 n,d=Decimal(x['mc2_aic_num']),Decimal(x['mc2_aic_den']); n2,d2=Decimal(x['block_aic_num']),Decimal(x['block_aic_den'])
 if not (0<=n<=n2 and 0<d<=d2):raise RuntimeError('DRAFT_SUBSET_COUNTER')
 if Decimal(x['mc2_duration_us'])*1000!=int(x['mc2_end_ns'])-int(x['mc2_start_ns']) or Decimal(x['bracket_duration_us'])*1000!=key[2]-key[1]:raise RuntimeError('DURATION_MISMATCH')
 dn+=n;dd+=d;bn+=n2;bd+=d2;dt+=Decimal(x['mc2_duration_us']);bt+=Decimal(x['bracket_duration_us'])
draftrows=[x for x in metrics if x['kind']=='EAGER_DRAFT_POOLED']
if len(draftrows)!=15 or {x['rank'] for x in draftrows}!=ranks or sum((Decimal(x['aic_mac_proxy_numerator']) for x in draftrows),Decimal(0))!=bn or sum((Decimal(x['aic_cycles_denominator']) for x in draftrows),Decimal(0))!=bd:raise RuntimeError('DRAFT_MODULE_AGG_MISMATCH')
targetrows=[x for x in metrics if x['rank'] in ranks and x['module']=='target.layers.0.forward']
if len(targetrows)!=15 or {x['rank'] for x in targetrows}!=ranks or any(int(x['calls'])!=3 for x in targetrows):raise RuntimeError('TARGET_AGG_COVERAGE')
tkeys={(x['rank'],x['run_in_model']):x for x in tc if x['rank'] in ranks and x['layer']==0 and x['run_in_model'] in (1,2,3)}
trows=[x for x in tm if x['rank'] in ranks]
if len(tkeys)!=45 or len(trows)!=45 or {(x['rank'],x['run']) for x in trows}!=set(tkeys):raise RuntimeError('TARGET_WINDOW_COVERAGE')
for x in trows:
 if x['stream']!=tkeys[(x['rank'],x['run'])]['main_stream_id']:raise RuntimeError('TARGET_STREAM_MISMATCH')
tn=sum((Decimal(x['aic_cycles'])*Decimal(x['aic_mac_ratio']) for x in trows),Decimal(0));td=sum((Decimal(x['aic_cycles']) for x in trows),Decimal(0)); tbn=sum((Decimal(x['aic_mac_proxy_numerator']) for x in targetrows),Decimal(0));tbd=sum((Decimal(x['aic_cycles_denominator']) for x in targetrows),Decimal(0))
for rank in ranks:
 ss=[x for x in trows if x['rank']==rank];tt=next(x for x in targetrows if x['rank']==rank)
 rn=sum((Decimal(x['aic_cycles'])*Decimal(x['aic_mac_ratio']) for x in ss),Decimal(0));rd=sum((Decimal(x['aic_cycles']) for x in ss),Decimal(0))
 if rn>Decimal(tt['aic_mac_proxy_numerator']) or rd>Decimal(tt['aic_cycles_denominator']):raise RuntimeError('TARGET_SUBSET_COUNTER')
def out(n,d,total_n,total_d):
 if not 0<d<total_d or not 0<=n<=total_n:raise RuntimeError('INVALID_PARTITION')
 return {'mc2_numerator':str(n),'mc2_cycles':str(d),'total_numerator':str(total_n),'total_cycles':str(total_d),'mc2_denominator_share_pct':str(100*d/total_d),'mc2_mac_proxy_pct':str(100*n/d),'other_mac_proxy_pct':str(100*(total_n-n)/(total_d-d)),'total_mac_proxy_pct':str(100*total_n/total_d)}
print(json.dumps({'scope':'Saved strict162 pooled draft versus target layer0 45 windows, same15 ranks and main-stream positive-AIC rows', 'draft':out(dn,dd,bn,bd),'target_layer0':out(tn,td,tbn,tbd),'draft_mc2_duration_sum_over_bracket_duration_sum_pct':str(100*dt/bt),'input_sha256':{k:v[1] for k,v in inputs.items()},'limits':['Cycle-weighted attribution is reported-counter arithmetic, not physical work, causality, idle or end-to-end saving.', '162 selected draft windows and45 target windows differ in model work, sample and branch selection.', 'Draft raw CSV source hash is not independently established here; stage159 verifies its own source.', 'Only main stream; unrepresented and nonpositive AIC rows excluded.', 'No device calls.']},indent=2))
