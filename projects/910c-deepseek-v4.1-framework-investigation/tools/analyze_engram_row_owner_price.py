#!/usr/bin/env python3
"""Pay original-call matched transport in N-owner WKV/gate sensitivities."""
from pathlib import Path
from hashlib import sha256
import json,zlib,itertools,statistics
from analyze_c2_producer_windows import price_exposure

ROOT=Path('audit/execution-search');BASE=Path('/Users/wio/work/branch-fusion-audit-20261104')
def main():
 compute_path=BASE/'.cache/first-token-v168-protocol-bodies.json.zlib';comm_path=BASE/'.cache/first-token-v168-collectives.json.zlib';arrival_path=BASE/'full-review/PREFILL-OA-JOINT-ARRIVAL.json';winpath=ROOT/'engram-execution-windows.json';ordinary_path=BASE/'full-review/INITIAL-HC-AND-ENGRAM-REQUESTS.json'
 compute=json.loads(zlib.decompress(compute_path.read_bytes()));comm=json.loads(zlib.decompress(comm_path.read_bytes()));arrival=json.loads(arrival_path.read_text());ws=json.loads(winpath.read_text())['windows'];oldfront=json.loads((ROOT/'engram-execution-frontier.json').read_text());pins=oldfront['source_profile_pins']
 assert sha256(compute_path.read_bytes()).hexdigest()==pins[str(compute_path)]
 pools={}
 for c in comm['profiles']:
  for w in c['windows']:
   for a in w['communications']:
    if a[4]=='hcom_alltoall_':pools.setdefault((c['DP'],c['TP'],a[7],a[8]),[]).append((a[1]-a[0])/1000)
 records=[]
 for p,c in zip(compute['profiles'],comm['profiles']):
  assert (p['database'],p['sha256'],p['DP'],p['TP'])==(c['database'],c['sha256'],c['DP'],c['TP'])
  for body,cw in zip(p['bodies'],c['windows']):
   assert body['F21_carrier_rows']==cw['carrier_rows'];by={r[0]:r for r in body['rows']};pres=[by[i] for i in body['HcPre_TASK_ROWS']];stream,={r[3] for r in pres};posts=sorted([r for r in body['rows'] if r[3]==stream and p['regimes'][r[-1]][1]=='HcPost'],key=lambda r:r[1]);assert len(posts)==80
   for layer in [1,14]:
    pre=pres[2*layer];post=posts[2*layer];w,=[r for r in ws if (r['dp'],r['tp'],r['carrier'],r['layer'])==(p['DP'],p['TP'],body['F21_carrier_rows'],layer)]
    matches=[r for r in cw['communications'] if pre[1]<=r[0]<r[1]<=post[1] and r[4]=='hcom_alltoall_'];assert len(matches)<=1
    count=w['local_row_capacity']*4096
    if matches:
     a2a,=matches;assert a2a[7]==count;ref=(a2a[1]-a2a[0])/1000;dtype=a2a[8];mode='SAME_NEXT_ATTENTION_PARENT';bounds=[ref,ref]
    else:
     # An old dummy/no-metadata attention can zero its output with no projection/A2A.
     # NewN-owner transport would still be additional work, so never invent a zero cost.
     candidates=[(dt,v) for (dp,tp,n,dt),v in pools.items() if (dp,tp,n)==(p['DP'],p['TP'],count)];assert len(candidates)==1,(p['DP'],p['TP'],count,candidates)
     dtype,values=candidates[0];ref=statistics.median(values);bounds=[min(values),max(values)];mode='ABSENT_OLD_A2A_SAME_RANK_COUNT_DTYPE_OTHER_PARENT_PROXY'
    records.append(dict(dp=p['DP'],tp=p['TP'],carrier=w['carrier'],layer=layer,original_local_rows=w['local_row_capacity'],next_attention_HCPre=w['next_pre_task'],WKV_us=w['wkv_us'],gate_us=w['gate_us'],other_body_us=w['body_us']-w['wkv_us']-w['gate_us'],whole_body_us=w['body_us'],reference_attention_HCCL_a2a_us=ref,reference_binding=mode,reference_range_us=bounds,reference_count_per_peer=count,reference_dtype_code=dtype,new_N128_owner_count_per_peer=w['local_row_capacity']*3200))
 assert len(records)==128
 oldstages=[]
 for s in arrival['sections']:
  for layer in [1,14]:
   gs=[[dict(tp=r['tp'],native_pair_us=r['whole_body_us'],branch='engram') for r in records if (r['dp'],r['carrier'],r['layer'])==(dp,s['M'],layer)] for dp in [0,1]]
   assert all(sorted(x['tp'] for x in g)==list(range(8)) for g in gs);oldstages.append(dict(carrier=s['M'],layer=layer,DP0_minus_DP1_us=s['layers'][layer]['DP0_minus_DP1_us'],groups=gs))
 context=dict(carrier_order=[s['M'] for s in arrival['sections']],requests=[{k:r[k] for k in ['slot','DP','first_eager_M','TTFT_ms']} for r in arrival['requests']])
 free=price_exposure(dict(**context,stage_rows=oldstages))['all_FP32_pairs/remaining0'];expected=oldfront['first_token_conditional_free_work']['body_us']['remaining_0'];assert abs(free['maximum_request_percent']-expected['maximum_request_percent'])<1e-10
 cases=[]
 for wk,gate,transport,h in itertools.product([.5,.25,.125],[1,.5,.25,0],[0,.25,1],[0,5,20]):
  stages=[]
  for s in arrival['sections']:
   for layer in [1,14]:
    gs=[]
    for dp in [0,1]:
     rr=[r for r in records if (r['dp'],r['carrier'],r['layer'])==(dp,s['M'],layer)]
     gs.append([dict(tp=r['tp'],native_pair_us=(1-wk)*r['WKV_us']+(1-gate)*r['gate_us']+r['other_body_us']-transport*r['reference_attention_HCCL_a2a_us']-h,branch='engram') for r in rr])
    stages.append(dict(carrier=s['M'],layer=layer,DP0_minus_DP1_us=s['layers'][layer]['DP0_minus_DP1_us'],groups=gs))
  price=price_exposure(dict(**context,stage_rows=stages))['all_FP32_pairs/remaining0'];cases.append(dict(WKV_remaining_fraction=wk,gate_remaining_fraction=gate,new_exchange_reference_fraction=transport,new_pack_launch_us_each=h,other_old_body_free=True,conditional_C8_TTFT=price))
 # Samewindow newexchange reference from the already-bound nextattention AIVcall.
 epath=ROOT/'engram-joint-entry-price.json';hpath=ROOT/'hc-both-delivery-price.json';qpath=ROOT/'qkv-joint-delivery-price.json';opath=ROOT/'o-projection-parent-price.json'
 e=json.loads(epath.read_text());hc=json.loads(hpath.read_text());q=json.loads(qpath.read_text());op=json.loads(opath.read_text());balanced=[]
 for p in e['profiles']:
  hr=next(x for x in hc['profiles'] if x['ep']==p['ep']);hm={r['attention_HCPre']:r for r in hr['deliveries']};om={r['HcPost']:r for r in next(x for x in op['profiles'] if x['ep']==p['ep'])['deliveries']};qm={r['smla']:r for r in next(x for x in q['profiles'] if x['ep']==p['ep'])['deliveries']};rr=[]
  for r in p['deliveries']:
   qr=qm[om[hm[r['next_attention_HCPre']]['attention_Post']]['smla']];assert qr['rows']==24
   rr.append(dict(WKV_row=r['WKV_csv_row'],next_HCPre=r['next_attention_HCPre'],WKV_us=r['WKV_us'],other_span_us=r['keep_original_WKV_other_span_us'],reference_old_attention_AIV_a2a_us=qr['same_payload_attention_a2a_reference_us']))
  scenarios=[]
  for wk,gate,transport,h in itertools.product([.5,.25,.125],[1,.5,.25,0],[0,.25,1],[0,5,20]):
   net=sum((1-wk)*r['WKV_us']+(1-gate)*r['other_span_us']-transport*r['reference_old_attention_AIV_a2a_us']-h for r in rr);scenarios.append(dict(WKV_remaining_fraction=wk,gate_and_other_span_remaining_fraction=gate,new_exchange_reference_fraction=transport,new_pack_launch_us_each=h,conditional_net_us=net,conditional_window_pct=net/p['window_us']*100))
  balanced.append(dict(ep=p['ep'],window_us=p['window_us'],bound_sites=len(rr),reference_records=rr,scenarios=scenarios))
 ordinary=json.loads(ordinary_path.read_text());ordinary_rows=[dict(request=r['request'],prompt=r['prompt'],verification_count=r['verifications'],original_WKV_gate_allfree_model_ms_percent=r['estimates_ms_and_percent']['WKV_gate_entire_old_free'],qualification='Historical response proxy: promptinterpolation + oldG6-scheduled/24physical graphcost and observedverificationcounts. Not a matched newAllToAll/kernel prediction or actualresponsegain.') for r in ordinary['ordinary']]
 # Bound the standalone WKV component inside that SAME historical ordinary model.
 # Positive sums, rankmax, median and in-bracket linear interpolation preserve a
 # uniform component ratio bound. This gives no credit for inputcopies/gate/owner.
 ratios=[dict(family='eager',dp=w['dp'],tp=w['tp'],carrier=w['carrier'],layer=w['layer'],rho=w['wkv_us']/(w['wkv_us']+w['gate_us'])) for w in ws]
 gatepath=BASE/'full-review/ENGRAM-ROTATED-GATE.json';g=json.loads(gatepath.read_text())['profile']
 for row in g['rows']:
  w=dict(zip(g['columns'],row));ratios.append(dict(family='graph',profile=w['profile'],layer=w['layer'],WKV_task=w['WKV_task'],rho=w['WKV']/(w['WKV']+w['whole_gate'])))
 padmanifest=BASE/'full-review/runs/frozen-padded-engram-arrivals.json';manifest=json.loads(padmanifest.read_text());padpins=[]
 for pin in manifest['rank_files']:
  path=BASE/pin['path'];assert sha256(path.read_bytes()).hexdigest()==pin['sha256'];padpins.append(dict(path=str(path),sha256=pin['sha256']))
  panel=json.loads(zlib.decompress(path.read_bytes()))
  for wave,entries in panel.items():
   for layer,entry in zip([1,14],entries):
    terms=entry[1];assert len(terms)==38;ratios.append(dict(family='padded_graph',rank=pin['rank'],wave=wave,layer=layer,rho=terms[0]/sum(terms)))
 assert len(ratios)==544;rho=max(r['rho'] for r in ratios)
 for row in ordinary_rows:
  oldms,oldpct=row['original_WKV_gate_allfree_model_ms_percent'];row['standalone_WKV_free_component_model_upper_ms_percent']=[rho*oldms,rho*oldpct];row['standalone_WKV_remaining_one_eighth_zero_newcost_model_upper_percent']=.875*rho*oldpct
 ratio_bound=dict(samples=len(ratios),maximum_rho=rho,witness=max(ratios,key=lambda r:r['rho']),ratio_samples=ratios,standalone_free_max_response_model_percent=max(r['standalone_WKV_free_component_model_upper_ms_percent'][1] for r in ordinary_rows),remaining_one_eighth_max_response_model_percent=max(r['standalone_WKV_remaining_one_eighth_zero_newcost_model_upper_percent'] for r in ordinary_rows),scope='Within the existing ordinary positive-interpolation/median/rankmax proxy only. WKV primitive only; no copy/lookup/gate/host credit. Close-to1% free bound is not a hardware/performanceconfidence guarantee or a bound on changed producer/schedule. No newtransport cost credited as free in a real implementation.',proof='Each observed WKV <= rho*(WKV+gate). Positivesums, rankmax, median, nonnegative verificationcount and interpolation inside sourcebrackets preserve this inequality. All sourcepool544samples and16rankcache SHA retained.')
 result=dict(status='ENGRAM_N_OWNER_PAID_REFERENCE_AND_REQUEST_SCREEN',original_C8_TTFT_allfree_reproduced=free,source_consistent_C8_prefill_records=records,C8_scenarios=cases,balanced_profiles=balanced,ordinary_models=ordinary_rows,ordinary_standalone_WKV_ratio_bound=ratio_bound,
  findings=['128firsttoken WKV/gate records reproduce oldwholefree maximumC8TTFT1.115388%; reference_binding distinguishes originalnextparentA2A vs no-originalA2A same-rank/count/dtype proxy. 48balanced completechains match their ownnextattention.','Old dummy/no-metadata attention can have AG/zero/RS but noA2A or projection; this is not a localQ0 equivalence. NewNowner transport cannot inherit a nonexistent oldcost or freefallback.','Allreference oldAllToAll cost is ADDED for newtransport, not removed; payloadratio3200/4096 is metadata, not a latency scaling rule. HCCL/AIV references include synchronization/arrival, are not intrinsic transit or a bound for a newproducer.','Retaining25%gate and12.5%WKV remaining canstillmissold1% C8TTFT reference evenwithzero transport/packing. Concrete childscenarios belowreference are stopped underthatbudget; no newuserthreshold or universalwholeinput bound.','Sixordinary allfree responseproxy spans.82471–1.41841%; logicalB1 doesnot establish physicalM1. Nativecost/resource/transport and requestexposure gaps remain for differentlargerparent.'],
  limits=['These sensitivities preserve oldexternalorder/EP arrivals and optimisticallygrant eachgroup its mostfavorable ranknet. They are not servicebefore/after or universalbound on changed scheduling.','WKV fractions are assumptions, not8xcompute speed. Geometryretains originalrowpanel paddedMAC; inter-panel B reuse and activeK/Fixpipe stillunqualified. gatefraction0 is deliberatelyunphysical allfree screen.','No nativeAllToAll or originalM3/M24 equivalence executed. Existing EDP rowexchange is differentgroup/path.','Do not combine Nowner with directSPowner credit without restoringmissingglobalK rows or changing numericpartition; do not add differentprofiles/window/request estimates.'],
  inputs=[dict(path=str(p),sha256=sha256(p.read_bytes()).hexdigest()) for p in [compute_path,comm_path,arrival_path,winpath,ordinary_path,epath,hpath,qpath,opath,gatepath,padmanifest]]+padpins,analyzer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),whole_chain_complete=False,new_primitive_admissions=0,new_NPU_work=0,new_profile=0,service_requests=0,remote_writes=[])
 (ROOT/'engram-row-owner-price.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 sample=[c for c in cases if c['WKV_remaining_fraction']==.125 and c['gate_remaining_fraction'] in [.25,0] and c['new_exchange_reference_fraction'] in [0,.25,1] and c['new_pack_launch_us_each']==5]
 print(json.dumps(dict(prefill_records=len(records),reference_binding_counts={k:sum(r['reference_binding']==k for r in records) for k in {r['reference_binding'] for r in records}},C8_scenarios=len(cases),balanced_scenarios=sum(len(p['scenarios']) for p in balanced),old_maximum_free_TTFT_percent=free['maximum_request_percent'],selected_C8=[{**{k:v for k,v in c.items() if k!='conditional_C8_TTFT'},'max_TTFT_percent':c['conditional_C8_TTFT']['maximum_request_percent']} for c in sample],ordinary_allfree_percent=[x['original_WKV_gate_allfree_model_ms_percent'][1] for x in ordinary_rows],ordinary_ratio_bound={k:v for k,v in ratio_bound.items() if k!='ratio_samples'}),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
