"""Recover captured target layer brackets using source sequence plus independent checks.
Reads CSV only. Refuses ambiguous/incomplete graphs; never assigns modules by name alone.
"""
from pathlib import Path
from collections import defaultdict,Counter
from decimal import Decimal
import bisect,csv,json
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'captured-layer-attribution';O.mkdir(exist_ok=False)
def ns(s):return int(Decimal(s.strip())*1000)
def d(s):
 try:return Decimal(s)
 except:return Decimal(0)
def union_time(items):
 total=0;end=None
 for a,b in sorted(items):
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total
results=[];validation=[];all_calls=[];failed=[]
for rankpath in sorted(ROOT.glob('*_ascend_pt')):
 rows=[];markers=defaultdict(list);path=rankpath/'ASCEND_PROFILER_OUTPUT/kernel_details.csv';total_rows=0
 for row in csv.DictReader(path.open()):
  total_rows+=1
  model=row['Model ID']
  if model in ['4294967295','N/A','']:continue
  row['_start']=ns(row['Start Time(us)']);row['_end']=row['_start']+ns(row['Duration(us)']);rows.append(row)
  if row['Type'] in ['HcPre','HcPost']:markers[(model,row['Stream ID'])].append(row)
  assert len(rows)<=100000,'Per-rank retention cap exceeded; keep existing evidence and narrow explicitly'
 rank_validation={'rank':rankpath.name,'all_csv_rows':total_rows,'captured_rows':len(rows),'models':[]};assigned=set();calls=[]
 try:
  for (model,stream),ms in markers.items():
   ms.sort(key=lambda r:r['_start']);types=[r['Type'] for r in ms]
   assert len(ms)%160==0,('Complete40layer graph marker count missing',len(ms))
   assert types==['HcPre','HcPost','HcPre','HcPost']*(len(ms)//4),('Unexpected mHC ordering',Counter(types))
   blocks=[]
   for i in range(0,len(ms),4):
    a,b,c,e=ms[i:i+4];assert a['_end']<=b['_start'] and b['_end']<=c['_start'] and c['_end']<=e['_start']
    block={'model':model,'stream':stream,'layer':(i//4)%40,'run':(i//4)//40,'start':a['_start'],'end':e['_end'],'attention_end':b['_end'],'ffn_start':c['_start'],'rows':[],'marker_task_ids':[x['Task ID'] for x in [a,b,c,e]]};blocks.append(block)
   starts=[b['start'] for b in blocks]
   for row_idx,row in enumerate(rows):
    if row['Model ID']!=model:continue
    bi=bisect.bisect_right(starts,row['_start'])-1
    if bi>=0 and row['_end']<=blocks[bi]['end']:
     blocks[bi]['rows'].append((row_idx,row))
   for block in blocks:
    rs=[x for _,x in block['rows']];moe=[x for x in rs if x['Type']=='DispatchFFNCombine'];smla=[x for x in rs if x['Type']=='SparseFlashMla'];qli=[x for x in rs if 'QuantLightningIndexer' in x['Type']]
    assert len(moe)==1 and moe[0]['Stream ID']==stream,('MoE source invariant failed',block['layer'],len(moe))
    assert len(smla)==1,('Attention source invariant failed',block['layer'],len(smla))
    shapes=smla[0]['Input Shapes'].strip('"').split(';');has_cmp=bool(shapes[2])
    assert has_cmp==(block['layer']>=2),('Compression-role mismatch',block['layer'],has_cmp)
    expected_indexer=block['layer'] in [2,8,14,20,24,28,32,36]
    assert bool(qli)==expected_indexer,('Indexer-source-role mismatch',block['layer'],len(qli))
    assert all(x['_end']<=block['attention_end'] for x in smla+qli)
    assert moe[0]['_start']>=block['ffn_start']
    # Keep ambiguous metadata outside module attribution, even when it overlaps a bracket.
    kept=[(ix,x) for ix,x in block['rows'] if 'metadata' not in x['Type'].lower()]
    for ix,_ in kept:assert ix not in assigned;assigned.add(ix)
    AICden=sum((d(x['aic_total_cycles']) for _,x in kept),Decimal(0));AICnum=sum((d(x['aic_total_cycles'])*d(x['aic_mac_ratio']) for _,x in kept),Decimal(0));AIVden=sum((d(x['aiv_total_cycles']) for _,x in kept),Decimal(0));AIVnum=sum((d(x['aiv_total_cycles'])*d(x['aiv_vec_ratio']) for _,x in kept),Decimal(0))
    call={'rank':rankpath.name,'model_id':model,'main_stream_id':stream,'run_in_model':block['run'],'layer':block['layer'],'window_start_ns':block['start'],'window_end_ns':block['end'],'window_ns':block['end']-block['start'],'known_task_union_ns':union_time([(x['_start'],x['_end']) for _,x in kept]),'attributed_rows':len(kept),'aic_cycle_denominator':str(AICden),'aic_mac_cycle_proxy_numerator':str(AICnum),'aiv_cycle_denominator':str(AIVden),'aiv_vec_cycle_proxy_numerator':str(AIVnum),'smla_count':len(smla),'indexer_count':len(qli),'marker_task_ids':block['marker_task_ids'],'semantic_mapping':'SOURCE_SEQUENCE_AND_ROLE_INVARIANTS','counter_rounding_uncertainty':'Ratios originate in rounded vendor CSV; report-derived counter proxy, not MFU or peak efficiency'};calls.append(call)
   rank_validation['models'].append({'model_id':model,'main_stream_id':stream,'complete_target_forward_runs':len(blocks)//40,'decoder_calls':len(blocks),'independent_invariants':['4 mHC markers per layer in source order','40layers per target graph; DSpark is eager','exactly1MC2 and1SparseFlashMla per layer','compressedKV present iff layer>=2','QLI present at frozen8index source IDs','attention-before-MoE phase ordering'],'status':'PASS'})
  assert calls,'No fully validated captured target decoder graph'
  bylayer=defaultdict(list)
  for call in calls:bylayer[call['layer']].append(call)
  assert len(bylayer)==40
  for layer,items in sorted(bylayer.items()):
   den=sum((Decimal(c['aic_cycle_denominator']) for c in items),Decimal(0));num=sum((Decimal(c['aic_mac_cycle_proxy_numerator']) for c in items),Decimal(0));vden=sum((Decimal(c['aiv_cycle_denominator']) for c in items),Decimal(0));vnum=sum((Decimal(c['aiv_vec_cycle_proxy_numerator']) for c in items),Decimal(0))
   results.append({'rank':rankpath.name,'module':'target.layers.'+str(layer)+'.forward','layer':layer,'case':'CAPTURED_TARGET_DECODE_ONLY','calls':len(items),'aic_reported_cycle_weighted_mac_ratio':float(num/den) if den else None,'aiv_reported_cycle_weighted_vec_ratio':float(vnum/vden) if vden else None,'aic_cycle_denominator':str(den),'aic_mac_cycle_proxy_numerator':str(num),'aiv_cycle_denominator':str(vden),'aiv_vec_cycle_proxy_numerator':str(vnum),'window_ns_sum':sum(c['window_ns'] for c in items),'known_task_union_ns_sum':sum(c['known_task_union_ns'] for c in items),'known_task_coverage_fraction':sum(c['known_task_union_ns'] for c in items)/sum(c['window_ns'] for c in items),'semantic_mapping':'SOURCE_SEQUENCE_AND_ROLE_INVARIANTS','not_MFU_or_full_device_activity':True})
  all_calls.extend(calls);rank_validation.update(status='PASS',assigned_captured_rows=len(assigned),unassigned_captured_rows=len(rows)-len(assigned),unassigned_types=dict(Counter(row['Type'] for ix,row in enumerate(rows) if ix not in assigned)))
 except Exception as ex:
  rank_validation.update(status='FAIL_UNASSIGNED',error=str(ex));failed.append(rank_validation)
 validation.append(rank_validation);print(json.dumps({k:v for k,v in rank_validation.items() if k in ['rank','status','models','error']}),flush=True)
(O/'source-mapping-validation.json').write_text(json.dumps({'validation':validation,'failures':len(failed),'no_device_work':True},indent=2)+'\n')
with (O/'module_metrics.csv').open('w') as f:
 if results:
  writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
(O/'module-call-metrics.json').write_text(json.dumps(all_calls,indent=2)+'\n')
pooled=[]
for layer in range(40):
 items=[x for x in results if x['layer']==layer]
 if not items:continue
 den=sum((Decimal(x['aic_cycle_denominator']) for x in items),Decimal(0));num=sum((Decimal(x['aic_mac_cycle_proxy_numerator']) for x in items),Decimal(0))
 pooled.append({'module':'target.layers.'+str(layer)+'.forward','layer':layer,'rank_count':len(items),'calls':sum(x['calls'] for x in items),'counter_weighted_mac_ratio':float(num/den) if den else None,'metric_scope':'AIC counters of attributed kernels inside source-validated captured target layer calls; not whole-NPU peak efficiency'})
summary={'status':'VALIDATED_CAPTURED_TARGET_SUBSET' if not failed else 'PARTIAL_MAPPING_WITH_FAILURES','ranks_passed':len(validation)-len(failed),'ranks_failed':len(failed),'module_metric_rows':len(results),'call_rows':len(all_calls),'lowest_captured_target_groups':sorted(pooled,key=lambda x:x['counter_weighted_mac_ratio'] if x['counter_weighted_mac_ratio'] is not None else 2)[:5],'pooled_all_40_modules':pooled,'excluded_scope':['Eager prefill','Eager DSpark','External Engram calls and root heads','Metadata tasks lacking consumer ownership'],'caveats':['Source-derived semantic attribution, not MSTX labels','All validation failures are unassigned, never forced','Task-counter proxy uses rounded PMU ratios; not exact hardware utilization or achievable speedup','Module wall-time coverage is not primary rank metric; overlapping unrelated activity excluded where known']}
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='pooled_all_40_modules'},indent=2))
