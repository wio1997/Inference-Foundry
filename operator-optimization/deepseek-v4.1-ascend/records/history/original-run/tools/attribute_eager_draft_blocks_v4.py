"""Bounded source-invariant attribution for eager draft blocks.

Counters only from the bracket's own stream; cross-stream tasks are excluded,
not guessed from temporal overlap. This yields a distinct metric scope from
the existing all-attributable-stream target ranking.
"""
from pathlib import Path
from decimal import Decimal,InvalidOperation
from collections import defaultdict,Counter
import bisect,csv,json
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'eager-draft-mainstream-attribution-v4';O.mkdir(exist_ok=False)
mapping=json.loads((T/'captured-layer-attribution/module-call-metrics.json').read_text())
def ns(s):return int(Decimal(s.strip())*1000)
def dec(s):
 try:
  v=Decimal(s);return v if v.is_finite() else None
 except (InvalidOperation,TypeError):return None
def counters(rows):
 num=Decimal(0);den=Decimal(0);missing=0;invalid=0
 for x in rows:
  c=dec(x['aic_total_cycles']);r=dec(x['aic_mac_ratio'])
  if c is None or c<=0:missing+=1;continue
  if r is None or not (Decimal(0)<=r<=Decimal(1)):invalid+=1;continue
  den+=c;num+=c*r
 return num,den,missing,invalid
results=[];details=[];rank_audit=[]
for rp in sorted(ROOT.glob('*_ascend_pt')):
 rank=rp.name;rows=[];markers=defaultdict(list);streams=defaultdict(list)
 for row in csv.DictReader((rp/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
  row['_start']=ns(row['Start Time(us)']);row['_end']=row['_start']+ns(row['Duration(us)']);rows.append(row)
  streams[(row['Model ID'],row['Stream ID'])].append(row)
  if row['Type'] in ['HcPre','HcPost']:markers[(row['Model ID'],row['Stream ID'])].append(row)
 for rs in streams.values():rs.sort(key=lambda r:r['_start'])
 for rs in markers.values():rs.sort(key=lambda r:r['_start'])
 mods=[m for m in mapping if m['rank']==rank and m['run_in_model'] in [1,2,3]]
 assert len(mods)==120
 buckets=defaultdict(list)
 for m in mods:
  source=streams[(m['model_id'],m['main_stream_id'])]
  rs=[x for x in source if x['_start']>=m['window_start_ns'] and x['_end']<=m['window_end_ns'] and 'metadata' not in x['Type'].lower()]
  num,den,miss,bad=counters(rs);assert bad==0
  buckets[m['layer']].append((num,den,miss,m['window_ns']))
 for layer,items in sorted(buckets.items()):
  num=sum((x[0] for x in items),Decimal(0));den=sum((x[1] for x in items),Decimal(0))
  results.append({'rank':rank,'module':'target.layers.'+str(layer)+'.forward','kind':'CAPTURED_TARGET','calls':len(items),
                  'aic_mac_proxy_numerator':str(num),'aic_cycles_denominator':str(den),'ratio':float(num/den) if den else None,
                  'skipped_no_positive_cycles':sum(x[2] for x in items),'window_ns_sum':sum(x[3] for x in items)})
 windows=[]
 for run in [1,2]:
  left=next(m for m in mods if m['run_in_model']==run and m['layer']==39)
  right=next(m for m in mods if m['run_in_model']==run+1 and m['layer']==0)
  assert left['window_end_ns']<right['window_start_ns'];windows.append((left['window_end_ns'],right['window_start_ns']))
 draft=[];rejected=Counter();mc2count=0
 for row in rows:
  if row['Model ID']!='4294967295' or row['Type']!='DispatchFFNCombine':continue
  ss=row['Input Shapes'].strip('"').split(';')
  try:is_draft=int(ss[1].split(',')[0])==8 and int(ss[3].split(',')[-1])==3
  except (ValueError,IndexError):is_draft=False
  if not is_draft or not any(a<=row['_start'] and row['_end']<=b for a,b in windows):continue
  mc2count+=1;key=(row['Model ID'],row['Stream ID']);ms=markers[key];starts=[x['_start'] for x in ms]
  i=bisect.bisect_right(starts,row['_start'])
  if i<3 or i>=len(ms):rejected['insufficient_markers']+=1;continue
  a,b,c,d=ms[i-3:i+1]
  if [x['Type'] for x in [a,b,c,d]]!=['HcPre','HcPost','HcPre','HcPost'] or c['_end']>row['_start'] or row['_end']>d['_start']:
   rejected['marker_order_or_moe_boundary']+=1;continue
  same=streams[key];rs=[x for x in same if x['_start']>=a['_start'] and x['_end']<=d['_end'] and 'metadata' not in x['Type'].lower()]
  m=[x for x in rs if x['Type']=='DispatchFFNCombine'];smla=[x for x in rows if x['Model ID']==row['Model ID'] and x['Type']=='SparseFlashMla' and x['_start']>=a['_start'] and x['_end']<=b['_end']]
  if len(m)!=1:rejected['moe_'+str(len(m))+'_attention_'+str(len(smla))]+=1;continue
  attention_region=[x for x in same if x['_start']>=a['_start'] and x['_end']<=b['_end']]
  attention_types=Counter(x['Type'] for x in attention_region)
  attention_allstreams=Counter(x['Type'] for x in rows if x['Model ID']==row['Model ID'] and x['_start']>=a['_start'] and x['_end']<=b['_end'])
  if len(smla)==1:
   validation='ONE_SPARSE_FLASH_MLA'
  elif len(smla)==0 and all(attention_types.get(x,0)>0 for x in ['ZerosLike','ConcatD']) and all(attention_allstreams.get(x,0)>0 for x in ['hcom_allGather_','hcom_reduceScatter_']) and not any(attention_types.get(x,0)>0 for x in ['MatMulV2','QuantBatchMatmulV3','InplacePartialRotaryMul']):
   validation='EMPTY_LOCAL_QUERY_INFERRED_NOT_DIRECTLY_OBSERVED'
  else:
   rejected['moe_'+str(len(m))+'_attention_'+str(len(smla))+'_other_pattern']+=1;continue
  num,den,miss,bad=counters(rs)
  if bad:rejected['invalid_pmu_ratio']+=1;continue
  draft.append({'rank':rank,'start_ns':a['_start'],'end_ns':d['_end'],'main_stream_id':row['Stream ID'],
                'model_id':row['Model ID'],'mc2_task_id':row['Task ID'],'aic_mac_proxy_numerator':str(num),
                'aic_cycles_denominator':str(den),'skipped_no_positive_cycles':miss,'attention_count_allstreams':len(smla),
                'source_role':'DRAFT_BLOCK_TOP3_LOCAL_EXPERTS8_WITH_4_MHC_MARKERS','attention_validation':validation,
                'block_identity_within_3':'UNKNOWN','not_cross_stream_attribution':True})
 draft.sort(key=lambda x:x['start_ns'])
 for a,b in zip(draft,draft[1:]):assert a['end_ns']<=b['start_ns'],'overlapping draft brackets'
 if draft:
  num=sum((Decimal(x['aic_mac_proxy_numerator']) for x in draft),Decimal(0));den=sum((Decimal(x['aic_cycles_denominator']) for x in draft),Decimal(0))
  results.append({'rank':rank,'module':'draft.decoder_block.pooled','kind':'EAGER_DRAFT_POOLED',
     'calls':len(draft),'aic_mac_proxy_numerator':str(num),'aic_cycles_denominator':str(den),'ratio':float(num/den) if den else None,
     'skipped_no_positive_cycles':sum(x['skipped_no_positive_cycles'] for x in draft),'window_ns_sum':sum(x['end_ns']-x['start_ns'] for x in draft)})
 details.extend(draft);rank_audit.append({'rank':rank,'draft_mc2_examined':mc2count,'draft_valid_brackets':len(draft),'rejected':dict(rejected),
                                        'target_brackets':len(mods),'windows':windows})
print('rank audit complete',len(rank_audit),flush=True)
with (O/'module_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
with (O/'draft_call_metrics.json').open('w') as f:json.dump(details,f,indent=2)
pooled=[]
for module in sorted(set(x['module'] for x in results)):
 rs=[x for x in results if x['module']==module];num=sum((Decimal(x['aic_mac_proxy_numerator']) for x in rs),Decimal(0));den=sum((Decimal(x['aic_cycles_denominator']) for x in rs),Decimal(0))
 pooled.append({'module':module,'kind':rs[0]['kind'],'rank_count':len(rs),'calls':sum(x['calls'] for x in rs),'ratio':float(num/den) if den else None,
                'counter_numerator':str(num),'counter_denominator':str(den),'mean_window_us':sum(x['window_ns_sum'] for x in rs)/sum(x['calls'] for x in rs)/1000})
good=[x for x in pooled if x['rank_count']==16 and x['ratio'] is not None]
summary={'status':'SENSITIVITY_WITH_SOURCE_CONSISTENT_EMPTY_QUERY_HYPOTHESIS','metric':'CSV AIC-cycle-weighted MAC-ratio proxy on the same main stream only; not NPU utilization, MFU, or cross-stream inclusive metric',
 'rank_count':len(rank_audit),'draft_call_count':len(details),'draft_attention_validation_counts':dict(Counter(x['attention_validation'] for x in details)),'rank_audit':rank_audit,
 'ranked_groups':sorted(good,key=lambda x:x['ratio']),
 'limitations':['Some brackets have no SparseFlashMla, include zero/concat on main stream and collectives in overlapping streams, and match a source-consistent empty-query pattern; metadata and causal ownership are unobserved, so classification is inferred, not proven.',
                'Draft three block identities pooled, not individually distinguished.',
                'No task on auxiliary streams assigned; target and draft comparator main-stream only.',
                'Source-role mapping derives from MoE expert shapes/top3, four mHC markers, and exactly one attention; no explicit module marker or request ID.',
                'Two windows between captured target runs1-2 and2-3 may include multiple eager forwards; no assumption of corresponding cross-DP iterations.',
                'Prefill, Engram, head and metadata not fully attributed; sampled denominator is kernel-cycle-only.'],
 'no_device_work':True}
assert len(json.dumps(summary).encode())<32768
(O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['rank_audit','ranked_groups']},indent=2))
