"""Source-role sensitivity for existing eager target windows; CPU-only."""
from pathlib import Path
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
import bisect,csv,json
R=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
O=T/'eager-target-attribution-v2';O.mkdir(exist_ok=False)
def ns(x):return int(Decimal(x.strip())*1000)
def dec(s):
 try:
  x=Decimal(s);return x if x.is_finite() else None
 except (InvalidOperation,TypeError):return None
def pmu(rows):
 num=den=Decimal(0);skip=0
 for x in rows:
  c=dec(x['aic_total_cycles']);v=dec(x['aic_mac_ratio'])
  if c is None or c<=0:skip+=1;continue
  assert v is not None and 0<=v<=1;num+=c*v;den+=c
 return num,den,skip
allcalls=[];audit=[];roles={2,8,14,20,24,28,32,36}
for rp in sorted(R.glob('*_ascend_pt')):
 streams=defaultdict(list);markers=defaultdict(list);mc=[];att=[];qli=[]
 for x in csv.DictReader((rp/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
  if x['Model ID']!='4294967295':continue
  x['_s']=ns(x['Start Time(us)']);x['_e']=x['_s']+ns(x['Duration(us)']);streams[x['Stream ID']].append(x)
  if x['Type'] in ('HcPre','HcPost'):markers[x['Stream ID']].append(x)
  if x['Type']=='SparseFlashMla':att.append(x)
  if 'QuantLightningIndexer' in x['Type']:qli.append(x)
  if x['Type']=='DispatchFFNCombine':
   sh=x['Input Shapes'].strip('"').split(';')
   try:keep=int(sh[1].split(',')[0])==24 and int(sh[3].split(',')[-1])==6
   except (ValueError,IndexError):keep=False
   if keep:x['_m']=int(sh[0].split(',')[0]);mc.append(x)
 for v in streams.values():v.sort(key=lambda x:x['_s'])
 for v in markers.values():v.sort(key=lambda x:x['_s'])
 mc.sort(key=lambda x:x['_s']); assert len(mc)==160
 local=[];counts=Counter();shape_counts=Counter()
 for ordinal,x in enumerate(mc):
  ms=markers[x['Stream ID']];i=bisect.bisect_right([z['_s'] for z in ms],x['_s']);assert i>=3 and i<len(ms)
  a,b,c,d=ms[i-3:i+1];assert [z['Type'] for z in (a,b,c,d)]==['HcPre','HcPost','HcPre','HcPost']
  assert a['_e']<=b['_s'] and b['_e']<=c['_s'] and c['_e']<=x['_s'] and x['_e']<=d['_s']
  rows=[z for z in streams[x['Stream ID']] if a['_s']<=z['_s'] and z['_e']<=d['_e'] and 'metadata' not in z['Type'].lower()]
  moes=[z for z in rows if z['Type']=='DispatchFFNCombine'];assert len(moes)==1
  ats=[z for z in att if a['_s']<=z['_s'] and z['_e']<=b['_e']]
  qs=[z for z in qli if a['_s']<=z['_s'] and z['_e']<=b['_e']]
  layer=ordinal%40;compressed=None
  if not ats:state='NO_EXPORTED_ATTENTION_UNRESOLVED'
  elif len(ats)==1 and ats[0]['Stream ID']==x['Stream ID']:
   sh=ats[0]['Input Shapes'].strip('"').split(';');assert len(sh)>2
   compressed=bool(sh[2])
   if compressed==(layer>=2) and bool(qs)==(layer in roles):state='ATTENTION_ROLE_CONSISTENT_ORDINAL_LAYER_HYPOTHESIS'
   else:state='ROLE_MISMATCH_UNRESOLVED'
  else:state='OTHER_ATTENTION_UNRESOLVED'
  num,den,skip=pmu(rows);rec={'rank':rp.name,'mc2_ordinal':ordinal,'hypothesized_layer':layer,'model_id':x['Model ID'],'stream_id':x['Stream ID'],'mc2_task_id':x['Task ID'],'physical_mc2_input_rows':x['_m'],'window_start_ns':a['_s'],'window_end_ns':d['_e'],'window_ns':d['_e']-a['_s'],'attention_state':state,'attention_count_same_model_all_streams':len(ats),'compressed_input_present':compressed,'indexer_count_same_model_all_streams':len(qs),'aic_cycles_denominator':str(den),'aic_mac_cycle_proxy_numerator':str(num),'skipped_no_positive_cycles':skip}
  local.append(rec);counts[state]+=1;shape_counts[str(x['_m'])]+=1
 for a,b in zip(local,local[1:]):assert a['window_end_ns']<=b['window_start_ns']
 
 valid=[x for x in local if x['compressed_input_present'] is not None]
 shifts=[shift for shift in range(40) if all(x['compressed_input_present']==((x['mc2_ordinal']+shift)%40>=2) and bool(x['indexer_count_same_model_all_streams'])==((x['mc2_ordinal']+shift)%40 in roles) for x in valid)]
 allcalls.extend(local);audit.append({'rank':rp.name,'target_mc2_windows':len(local),'state_counts':dict(counts),'m_counts':dict(shape_counts),'ordinal_layer_offsets_consistent_with_roles':shifts,'role_checked_windows':len(valid)})
assert len(audit)==16 and len(allcalls)==2560
with (O/'calls.json').open('w') as f:json.dump(allcalls,f,indent=2)
b=defaultdict(list)
for x in allcalls:
 if x['attention_state']=='ATTENTION_ROLE_CONSISTENT_ORDINAL_LAYER_HYPOTHESIS':b[(x['physical_mc2_input_rows'],x['hypothesized_layer'])].append(x)
metrics=[]
for (m,layer),rs in sorted(b.items()):
 num=sum((Decimal(x['aic_mac_cycle_proxy_numerator']) for x in rs),Decimal(0));den=sum((Decimal(x['aic_cycles_denominator']) for x in rs),Decimal(0))
 metrics.append({'physical_mc2_input_rows':m,'hypothesized_layer':layer,'calls':len(rs),'rank_count':len({x['rank'] for x in rs}),'counter_weighted_mac_ratio':float(num/den) if den else None,'aic_cycles_denominator':str(den),'aic_mac_cycle_proxy_numerator':str(num),'mean_window_us':sum(x['window_ns'] for x in rs)/len(rs)/1000})
with (O/'shape-layer-metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
counts=Counter(x['attention_state'] for x in allcalls);byshape=Counter((x['physical_mc2_input_rows'],x['attention_state']) for x in allcalls)
o={'status':'EAGER_TARGET_ORDINAL_ROLE_HYPOTHESIS_NOT_EXPLICIT_LAYER_IDS','ranks':16,'all_windows':len(allcalls),'states':dict(counts),'shape_state_counts':[{'m':m,'state':s,'calls':n} for (m,s),n in sorted(byshape.items())], 'rank_audit':audit,'metric':'sum(AIC cycles * CSV MAC ratio)/sum(AIC cycles) over same-stream kernels contained in marker bracket; not whole-device busy rate or MFU', 'limitations':['Layer identity uses source role invariants and per-rank target-MC2 ordinal modulo 40; every candidate offset 0-39 was checked, but this is not an explicit layer marker. A dropped/partial initial forward could alter the inference; role agreement applies only visible attention windows.','Attention and indexer detection search all streams in a time bracket; same-stream attention is required for the role-consistent subset, but time overlap alone does not prove dependencies.','720 expected windows lacking exported attention remain unresolved, not automatically empty-query or prefill.','Physical M is shape, not phase or request token count; compare cases separately.','Vector-only kernels with no AIC cycles do not appear in the proxy denominator; no all-NPU utilization ranking.'],'no_device_work':True}
assert len(json.dumps(o).encode())<32768
(O/'summary.json').write_text(json.dumps(o,indent=2)+'\n');print(json.dumps({'ranks':16,'windows':len(allcalls),'states':dict(counts),'shape_state_counts':o['shape_state_counts']},indent=2))
