"""Inspect two bounded existing-trace examples; do not force attribution."""
from pathlib import Path
from decimal import Decimal
from collections import defaultdict,Counter
import bisect,csv,json
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'draft-missing-attention-inspection-v2';O.mkdir(exist_ok=False)
mapping=json.loads((T/'captured-layer-attribution/module-call-metrics.json').read_text())
doc=[]
for ep in [0,15]:
 matches=list(ROOT.glob('*_ep'+str(ep)+'_rank*_ascend_pt'));assert len(matches)==1
 rp=matches[0];rank=rp.name;rows=[];markers=defaultdict(list)
 for r in csv.DictReader((rp/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
  r['_s']=int(Decimal(r['Start Time(us)'].strip())*1000);r['_e']=r['_s']+int(Decimal(r['Duration(us)'])*1000);rows.append(r)
  if r['Type'] in ['HcPre','HcPost']:markers[(r['Model ID'],r['Stream ID'])].append(r)
 for v in markers.values():v.sort(key=lambda x:x['_s'])
 mods=[m for m in mapping if m['rank']==rank and m['run_in_model'] in [1,2,3]]
 windows=[(next(m['window_end_ns'] for m in mods if m['run_in_model']==i and m['layer']==39),next(m['window_start_ns'] for m in mods if m['run_in_model']==i+1 and m['layer']==0)) for i in [1,2]]
 found={}
 for r in rows:
  if r['Type']!='DispatchFFNCombine' or r['Model ID']!='4294967295' or not any(a<=r['_s'] and r['_e']<=b for a,b in windows):continue
  s=r['Input Shapes'].strip('"').split(';')
  try:ok=int(s[1].split(',')[0])==8 and int(s[3].split(',')[-1])==3
  except (IndexError,ValueError):ok=False
  if not ok:continue
  m=markers[(r['Model ID'],r['Stream ID'])];i=bisect.bisect_right([x['_s'] for x in m],r['_s'])
  if i<3 or i>=len(m):continue
  a,b,c,d=m[i-3:i+1]
  if [x['Type'] for x in [a,b,c,d]]!=['HcPre','HcPost','HcPre','HcPost']:continue
  attention=[x for x in rows if x['Model ID']==r['Model ID'] and x['Type']=='SparseFlashMla' and a['_s']<=x['_s'] and x['_e']<=b['_e']]
  k='present' if len(attention)==1 else 'missing'
  if k in found:continue
  region=[x for x in rows if a['_s']<=x['_s'] and x['_e']<=b['_e']]
  names=Counter(x['Type'] for x in region)
  interesting=[x for x in region if any(z in x['Type'].lower() for z in ['flash','attention','mla','query','index'])]
  found[k]={'ep':ep,'rank':rank,'state':k,'attention_interval_us':(b['_e']-a['_s'])/1000,
            'stream_id':r['Stream ID'],'mc2_input_shapes':r['Input Shapes'],
            'type_counts':dict(names),'region_events':[{'type':x['Type'],'model_id':x['Model ID'],'stream_id':x['Stream ID']} for x in region[:60]],'selected_events':[{'type':x['Type'],'model_id':x['Model ID'],'stream_id':x['Stream ID'],
                   'start_relative_us':(x['_s']-a['_s'])/1000,'duration_us':(x['_e']-x['_s'])/1000,
                   'input_shapes':x['Input Shapes'][:400]} for x in interesting[:60]]}
  if len(found)==2:break
 doc.extend(found.values())
out={'status':'BOUNDED_REJECTION_CONTEXT_ONLY','examples':doc,'no_invariant_relaxed':True,
     'limitations':'Absence in this exported window does not prove no attention computation, its target identity, or an optimization opportunity.',
     'no_device_work':True}
assert len(json.dumps(out).encode())<32768
(O/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
