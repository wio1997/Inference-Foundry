#!/usr/bin/env python3
import json,pathlib,ijson,collections
from decimal import Decimal as D
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json')
w={'short_L9':(D('1793241328964372.235'),D('1793241328966065.515')),'long_L30':(D('1793241329714147.695'),D('1793241329714147.715'))}
out={k:{'other_tracks_X_counts':collections.Counter(),'same_track_nearby_events':[],'other_tracks_event_wait':[]} for k in w}
with p.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict) or str(e.get('pid'))!='2179523' or 'ts' not in e or e.get('ph')!='X':continue
  t=D(str(e['ts'])); end=t+D(str(e.get('dur',0)));tid=str(e.get('tid')); name=str(e.get('name',''))
  for k,(a,b) in w.items():
   if tid=='35' and 'EVENT' in name.upper() and end>=a-D(300) and t<=b+D(300): out[k]['same_track_nearby_events'].append({q:e.get(q) for q in ('name','tid','ts','dur','args')})
   if tid!='35' and t<b and end>a:
    out[k]['other_tracks_X_counts'][(tid,name)]+=1
    if 'EVENT' in name.upper() or 'WAIT' in name.upper():out[k]['other_tracks_event_wait'].append({q:e.get(q) for q in ('name','tid','ts','dur','args')})
for k in out:out[k]['other_tracks_X_counts']=[{'tid':tid,'name':n,'count':cnt} for (tid,n),cnt in out[k]['other_tracks_X_counts'].items()]
print(json.dumps({'windows_us':{k:list(map(str,v)) for k,v in w.items()},'result':out,'limitations':'Track timestamps only; intersection is not dependency or proof of clock alignment. Same-track event window extends ±300us. Exported X events only, no claim of complete hardware activity.'},indent=2,default=str))
