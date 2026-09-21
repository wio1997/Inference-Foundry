"""Summarize TP0 torch-NPU kernels by the request phase timestamps."""
import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('/data/wio/Inference_Foundry/evidence/20260920_scope_profile')
phase = json.loads((root/'run1/phase_times.json').read_text())
path = glob.glob(str(root/'raw/dp0_pp0_tp0*/ASCEND_PROFILER_OUTPUT/kernel_details.csv'))[0]
d = pd.read_csv(path, usecols=['Task ID','Type','Start Time(us)','Duration(us)'])
d = d.dropna(subset=['Task ID','Type','Start Time(us)','Duration(us)'])
d['category'] = np.where(d['Type'].str.startswith('hcom_'), 'communication', 'compute_or_copy')

def union(start, end):
    if len(start)==0: return 0.0, 0.0
    ix=np.argsort(start); start=start[ix]; end=end[ix]
    left=right=start[0]; total=longest=0.0
    for s,e in zip(start[1:],end[1:]):
        if s>right:
            total+=right-left; longest=max(longest,s-right); left=s; right=e
        else: right=max(right,e)
    return total+right-left,longest

out={'source':path,'filter':'non-null Task ID; intervals clipped to phase wall; overlapping streams unioned','phases':{}}
for name in ('warm','cold'):
    t0=pd.Timestamp(phase[name+'_start_utc']).timestamp()*1e6
    t1=pd.Timestamp(phase[name+'_end_utc']).timestamp()*1e6
    x=d[(d['Start Time(us)']+d['Duration(us)']>t0)&(d['Start Time(us)']<t1)].copy()
    x['start']=np.maximum(x['Start Time(us)'],t0)
    x['end']=np.minimum(x['Start Time(us)']+x['Duration(us)'],t1)
    result={'wall_s':(t1-t0)/1e6,'events':len(x),'categories':{}}
    for cat,frame in [('all',x),*list(x.groupby('category'))]:
        busy,gap=union(frame['start'].to_numpy(),frame['end'].to_numpy())
        result['categories'][cat]={'events':len(frame),'union_busy_s':busy/1e6,'longest_idle_gap_ms':gap/1e3,'sum_duration_s_overlaps':(frame['end']-frame['start']).sum()/1e6}
    result['top_types_sum_s_overlaps']=(x.assign(clipped_duration=x['end']-x['start']).groupby('Type')['clipped_duration'].sum().sort_values(ascending=False).head(20)/1e6).to_dict()
    out['phases'][name]=result
(root/'run1/device_phase_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out['phases'],indent=2))
