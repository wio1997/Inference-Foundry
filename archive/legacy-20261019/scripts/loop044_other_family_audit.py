#!/usr/bin/env python3
"""Exact-name census of Run107 target kernels outside Run143 named families."""
import glob,json,statistics
from collections import defaultdict
from pathlib import Path
from loop044_target_union_audit import family,excluded,root
rows=[]
for rank in range(8):
    paths=glob.glob(str(root/f'evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
    assert len(paths)==1
    events=json.loads(Path(paths[0]).read_text())
    scopes=sorted((e for e in events if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
    assert len(scopes)==2
    for cycle,scope in zip((64,65),scopes):
        if (rank,cycle) in excluded:continue
        start=float(scope['ts']);end=start+float(scope['dur'])
        counts=defaultdict(int);sums=defaultdict(float);all_events=[]
        for e in events:
            t=float(e.get('ts',-1));task=e.get('args',{}).get('Task Type','')
            if not(start<=t<end) or not(task.startswith('KERNEL_') or task.startswith('SDMA_')):continue
            name=e.get('name','');group=family(name,task)
            key=name if group=='other_compute' else '@'+group
            dur=float(e['dur']);a=t;b=min(t+dur,end)
            counts[key]+=1;sums[key]+=dur/1000
            all_events.append((a,1,key));all_events.append((b,-1,key))
        active=defaultdict(int);exclusive=defaultdict(float);prev=None
        for t,delta,key in sorted(all_events):
            if prev is not None and t>prev:
                live=[k for k,v in active.items() if v>0]
                if len(live)==1:exclusive[live[0]]+=(t-prev)/1000
            active[key]+=delta;prev=t
        rows.append({'rank':rank,'cycle':cycle,'counts':dict(counts),'kernel_sum_ms':dict(sums),'exclusive_ms':dict(exclusive)})
assert len(rows)==15
names=sorted(set(k for r in rows for k in r['counts'] if not k.startswith('@')))
def desc(values):return {'median':statistics.median(values),'min':min(values),'max':max(values)}
summary={name:{'count':desc([r['counts'].get(name,0) for r in rows]),'kernel_sum_ms':desc([r['kernel_sum_ms'].get(name,0) for r in rows]),'exclusive_ms':desc([r['exclusive_ms'].get(name,0) for r in rows])} for name in names}
order=sorted(names,key=lambda n:summary[n]['exclusive_ms']['median'],reverse=True)
out={'run':'run144','source':'Run107 15 valid synchronized target windows; exact trace kernel name','names_count':len(names),'ranked_names':[{"name":n,**summary[n]} for n in order],'windows':rows,'interpretation':'Exact-name exclusive device timeline coverage is descriptive, not the gain from removal. Names may be kernel implementations shared by multiple source operations; source mapping is required.'}
p=root/'evidence/20260925_loop044_target/run144/other_family_audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'names_count':len(names),'top25':out['ranked_names'][:25]},indent=2))
