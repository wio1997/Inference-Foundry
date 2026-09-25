#!/usr/bin/env python3
"""Timeline coverage by target kernel family in Run107 valid windows."""
import glob,json,statistics
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parents[1]
prior=json.loads((root/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
excluded={(x['rank'],x['cycle']) for x in prior['excluded_windows']}
name_map={
 'gmm1':lambda n:n.startswith('aclnnGroupedMatmulSwigluQuantWeightNzV2_'),
 'gmm2':lambda n:n.startswith('aclnnGroupedMatmulWeightNz_'),
 'quant_matmul':lambda n:n.startswith('aclnnQuantMatmulWeightNz_'),
 'compressor':lambda n:n=='Compressor',
 'hc_pre':lambda n:n=='HcPre',
 'hc_post':lambda n:n=='HcPost',
 'scatter_sk':lambda n:n.startswith('aclnnScatterNdUpdateSk_'),
 'rms_family':lambda n:n in ('RmsNorm','RmsNormDynamicQuant','triton_rms_kernel'),
 'communication':lambda n:n.startswith('hcom_'),
}
def family(n,task):
    if task.startswith('SDMA_'):return 'sdma'
    for k,pred in name_map.items():
        if pred(n):return k
    return 'other_compute'
def union(intervals):
    intervals=sorted(intervals)
    if not intervals:return 0.0
    total=0.0;a,b=intervals[0]
    for c,d in intervals[1:]:
        if c>b:
            total+=b-a;a,b=c,d
        else:b=max(b,d)
    return total+b-a
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
        intervals=defaultdict(list);kernel_counts=defaultdict(int);kernel_sums=defaultdict(float)
        for e in events:
            t=float(e.get('ts',-1))
            if not(start<=t<end):continue
            task=e.get('args',{}).get('Task Type','')
            if not(task.startswith('KERNEL_') or task.startswith('SDMA_')):continue
            dur=float(e['dur']);name=e.get('name','')
            group=family(name,task)
            intervals[group].append((t,min(t+dur,end)))
            kernel_counts[group]+=1;kernel_sums[group]+=dur/1000
        assert kernel_counts['gmm1']==43 and kernel_counts['gmm2']==43
        assert kernel_counts['quant_matmul']==236
        assert kernel_counts['scatter_sk']==126
        changes=[]
        for group,ivals in intervals.items():
            for a,b in ivals:
                changes.extend(((a,1,group),(b,-1,group)))
        changes.sort(key=lambda x:x[0])
        active=defaultdict(int);covered=defaultdict(float);exclusive=defaultdict(float)
        all_covered=0.0;prev=None
        for t,delta,group in changes:
            if prev is not None and t>prev:
                live=[k for k,v in active.items() if v>0]
                span=t-prev
                if live:all_covered+=span
                for k in live:covered[k]+=span
                if len(live)==1:exclusive[live[0]]+=span
            active[group]+=delta;prev=t
        row={'rank':rank,'cycle':cycle,'kernel_counts':dict(kernel_counts),
             'kernel_sum_ms':dict(kernel_sums),
             'union_ms':all_covered/1000,
             'family_coverage_ms':{k:v/1000 for k,v in covered.items()},
             'family_exclusive_ms':{k:v/1000 for k,v in exclusive.items()}}
        rows.append(row)
assert len(rows)==15
families=sorted(set(k for row in rows for k in row['kernel_counts']))
def desc(xs):return {'median':statistics.median(xs),'min':min(xs),'max':max(xs)}
summary={k:{'count':desc([r['kernel_counts'].get(k,0) for r in rows]),
            'kernel_sum_ms':desc([r['kernel_sum_ms'].get(k,0) for r in rows]),
            'coverage_ms':desc([r['family_coverage_ms'].get(k,0) for r in rows]),
            'exclusive_coverage_ms':desc([r['family_exclusive_ms'].get(k,0) for r in rows])}
         for k in families}
out={'run':'run143','source':'Run107 synchronized target trace_view; 15 valid rank-cycle windows',
     'device_union_ms':desc([r['union_ms'] for r in rows]),
     'prior_target_window_union_ms':prior['device_union_ms'],
     'families':summary,'windows':rows,
     'interpretation':'Coverage is a trace timeline census. Exclusive coverage means no other classified kernel family executes in that interval; it is not the latency gain from deleting or shortening that family because dependencies and launch timing are not reconstructed. Communication includes peer wait and profile synchronization. Kernel sums may overlap.',
     'comparison':'Run98 unprofiled target event median ~46.575ms. Run107 synced device union is diagnostic and cannot be used as formal E2E or direct attainable bound.'}
p=root/'evidence/20260925_loop044_target/run143/union_audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'device_union_ms':out['device_union_ms'],'families':summary},indent=2))
