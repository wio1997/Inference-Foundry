#!/usr/bin/env python3
"""Run107 target HCCL duration and cross-rank arrival/end skew audit."""
import glob,json,statistics
from collections import defaultdict
from pathlib import Path
root=Path(__file__).resolve().parents[1]
prior=json.loads((root/'evidence/20260924_loop038_cycle/run107/target_window.json').read_text())
excluded={(x['rank'],x['cycle']) for x in prior['excluded_windows']}
rows=[];by_cycle=defaultdict(dict)
for rank in range(8):
    paths=glob.glob(str(root/f'evidence/20260924_loop038_cycle/run107/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
    assert len(paths)==1
    events=json.loads(Path(paths[0]).read_text())
    scopes=sorted((e for e in events if e.get('name')=='extreme::target' and e.get('cat')=='cpu_op'),key=lambda e:float(e['ts']))
    assert len(scopes)==2
    for cycle,scope in zip((64,65),scopes):
        if (rank,cycle) in excluded:continue
        start=float(scope['ts']);end=start+float(scope['dur'])
        comm=[];gmm=[]
        for e in events:
            t=float(e.get('ts',-1))
            if not start<=t<end:continue
            task=e.get('args',{}).get('Task Type','')
            if not task.startswith('KERNEL_'):continue
            name=e.get('name','');d=float(e['dur'])/1000
            if name.startswith('hcom_'):comm.append({'name':name,'start_us':t,'end_us':t+float(e['dur']),'duration_ms':d})
            if 'GroupedMatmul' in name:gmm.append({'name':name,'duration_ms':d})
        comm.sort(key=lambda x:x['start_us'])
        assert len(comm)==260 and len(gmm)==86,(rank,cycle,len(comm),len(gmm))
        row={'rank':rank,'cycle':cycle,'comm':comm,'comm_duration_ms':sum(e['duration_ms'] for e in comm),'first_reduce_scatter_ms':comm[0]['duration_ms'],'remaining_comm_ms':sum(e['duration_ms'] for e in comm[1:]),'gmm_duration_ms':sum(e['duration_ms'] for e in gmm)}
        rows.append(row);by_cycle[cycle][rank]=row
def desc(v):return {'median':statistics.median(v),'min':min(v),'max':max(v)}
per_kind={}
for k in ('hcom_allGather_','hcom_reduceScatter_','hcom_alltoall_'):
    per_kind[k]={'duration_ms':desc([sum(e['duration_ms'] for e in r['comm'] if e['name']==k) for r in rows]),
                 'count':desc([sum(e['name']==k for e in r['comm']) for r in rows])}
ordinal=[]
for cycle,ranks in by_cycle.items():
    for index in range(260):
        ev=[r['comm'][index] for r in ranks.values()]
        names={e['name'] for e in ev}
        if len(names)!=1:continue
        starts=[e['start_us'] for e in ev];ends=[e['end_us'] for e in ev];dur=[e['duration_ms'] for e in ev]
        ordinal.append({'cycle':cycle,'ordinal':index,'name':ev[0]['name'],'ranks':len(ev),
          'start_skew_ms':(max(starts)-min(starts))/1000,'end_skew_ms':(max(ends)-min(ends))/1000,
          'duration_spread_ms':max(dur)-min(dur),'duration_median_ms':statistics.median(dur),
          'latest_start_rank':list(ranks)[starts.index(max(starts))],
          'latest_end_rank':list(ranks)[ends.index(max(ends))]})
assert len(ordinal)==520
out={'run':'run145','source':'Run107 15 valid target windows, 2 cycles, 260 ordered HCCL kernels per rank-cycle',
     'per_rank_comm_duration_ms':desc([r['comm_duration_ms'] for r in rows]),
     'per_rank_gmm_duration_ms':desc([r['gmm_duration_ms'] for r in rows]),
     'first_reduce_scatter_ms':desc([r['first_reduce_scatter_ms'] for r in rows]),
     'remaining_comm_ms':desc([r['remaining_comm_ms'] for r in rows]),
     'by_kind':per_kind,
     'ordinal_start_skew_ms':desc([x['start_skew_ms'] for x in ordinal]),
     'ordinal_end_skew_ms':desc([x['end_skew_ms'] for x in ordinal]),
     'ordinal_duration_spread_ms':desc([x['duration_spread_ms'] for x in ordinal]),
     'ordinals':ordinal,'windows':[{'rank':r['rank'],'cycle':r['cycle'],'comm_duration_ms':r['comm_duration_ms'],'first_reduce_scatter_ms':r['first_reduce_scatter_ms'],'remaining_comm_ms':r['remaining_comm_ms'],'gmm_duration_ms':r['gmm_duration_ms']} for r in rows],
     'interpretation':'HCCL kernel duration may include waiting for peers. Cross-rank timestamp skew is diagnostic only and cannot be subtracted from target wall time without a causal execution model. Ordinal matching assumes same operation order and checks name equality.'}
p=root/'evidence/20260925_loop044_target/run145/comm_gmm_audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('ordinals','windows')},indent=2))
print('top_duration_spread',json.dumps(sorted(ordinal,key=lambda x:x['duration_spread_ms'],reverse=True)[:12],indent=2))
