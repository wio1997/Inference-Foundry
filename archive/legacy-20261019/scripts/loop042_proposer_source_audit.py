#!/usr/bin/env python3
"""Offline host phase covariance and proposer source audit after Run137."""
import glob,json,math,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
phase=json.loads((root/'evidence/20260925_loop042_phase/run137/phase_analysis.json').read_text())
run98=json.loads((root/'evidence/20260925_loop039_priority/run119/steady_priority.json').read_text())
cs=phase['cycles']
def corr(xs,ys):
    mx=statistics.mean(xs);my=statistics.mean(ys)
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den=math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
    return num/den
pair=[]
for c in cs:
    for r in range(8):
        d=c['stage_duration_ms'];pair.append((d['target'][str(r)],d['proposer'][str(r)]))
by_rank={}
for r in range(8):
    target=[c['stage_duration_ms']['target'][str(r)] for c in cs]
    proposer=[c['stage_duration_ms']['proposer'][str(r)] for c in cs]
    by_rank[str(r)]={'target_host_median_ms':statistics.median(target),'proposer_host_median_ms':statistics.median(proposer),
                     'sum_host_median_ms':statistics.median(x+y for x,y in zip(target,proposer))}
t=[x for x,y in pair];p=[y for x,y in pair]
names=['extreme::dspark_host_mirror_commit','extreme::dspark_host_mirror_launch',
       'extreme::dspark_refresh_common','extreme::dspark_context_slots',
       'extreme::dspark_prepare_inputs','extreme::dspark_pack_hidden',
       'extreme::dspark_model']
phase_rows={r:{x['cycle']:dict(x['marks']) for x in json.loads((root/f'evidence/20260925_loop042_phase/run137/phase/rank{r}.json').read_text())['phase_rows']} for r in range(8)}
latest_end=[max(phase_rows[r][c]['draft_commit'] for r in range(8)) for c in range(64,256)]
latest_end_cadence_ms=[(b-a)/1e6 for a,b in zip(latest_end,latest_end[1:])]
scopes={name:[] for name in names}
for rank in range(8):
    paths=glob.glob(str(root/f'evidence/20260924_loop038_cycle/run106/profile/rank{rank}_*/ASCEND_PROFILER_OUTPUT/trace_view.json'))
    assert len(paths)==1
    events=json.loads(Path(paths[0]).read_text())
    for name in names:
        durations=[float(e['dur'])/1000 for e in events if e.get('cat')=='cpu_op' and e.get('name')==name]
        assert len(durations)==2,(rank,name,len(durations))
        scopes[name]+=durations
scope_medians={name:statistics.median(vals) for name,vals in scopes.items()}
out={'run':'run138','source':'Run137 legal 8-rank host timestamps cycles64-255 plus Run98 unprofiled NPU events',
     'host_target_proposer_duration_correlation':corr(t,p),
     'latest_rank_end_cadence_median_ms':statistics.median(latest_end_cadence_ms),
     'latest_rank_end_total_span_ms_per_cycle':(latest_end[-1]-latest_end[0])/1e6/(len(latest_end)-1),
     'run106_profiled_host_dspark_scope_median_ms':scope_medians,
     'by_rank':by_rank,
     'host_target_plus_proposer_median_ms':statistics.median(x+y for x,y in pair),
     'host_target_median_ms':statistics.median(t),'host_proposer_median_ms':statistics.median(p),
     'device_target_median_ms_run98':run98['runtime_stage_ms']['target']['median'],
     'device_proposer_median_ms_run98':run98['runtime_stage_ms']['proposer']['median'],
     'source_findings':[
       'bootstrap/vllm_dspark_handoff.py _execute commits a one-cycle-delayed 12-count Host mirror via event synchronize, refreshes metadata and draft slots, prepares inputs, packs target hidden, then calls borrowed proposer._propose.',
       'Run98 NPU event proposer median is ~6.4ms, while Run137 Host proposer scope median is ~42ms; asynchronous execution and waiting on prior target work make these different quantities.',
       'DSpark borrowed proposer is eager (DirectDSparkHandoff rejects use_cuda_graph). Run106 profiled CPU scopes show model/_propose dominates Host proposer time, but these two profiled cycles are not steady unprofiled evidence; deeper model subphase timing is needed.'
     ],
     'interpretation':'Across rank medians, slower Host target scope corresponds to faster Host proposer scope and the sum is near 47ms, close to Run98 target NPU event duration. Host proposer 42ms cannot be treated as removable overhead. Run106 model/_propose scope dominates profiled Host proposer; bounded subphase wall and thread CPU timing must localize wait before an edit.',
     'limits':'Run98 and Run137 are separate cohorts; no exact shared-cycle device/host alignment. Host timestamps are asynchronous dispatch marks and include waits. Cross-rank cycle_wall is an envelope, not repeatable steady cadence; use latest-rank end cadence for steady progress.'}
path=root/'evidence/20260925_loop042_phase/run138/proposer_audit.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ('host_target_proposer_duration_correlation','by_rank','host_target_plus_proposer_median_ms','host_target_median_ms','host_proposer_median_ms','device_target_median_ms_run98','device_proposer_median_ms_run98')},indent=2))
