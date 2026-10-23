#!/usr/bin/env python3
"""Partition observed native kernel time and audit graph task connections."""
import bisect
import collections
import csv
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
import sys
from e33_clock_map import ClockMap

ROOT=Path(__file__).resolve().parents[1]


def family(name,kind):
    x=(kind+' '+name).lower()
    if 'mega_moe' in x:return 'mega_moe_fused'
    if any(k in x for k in ['engram','dequantized_gather']):return 'engram'
    if any(k in x for k in ['hcom_','hccllaunch','ccl_kernel']):return 'hccl_native_tasks'
    if any(k in x for k in ['groupedmatmul','swiglugroupquant','moedistribute','moefinalizerouting','moegating','histogramv2','tokenunpermute']):return 'moe_separate_kernels'
    if any(k in x for k in ['mqsmla','sparse_attn','sparse_mqa','lightning_indexer','indexer_']):return 'attention_and_indexer'
    if any(k in x for k in ['mhc','hcpre','hc_prenorm','sinkhorn','hc_']):return 'mhc'
    if any(k in x for k in ['gemm_impl','matmul','mat_mul','linear_']):return 'other_gemm'
    if any(k in x for k in ['norm','rope','rotary','hadamard','cast_store_kvcache']):return 'norm_rope_cache'
    if any(k in x for k in ['sample','sampling','verification','rejection','topk','top_k','softmax']):return 'sampling_topk_softmax'
    if any(k in x for k in ['quant','cast_kernel']):return 'other_quantization'
    return 'other_named_kernels'


def partition(intervals,start,end):
    events=[]
    for category,values in intervals.items():
        for a,b in values:events.append((a,category,1));events.append((b,category,-1))
    events.sort();active=collections.Counter();combinations=collections.Counter();last=start;i=0
    while i<len(events):
        point=events[i][0]
        if point>last:
            key=tuple(sorted(k for k,v in active.items() if v>0));combinations[key]+=point-last;last=point
        while i<len(events) and events[i][0]==point:
            _,cat,delta=events[i];active[cat]+=delta;assert active[cat]>=0;i+=1
    if last<end:combinations[()]+=end-last
    assert sum(combinations.values())==end-start
    return [dict(families=list(k),wall_ns=v) for k,v in sorted(combinations.items(),key=lambda kv:-kv[1])]


def graph_audit(db,clock,spans,rank,start,end):
    db.row_factory=sqlite3.Row
    raw=db.execute('''SELECT a.connectionId,a.startNs as hostStart,a.endNs as hostEnd,t.modelId,
           min(t.startNs) as deviceStart,max(t.endNs) as deviceEnd,count(*) as markerTasks
           FROM TASK t JOIN CANN_API a ON t.connectionId=a.connectionId
           JOIN STRING_IDS s ON a.name=s.id WHERE s.value='aclmdlRIExecuteAsync'
           GROUP BY a.connectionId,t.modelId ORDER BY deviceStart''').fetchall()
    graphs=[];models=collections.defaultdict(list)
    relevant=[x for x in spans if x.get('dp_rank')==rank and any(k in x['source'] for k in ['worker.NPUWorker.execute_model','execute_main_model','execute_dflash','execute_draft','engine_step'])]
    for row in raw:
        d=dict(row);d['host_start_wall_ns']=clock.wall_ns(d['hostStart']);d['host_end_wall_ns']=clock.wall_ns(d['hostEnd'])
        d['host_enclosing_sources']=sorted({x['source'] for x in relevant if x['wall_ns']<=d['host_start_wall_ns'] and d['host_end_wall_ns']<=x['wall_end_ns']})
        d['observed_kernel_count']=0;d['kernel_families']=collections.Counter();d['grouped_mm_kernels']=0;d['mega_moe_kernels']=0
        d['inside_client_window']=d['host_start_wall_ns']>=start and d['host_end_wall_ns']<=end
        graphs.append(d);models[d['modelId']].append(d)
    starts={k:[v['deviceStart'] for v in values] for k,values in models.items()};unassigned=[];assigned=0
    for row in db.execute('''SELECT t.startNs,t.endNs,t.modelId,t.globalTaskId,s.value as name
           FROM TASK t JOIN COMPUTE_TASK_INFO i ON t.globalTaskId=i.globalTaskId
           JOIN STRING_IDS s ON i.name=s.id WHERE t.modelId<>4294967295'''):
        d=dict(row);model=d['modelId'];index=bisect.bisect_right(starts.get(model,[]),d['startNs'])-1
        if index<0 or d['endNs']>models[model][index]['deviceEnd']:
            unassigned.append(d);continue
        g=models[model][index];g['observed_kernel_count']+=1;g['kernel_families'][family(d['name'],'')]+=1
        g['grouped_mm_kernels']+=int('groupedmatmul' in d['name'].lower());g['mega_moe_kernels']+=int('mega_moe' in d['name'].lower());assigned+=1
    for g in graphs:g['kernel_families']=dict(g['kernel_families'])
    return dict(graph_instances=graphs,graph_replay_launches=len(graphs),graph_model_ids=sorted(models),
                graph_compute_tasks_assigned=assigned,graph_compute_tasks_unassigned=len(unassigned),unassigned_samples=unassigned[:20],
                method='Match native graph modelId and task interval to the same-model RIExecuteAsync marker window; CPU phase enclosure is checked independently.',
                limits=['A graph launch or modelId is not, by itself, proof of useful requests, target/draft semantics, or dummy work.',
                        'CPU spans do not currently wrap vLLM sample_tokens or execute_dummy_batch; unmatched phase attribution is retained.',
                        'Tasks outside every graph marker are preserved as unassigned, not forced into a phase.'])


def main():
    from e33_native_census import native_clock_rows,profiles,union_ns
    tag,mode=sys.argv[1:];run=f'E33-{tag}-L32K-C32-D1-{mode}-P';framework='di' if tag=='DI' else 'vllm'
    base=ROOT/f'evidence-e33/analysis/{run}-calibrated1';summary=json.loads((base/'summary.json').read_text());start,end=summary['client_window_wall_ns']
    out=ROOT/f'evidence-e33/analysis/{run}-timeline1';out.mkdir(exist_ok=False)
    diag=ROOT/(f'evidence-e33/di-detailed/{run}' if tag=='DI' else 'evidence-e33/vllm-detailed')
    anchors,spans,_,_,_=native_clock_rows(run,diag,framework)
    results=[]
    for profile in profiles(ROOT/f'evidence-e33/control/{run}/exported-traces.json'):
        clock=ClockMap.from_profile(profile);native=profile/'ASCEND_PROFILER_OUTPUT';dbpath=next(native.glob('ascend_pytorch_profiler_*.db'))
        db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True);rank=db.execute('SELECT rankId FROM RANK_DEVICE_MAP').fetchone()[0]
        intervals=collections.defaultdict(list);stats=collections.defaultdict(lambda:dict(count=0,sum_duration_ns=0));types=collections.Counter();attrs=collections.Counter()
        for row in csv.DictReader((native/'kernel_details.csv').open()):
            a=int(Decimal(row['Start Time(us)'].strip())*1000);b=a+int(Decimal(row['Duration(us)'].strip())*1000)
            a=max(start,clock.wall_ns(a));b=min(end,clock.wall_ns(b))
            if b<=a:continue
            category=family(row['Name'],row['Type']);intervals[category].append((a,b));g=stats[category];g['count']+=1;g['sum_duration_ns']+=b-a;types[row['Type']]+=1
            for k in ['Input Shapes','Input Data Types','Input Formats']:
                if row.get(k,'').strip() not in ['', 'N/A','NA','None','[]']:attrs[k]+=1
        combinations=partition(intervals,start,end)
        for cat,g in stats.items():g['union_ns']=union_ns(intervals[cat]);g['exclusive_ns']=next((x['wall_ns'] for x in combinations if x['families']==[cat]),0)
        graph=graph_audit(db,clock,spans,rank,start,end);db.close()
        result=dict(run_id=run,rank=rank,client_duration_ns=end-start,observed_records=sum(types.values()),operator_type_count=len(types),native_attribute_record_counts=dict(attrs),
                    families=dict(stats),disjoint_wall_partition=combinations,graph=graph,
                    limits=['Family labels are a name/source-guided grouping, not proof that the underlying mathematics or work count is equal.',
                            'Mega MoE includes native communication and compute inside one fused kernel; its internal phases cannot be split by this outer timeline.',
                            'HCCL task wall intervals include waiting and may overlap other kernels.',
                            'Intervals with no exported kernel are unclassified wall time, not automatically CPU overhead or NPU idle.'])
        (out/f'rank{rank:02d}-timeline.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
        print(json.dumps(dict(run_id=run,rank=rank,graphs=graph['graph_replay_launches'],unassigned_graph_tasks=graph['graph_compute_tasks_unassigned'],families=len(stats))),flush=True)
    (out/'summary.json').write_text(json.dumps(dict(run_id=run,ranks=results),indent=2)+'\n')


if __name__=='__main__':main()
