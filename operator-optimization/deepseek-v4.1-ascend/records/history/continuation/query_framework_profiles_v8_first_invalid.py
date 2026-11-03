from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,statistics

N=Path(__file__).parent;O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')


def stat(v):
    return dict(n=len(v),min=min(v),median=statistics.median(v),mean=statistics.mean(v),max=max(v)) if v else None


def union(v):
    end=None;total=0
    for a,b in sorted(v):
        if b<=a:continue
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total


def main():
    files={name:R/name for name in ['publication-draft/framework-cpu.json','continuation-20261030/worker-validation.json','continuation-20261030/prepare-input-analysis.json','continuation-20261030/graph-executions.json','continuation-20261030/iteration-gaps.json']}
    inputs={name:json.loads(p.read_text()) for name,p in files.items()}
    hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
    prep=inputs['continuation-20261030/prepare-input-analysis.json']
    graph=inputs['continuation-20261030/graph-executions.json']
    gaps=inputs['continuation-20261030/iteration-gaps.json']
    by_source={g['source_dir']:g for g in graph['rows']}
    gaps_by_source={g['source_dir']:g for g in gaps['rows']}
    roots=[];names=defaultdict(lambda:dict(count=0,inclusive_sum_ns=0,exclusive_sum_ns=0,root_presence=0))
    for rank in prep['rows']:
        source=rank['source_dir'];assert source in by_source and source in gaps_by_source
        assert by_source[source]['rank']==rank['rank']==gaps_by_source[source]['rank']
        main_ex=[e for ms in by_source[source]['model_streams'] for e in ms['executions'] if e['main_shape_compatible_40_moe']]
        gaplist=gaps_by_source[source]['gaps']
        for s in rank['prepare_input_scopes']:
            assert s['duration_ns']==s['e_ns']-s['s_ns']
            assert s['duration_ns']==s['descendant_union_ns']+s['root_exclusive_recorded_scope_ns']
            gap_matches=[dict(gap_ordinal=g['ordinal'],overlap_ns=max(0,min(s['e_ns'],g['end_epoch_ns'])-max(s['s_ns'],g['start_epoch_ns']))) for g in gaplist]
            gap_matches=[x for x in gap_matches if x['overlap_ns']>0]
            main_matches=[dict(ordinal=e['ordinal'],overlap_ns=max(0,min(s['e_ns'],e['end_epoch_ns'])-max(s['s_ns'],e['start_epoch_ns']))) for e in main_ex]
            main_matches=[x for x in main_matches if x['overlap_ns']>0]
            roots.append(dict(rank=rank['rank'],source_dir=source,ordinal=s['ordinal'],s_ns=s['s_ns'],e_ns=s['e_ns'],duration_ns=s['duration_ns'],
                              descendant_count=s['descendant_scope_count'],descendant_union_ns=s['descendant_union_ns'],root_uncovered_ns=s['root_exclusive_recorded_scope_ns'],
                              exclusive_partition_sum_ns=sum(v['exclusive_sum_ns'] for v in s['by_name'].values()),containment_violations=s['containment_violations'],
                              original_gap_matches=s['gaps'],recomputed_gap_matches=gap_matches,exported_main_matches=main_matches,
                              by_name=s['by_name']))
            for name,v in s['by_name'].items():
                for k in ['count','inclusive_sum_ns','exclusive_sum_ns']:names[name][k]+=v[k]
                names[name]['root_presence']+=1
    assert len(roots)==24
    per_rank=[]
    for rank in sorted({r['rank'] for r in roots}):
        rs=[r for r in roots if r['rank']==rank]
        per_rank.append(dict(rank=rank,roots=len(rs),duration_ns=stat([r['duration_ns'] for r in rs]),
                             root_uncovered_ns=stat([r['root_uncovered_ns'] for r in rs]),descendant_count=stat([r['descendant_count'] for r in rs]),
                             root_uncovered_fraction=stat([r['root_uncovered_ns']/r['duration_ns'] for r in rs]),
                             copy_H2D_calls=stat([r['by_name'].get('acl_memcpy_host_to_device',{}).get('count',0) for r in rs]),
                             pin_memory_calls=stat([r['by_name'].get('aten::_pin_memory',{}).get('count',0) for r in rs]),
                             Event_synchronize_count=sum(r['by_name'].get('Event::synchronize',{}).get('count',0) for r in rs),
                             containment_violations=sum(r['containment_violations'] for r in rs),
                             roots_with_exported_main_overlap=sum(bool(r['exported_main_matches']) for r in rs),
                             roots_with_recomputed_gap_overlap=sum(bool(r['recomputed_gap_matches']) for r in rs)))
    framework=inputs['publication-draft/framework-cpu.json'];worker=inputs['continuation-20261030/worker-validation.json']
    worker_requests=[]
    for i,r in enumerate(worker['request_timings']):
        nonempty=[c['at_ns'] for c in r['chunks'] if c['nonempty']]
        deltas=[b-a for a,b in zip(nonempty,nonempty[1:])]
        worker_requests.append(dict(index=i,elapsed_s=(r['end_ns']-r['start_ns'])/1e9,first_visible_s=(r['first_nonempty_ns']-r['start_ns'])/1e9,
                                    prompt_tokens=r['usage']['prompt_tokens'],generation_tokens=r['usage']['completion_tokens'],nonempty_SSE_lines=len(nonempty),
                                    mean_output_tokens_per_nonempty_line=r['usage']['completion_tokens']/len(nonempty) if nonempty else None,
                                    line_gap_ns=stat(deltas)))
    engine=dict(run_id=framework['run_id'],case_valid=framework['case_valid'],request_elapsed_s=framework['request_elapsed_s'],proc_interval_s=framework['proc_interval_s'],
                engines={name:{k:v for k,v in data.items() if k not in ['leaf_top']} for name,data in framework['engines'].items()},
                leaf_sample_weights={name:data['leaf_top'] for name,data in framework['engines'].items()},counters=framework['total_counters'],
                scope=framework['scope'],limits=framework['limitations'])
    counters=worker['counters']
    worker_info=dict(run_id=worker['run_id'],sampler_exit_codes={k:v['exit_code'] for k,v in worker['samplers'].items()},
                     stack_files_present=worker['stack_files_present'],request_count=len(worker_requests),requests=worker_requests,
                     counters=counters,proc_schema=list(worker['proc']) if isinstance(worker['proc'],dict) else type(worker['proc']).__name__,
                     limits=['No successful worker Python stack file was produced; proc readings cannot supply per-function samples.',
                             'SSE line timing is output-delivery granularity, not token-level TPOT, model-step latency or server serialization cost.'])
    name_rows=[dict(name=k,**v) for k,v in sorted(names.items(),key=lambda kv:-kv[1]['exclusive_sum_ns'])]
    summary=dict(generated_utc=datetime.now(timezone.utc).isoformat(),input_sha256=hashes,prepare_roots=len(roots),prepare_rank_summaries=per_rank,
                 prepare_name_count=len(name_rows),recorded_descendant_count=sum(r['descendant_count'] for r in roots),
                 partition_checks_pass=sum(r['descendant_union_ns']==r['exclusive_partition_sum_ns'] for r in roots),
                 original_recomputed_gap_match_disagreements=sum(r['original_gap_matches']!=r['recomputed_gap_matches'] for r in roots),
                 root_exclusive_is_cpu_active=False,exported_host_device_overlap_is_causal_proof=False,
                 engine_run_id=engine['run_id'],worker_run_id=worker_info['run_id'],worker_stack_files=worker_info['stack_files_present'],
                 worker_request_summary=worker_requests,all_runs_joined=False,new_source_read=False,new_device_or_inference_work=False,overall_complete=False)
    result=dict(summary=summary,prepare_scope_results=roots,prepare_name_aggregate=name_rows,engine=engine,worker=worker_info,
                existing_source_snapshot_sha256={k:v['sha256'] for k,v in prep['sources'].items()},
                limits=['Recorded scope inclusive/exclusive walls are not CPU-active work or removable time; cross-rank sums do not yield request latency.',
                        'Recomputed intersections use exported epoch coordinates; they do not independently calibrate CPU/device clocks or prove dependency/overlap.',
                        'Engine CPU run, worker validation run and official profiling run are distinct observations and remain separate.',
                        'Name aggregates query all recorded names, not only TopN; names alone do not identify metadata fields, aliases or opportunities.'])
    (O/'framework-profile-audit-v8.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (O/'framework-profile-summary-v8.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,prepare_names_top12=name_rows[:12]),ensure_ascii=False))


if __name__=='__main__':main()
