from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,socket,statistics
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
ROOT=P/'analysis/native-execution-identity-v14'
b=(ROOT/'index.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4';index=json.loads(b)
OUT=ROOT/'graph-blocks-PMU-and-MC2-v14.json';assert not OUT.exists()
graphrows=[];pmustats=[];MC2=[];type_metrics=defaultdict(list)
def union(ts):
    end=None;total=0
    for a,b in sorted((t['startNs'],t['endNs'])for t in ts):
        if end is None or a>end:total+=b-a;end=b
        elif b>end:total+=b-end;end=b
    return total
def counters(ids,pmu_by_task):
    ps=[p for i in ids for p in pmu_by_task.get(i,[])]
    out=dict(tasks=len(ids),PMU_execution_groups=len(ps))
    for engine,ratio in [('aic','mac_ratio'),('aiv','vec_ratio')]:
        good=[]
        for p in ps:
            d={v['metric_name']:v['value']for v in p['metric_rows']}
            c,v=d.get(engine+'_total_cycles'),d.get(engine+'_'+ratio)
            if isinstance(c,(int,float))and c>0 and isinstance(v,(int,float))and 0<=v<=1:good.append((c,v))
        den=sum(c for c,v in good);num=sum(c*v for c,v in good)
        out[engine]=dict(valid_groups=len(good),reported_cycles=den,reported_weighted_ratio_numerator=num,cycle_weighted_reported_ratio=num/den if den else None)
    return out
for ref in index['databases']:
    b=Path(ref['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==ref['sha256'];d=json.loads(b);di=d['database_index']
    tasks={t['task_record_rowid']:t for t in d['TASK_records']};defs=defaultdict(list)
    for c in d['COMPUTE_definitions']:defs[c['globalTaskId']].append(c)
    assert all(len(v)==1 for v in defs.values())
    def definition(t):return defs[t['globalTaskId']][0]
    pmu_by_task=defaultdict(list);bad=[]
    for p in d['PMU_execution_identity_candidates']:
        if len(p['containing_TASK_rows'])==1:pmu_by_task[p['containing_TASK_rows'][0]].append(p)
        else:bad.append(p)
    compute_ids={t['task_record_rowid']for t in tasks.values()if t['globalTaskId']in defs}
    pmu_compute=set(pmu_by_task)&compute_ids
    missing=compute_ids-set(pmu_by_task)
    pmustats.append(dict(database_index=di,PMU_groups=len(d['PMU_execution_identity_candidates']),unique_native_id_and_interval_groups=sum(len(v)for v in pmu_by_task.values()),ambiguous_groups=len(bad),PMU_assigned_TASK_rows=len(pmu_by_task),PMU_groups_per_TASK_hist=dict(Counter(len(v)for v in pmu_by_task.values())),PMU_on_compute_TASK_rows=len(pmu_compute),PMU_on_noncompute_TASK_rows=len(set(pmu_by_task)-compute_ids),compute_missing_PMU_rows=len(missing),missing_PMU_op_type_counts=dict(Counter(definition(tasks[i])['opType_text']for i in missing)),all_compute_counters=counters(compute_ids,pmu_by_task)))
    for t in tasks.values():
        if t['globalTaskId']not in defs:continue
        c=definition(t)
        type_metrics[c['opType_text']].append(dict(database_index=di,task_record_rowid=t['task_record_rowid'],duration_ns=t['endNs']-t['startNs'],counters=counters([t['task_record_rowid']],pmu_by_task)))
        if c['opType_text']=='DispatchFFNCombine':
            MC2.append(dict(database_index=di,task_record_rowid=t['task_record_rowid'],globalTaskId=t['globalTaskId'],modelId=t['modelId'],startNs=t['startNs'],endNs=t['endNs'],native_API_rowids=t['native_linked_API_rowids'],definition=c,counters=counters([t['task_record_rowid']],pmu_by_task)))
    for gi,g in enumerate(d['graph_API_control_bindings']):
        if not g['control_pair_valid']:continue
        ts=[tasks[i]for i in g['compute_task_rowids']]
        markers=sorted([t for t in ts if definition(t)['opType_text']in ['HcPre','HcPost']],key=lambda t:(t['startNs'],t['streamId'],t['taskId']))
        good=[definition(t)['opType_text']for t in markers]==['HcPre','HcPost','HcPre','HcPost']*40
        blocks=[];assigned=set()
        if good:
            for k in range(40):
                ms=markers[k*4:k*4+4];lo,hi=ms[0]['startNs'],ms[-1]['endNs']
                bt=[t for t in ts if lo<=t['startNs'] and t['endNs']<=hi];ids=[t['task_record_rowid']for t in bt]
                assert not set(ids)&assigned;assigned.update(ids)
                ops=Counter(definition(t)['opType_text']for t in bt)
                blocks.append(dict(block_ordinal=k,startNs=lo,endNs=hi,span_ns=hi-lo,compute_interval_union_ns=union(bt),task_record_rowids=ids,op_type_counts=dict(ops),one_attention_one_MoE=ops['SparseFlashMla']==ops['DispatchFFNCombine']==1,counters=counters(ids,pmu_by_task)))
        outside=[t for t in ts if t['task_record_rowid']not in assigned]
        row=dict(database_index=di,graph_ordinal=gi,CANN_API=g['CANN_API'],model_id=g['model_id'],prepare_root_keys=g['prepare_root_keys'],phase_records=g['phase_records'],exact_marker_count=len(g['same_thread_markers']),compute_executions=len(ts),marker_sequence_pass=good,blocks=blocks,outside_marker_tasks=[dict(task=t,definition=definition(t))for t in outside],graph_counters=counters(g['compute_task_rowids'],pmu_by_task),outside_marker_counters=counters([t['task_record_rowid']for t in outside],pmu_by_task),compute_interval_union_ns=union(ts),graph_control_pair_ns=g['control_pair_end_ns']-g['control_pair_start_ns'])
        graphrows.append(row)
cohorts=defaultdict(list)
for g in graphrows:
    if len(g['phase_records'])==1 and g['phase_records'][0].get('schedule_position_cohort_key'):cohorts[g['phase_records'][0]['schedule_position_cohort_key']].append(g)
cohortrows=[]
for key,gs in cohorts.items():
    cohortrows.append(dict(key=key,graph_executions=len(gs),ep_ranks=sorted(g['phase_records'][0]['ep_rank']for g in gs),phases=sorted({g['phase_records'][0]['phase']for g in gs}),members=[dict(database_index=g['database_index'],graph_ordinal=g['graph_ordinal'],root_key=g['prepare_root_keys'][0],ep_rank=g['phase_records'][0]['ep_rank'],compute_interval_union_ns=g['compute_interval_union_ns'],graph_control_pair_ns=g['graph_control_pair_ns'],graph_counters=g['graph_counters'])for g in gs]))
signature_groups=defaultdict(list)
for m in MC2:
    c=m['definition'];key=json.dumps({k:c.get(k)for k in ['inputShapes_text','outputShapes_text','inputDataTypes_text','outputDataTypes_text','blockNum','mixBlockNum']},sort_keys=True)
    signature_groups[key].append(m)
mc2groups=[dict(signature=json.loads(k),executions=len(v),databases=len({m['database_index']for m in v}),model_ids=sorted({m['modelId']for m in v}),duration_ns=dict(minimum=min(m['endNs']-m['startNs']for m in v),median=statistics.median(m['endNs']-m['startNs']for m in v),maximum=max(m['endNs']-m['startNs']for m in v)),execution_keys=[[m['database_index'],m['task_record_rowid']]for m in v])for k,v in signature_groups.items()]
op_summaries=[]
for op,rows in sorted(type_metrics.items(),key=lambda item:str(item[0])):
    out=dict(op_type=op,executions=len(rows),duration_ns=dict(minimum=min(r['duration_ns']for r in rows),median=statistics.median(r['duration_ns']for r in rows),maximum=max(r['duration_ns']for r in rows)))
    for engine in ['aic','aiv']:
        den=sum(r['counters'][engine]['reported_cycles']for r in rows);num=sum(r['counters'][engine]['reported_weighted_ratio_numerator']for r in rows)
        out[engine+'_reported_proxy']=num/den if den else None
    op_summaries.append(out)
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(index['databases']),compute_executions=sum(r['summary']['compute_executions']for r in index['databases']),PMU_unique_native_id_and_interval_groups=sum(s['unique_native_id_and_interval_groups']for s in pmustats),PMU_assigned_TASK_rows=sum(s['PMU_assigned_TASK_rows']for s in pmustats),PMU_on_noncompute_TASK_rows=sum(s['PMU_on_noncompute_TASK_rows']for s in pmustats),compute_missing_PMU_rows=sum(s['compute_missing_PMU_rows']for s in pmustats),graph_executions=len(graphrows),marker_sequence_pass=sum(g['marker_sequence_pass']for g in graphrows),marker_blocks=sum(len(g['blocks'])for g in graphrows),blocks_with_one_attention_one_MoE=sum(b['one_attention_one_MoE']for g in graphrows for b in g['blocks']),graph_phase_counts=dict(Counter(g['phase_records'][0]['phase']if len(g['phase_records'])==1 else 'PHASE_UNBOUND'for g in graphrows)),bound_graph_cohorts=len(cohortrows),cohort_ep_sets=[c['ep_ranks']for c in cohortrows],MC2_executions=len(MC2),MC2_shape_regimes=len(mc2groups),op_types=len(op_summaries),no_target_source_read=True,no_DB_or_device_work=True,whole_modules_closed=0,overall_complete=False,
    limits=['PMU association uses native globalTaskId plus unique TASK interval containing stored timestamp; not nearest time. Counter definitions/units/averaging are not inferred by association.', 'MAC/vector proxies pool reported cycles and ratios only; not physical peak efficiency, MFU, CPU cost, request exposure or causal ranking.', '40four-marker blocks are temporal structural partitions; source layer identity/all tensor consumers are not proved by markers.', 'Unbound graph executions stay unbound; cohorts use exact recorded request/position metadata, never timestamps or graph ordinal.', 'MC2 compound durations/PMU remain separate from internal routing/expert/link work; no physical valid-mask/row-count claim from static shapes.'])
payload=(json.dumps(dict(summary=summary,PMU_execution_binding=pmustats,graph_executions=graphrows,graph_cohorts=cohortrows,MC2_executions=MC2,MC2_shape_groups=mc2groups,op_reported_counter_summaries=op_summaries),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest(),MC2_regimes=[{k:v for k,v in g.items()if k!='execution_keys'}for g in mc2groups],missing_PMU_ops=dict(sum((Counter(s['missing_PMU_op_type_counts'])for s in pmustats),Counter()))),ensure_ascii=False))
