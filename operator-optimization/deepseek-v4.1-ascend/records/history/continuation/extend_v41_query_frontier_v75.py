"""Same-boundary QB reference comparisons and measured query-join brackets."""
from collections import defaultdict,Counter
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_projection_ceiling_v61 import stat


def main():
    assert not(O/'V41-QB-query-RoPE-frontier-v75.json').exists()
    refs=read(O/'projection-work-and-reference-bounds-v61.json')['regimes']
    qb=[r for r in refs if r['role']=='Q_B']
    assert len(qb)==8 and sum(r['executions']for r in qb)==4367
    comparison=[]
    for r in qb:
        native_us=r['measured_native_ns']['median']/1000
        comparison.append(dict(regime_id=r['regime_id'],origin=r['origin'],context=r['context'],M=r['M'],executions=r['executions'],measured_QB_TASK_ns=r['measured_native_ns'],QB_direct_INT8_dense_ops=r['conventional_dense_ops_2perMAC'],published_INT8_reference_compute_us=r['reference_compute_us'],cold_QB_value_IO_reference_us=r['reference_cold_io_us'],two_resource_cold_reference_us=r['conditional_reference_floor_us'],reference_dense_OP_rate_fraction=r['reference_compute_us']/native_us,cold_reference_to_native_TASK_ratio=r['conditional_reference_floor_us']/native_us,ratio_is_not_MFU_or_measured_physical_efficiency=True,boundary='One native QB ComputeInfo task only,not QRnorm/RoPE/attention or whole request. Same boundary as v61 baseline QB reference;new fused-subgraph reference is kept separate.',cold_traffic_or_current_clock_capacity_attested=False))
    source_binding=read(O/'V41-QB-native-source-binding-v75.json')
    assert source_binding['bound_direct_TASKs']==1167 and not source_binding['unbound_direct_TASKs']
    old=read(O/'QB-to-query-RoPE-native-visibility-v74.json')['graph_sequences']
    bydb=defaultdict(list)
    for row in old:bydb[row['database_index']].append(row)
    records=[];gap=[];rd=[];toy=[];waittypes=Counter()
    for db,rows in sorted(bydb.items()):
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');tasks={t['task_record_rowid']:t for t in e['TASK_records']}
        for row in rows:
            q=tasks[row['Q_B_TASK_rowid']];r=tasks[row['query_RoPE_TASK_rowid']]
            assert q['streamId']==r['streamId']and q['endNs']<=r['startNs']
            waits=[]
            for stage in row['between_native_stages']:
                t=tasks[stage['TASK_rowid']];waittypes[t['task_type_name']]+=1
                if t['task_type_name']=='EVENT WAIT':waits.append(dict(TASK_rowid=t['task_record_rowid'],startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs']))
            g=r['startNs']-q['endNs'];dur=r['endNs']-r['startNs'];budget=min(g,dur)
            gap.append(g);rd.append(dur);toy.append(budget)
            records.append(dict(database_index=db,graph_ordinal=row['graph_ordinal'],block_ordinal=row['block_ordinal'],QB_TASK_rowid=q['task_record_rowid'],query_RoPE_TASK_rowid=r['task_record_rowid'],QB_duration_ns=q['endNs']-q['startNs'],query_RoPE_duration_ns=dur,QB_end_to_RoPE_start_ns=g,measured_bracket_startQB_to_endRoPE_ns=r['endNs']-q['startNs'],middle_EVENT_WAITs=waits,fixed_duration_ideal_shift_budget_ns=budget,actual_wait_event_producer_or_alias_identity_proved=False,actual_critical_path_gain_ns=None))
    assert len(records)==3200
    write(O/'V41-QB-reference-to-native-comparisons-v75.json',dict(comparisons=comparison,regimes=8,native_QB_executions=4367,scope='One native QB boundary matched to its own declared direct/cold reference;do not compare a QB+Rope fusion lower bound to QB-only duration.',full_physical_numeric_ceiling_proved=False,overall_complete=False))
    write(O/'V41-QB-query-RoPE-frontier-v75.json',dict(records=records,graph_sequences=3200,middle_native_TASK_types=dict(waittypes),gap_ns=stat(gap),query_RoPE_native_ns=stat(rd),fixed_duration_ideal_shift_budget_ns=stat(toy),source_control_order='V4.1 multistream_preprocess:QB matmul at491,main.wait_stream(aux) at492,return at495;outer_preprocess applies query RoPE at373. The source join precedes an operation whose q/C-S values are prepared earlier.',candidate_refinement='Existing F07 epilogue/consumer fusion can also study query RoPE before the KV-stream join. Keep required cache/attention join before its true consumers;all q/C-S/KV alias and source generation/readiness/stream-error contracts must be checked.',unchanged_native_duration_toy_model='With QB and KV times fixed and independent resources,move rope of duration r into observed gap g:maximum placement budget is min(g,r). These measured durations are not unavoidable lower bounds for a new kernel. This toy budget is not a measured saving,request critical-path gain,or a physical performance limit.',resource_gates=['Query RoPE and KV norm/RoPE/scatter may compete for the same Vector/memory pool','Native event producer,buffer aliases/C-S generations and current graph-output publication must be proven','New fused QB can change GEMM tiling,conversion,queue/launch/consumer readiness;price the whole graph','Do not sum these observations across cards/graphs/iterations as request time saved'],actual_kernel_or_schedule_changed=False,actual_request_gain=None,overall_complete=False))
    print(dict(QB_baseline_regimes=8,QB_native_executions=4367,graph_frontiers=3200,middle_TASK_types=dict(waittypes),gap_ns=stat(gap),rope_ns=stat(rd),fixed_duration_ideal_shift_ns=stat(toy),real_gain=None))


if __name__=='__main__':main()
