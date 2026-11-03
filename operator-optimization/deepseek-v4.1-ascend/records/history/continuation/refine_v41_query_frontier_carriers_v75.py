"""Use native runtime task carriers for a conservative join placement bracket."""
from collections import defaultdict,Counter
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_projection_ceiling_v61 import stat


def main():
    before=O/'V41-QB-query-RoPE-frontier-before-carrier-refinement-v75.json'
    assert not before.exists()
    path=O/'V41-QB-query-RoPE-frontier-v75.json';p=read(path);write(before,p)
    bydb=defaultdict(list)
    for row in p['records']:bydb[row['database_index']].append(row)
    carriers=[];gaps=[];budgets=[];unmatched=[]
    for db,rows in sorted(bydb.items()):
        e=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        tasks={t['task_record_rowid']:t for t in e['TASK_records']};by_identity=defaultdict(list)
        for t in e['TASK_records']:
            if t['task_type_name']=='AI_CORE':by_identity[(t['globalPid'],t['streamId'],t['taskId'])].append(t)
        for row in rows:
            q=tasks[row['QB_TASK_rowid']];r=tasks[row['query_RoPE_TASK_rowid']]
            possible=[t for t in by_identity[(q['globalPid'],q['streamId'],q['taskId'])]if t['startNs']<=q['startNs']and q['endNs']<=t['endNs']<=r['startNs']]
            row['previous_ComputeInfo_end_toy_budget_ns']=row.pop('fixed_duration_ideal_shift_budget_ns')
            if len(possible)!=1:
                row.update(runtime_QB_carrier_proved=False,fixed_duration_ideal_shift_budget_ns=None)
                unmatched.append(dict(database_index=db,QB_TASK_rowid=q['task_record_rowid'],candidate_count=len(possible)));continue
            t=possible[0];gap=r['startNs']-t['endNs'];budget=min(gap,row['query_RoPE_duration_ns'])
            row.update(runtime_QB_carrier_proved=True,runtime_QB_carrier_TASK_rowid=t['task_record_rowid'],runtime_QB_carrier_startNs=t['startNs'],runtime_QB_carrier_endNs=t['endNs'],runtime_QB_carrier_duration_ns=t['endNs']-t['startNs'],carrier_extra_after_ComputeInfo_end_ns=t['endNs']-q['endNs'],conservative_carrier_end_to_RoPE_start_ns=gap,fixed_duration_ideal_shift_budget_ns=budget,carrier_and_ComputeInfo_are_not_independent_work=True)
            carriers.append((db,t['task_record_rowid']));gaps.append(gap);budgets.append(budget)
    assert len(carriers)+len(unmatched)==3200 and len(carriers)==len(set(carriers))
    p['previous_ComputeInfo_end_gap_stat']=p.pop('gap_ns');p['previous_ComputeInfo_end_ideal_shift_stat']=p.pop('fixed_duration_ideal_shift_budget_ns')
    p.update(runtime_QB_carriers_bound=len(carriers),runtime_QB_carriers_unbound=unmatched,conservative_carrier_gap_ns=stat(gaps),fixed_duration_ideal_shift_budget_ns=stat(budgets),carrier_identity_rule='Same native globalPid/streamId/taskId,AI_CORE carrier encloses QB ComputeInfo interval and completes before query RoPE. No extra operator is inferred and its duration is not added to QB ComputeInfo.',earliest_placement_reference='After the matched runtime QB carrier completes;ComputeInfo end alone was an earlier annotation boundary. Fixed-duration placement example only,not proof of actual new-kernel readiness/critical-path gain.',fixed_native_rope_fits_each_matched_carrier_gap=all(r['conservative_carrier_end_to_RoPE_start_ns']>=r['query_RoPE_duration_ns']for r in p['records']if r['runtime_QB_carrier_proved']))
    write(path,p)
    clauses=read(O/'candidate-relation-clauses-v75.json');assert len(clauses['clauses'])==43
    assert not any({r['a'],r['b']}=={'F07','F08'}for r in clauses['clauses'])
    clauses['clauses'].append(dict(id='REL75-01',a='F07',b='F08',relation='QUERY_ROPE_BEFORE_KV_JOIN;SHARED_VECTOR_RESOURCE_GATE',operator_claim='Current V4.1 joins the KV stream after QB and only then rotates the query. Study moving pair-local query RoPE before that join,or fusing it into QB,while retaining KV publication/join before actual attention/compressor/cache consumers. Query C/S generation and all aliases/readiness must be preserved;Vector/memory contention can erase a toy overlap budget.',evidence=['V41-query-operator-path-contract-v75.json','V41-QB-native-source-binding-v75.json','V41-QB-query-RoPE-frontier-v75.json'],applicable_domain='Positive-query V4.1 target/draft branch. No-query ranks still execute required KV/cache/global collective work. No legacy head512 norm barrier in this query chain.',conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,new_main_candidate=False,new_schedule_subvariant_recorded=True))
    clauses['clause_count']=clauses['selected_pair_count']=44
    clauses['higher_order_groups'].append(dict(id='HG75-Q-KV-JOIN',members=['F06','F07','F08','F13'],contract='Keep local query vs global KV row domains and distinct producer outputs;query RoPE may precede the existing control join only with C/S/alias/readiness proof. Required KV cache/attention frontier and finite shared Cube/Vector/memory capacity remain.',scope_state='SCOPED_COMPOSITION_CONSTRAINTS_RECORDED;ALL_VARIANTS_AND_HIGHER_ORDER_CLOSURE_OPEN',all_subsets_analyzed=False,legal_joint_fusion_proved=False))
    write(O/'candidate-relation-clauses-v75.json',clauses)
    graph=read(O/'candidate-relation-coverage-v75.json')
    row=next(r for r in graph['pairs']if(r['a'],r['b'])==('F07','F08'));assert not row['clauses']
    row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=['REL75-01'])
    graph.update(scoped_conditional_pairs=44,remaining_pair_relations=551,higher_order_groups=9,scope='44selected pair clauses/9groups/all35nodes;3old clauses/1group corrected for actual V4.1 entry and one query-before-KV-join relation added.551centralpairreconciliations and full-model/all-variant gates remain open.')
    write(O/'candidate-relation-coverage-v75.json',graph)
    write(O/'V41-query-RoPE-placement-variant-v75.json',dict(candidate='F07',related_candidates=['F06','F08','F13'],variant_id='F07-QUERY-ROPE-BEFORE-KV-JOIN',current_source_sequence=['query C/S prepared','QB matmul','KV aux-stream join','query RoPE','index/compression/attention consumers'],candidate_sequence=['query C/S prepared','QB completion','query RoPE','KV aux-stream join before actual KV-dependent consumers'],related_existing_variant='QB-RoPE epilogue fusion;this scheduling subvariant can be evaluated separately from replacing the GEMM kernel.',new_main_candidate=False,conditions=['Positive query rows and unchanged source BF16/RINT values','Fresh/live q versus KV/cache write alias and C/S/table-generation proof','All other query observers see identical readiness/publication','Same global collective participation/order;no-query cache work retained','Price Vector/memory sharing with KV norm/RoPE/scatter and actual queue/graph-capture protocol'],implementation=False,actual_measured_speedup=None,overall_complete=False))
    print(dict(runtime_carriers_bound=len(carriers),unbound=len(unmatched),conservative_gap_ns=stat(gaps),fixed_duration_ideal_shift_ns=stat(budgets),all_matched_gaps_fit_native_rope=p['fixed_native_rope_fits_each_matched_carrier_gap'],relation_pairs=44,central_pending=551,higher_order_groups=9,real_gain=None))


if __name__=='__main__':main()
