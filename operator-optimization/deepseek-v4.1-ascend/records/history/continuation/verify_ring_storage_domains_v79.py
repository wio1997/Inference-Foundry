"""Verify state accounting/coverage claims;not all-goal completion or native tests."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import hashlib,zipfile
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read


def main():
    cp=read(O/'REVIEW-CHECKPOINT-v79.json');assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    mf=read(O/'checkpoint-manifest-v79.json')['artifacts']
    for r in mf:
        b=Path(r['path']).read_bytes();assert len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256'],r['path']
    masters=read(O/'candidate-record-index-v79.json')['model_candidates'];assert len(masters)==35
    for r in masters:assert hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256']
    src=read(O/'ring-storage-source-contract-v79.json');assert len(src['source_files'])==7 and not src['new_remote_source_read']
    archive=src['original_archive'];assert hashlib.sha256(Path(archive['path']).read_bytes()).hexdigest()==archive['sha256']
    with zipfile.ZipFile(archive['path'])as z:
        for r in src['source_files']:
            b=Path(r['path']).read_bytes();assert hashlib.sha256(b).hexdigest()==r['sha256']
            if 'original_archive_member'in r:assert b==z.read(r['original_archive_member'])
    assert src['owner_topology']['C2_ring_layers']==[2,8,14]and src['owner_topology']['number_of_ring_objects']==3 and src['owner_topology']['source_layers_with_ratio2_in_backbone']==18
    assert src['global_pool_unit']['scheduled_B_is_not_resident_owner_bound']and src['global_pool_unit']['actual_resident_owner_upper_bound']is None
    assert len(src['observed_state_views'])==6 and sum(r['executions']for r in src['observed_state_views'])==144
    assert src['direct_state_logical_blocks']==17306 and src['graph_pool_update_records']==480 and not src['graph_state_N_or_actual_pointer_attested']
    c=read(O/'ring-storage-capacity-models-v79.json');assert c['pool_divisor_bytes']==540928
    assert[p['capacity']for p in c['tuple_pages']]==[131072]*3+[147712]
    assert sum(p['capacity']for p in c['tuple_pages'])==540928
    assert all(p['capacity']==p['capacity_after_even_compaction']==p['capacity_after_state_detachment']for p in c['tuple_pages'])
    assert c['ring32_bytes_per_owner_all_three_layers']==393216 and c['ring16_bytes_per_owner_all_three_layers']==196608
    assert c['direct_typed_logical_ring_view_bytes_per_object']==17306*32*1024*4==2268332032
    assert not c['three_logical_ring_views_are_additive_independent_allocations']
    assert c['standalone_even_compaction_num_blocks_change']==c['standalone_even_compaction_pool_divisor_change']==0
    models=c['private_bank_models'];assert len(models)==68
    # Check all breakpoints of floor((B-D)/C),not an implementation-mirroring test.
    for r in models:
        U,S,rows,z=r['bank_owner_capacity_U'],r['current_resident_owners_S'],r['ring_rows'],r['reserved_invalid_slot']
        D=(U+z)*3*rows*1024*4;C=r['normal_pool_divisor_bytes'];assert D==r['state_payload_bytes']
        assert 0<=S<=U and r['max_net_free_global_ID_change']<=S
        remainders={0,C-1,D%C,max(0,D%C-1)}
        values=[]
        for rem in remainders:
            B=1000*C+rem;n0=B//C;n1=(B-D)//C
            values.append(S-(n0-n1))
        assert min(values)==r['min_net_free_global_ID_change']and max(values)==r['max_net_free_global_ID_change']
    def row(S,slots):return next(r for r in models if r['bank_owner_capacity_U']==4 and r['current_resident_owners_S']==S and r['ring_rows']==slots and r['reserved_invalid_slot']==0)
    assert[(row(s,k)['min_net_free_global_ID_change'],row(s,k)['max_net_free_global_ID_change'])for s,k in[(2,32),(4,32),(2,16),(4,16)]]==[(-1,0),(1,2),(0,1),(2,3)]
    assert c['no_projection_GEMM_FLOP_reduction']and c['state_allocation_size_is_not_per_call_traffic']and c['actual_native_memory_bandwidth_or_operator_speedup']is None
    p=read(O/'ring-state-layout-and-lifetime-proof-v79.json');slots=p['compact_slot_table'];assert len(slots)==32
    assert {s['new_physical_row']for s in slots}==set(range(32))
    assert {s['new_physical_row']for s in slots if s['owner_half']==0}.isdisjoint({s['new_physical_row']for s in slots if s['owner_half']==1})
    assert all(s['new_physical_row']==16*s['owner_half']+s['old_even_row']//2 for s in slots)
    assert len(p['ABA_counterexample'])==3 and len(p['required_lifetime_contract'])==5 and len(p['variants'])==5
    assert p['alternatives_not_additive']and not p['actual_native_or_performance_validation']and not p['whole_candidate_closed']
    x=read(O/'M22-adjacent-operator-boundary-crosswalk-v79.json');assert len(x['records'])==x['record_count']==900
    assert x['source_family_counts']=={'M22-S06':168,'M22-S07':149,'M22-S08':323,'M22-S09':260}
    prior=read(O/x['source_inventory'])['static_branch_sites'];priorby={r['id']:r for r in prior}
    assert hashlib.sha256((O/x['source_inventory']).read_bytes()).hexdigest()==x['source_inventory_sha256']
    assert len({r['site_id']for r in x['records']})==900
    for r in x['records']:
        a=priorby[r['site_id']];assert r['original_review_status']==a['review_status']=='UNREVIEWED'
        assert r['source_sha256']==a['source_sha256']==x['worker_snapshot_sha256']and r['condition']==a['condition']
        assert not r['new_native_operator_or_conditional_branch_proof']
    assert x['new_guard_dataflows_claimed']==0 and x['exact_prior_local_guard_dataflows_preserved']==110 and x['remaining_unrouted_static_navigation_sites']==91
    before={r['id']:r for r in read(O/'module-phase-branch-matrix-v78.json')['modules']};after={r['id']:r for r in read(O/'module-phase-branch-matrix-v79.json')['modules']}
    inv=read(O/'ring-storage-branch-inventory-v79.json');mods={'M02','M08','M21','M22','M24'}
    assert sum(len(m['branches'])for m in after.values())==342 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in after.values())==24
    assert len(after['M02']['branches'])==12 and len(after['M08']['branches'])==13 and after['M08']['branches'][-1]['id']=='M08-OP-13'
    assert set(inv['old_M02_record_to_new_families'])==set(map(str,range(1,12)))and len(inv['extended_existing_families'])==11
    for mid in before:
        if mid not in mods:assert before[mid]==after[mid]
        else:
            assert read(Path(after[mid]['legacy_branch_history']))['module']==before[mid]
            assert after[mid]['branches']==inv['modules'][mid]
            if mid!='M02':
                assert[r['id']for r in before[mid]['branches']]==[r['id']for r in after[mid]['branches'][:len(before[mid]['branches'])]]
                for a,b in zip(before[mid]['branches'],after[mid]['branches']):
                    assert b['evidence'][:len(a['evidence'])]==a['evidence']
                    if a['id']not in inv['extended_existing_families']:assert a==b
    oldrel=read(O/'candidate-relation-clauses-v78.json');rel=read(O/'candidate-relation-clauses-v79.json')
    assert rel['clauses'][:45]==oldrel['clauses']and len(rel['clauses'])==rel['clause_count']==46
    assert rel['clauses'][-1]['id']=='REL79-01'and(rel['clauses'][-1]['a'],rel['clauses'][-1]['b'])==('F31','F34')
    assert len(rel['higher_order_groups'])==10
    for a,b in zip(oldrel['higher_order_groups'],rel['higher_order_groups']):
        if a['id']!='HG73-C2':assert a==b
        else:assert b['members']==a['members']+['F31']and not b['state_representation_refinement_v79']['all_combinations_analyzed']
    cg=read(O/'candidate-relation-coverage-v79.json');nodes={n['id']for n in cg['nodes']}
    assert len(nodes)==35 and len(cg['pairs'])==595 and cg['scoped_conditional_pairs']==46 and cg['remaining_pair_relations']==549
    assert {(r['a'],r['b'])for r in cg['pairs']}==set(combinations(sorted(nodes),2))and all(not r['whole_pair_all_variants_closed']for r in cg['pairs'])
    cov=read(O/'operator-analysis-coverage-ledger-v79.json');assert cov['regimes']==read(O/'operator-analysis-coverage-ledger-v78.json')['regimes']
    rv=[r for r in cov['regimes']if r['state'].startswith('SCOPED_')];pd=[r for r in cov['regimes']if r['state'].startswith('REGIME_')]
    assert(len(rv),sum(r['executions']for r in rv))==(2073,285598)and(len(pd),sum(r['executions']for r in pd))==(9,1352)
    assert cov['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    for q in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:assert read(q)['latest_profile_first_review']==cp
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    runs=read(O/'task-run-registration-v79.json')['runs'];assert len(runs)==3 and all(r['status']=='pass'for r in runs)
    print(dict(status='PROGRESS',manifest_artifacts_verified=len(mf),source_files=7,source_ring_objects=3,common_IDs_per_resident_owner=1,
        normal_divisor=540928,bank_models=68,layout_slots=32,M22navigation_routed=900,new_M22_guard_proofs=0,prior_M22guard_proofs=110,
        current_records=342,semantic_modules=24,other21modules_identical=True,relations=46,central_pending=549,higher_order_groups=10,
        profile_reviewed=[2073,285598],profile_pending=[9,1352],whole_modules_closed=0,overall_complete=False))


if __name__=='__main__':main()
