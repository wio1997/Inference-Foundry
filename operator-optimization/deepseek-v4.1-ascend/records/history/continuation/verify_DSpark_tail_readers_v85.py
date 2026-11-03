"""Verify saved profile identity, explicit operator domains and honest prior-proof scope."""
from collections import defaultdict,Counter
from pathlib import Path
import hashlib,json,sys,zipfile,struct,math,re
from finalize_origin_coverage_v9 import N,O,R,W,T as TASK_DIR,L,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import dimensions

def receipt(item):
    b=Path(item['path']).read_bytes()
    assert hashlib.sha256(b).hexdigest()==item['sha256'],item['path']
    if 'bytes'in item:assert len(b)==item['bytes']

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def bf16(x):
    bits=struct.unpack('I',struct.pack('f',x))[0]
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('f',struct.pack('I',bits))[0]

def main(analysis_only=False):
    src=read(O/'DSpark-tail-source-index-v85.json');assert len(src['source_files'])==8
    receipt(src['original_archive']);receipt(src['saved_runtime_script'])
    with zipfile.ZipFile(src['original_archive']['path'])as archive:
        for row in src['source_files']:
            receipt(row);assert Path(row['path']).read_bytes()==archive.read(row['archive_member'])
            lines=Path(row['path']).read_text().splitlines()
            for clause in row['clauses']:assert lines[clause['first']-1:clause['last']]==clause['source']
    references=src['symbol_reference_inventory'];assert len(references)==10 and len({r['path']for r in references})==4
    assert src['saved_runtime_script']['selected_arguments']==dict(spec_method='dspark',num_speculative_tokens=5,max_num_batched_tokens=8192,TP=8)
    rec=read(O/'DSpark-tail-prior-evidence-reconciliation-v85.json')
    for row in rec['immutable_prior_input_receipts']:receipt(row)
    old=read(O/'F26-consumer-contract-v11.json');installed=json.loads(read(N/'installed-fullhc-consumers166-v11.json')['stdout'])['result']
    assert len(installed['matches'])==90 and len(installed['matched_files'])==26
    pins={r['archive_member']:r['sha256']for r in src['source_files']}
    assert all(pins[k]==v for k,v in old['source_sha256'].items())
    assert rec['prior_answer_first_established_version']==11 and rec['old_conditional_observer_guards_were_not_disproved']
    closure=read(O/'DSpark-tail-reader-closure-v85.json')
    assert closure['prior_direct_drafter_question_established_v11'] and not closure['all_snapshot_observers_closed']
    assert not closure['new_full_dead_buffer_proof'] and not closure['native_aliases_or_loaded_binary_observers_proved']
    assert closure['BF16_full_HC_buffer_requested_bytes']==8192*4*5120*2==335544320
    assert closure['other_paths_to_preserve']==old['other_paths_to_preserve']
    assert len(closure['eight_consumer_guard_cases'])==8
    for row in closure['eight_consumer_guard_cases']:
        rebind=row['buffer_exists']and row['method_is_mtp']
        assert row['Ascend_getter_result_rebinds_hidden']==rebind
        assert row['raw_values_reach_drafter_via_this_argument']==(rebind and not row['aux_output_path'])
        assert not row['runtime_hit_claimed']
    variants=read(O/'DSpark-tail-reorganization-variants-v85.json')
    assert len(variants['variants'])==5 and not variants['source_observer_question_closed']
    assert all(not v['all_snapshot_observers_closed']and len(v['conditions'])>=8 for v in variants['variants'])
    assert 'v11 F26a'in variants['variants'][0]['origin_and_novelty']and 'v11 F26b'in variants['variants'][1]['origin_and_novelty']
    profile=read(O/'DSpark-tail-native-boundaries-v85.json')
    assert(profile['graph_count'],profile['root_bound_graphs'],profile['phase_unbound_graphs'])==(80,64,16)
    assert profile['all_target_tail_groups']==104 and profile['target_tail_N_counts']=={'24':80,'6400':8,'6528':8,'6672':8}
    assert profile['direct_raw_HC_AG_without_ComputeInfo']==24
    oldops=read(O/'communication-native-operator-population-v66.json')['operations']
    idx={(r['database_index'],r['COMM_OP_rowid']):r for r in oldops}
    typed=profile['raw_HC_and_premix_operations'];assert len(typed)==208
    counts=Counter()
    for row in typed:
        original=idx[(row['database_index'],row['COMM_OP_rowid'])]
        assert all(row[k]==v for k,v in original.items())
        raw=row['semantic_role']=='MTP_FULL_HC_STATE_ALLGATHER';counts[raw]+=1
        assert row['rankSize']==8 and row['count']==row['local_rows_from_actual_count']*(20480 if raw else 4)
        assert row['physical_gather_rows']==8*row['local_rows_from_actual_count']
        assert row['dtype']==('BFP16'if raw else 'FP32')
    assert counts=={True:104,False:104}
    assignments=defaultdict(dict)
    for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
        assignments[a[0]][a[1]]=a
    bydb=defaultdict(list)
    for row in profile['graph_boundaries']:bydb[row['database_index']].append(row)
    assert len({(r['database_index'],r['graph_ordinal'])for r in profile['graph_boundaries']})==80
    for db,rows in bydb.items():
        native=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json')
        tasks={r['task_record_rowid']:r for r in native['TASK_records']};defs={r['globalTaskId']:r for r in native['COMPUTE_definitions']}
        for row in rows:
            group=[tasks[tid]for tid,a in assignments[db].items()if a[2]=='GRAPH'and a[8]==row['graph_ordinal']and a[9]==39]
            posts=sorted([t for t in group if defs[t['globalTaskId']]['opType_text']=='HcPost'],key=lambda t:t['startNs'])
            merges=[t for t in group if defs[t['globalTaskId']]['opType_text']=='ForeachAddList']
            assert len(posts)==2 and len(merges)==1
            assert row['last_FFN_HcPost_TASK']==posts[-1]['task_record_rowid']and row['last_shared_routed_merge_TASK']==merges[0]['task_record_rowid']
            copy=tasks[row['snapshot_copy_TASK']];definition=defs[copy['globalTaskId']]
            assert definition['opType_text']=='TensorMove'and definition['outputDataTypes_text']=='DT_BF16'
            assert dimensions(definition['outputShapes_text'])[0]==[24,20480]
            assert row['snapshot_copy_duration_ns']==copy['endNs']-copy['startNs']
            assert [t['operator']for t in row['tail_TASKs']]==['TensorMove','Cast','Mul','ReduceSum','Cast','RmsNorm']
            for t in row['tail_TASKs']:
                assert tasks[t['TASK_rowid']]['endNs']-tasks[t['TASK_rowid']]['startNs']==t['duration_ns']
    work=read(O/'DSpark-tail-work-communication-models-v85.json');assert len(work['models'])==4
    ceilings=read(O/'DSpark-tail-ceiling-domains-v85.json');assert len(ceilings['models'])==4
    for row,bound in zip(work['models'],ceilings['models']):
        m,n,h,H,p=row['M'],row['N'],row['h'],row['H'],row['P'];assert n==p*m and p==8 and h==4 and H==5120
        assert row['snapshot_copy_logical_read_write_bytes']==4*n*h*H
        raw=2*(p-1)*m*h*H;mix=4*(p-1)*m*h;new=2*(p-1)*m*H
        assert(row['ring_reference_raw_HC_send_bytes'],row['ring_reference_premix_send_bytes'],row['ring_reference_collapsed_or_normalized_send_bytes'])==(raw,mix,new)
        assert row['current_lastPost_plus_globalTail_FP32_ordinary_ops']==36*m*H+(11*H+1)*n
        assert row['existing_None_localCollapse_globalNorm_FP32_ordinary_ops']==43*m*H+(4*H+1)*n
        assert row['localCollapse_localNorm_FP32_ordinary_ops']==(47*H+1)*m
        assert bound['fused_local_collapse_norm_boundary_value_payload_bytes']==2*m*(h+1)*H+4*m*h+2*H
        assert bound['explicit_rstd_reference_total_bytes']==row['fused_localCollapseNorm_noncollective_value_reference_bytes']
        assert bound['actual_numeric_time_bound_us']is None and (raw+mix)/new==4.0015625
    witness=read(O/'DSpark-tail-rounding-witnesses-v85.json');post=witness['post_BF16_before_collapse']
    assert [bf16(v)for v in post['unrounded_post_values']]==post['rounded_post_values']==[258,-255,1,1]
    assert bf16(sum(post['rounded_post_values']))==post['required_Y']==5
    assert bf16(sum(post['unrounded_post_values']))==post['bypass_round_Y']==4
    norm=witness['collapse_BF16_before_RMS'];assert norm['required_output']==[1,.99609375]and norm['bypass_output']==[1,1]
    transfer=read(O/'DSpark-tail-transfer-credit-v85.json')
    for row in transfer['conditional_private_interface_models']:
        m=row['M'];assert row['merge_Post_localC_private_interface_budget_bytes']==92*m*5120
        assert row['merge_Post_localC_localNorm_private_interface_budget_bytes']==96*m*5120
        assert row['above_plus_VectorDownAdd_private_S_budget_bytes']==100*m*5120
        assert row['previous_input_residual_and_coefficients_retained']and row['final_normalized_or_collapsed_local_write_to_AG_is_retained']
    budget=read(O/'tail-shared-memory-budget-v85.json');assert len(budget['models'])==6
    for item in budget['input_receipts']:receipt(item)
    for row in budget['models']:
        c=row['pool_divisor_bytes'];assert c==540928
        delta=row['assumed_usable_budget_credit_bytes']-row['extra_private_ring_bytes']-row['extra_WKV_cache_bytes']-row['other_extra_metadata_workspace_bytes_assumed']
        vals=[row['released_old_common_ID_reservations']+(r+delta)//c for r in[0,c-1]]
        assert vals==[row['min_net_free_global_ID_change'],row['max_net_free_global_ID_change']]
        assert not row['actual_device_budget_or_capacity_gain_proved']
    assert budget['correctly_shared_remainder_joint_range']==[597,598]
    additions=read(O/'DSpark-tail-composition-clauses-v85.json');mem=read(O/'tail-shared-memory-relations-v85.json')
    new=additions['clauses']+mem['clauses'];assert len(new)==11
    previous=read(O/'candidate-relation-clauses-v84.json')
    assert len({tuple(sorted([c['a'],c['b']]))for c in previous['clauses']+new})==95
    assert len(additions['higher_order_groups']+mem['higher_order_groups'])==3
    assert all(not c['all_snapshot_observers_closed']and c['remaining_observer_guards']for c in new)
    if analysis_only:
        assert not(O/'REVIEW-CHECKPOINT-v85.json').exists()
        write(O/'DSpark-tail-analysis-preflight-v85.json',dict(status='PASS',prior_direct_answer_version=11,
          all_observers_closed=False,source_files=8,profile=[104,208,80,64,16],typed_shape_models=4,
          conditional_variants=5,shared_budget_models=6,new_pairs=11,new_groups=3,new_device_execution=False,overall_complete=False))
        print(dict(preflight='PASS',prior_answer=11,all_observers_closed=False,profile=[104,208,80,64,16],new_pairs=11));return
    manifest=read(O/'checkpoint-manifest-v85.json')
    for item in manifest['artifacts']:receipt(item)
    relations=read(O/'candidate-relation-clauses-v85.json');cov=read(O/'candidate-relation-coverage-v85.json')
    assert len(relations['clauses'])==95 and len(relations['higher_order_groups'])==19
    assert(cov['scoped_conditional_pairs'],cov['remaining_pair_relations'],cov['higher_order_groups'])==(95,500,19)
    for oldclause in previous['clauses']:
        now=next(c for c in relations['clauses']if c['id']==oldclause['id']).copy();now.pop('DSpark_reader_refinement_v85',None);assert now==oldclause
    nav=read(O/'candidate-pair-navigation-v85.json');assert len(nav['pairs'])==595 and sum(bool(p['conditional_clause_ids'])for p in nav['pairs'])==95
    for row in read(O/'candidate-operator-footprints-v85.json')['candidates']:
        p=O/(row['id']+'-master-before-v85.json')
        if not p.exists():p=R/'candidates'/(row['id']+'.json')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['captured_master_sha256']
    before={m['id']:m for m in read(O/'module-phase-branch-matrix-v84.json')['modules']}
    after={m['id']:m for m in read(O/'module-phase-branch-matrix-v85.json')['modules']}
    changed={'M02','M06','M13','M14','M15','M18','M19','M22','M23','M24'}
    assert len(after)==26 and sum(len(m['branches'])for m in after.values())==345
    assert sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in after.values())==24
    for mid,mod in after.items():
        assert not mod.get('closed',False)
        oldids=[b.get('id')for b in before[mid]['branches']];assert [b.get('id')for b in mod['branches']][:len(oldids)]==oldids
        if mid not in changed:assert mod['branches']==before[mid]['branches']
    assert after['M15']['branches'][-1]['id']=='M15-OP-09'
    candidates=read(O/'candidate-profile-ledger-v85.json')['candidates'];assert len(candidates)==35
    assert all(not c['legal_fusion_proved']and not c['candidate_closed']for c in candidates)
    assert len(read(O/'framework-candidate-ledger-v85.json')['candidates'])==7
    oldops=read(O/'operator-analysis-coverage-ledger-v84.json');ops=read(O/'operator-analysis-coverage-ledger-v85.json')
    assert ops['regimes']==oldops['regimes']and ops['unchanged_global_uncertainty']==oldops['unchanged_global_uncertainty']
    assert(ops['scoped_conditional_review_regimes'],ops['scoped_conditional_review_executions'])==(2073,285598)
    assert(ops['remaining_regime_reconciliation_count'],ops['remaining_execution_reconciliation_count'])==(9,1352)
    cp=read(O/'REVIEW-CHECKPOINT-v85.json');assert cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0
    summary=read(O/'DSpark-tail-summary-v85.json');assert summary['direct_drafter_content_answer_first_version']==11 and not summary['all_snapshot_observers_closed']
    for p in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        assert read(p)['latest_profile_first_review']['candidate_ledger']==str(O/'candidate-profile-ledger-v85.json')
    assert read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING'
    print(dict(checkpoint=85,status='PASS',manifest_artifacts=len(manifest['artifacts']),prior_answer=11,all_observers_closed=False,
      profile=[104,208,80,64,16],pair_coverage=[95,19,500],branches=[345,24],profile_review=[2073,285598],pending=[9,1352],whole_modules_closed=0,overall_complete=False))

if __name__=='__main__':main('--analysis-only'in sys.argv)
