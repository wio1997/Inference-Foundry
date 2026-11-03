"""Audit this bounded ownership/cost refinement; not a whole-goal completion test."""
from collections import Counter
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N, O, R, W, T as TASK_DIR, L, read


def main():
    cp = read(O / 'REVIEW-CHECKPOINT-v72.json')
    assert cp['status'] == 'PROGRESS' and cp['whole_modules_closed'] == 0 and not cp['overall_complete']
    manifest = read(O / 'checkpoint-manifest-v72.json')['artifacts']
    for entry in manifest:
        b = Path(entry['path']).read_bytes()
        assert len(b) == entry['bytes'] and hashlib.sha256(b).hexdigest() == entry['sha256'], entry['path']
    masters = read(O / 'candidate-record-index-v72.json')['model_candidates']
    assert len(masters) == 35
    for entry in masters:
        assert hashlib.sha256(Path(entry['master_record']).read_bytes()).hexdigest() == entry['master_sha256']
    owners = read(O / 'pad-owner-invocations-v72.json')
    assert owners['owner_API_calls'] == len(owners['invocations']) == 1272
    assert owners['fields'] == {'MC2_HIDDEN':320, 'MC2_ROUTER':320, 'MC2_MASK':320, 'SP_HIDDEN':104, 'SP_IDS':104, 'SP_MASK':104}
    assert owners['native_compute_counts'] == {'Cast':1272, 'MemSet':1272, 'PadV3':1272}
    assert owners['native_noncompute_counts'] == {'KERNEL_MIX_AIV':1272}
    api_keys, task_keys, memset_keys = [], [], []
    for owner in owners['invocations']:
        assert owner['owner_API_table'] == 'PYTORCH_API' and owner['owner_API_name'] == 'aclnnConstantPadNd'
        assert not owner['MemSet_operand_address_or_length_proved'] and owner['owner_join_is_not_temporal_adjacency']
        api_keys.append((owner['database_index'], owner['owner_API_rowid']))
        memset_keys.append((owner['database_index'], owner['MemSet_TASK_rowid']))
        assert owner['native_noncompute_type_counts'] == {'KERNEL_MIX_AIV':1}
        ops = Counter(s['operator'] or s['task_type'] for s in owner['stages'])
        want = {'MemSet':1, 'PadV3':1, 'KERNEL_MIX_AIV':1}
        if owner['field'].endswith('MASK'):
            want['Cast'] = 3
        assert ops == want
        for stage in owner['stages']:
            assert stage['connectionId'] in owner['owner_runtime_connection_ids']
            task_keys.append((owner['database_index'], stage['TASK_rowid']))
    assert len(api_keys) == len(set(api_keys)) == len(memset_keys) == len(set(memset_keys)) == 1272
    assert len(task_keys) == len(set(task_keys)) == 5088
    refinement = read(O / 'pad-helper-regime-refinement-v72.json')
    assert (refinement['regime_count'], refinement['native_execution_count'], refinement['pending_regimes_promoted']) == (7,1272,0)
    assert set(memset_keys) == {tuple(k) for r in refinement['regimes'] for k in r['native_TASK_keys']}
    reconcile = read(O / 'pad-owner-API-namespace-reconciliation-v72.json')
    assert reconcile['all1272_owners_independently_rejoined'] and reconcile['byte_or_operand_gap_not_resolved_by_owner_name']
    costs = read(O / 'pad-fusion-costs-v72.json')
    assert (costs['native_TASK_total'],costs['ComputeInfo_TASK_total'],costs['nonCompute_TASK_total']) == (5088,3816,1272)
    mc2 = costs['MC2']
    assert (mc2['N'],mc2['P'],mc2['H'],mc2['E']) == (800,816,5120,384)
    assert mc2['joint_output_reference_bytes'] == 19031632 and mc2['eliminated_known_BOOL_data_and_output_cast_reference_bytes'] == 3232
    assert mc2['scalar_cast_bytes'] is None and mc2['MemSet_address_bytes_and_overlap'] is None
    assert mc2['not_a_native_three_pad_invocation_join'] and mc2['native_MC2_816_to_800_ABI_change_not_proved']
    for row in mc2['fields']:
        assert row['one_read_full_write_reference_bytes'] == row['input_bytes'] + row['output_bytes']
    assert sum(x['occurrences'] for x in costs['draft']) == 104
    previous_costs = read(O / 'embedding-initialization-fusion-costs-v67.json')['draft_initialization_models']
    for current, previous in zip(costs['draft'], previous_costs):
        assert current['N'] == previous['N'] and current['local_rows'] == current['P']//8
        assert current['baseline_global_Tile_plus_HC4_Pad_value_bytes'] == previous['current_global_Tile_plus_HC_Pad_value_bytes']
        assert current['existing_delayed_Tile_budget_bytes'] == previous['delayed_Tile_interface_budget_bytes']
        assert sum(r['valid_owned_rows'] for r in current['rank_models']) == current['N']
        for a,b in zip(current['rank_models'],previous['rank_models']):
            assert a['joint_local_SP_HC4_value_bytes'] == b['new_direct_HC_init_value_bytes']
            assert a['HC_interface_budget_bytes'] == b['HC_interface_budget_bytes']
            assert a['provided_mask_read_write_reference_bytes'] == a['valid_owned_rows'] + current['local_rows']
    assert costs['draft'][1]['rank_models'][0]['HC_interface_budget_bytes'] == 2672640
    assert costs['draft'][1]['rank_models'][6]['joint_local_SP_HC4_value_bytes'] == 143360
    assert costs['draft'][1]['rank_models'][7]['joint_local_SP_HC4_value_bytes'] == 122880
    theory = costs['theory']
    assert theory['status'] == 'CONDITIONAL_CEILING' and not theory['actual_physical_traffic_bound_proved']
    assert theory['actual_numeric_request_ceiling'] is None and theory['MC2_pad3_reference_us'] == 19031632/1600000
    variants = read(O / 'pad-fusion-variants-v72.json')
    assert variants['count'] == len(variants['variants']) == 9 and variants['new_model_candidates'] == 0
    assert all(not v['candidate_newly_discovered'] and not v['legal_fusion_proved'] and v['measured_speedup'] is None for v in variants['variants'])
    math = read(O / 'pad-value-map-validation-v72.json')
    assert (math['symbolic_row_map_cases'],math['supplied_mask_cases'],math['all_BF16_payload_copy_witnesses']) == (26520,79560,65536)
    assert not math['source_native_or_Torch_tests'] and not math['actual_persistent_bank_reader_protocol_proved']
    source = read(O / 'pad-operator-contract-v72.json')
    assert len(source['source_files']) == 3 and not source['new_remote_source_read']
    for row in source['source_files']:
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['sha256']
    cov = read(O / 'operator-analysis-coverage-ledger-v72.json')
    prev = read(O / 'operator-analysis-coverage-ledger-v71.json')
    assert cov['regimes'] == prev['regimes']
    reviewed = [r for r in cov['regimes'] if r['state'].startswith('SCOPED_')]
    pending = [r for r in cov['regimes'] if r['state'].startswith('REGIME_')]
    assert (len(reviewed),sum(r['executions'] for r in reviewed)) == (2073,285598)
    assert (len(pending),sum(r['executions'] for r in pending)) == (9,1352)
    assert Counter('|'.join(r['module_candidates']) for r in pending) == {'M19':4,'M12':3,'M05|M21':2}
    assert cov['unchanged_global_uncertainty'] == {'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    assert cov['new_pending_regimes_reconciled'] == cov['new_pending_executions_reconciled'] == 0
    before = {m['id']:m for m in read(O / 'module-phase-branch-matrix-v71.json')['modules']}
    after = {m['id']:m for m in read(O / 'module-phase-branch-matrix-v72.json')['modules']}
    inventory = read(O / 'pad-branch-inventory-v72.json')
    for mid in before:
        if mid not in ['M12','M19']:
            assert before[mid] == after[mid]
            continue
        assert after[mid]['branches'] == inventory['modules'][mid]
        assert read(Path(after[mid]['legacy_branch_history']))['module'] == before[mid]
        assert [b['id'] for b in before[mid]['branches']] == [b['id'] for b in after[mid]['branches']]
        for a,b in zip(before[mid]['branches'],after[mid]['branches']):
            assert b['evidence'][:len(a['evidence'])] == a['evidence']
            if a['id'] not in inventory['extended_families']:
                assert a == b
    for path in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        assert read(path)['latest_profile_first_review'] == cp
    assert read(TASK_DIR/'index.json')['status'] == 'ACTIVE' and read(L/'state.json')['status'] == 'EVALUATING'
    runs = read(O / 'task-run-registration-v72.json')['runs']
    assert len(runs) == 3 and all(r['status'] == 'pass' for r in runs)
    assert len(read(O / 'framework-candidate-ledger-v72.json')['candidates']) == 7
    print(dict(manifest_artifacts_verified=len(manifest), candidate_master_hashes_verified=35, owners=1272, native_TASKs=5088, pending_promoted=0, reviewed=[2073,285598], pending=[9,1352], retained_M12_M19_families=[15,13], other_modules_identical=24, checkpoint_status='PROGRESS', whole_modules_closed=0, overall_complete=False))


if __name__ == '__main__':
    main()
