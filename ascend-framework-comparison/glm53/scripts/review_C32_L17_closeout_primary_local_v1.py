"""Independent final closeout acceptance; local saved evidence only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}


def read(path):
    p = ROOT / path
    SOURCES[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_bytes())


def verify(path, expected):
    actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert actual == expected, path
    SOURCES[path] = actual


def diff(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            if key not in a or key not in b:
                result.append((path + (key,), a.get(key), b.get(key)))
            else:
                result.extend(diff(a[key], b[key], path + (key,)))
        return result
    return [] if a == b else [(path, a, b)]


def main():
    index_path = 'results/C32-L17-closeout-integrity-index-v2.json'
    idx = read(index_path)
    indexed_paths = []

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get('path'), str) and isinstance(o.get('sha256'), str):
                verify(o['path'], o['sha256']); indexed_paths.append(o['path'])
            for value in o.values():
                walk(value)
        elif isinstance(o, list):
            for value in o:
                walk(value)

    walk(idx)
    for path, expected in idx['final_document_hashes'].items():
        verify(path, expected)
    for path in [
        'results/C32-L17-native-formal-decision-v1.json',
        'results/C32-L17-formal-primary-independent-review-v1.json',
        'results/model-request-ledger-through-L17-formal-v2.json',
        'results/C32-L17-current-comparison-v1.json',
    ]:
        o = read(path)
        for p, h in o.get('source_evidence_sha256', o.get('source_sha256', {})).items():
            verify(p, h)
    ledger = read('results/model-request-ledger-through-L17-formal-v2.json')
    previous = read(ledger['previous_ledger'])
    t = ledger['totals']; n = ledger['new_qualified_run']
    assert t['backend_HTTP_attempts'] == 6729
    assert t['qualified_successful_backend_inferences'] == 6492
    assert t['backend_attempts_without_qualified_success'] == 173
    assert t['pending_native_backend_inferences'] == 64
    assert t['backend_HTTP_attempts'] == sum(t[k] for k in ['qualified_successful_backend_inferences','backend_attempts_without_qualified_success','pending_native_backend_inferences'])
    assert t['successful_backend_output_tokens'] == sum(t[k] for k in ['all_formal_shaped_D_output_tokens','functional_output_tokens','warm_and_P_control_output_tokens']) == 1432790
    observed_outputs = sum(t[k] for k in ['successful_backend_output_tokens','unqualified_observed_successful_backend_output_tokens','pending_native_successful_output_tokens'])
    assert observed_outputs == 1432918
    assert ledger['new_pending_native_actual'] == previous['new_pending_native_actual']
    assert n['unique_rids_sha256'] == '5e4681056a30235d8141c21a2efc87808f6cf000943f0d1555c464c2d10f08ce'
    all_ids = ledger['unique_rids_all_460']
    assert len(all_ids['rids']) == len(set(all_ids['rids'])) == 460
    assert hashlib.sha256(('\n'.join(sorted(all_ids['rids']))+'\n').encode()).hexdigest() == all_ids['sha256'] == 'acbc23b784e4a17ecd975cf83448bc5d9583d909b04a8765fda048fba63fd066'
    assert n['backend_HTTP_attempts'] == n['qualified_successful_backend_inferences'] == 320
    assert n['successful_output_tokens'] == 76992 and n['formal_successful_output_tokens'] == 76800
    assert n['D_rid_observed'] is False
    corrected_flags = []; document_diffs = {}
    correction_dir = 'runs/20261019T-L17-status-count-source-clarification-zcode-v1'
    for p in ['STATE.json','STATUS_BRIEF.json','diagnostics/model-profile/loops/L17-D-native-eager-live-inputs-v1.json']:
        # The clarification executor saved these three backups by basename.
        old = read(correction_dir+'/before/'+Path(p).name); new = read(p)
        changes = diff(old, new); document_diffs[p] = ['.'.join(path) for path, _, _ in changes]
        for path, a, b in changes:
            key = path[-1]
            if key.endswith('planned_counted_as_actual'):
                assert a is True and b is False
                corrected_flags.append((p, '.'.join(path)))
            elif key == 'actual_counts_verified_from_terminal_report':
                assert a is None and b is True
            elif key == 'count_source_note':
                assert a is None and 'Actual320' in b and 'planned320 was not used as evidence' in b
            elif key == 'revision':
                assert a == 'v12' and b == 'v13'
            elif key == 'documentation_clarification_only':
                assert a is None and b is True
            else:
                raise AssertionError(('unexpected semantic change',p,path,a,b))
        if p in ['STATE.json','STATUS_BRIEF.json']:
            assert new['status'] == 'L17_FORMAL_V2_VALID_MEASUREMENT_DECODE_TARGET_MISSED'
            assert new['objective_complete'] is False and new['current_blocker'] is None
            assert new['active_execution']['active'] is False
            assert new['active_execution']['actual_backend_HTTP_attempts'] == 320
            assert new['model_http_attempts_by_this_task'] == 6729 and new['successful_model_inferences'] == 6492
            assert new['latest_valid_measurement'] == 'results/sglang-C32-L17-native-formal-measurement-v2.json'
            assert new['performance_gain_confirmed'] is False
            verify(new['model_request_ledger'],new['model_request_ledger_sha256'])
    assert len(corrected_flags) == 9
    first_before = 'runs/20261019T-L17-formal-v2-closeout-zcode-v1/before/'
    for p in ['HANDOFF.md','RECOVERY_LATEST.md','PERFORMANCE_CHECKPOINT.md']:
        assert (ROOT/p).read_bytes().endswith((ROOT/first_before/p).read_bytes()), p
    decision = read('results/C32-L17-native-formal-decision-v1.json')
    assert decision['measurement_valid'] is True and decision['optimization_accepted'] is False
    assert decision['rollback_executed'] is False and decision['aggregate_SLA_pass'] is False
    assert decision['global_optimization_objective_complete'] is False
    read('runs/20261019T-L17-formal-v2-closeout-zcode-v1/receipt.json')
    read(correction_dir+'/receipt.json')
    SOURCES[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {
        'project_id':'glm53',
        'decision':'ACCEPTED_FINAL_L17_CLOSEOUT_VALID_MEASUREMENT_TARGET_MISSED',
        'measurement_valid':True,'optimization_accepted':False,'SLA_pass':False,
        'L17_loop_complete':True,'global_objective_complete':False,
        'indexed_path_hash_checks':len(indexed_paths),
        'all_index_and_current_source_hashes_verified':True,
        'independently_recounted_new_backend_RIDs':320,
        'independently_recounted_all_local_RIDs':460,
        'all_global_attempt_and_qualified_output_partitions_pass':True,
        'observed_output_partition':{'qualified':1432790,'old_unqualified_observed':64,'pending_native':64,'total':observed_outputs},
        'original_v1_native_pending64_unchanged':True,
        'documentation_count_source_flags_clarified':corrected_flags,
        'documentation_semantic_diff_paths':document_diffs,
        'historical_Markdown_full_bytes_retained':True,
        'runtime':'L17 NoCopy D candidate remains; no rollback; P and Meta retained.',
        'model_requests_SDK_init_profiles_services_Git_added_by_final_local_acceptance':0,
        'source_evidence_sha256':SOURCES,
    }
    p = ROOT/'results/C32-L17-closeout-primary-acceptance-v1.json'
    assert not p.exists()
    p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'decision':result['decision'],'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_hashes':len(SOURCES),'corrected_flags':len(corrected_flags)}))


if __name__ == '__main__':
    main()
