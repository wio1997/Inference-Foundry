"""Audit cached evidence capabilities without connecting, importing the model, or rereading SDK/model source."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import packet


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v48.json').exists()
    for a in read(O / 'checkpoint-manifest-v47.json')['artifacts']:
        b = Path(a['path']).read_bytes()
        assert len(b) == a['bytes'] and hashlib.sha256(b).hexdigest() == a['sha256'], a['path']
    labels = [
        'host-profile-inventory166-v12-v2', 'queue-schema166-v15',
        'native-execution-identity166-v14', 'graph-sidecar-candidates166-v14',
        'prepare-field-origins166-v12', 'event-resource-schema166-v13',
        'saved-events-resources166-v13-v2', 'CPU-surfaces166-v17',
        'saved-profile-surfaces166-v14', 'readonly-core-attributes166-v12-v2',
        'profiler-metric-formulas166-v15', 'profile-provenance166-v17-v2',
        'complete-PMU166-v18', 'PMU-reviewed166-v18',
        'compute-attrs166-v22',
    ]
    data = {label: packet(label, False) for label in labels}
    # Do not reserialize a displayed subset and compare it with a remote full-result SHA.
    # The immutable local receipt is pinned separately against its historical manifest.
    manifests = [read(p) for p in sorted(O.glob('checkpoint-manifest-v*.json'))]
    inputs = []
    for label in labels:
        path = N / (label + '.json')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pins = [a for m in manifests for a in m.get('artifacts', []) if a['path'] == str(path)]
        assert pins and any(a['sha256'] == digest and a['bytes'] == path.stat().st_size for a in pins), label
        x = data[label]
        inputs.append(dict(receipt=str(path), receipt_sha256=digest,
                           historical_manifest_match=True,
                           remote_full_result=x['full_result'], remote_full_result_sha256=x['result_sha256'],
                           remote_full_payload_rehashed_locally=False))

    utilities = []
    for name in ['inventory_host_profile_v12_166_v2.py', 'query_complete_PMU_fields_v18_166.py']:
        path = N / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        pins = [a for m in manifests for a in m.get('artifacts', []) if a['path'] == str(path)]
        if pins:
            assert any(a['sha256'] == digest for a in pins), name
        utilities.append(dict(path=str(path), sha256=digest, historical_manifest_match=bool(pins),
                              kind='CURRENT_LOCAL_ANALYSIS_QUERY_UTILITY;NOT_MODEL_OR_SDK_SOURCE',
                              remote_executed_script_identity_newly_attested=False))
    unit_contract = read(O / 'complete-PMU-resource-contract-v18.json')
    assert 'FREQ_TO_MHz=1e6' in unit_contract['reviewed_findings'][0]
    assert '1.8e9Hz' in unit_contract['reviewed_findings'][0]

    caps = data[labels[0]]
    assert caps['summary']['database_files'] == caps['summary']['profiler_info_files'] == 32
    assert caps['summary']['schema_variants'] == 1
    assert len({d['schema_key'] for d in caps['database_summary']}) == 1
    q = data['queue-schema166-v15']['first_database']
    api_columns = [c['name'] for c in q['queue_flow_schemas']['PYTORCH_API']]
    assert api_columns == ['startNs', 'endNs', 'globalTid', 'connectionId', 'name', 'sequenceNumber',
                           'fwdThreadId', 'inputDtypes', 'inputShapes', 'callchainId', 'type']
    metadata_tables = ['NPU_INFO', 'HOST_INFO', 'META_DATA', 'SESSION_TIME_INFO', 'RANK_DEVICE_MAP']
    assert set(metadata_tables) <= set(q['all_table_names'])
    # The v12 schema fingerprint covers exactly these six tables, not all DB tables.
    schema_fingerprint_tables = ['CANN_API', 'COMPUTE_TASK_INFO', 'CONNECTION_IDS',
                                 'PYTORCH_API', 'PYTORCH_CALLCHAINS', 'TASK']
    native = data['native-execution-identity166-v14']['summary']
    side = data['graph-sidecar-candidates166-v14']
    s = side['summary']
    assert s['candidate_count_hist'] == {'0': 16, '1': 64}
    assert s['previously_unbound_with_unique_candidate'] == 0
    first_config = side['first_profiler_info']['config']
    assert first_config['common_config']['record_shapes'] is True
    assert first_config['common_config']['with_stack'] is True
    assert first_config['common_config']['profile_memory'] is False
    assert first_config['experimental_config']['record_op_args'] is False
    attrs = data['readonly-core-attributes166-v12-v2']['result']
    assert len(attrs['rows']) == 16
    for row in attrs['rows']:
        for key, value in [('CUBE_CORE_NUM', 24), ('VECTOR_CORE_NUM', 48)]:
            assert row['attributes'][key]['returncode'] == 0 and row['attributes'][key]['value'] == value
    provenance = data['profile-provenance166-v17-v2']
    captured = []
    for group in provenance['distinct_profile_hardware_fields']:
        for device in group['fields'].get('DeviceInfo', []):
            captured.append(dict(records=group['records'], **device))
    assert {r['id'] for r in captured} == set(range(16))
    assert all(r['ai_core_num'] == 24 and r['aiv_num'] == 48 for r in captured)
    assert all(r['aic_frequency'] == r['aiv_frequency'] == '1800' for r in captured)
    npu_info = data['profiler-metric-formulas166-v15']['device_info']
    assert npu_info['NPU_INFO'] == [dict(id=0, name='Ascend910B')]
    pmu = data['complete-PMU166-v18']['summary']
    assert len(pmu['metric_name_counts']) == 28
    assert set(pmu['metric_name_counts'].values()) == {286101}
    assert pmu['all_PMU_rows'] == 8010828 and pmu['all_metric_groups'] == 286101
    assert sum(pmu['metric_configurations'].values()) == 32
    check = data['PMU-reviewed166-v18']['summary']
    for e in ['aic', 'aiv']:
        assert check['time_formula_test']['engines'][e]['matching_groups'] == 286101
        assert check['time_formula_test']['engines'][e]['nonzero_with_no_matching_block'] == 0
    assert check['pipe_time_equals_reported_total_time_times_ratio']['mismatches'] == 0
    attr = data['compute-attrs166-v22']['summary']
    assert attr['attribute_lookup_execution_states'] == {
        'EMPTY_OR_PLACEHOLDER': 165030, 'NONEMPTY_STRING_CANDIDATE': 121920}
    sync = data['saved-events-resources166-v13-v2']['summary']
    assert sync['corrected_valid_count_sync_pairs'] == 56
    cpu = data['CPU-surfaces166-v17']['summary']
    assert cpu['helper_databases'] == 32 and cpu['helper_PYTORCH_API_tables'] == 0
    now = datetime.now(timezone.utc).isoformat()
    proof = dict(
        generated_utc=now, input_receipts=inputs, schema_and_flag_query_utilities=utilities,
        table_schema_evidence=dict(first_queue_schema_database=q.get('database_index'),
            queue_schema_query_database_count=5, first_database_table_names=q['all_table_names'],
            PYTORCH_API_columns=api_columns,
            six_table_schema_fingerprint_database_count=32,
            six_table_schema_fingerprint_tables=schema_fingerprint_tables,
            fingerprint_scope_description_source='Current local copy of the inventory utility;its historical remote executed-script bytes not newly attested. Saved receipts report the fingerprint count.',
            does_not_fingerprint_all_metadata_or_other_raw_surfaces=True),
        collection_flags=dict(first_profiler_info=side['first_profiler_info'],
            first_record_only_flags=['profile_memory=False', 'record_op_args=False', '_host_sys=[]', '_l2_cache=False'],
            all_32_receipt_confirmed=dict(record_shapes=True, with_stack=True,
                aic_metrics='ACL_AICORE_PIPE_UTILIZATION', profiler_level='Level1', export_type=['text']),
            first_record_flags_not_promoted_to_all32=True),
        saved_hardware_evidence=dict(runtime_checked_utc=attrs['checked_utc'], runtime_devices=16,
            Cube_per_device=24, Vector_per_device=48, capture_info_records=captured,
            raw_aic_aiv_frequency='1800', unit_conversion_reference='complete-PMU-resource-contract-v18.json',
            documented_reference_conversion_Hz=1800000000,
            reference_and_empirical_agreement_not_exporter_path_or_physical_peak=True,
            first_database_NPU_INFO=npu_info,
            NPU_INFO_name_not_an_independent_marketing_hardware_or_peak_attestation=True,
            UBUF_query_zero_not_usable_capacity=True),
        native_census=native, graph_sidecar_candidate_census=s,
        PMU_census=pmu, PMU_consistency=check,
        corrected_event_pair_census=sync, worker_frontend_surface_census=cpu,
        attrInfo_decode_census=attr,
        alternate_raw_surfaces_inventory=data['saved-profile-surfaces166-v14']['summary'],
        loaded_binary_identity_proved=False, new_remote_query=False, new_model_source_read=False,
        new_model_import_or_device_or_service_work=False, whole_module_closed=False, overall_complete=False)
    write(O / 'saved-profile-local-proof-index-v48.json', proof)

    def row(i, field, status, available, limit, action, refs):
        return dict(id='EC%02d' % i, evidence_question=field, state=status,
                    available_evidence=available, conclusion_limit=limit,
                    next_action=action, evidence=refs, requires_new_capture_now=False)

    rows = [
        row(1, 'Executed API/TASK/ComputeInfo identity and hierarchical cost domain', 'SAVED_SUMMARIES_AVAILABLE;FULL_ROWS_REMOTE',
            '32 DBs;1108212 TASK records;286950 compute executions. TASK rowid is an execution key;globalTaskId may repeat.',
            'Ancestry, graph children and primary functional buckets are different overlapping populations;counts are not candidate savings.',
            'Recover pinned full parts only for a specific unresolved per-variant selector/join;do not reprofile a population already saved.',
            ['native-execution-identity166-v14', 'host-profile-inventory166-v12-v2']),
        row(2, 'Operator shape/dtype and caller stacks', 'SELECTED_SCHEMA_AND_SAVED_CHAINS_AVAILABLE',
            'All32 share the selected six-table schema;PYTORCH_API exposes inputShapes/inputDtypes/callchainId and execution/connection fields.',
            'Callchain ancestry and equal shape do not establish producer-consumer storage, alias or actual tensor values.',
            'Use existing fullchain selectors first;point-read a producer/consumer only when the precise semantic question survives the saved evidence.',
            ['host-profile-inventory166-v12-v2', 'queue-schema166-v15']),
        row(3, '80 graph executions and phase binding', '64_BOUND;16_NO_SIDECAR_ENVELOPE_CANDIDATE',
            'Graph containment and64 same-call marker agreements saved;candidate histogram0:16,1:64;no formerly-unbound graph has a unique sidecar candidate.',
            'Clock calibration alone is not established to supply the16 missing bindings;ordinal/nearest-time assignment remains prohibited.',
            'Read saved graph/control/marker provenance for a new positive binding mechanism;retain unbound population if none exists.',
            ['native-execution-identity166-v14', 'graph-sidecar-candidates166-v14']),
        row(4, 'Actual operator scalar values/attrInfo ownership', 'RAW_DECODE_AVAILABLE;OWN_OP_SEMANTICS_UNPROVED',
            '181 distinct attr strings;121920 executions have a nonempty string candidate and165030 empty/placeholder. Reuse across adjacent op types is documented.',
            'A STRING_IDS decode,shape or chronological neighbour does not prove this operator received a specific parameter value;empty is not default-zero.',
            'Use the v23 attribute-quality contract and an exact exporter/native ownership or typed argument source for the disputed parameter only.',
            ['compute-attrs166-v22', 'profile-attribute-quality-contract-v23.json']),
        row(5, 'Tensor pointer/storage/version, aliases and captured bank generations', 'NOT_EXPLICIT_IN_PYTORCH_API;FULL_CAPTURE_ABSENCE_NOT_PROVED',
            'The selected PYTORCH_API schema has no explicit pointer/storage/version columns. First profiler_info disables memory profiling and op-argument recording.',
            'Those first-record flags are not an all32 flag census;other raw files,attribute fields or native sidecars have not been proved unable to carry identifiers.',
            'Check pinned saved config/metadata and relevant existing raw surface before proposing new capture. Use source invariants conditionally;do not demand runtime proof to call a conditional variant analyzed.',
            ['queue-schema166-v15', 'graph-sidecar-candidates166-v14', 'saved-profile-surfaces166-v14']),
        row(6, 'Native event handle/generation and last-reader ordering', 'HOST_ORDER_AVAILABLE;GENERATION_UNKNOWN',
            '56 corrected sync pairs share a thread and do not overlap;no observed valid-count writer lies between them.',
            'Host sequence without an observed intervening writer does not prove native event generation equality or safe sync/cache deletion.',
            'Inspect saved event/API identities for the exact two operations;retain generation and all-reader obligations in conditional reorganization contracts.',
            ['saved-events-resources166-v13-v2', 'event-resource-schema166-v13']),
        row(7, 'PMU field coverage and missing groups', 'ALL_SAVED_PMU_FIELDS_READ;849_COMPUTE_WITHOUT_GROUP',
            '8010828 rows,28 metric names,286101 groups;all fields retained rather than four chosen ratios.286950-286101=849.',
            'Missing groups are not zero cost or recovery by rereading another PMU column;unsupported/non-emitting work may remain missing even under recollection.',
            'Reuse complete PMU for covered executions;qualify uncovered metadata/control work separately. New counters only if a stated conclusion needs them.',
            ['complete-PMU166-v18', 'PMU-reviewed166-v18']),
        row(8, 'Core counts, recorded clock and cycle/time consistency', 'CACHED_FACTS_AND_CONVERSION_AVAILABLE;PHYSICAL_PEAK_OPEN',
            '16 runtime devices report24Cube/48Vector;capture metadata covers IDs0..15 with24/48 and raw1800 clocks. CANN reference converts1800 to1.8e9Hz;both-engine formula checks match286101 groups.',
            'This is already available. It does not prove actual exporter branch,per-dtype operations/cycle,HBM peak,whole-module efficiency or removable latency.',
            'Do not requery core counts or relabel the reference clock unit unknown. Request only the missing denominator required by a particular same-scope comparison.',
            ['readonly-core-attributes166-v12-v2', 'profile-provenance166-v17-v2', 'complete-PMU-resource-contract-v18.json']),
        row(9, 'Physical HBM/network bytes, cache traffic and peak utilization', 'NOT_SUPPLIED_BY_THE28_PIPE_UTILIZATION_METRICS',
            'Saved PMU metric names are pipe total/cycles/time/ratios/icache fields;MTE fractions overlap. First config has_l2_cache=False.',
            'MTE busy cycles/logical bytes/COMM payload are not measured external traffic or saturated bandwidth. No claim that all other raw surfaces are empty.',
            'Use logical lower/upper bounds with explicit assumptions;inspect saved relevant counters first,then consider controlled collection only if physical ranking depends on missing traffic.',
            ['complete-PMU166-v18', 'PMU-reviewed166-v18', 'graph-sidecar-candidates166-v14']),
        row(10, 'Current EngineCore/process onCPU, readiness and queue cost', 'WORKER_FRONTEND_WALL_AVAILABLE;ENGINE_ONCPU_NOT_ESTABLISHED',
            '32 helper DBs have zero PYTORCH_API tables;32 worker API DBs and two frontend files are inventoried. Archived EngineCore samples are source169 evidence.',
            '32 files do not mean32 independently profiled processes. Worker ancestry/wall spans cannot supply current166 EngineCore onCPU or ready/queue cost.',
            'Maintain process-separated wall and onCPU denominators;check saved process traces for the named stage before requiring a new process capture.',
            ['CPU-surfaces166-v17', 'profile-provenance166-v17-v2']),
        row(11, 'Coverage of unexecuted condition branches and all consumers', 'PROFILE_IS_EXECUTED_POPULATION_ONLY;FINITE_SOURCE_GAPS_REMAIN',
            '96 prepare roots,matched markers,shapes and source-bound executed calls constrain the observed paths;M21/M22 stable branch inventories retain their distinct scopes.',
            'A missing executed branch is not a proof that it is impossible or effect-free. Conversely native validation of a declared conditional variant is not mandatory to finish its semantic analysis.',
            'Point-read only the unresolved eligibility/producer/consumer branch. Track conditional analysis completion separately from deployment eligibility and measured gain.',
            ['prepare-field-origins166-v12', 'M21-semantic-branch-inventory-v47.json', 'M22-semantic-branch-inventory-v46.json']),
        row(12, 'Saved DB metadata/rank/session identity and all32 capture flags', 'TABLES_KNOWN;SOME_VALUES_NOT_IN_LOCAL_RECEIPTS',
            'First DB contains NPU_INFO/HOST_INFO/META_DATA/SESSION_TIME_INFO/RANK_DEVICE_MAP;first NPU_INFO is0/Ascend910B. All32 metric/shape/stack flags already agree in saved PMU census.',
            'Six-table schema uniformity does not attest these five metadata schemas/values. Metadata may improve identity/clock provenance but is not promised to recover16 graph phases or tensor generations.',
            'A bounded saved-only metadata/config query is prepared,NOT EXECUTED;run only against an accessible original or hash-identical relocated capture.',
            ['queue-schema166-v15', 'profiler-metric-formulas166-v15', 'host-profile-inventory166-v12-v2']),
        row(13, 'Loaded binary/tiling linkage', 'SAVED_ARTIFACT_INVENTORY;FULL_LINKAGE_UNPROVED',
            'Saved-surface inventory includes73 files each in compiled artifact families;file existence/count has been inventoried independently of their loaded-execution identity.',
            'Presence/count does not identify every loaded binary/tiling or prove numerical equivalence. This is not a prerequisite to describing a clearly conditional source-level opportunity.',
            'Resolve binary/tiling identity only for a conclusion that depends on its internals;preserve original capture identity and evidence scope.',
            ['saved-profile-surfaces166-v14', 'host-profile-inventory166-v12-v2']),
    ]
    for r in rows:
        resolved = []
        for name in r['evidence']:
            p = O / name if name.endswith('.json') else N / (name + '.json')
            assert p.exists(), str(p)
            resolved.append(str(p))
        r['evidence'] = resolved
    audit = dict(generated_utc=now, rows=rows, evidence_question_count=len(rows),
        proof_index=str(O / 'saved-profile-local-proof-index-v48.json'),
        saved_only_query=str(N / 'query_saved_metadata_and_capture_flags_v48.py'),
        query_execution_state='PREPARED_NOT_EXECUTED;ORIGINAL_DATA_ROUTE_PENDING',
        explicit_corrections=[
            'Core counts and the reference/empirically checked frequency conversion are already cached;not new evidence to recollect.',
            '16 phase-unbound graphs have zero saved sidecar envelope candidates,not16 merely ambiguous unique candidates awaiting clock promotion.',
            'PYTORCH_API table field absence and first-record disabled flags do not prove full raw-capture absence.',
            'All28 saved PMU fields have been read;849 missing groups are not a four-field extraction omission.',
            'Conditional opportunity analysis may finish without implementation/native verification/benchmark;actual-effect and eligibility claims retain their own gaps.'],
        new_source_read=False, new_remote_query=False, new_device_or_service_work=False,
        candidates_added=0, modules_closed=0, all_theory_comparisons_complete=False,
        original_profile_access_restored=False, overall_complete=False)
    write(O / 'saved-profile-evidence-capability-audit-v48.json', audit)
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v47.json')
    gate.update(status='PROFILE_CAPABILITY_AUDIT;NO_NEW_SOURCE_OR_REMOTE_READ',
        previous_turn_classification='PROGRESS:v47access_dispatch_resource_units',
        profile_facts='13 scoped evidence questions audited;cache denominator facts preserved;not all missing evidence requires recapture.',
        question='For the next disputed fusion/reorganization claim,which exact identity/value/ordering/cost fact is absent from saved evidence?',
        allowed_source=[], answered_by=['saved-profile-evidence-capability-audit-v48.json', 'saved-profile-local-proof-index-v48.json'],
        closed_scope='Evidence capability and scope audit only;not any whole-module branch/consumer or native-generation closure.',
        not_closed='Original capture route,remaining finite branches/consumers/process denominators,16 phase bindings,849 missing PMU and physical capacity claims.',
        stop_rule='Use cached facts,then a hash-pinned saved-only query or a precise source question. No broad reread,blanket recollection or193dataset substitution.')
    write(O / 'PROFILE-CODE-QUESTION-GATE-v48.json', gate)
    print(json.dumps(dict(evidence_questions=len(rows), verified_input_receipts=len(inputs),
        cached_devices=len(attrs['rows']), graph_without_sidecar_candidate=16,
        all_saved_PMU_fields=28, missing_PMU=849, remote_query_executed=False,
        whole_modules_closed=0, overall_complete=False)))


if __name__ == '__main__':
    main()
