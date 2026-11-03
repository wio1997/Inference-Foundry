"""Validate scope/real capture capabilities/current evidence routing;no device work."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json, sys
from finalize_origin_coverage_v9 import N, O, T as TASK_DIR, L as LOOP_DIR, read, write
from analyze_evidence_batches_v155 import pin


def main():
    preflight='--preflight'in sys.argv;checks=0
    def check(value,why):
        nonlocal checks
        assert value,why;checks+=1
    for source in read(O/'evidence-consolidation-inputs-v158.json')['inputs']:check(pin(source['path'])==source,'input '+source['path'])
    for source in read(O/('checkpoint-manifest-v157.json'if preflight else'checkpoint-manifest-v158.json'))['artifacts']:check(pin(source['path'])==source,'sealed '+source['path'])
    capture=read(O/'all-profile-capture-capabilities-v158.json');actual=[];counts=Counter()
    for row in capture['profiles']:
        check(pin(row['source']['path'])==row['source'],'capture source digest');source=read(Path(row['source']['path']));check(row['common_config']==source['config']['common_config']and row['experimental_config']==source['config']['experimental_config'],'exact recorded configuration');check(source['rank_id']==row['rank_id'],'rank')
        common=source['config']['common_config'];experimental=source['config']['experimental_config'];check(common['profile_memory']is False and experimental['record_op_args']is False,'actual oldflags disabled')
        for key in['record_shapes','profile_memory','with_stack','with_modules','with_flops']:counts[key+'='+str(common[key])]+=1
        for key in['record_op_args','_aic_metrics','_l2_cache','_op_attr']:counts[key+'='+str(experimental[key])]+=1
        actual.append(row['source']['path'])
    check(len(set(actual))==32 and dict(counts)==capture['observed_config_counts'],'32 unique configurations and exact counts')
    inventory=read(O/'raw-profile-inventory-v155.json');streams=defaultdict(lambda:dict(files=0,bytes=0))
    for row in inventory['files']:
        if'/FRAMEWORK/'in row['relative']:
            value=streams[Path(row['relative']).name];value['files']+=1;value['bytes']+=row['bytes']
    check(dict(streams)==capture['framework_streams']and set(streams)=={'torch.op_range','torch.op_mark','torch.python_tracer_func','torch.python_tracer_hash'},'complete physical FRAMEWORK stream inventory')
    check(not capture['enabling_record_op_args_alone_proves_addresses_or_readers'],'new flag not an unverified capability proof')
    cross=read(O/'all-module-evidence-crosswalk-v158.json');matrix=read(Path(cross['parent_matrix']['path']));branches={(m['id'],b['id']):b for m in matrix['modules']for b in m['branches']};check(len(branches)==358,'current semantic row universe')
    spec=read(O/'operator-evidence-acquisition-spec-v158.json');facets=spec['facets'];seen=set();modules={x['module']:x for x in cross['module_rows']}
    for name,source in cross['evidence_reference_pins'].items():check(pin(source['path'])==source and Path(source['path']).name==name,'existing reference pin '+name)
    for row in cross['branch_rows']:
        key=(row['module'],row['branch_id']);check(key in branches and key not in seen,'unique original row');seen.add(key);original=branches[key]
        check((row['branch'],row['existing_state'],row['existing_scope'],row['existing_gap'],row['existing_evidence'])==(original['branch'],original['state'],original['scope'],original['gap'],original['evidence']),'full original branch/remaining evidence preserved')
        check(row['evidence_facets']==modules[row['module']]['shared_evidence_facets']and set(row['evidence_facets'])<=set(facets),'question routing exists')
        check(row['branch_specific_instantiation_still_required']and row['grouping_is_not_semantic_equivalence_or_complete_analysis']and row['empty_fields_do_not_prove_independence'],'no grouping closure')
        check(not row['whole_branch_completion_promoted']and not row['dispatch_as_individual_pair_or_branch_task'],'no new358tasklist')
    check(seen==set(branches)and len(modules)==26,'all current rows/modules retained')
    check(all(not m['whole_module_closed']and not m['analysis_inventory_exhaustiveness_proved']for m in modules.values()),'inventory nonexhaustive')
    check(cross['reference_checks_are_not_content_or_exhaustiveness_proofs'],'reference integrity scope')
    adapter=read(O/'host-adapter-collector-readiness-v127.json');check(not adapter['full_model_capture_ready']and not adapter['actual_CANN_interception_tested']and adapter['validated']['platform']=={'system':'Linux','machine':'aarch64'},'real adapter readiness')
    packages={x['id']:x for x in spec['packages']};check(len(packages)==8 and {f for x in packages.values()for f in x['facets']}==set(facets),'shared packages route all named question facets')
    check(packages['P3-EAGER-PROVIDER']['status']=='CPU_FAKE_RUNTIME_VALIDATED_REAL_CANN_UNTESTED','no native hook readiness promotion')
    check(packages['P4-CAPTURE-CREATION-UPDATE']['status']=='ACQUISITION_SPEC_ONLY_NOT_IMPLEMENTED_OR_RUNTIME_READY','captured observer not implemented')
    points=spec['observation_point_separation'];check(points['original_eager']==1272 and points['original_captured_BOOL']==80 and points['eager_and_capture_creation_points_are_distinct'],'distinct native evidence entry points')
    check(not points['minimum_full_model_run_count_proved']and spec['proposed_grouping_is_not_semantic_coverage_or_absolute_minimality_proof'],'not an unsupported global minimum')
    identity=spec['identity_rule'];check(identity['new_capture_is_not_original_capture']and identity['historical_9_1352_rows_retained'],'no transfer of new to historical evidence')
    check(len(spec['stages'])==3 and not spec['stages'][1]['ready_to_execute']and not spec['stages'][2]['ready_to_execute'],'runtime stages not ready')
    check(spec['stages'][1]['authorization']=='NOT_GRANTED;v109/v120grantsconsumed'and spec['stages'][2]['authorization']=='NOT_GRANTED','no pending/invented grant')
    bounds=spec['stages'][1]['bounds'];check(bounds['devices']==1 and bounds['selected_native_records_max']==2 and bounds['host_packet_bytes_max']==4096 and not bounds['model_loaded']and not bounds['replay'],'reviewable minimum smoke domain')
    reconciliation=read(O/'current-obligation-reconciliations-v158.json');base=read(Path(reconciliation['parent']['path']));check(set(reconciliation['reconciliations'])=={x['id']for x in base['obligations']}=={'G%02d'%i for i in range(15)},'all15 broad obligations retained')
    for row in reconciliation['reconciliations'].values():
        check(set(row['current_evidence_facets'])<=set(facets)and set(row['current_acquisition_packages'])<=set(packages),'actual named routes')
        check(not row['whole_domain_closed']and row['original_broad_remaining_and_history_preserved'],'no broad gap deletion')
    old_scope=read(O/'coverage-current-obligations-v154.json')['outside_independent_operator_research_scope'];check(spec['scope_rule']==old_scope,'existing user operator-effect scope reused')
    summary=read(O/'evidence-consolidation-summary-v158.json');check(summary['new_pair_clauses']==summary['new_independence_proofs']==summary['new_native_operand_or_phase_promotions']==summary['new_whole_module_closures']==0 and not summary['overall_complete'],'no new native/completion claim')
    if not preflight:
        cp=read(O/'REVIEW-CHECKPOINT-v158.json');check(cp['summary']==read(N/'RECOVERY.json')['latest_profile_first_review']['summary'],'recovery authority')
        effective=read(Path(cp['coverage_obligations']));old_byid={x['id']:x for x in base['obligations']}
        for row in effective['obligations']:
            original=old_byid[row['id']];check(all(row[k]==v for k,v in original.items()),'all old obligation fields preserved')
            check(row['current_evidence_routing_v158']==reconciliation['reconciliations'][row['id']],'canonical current routing applied')
        check(cp['candidate_ledger'].endswith('candidate-profile-ledger-v154.json')and cp['unchanged_operator_authorities_reused'],'unchanged semantic authority')
        check(read(TASK_DIR/'index.json')['status']=='ACTIVE'and read(LOOP_DIR/'state.json')['status']=='EVALUATING','Task/Loop status')
        pointer=read(N/'CURRENT-ANALYSIS-WORKFLOW.json');check(pointer['effective_checkpoint']==158 and pin(pointer['work_queue']['path'])==pointer['work_queue'],'workflow pointer')
    result=dict(status='PASS',checkpoint=158,preflight=preflight,checks=checks,verification_scope=['32 actual recorded profiler configurations and physical FRAMEWORK stream inventory','26modules/358original rows and451reference pins preserved;shared routing not semantic equivalence','Real CPU-only adapter/readiness limits and distinct eager versus capture observation points','All15obligations reconciled without scope deletion or native/completion promotion'],old_unchanged_operator_case_suites_reexecuted=False,new_device_work=False,new_native_or_whole_closures=0,overall_complete=False,observed_utc=datetime.now(timezone.utc).isoformat(),verifier=pin(Path(__file__)))
    output=O/('evidence-consolidation-preflight-v158.json'if preflight else'evidence-consolidation-postcheck-result-v158p.json');check(not output.exists(),'single result')
    if not preflight:
        manifest=O/'checkpoint-manifest-v158.json';result.update(sealed_manifest=pin(manifest),sealed_manifest_files=len(read(manifest)['artifacts']),recovery_summary_agrees=True,sealed_files_modified=False)
    write(output,result)
    if not preflight:write(N/'CURRENT-POSTCHECK.json',result)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
