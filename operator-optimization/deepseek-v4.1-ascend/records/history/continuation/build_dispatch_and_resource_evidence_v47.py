"""Conditional frozen dispatch proof, actual login outcomes and resource-denominator audit."""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json

from finalize_origin_coverage_v9 import N,O,W,read,write


def main():
    assert not(O/'REVIEW-CHECKPOINT-v47.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    source=read(O/'M21-hardware-adaptor-and-DSACP-source-v47.json')
    for f in source['files']:assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
    hp=next(f for f in source['files']if f['path'].endswith('/hardware_profile.py'))
    family={r['device_key'].split('.')[-1]:r['adaptor_family'].split('.')[-1]for r in hp['selected_profile_entries']}
    assert family==dict(A2='STANDARD',A3='STANDARD',_310P='COMPATIBILITY',A5='FP8_OPTIMIZED')
    adaptor=dict(STANDARD='BaseDeviceAdaptor',COMPATIBILITY='Ascend310PDeviceAdaptor',FP8_OPTIMIZED='A5DeviceAdaptor')
    cases=[]
    for device,f in family.items():
        packed='MixedQuantPackedCacheOps'if f=='FP8_OPTIMIZED'else None
        cases.append(dict(build_device_type=device,profile_family=f,adaptor=adaptor[f],normal_packed_getter_result=packed))
    assert next(x for x in cases if x['build_device_type']=='A3')['normal_packed_getter_result']is None
    facts=[
        dict(grade='DOCUMENTED_FROZEN_SOURCE',statement='DeviceConfig selects built-in __device_type__ by enum name;unsupported explicit name raises,not legacy fallback. Only absent type delegates old __soc_version__ resolver. get_device_config is cached;hardware profile and DeviceOperator binding are selected at module initialization.'),
        dict(grade='DOCUMENTED_FROZEN_SOURCE',statement='Normal profile mapping:A2/A3→STANDARD→BaseDeviceAdaptor;310P→COMPATIBILITY adaptor inheriting the None getter;A5→FP8_OPTIMIZED→A5DeviceAdaptor returning MixedQuantPackedCacheOps. Merely finding A5 classes in the package does not select their ABI.'),
        dict(grade='DERIVED_CONDITIONAL',statement='Under the documented A3 normal-build factory path,packed-cache getter returns None,so metadata uses standard/legacy path. Current positive legacy SMLA/QLI calls are consistent with this;they do not by themselves attest every live dependency/build value or external override.'),
        dict(grade='HISTORICAL_ONLY',statement='Saved source169 _build_info contains A3. This is not a fresh166generated build-info or hardware_profile/device_config hash attestation;those actual bindings remain explicitly unproved.'),
        dict(grade='HISTORICALLY_STAGE_HASH_MATCH',statement='utils.enable_dsa_cp reads validated AscendConfig singleton.enable_dsa_cp. v42 normalization by CLIflag,has_indexer,use_sequence_parallel_moe is already documented;do not substitute raw additional_config flag or local model name for normalized value.'),
        dict(grade='DERIVED_SCOPE_CONSTRAINT',statement='A3 use of an existing pure slot-coordinate helper is separate from selecting full A5 packed cache ABI. Preserve prior ratio1mask/physical-vs-logical completion counterexample;enabling A5 adaptor is not an established A3 fusion optimization.'),
    ]
    write(O/'M21-hardware-and-DSACP-dispatch-contract-v47.json',dict(generated_utc=stamp,facts=facts,normal_factory_cases=cases,
        unsupported_build_type='RuntimeError before normal profile selection',unknown_profile='RuntimeError',
        source_evidence='M21-hardware-adaptor-and-DSACP-source-v47.json',current166_generated_build_info_attested=False,
        current166_profile_dependency_hashes_attested=False,actual_backend_override_excluded=False,
        implementation=False,native_validation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))

    # rc0 from the bastion is not proof remote Python ran or target asset matched.
    login=[]
    for asset in ['193','166','166-via105']:
        path=N/('updated-login-'+asset+'-v47.json');r=read(path);raw=r['stdout'].strip();parsed=None
        if 'match asset failed'in raw or '未发现匹配的资产'in raw:
            status='BASTION_ASSET_NOT_MATCHED';remote_executed=False
        else:
            try:parsed=json.loads(raw)
            except json.JSONDecodeError:parsed=None
            remote_executed=isinstance(parsed,dict)and'hostname'in parsed
            status='REMOTE_IDENTITY_VERIFIED'if r['returncode']==0 and remote_executed else'UNVERIFIED_REMOTE_RESULT'
        login.append(dict(asset=asset,receipt=str(path),ssh_exit_code=r['returncode'],application_status=status,
            remote_python_executed=remote_executed,hostname=parsed.get('hostname')if parsed else None,
            known_original_profile_paths_present=any(p['exists']for p in parsed.get('saved_profile_paths',[]))if parsed else None))
    assert login[0]['application_status']=='REMOTE_IDENTITY_VERIFIED'and all(x['application_status']=='BASTION_ASSET_NOT_MATCHED'for x in login[1:])
    discovery=read(N/'193-existing-saved-materials-v47.json');assert discovery['returncode']==0
    d=json.loads(discovery['stdout']);assert d['hostname']=='bm-jn-zs-zone1-910C-64G-10-193'
    write(O/'original-profile-access-status-v47.json',dict(generated_utc=stamp,login_outcomes=login,
        asset193_limited_existing_material_discovery=d,
        not_an_exhaustive193_filesystem_search=True,original_dataset_access_restored=False,
        missing_input='Usable original166/105asset entry/account,or an accessible shared/backup location for existing profile;193recapture only if user chooses it as a new controlled evidence collection.',
        question_pending='Original asset or saved profile location after verified193login and166/105asset no-match.',
        password_handling='<REDACTED_CREDENTIAL>',
        no_other_sessions_modified=True,no_models_or_containers_transferred=True,no_new_model_device_service_health_work=True,
        source_capture_not_replaced_by193=True,whole_module_closed=False,overall_complete=False))

    # Correct global units:3136 are ambiguous module-assignment executions,not priced optimization rows.
    resource=read(O/'resource-coverage-summary-v20.json');cap=resource['capture_resource_coverage']
    assert cap['certainly_attributed']+cap['ambiguous_execution_count']==cap['compute_executions']==286950
    assert sum(resource['ambiguous_sets'].values())==3136 and cap['shape_role_context_regimes']==2082
    old=read(O/'candidate-profile-ledger-v20.json')['candidates'];current=read(O/'candidate-profile-ledger-v46.json')['candidates']
    assert len(old)==33 and len(current)==35
    theory=read(O/'module-theory-comparison-readiness-v20.json')['modules'];modules=read(O/'module-profile-ledger-v46.json')['modules'];by={m['id']:m for m in modules}
    zero=set(resource['zero_ComputeInfo_attribution_modules']);assert zero=={'M00','M01','M02','M03','M22','M25'}
    compute={'M07','M08','M11','M13','M16','M18'};compound={'M06','M09','M10','M12'};hierarchy={'M18','M19','M20','M21','M22','M24'}
    rows=[]
    for t in theory:
        mid=t['module']
        if mid in zero:
            family_label='CONTROL_PROCESS_AND_INDIRECT_DEVICE_COST'
            need=['Actual process/onCPU/wall/queue/producer-ready boundary for the chosen stage;configuration/source branch eligibility where relevant.',
                  'Child kernel cost only under explicit hierarchy,never zero cost from no primary ComputeInfo assignment.']
        elif mid in compute:
            family_label='CONDITIONAL_DTYPE_ALGORITHM_COMPUTE_AND_MEMORY'
            need=['Correct variant boundary/valid rows/weight+dtype+rounding/instruction algorithm;shape MAC is a candidate model,not universal work bound.',
                  'Capacity/core/frequency/units plus memory/communication and same-denominator exposure before numerical roofline/ceiling claim.']
        elif mid in compound:
            family_label='COMPOUND_DYNAMIC_WORK_MEMORY_AND_PROTOCOL'
            need=['Actual valid query/key/topK/expert/token/owner work and compound operator internal semantics as required by the candidate.',
                  'External shape product or MAC0 is not total attention/MoE/custom-op arithmetic;keep memory/native communication/waits in scope.']
        elif mid=='M23':
            family_label='COLLECTIVE_LAYER_TOPOLOGY_AND_DELIVERY'
            need=['Separate COMM_OP intervals,HCCL-named ComputeInfo tasks and internal compound communication;do not sum hierarchical times.',
                  'Actual rank/group/topology/protocol/count/dtype/valid payload and dependencies;logical tensor bytes are not wire bytes.']
        else:
            family_label='MIXED_VECTOR_MEMORY_STATE_AND_HOST'
            need=['Correct functional/ancestry/phase scope,actual valid data range,publication/alias/state and host/device readiness.',
                  'Bytes/operation model for the chosen algorithm and actual memory/CPU/launch/overlap denominator;MAC0 is not zero work.']
        rows.append(dict(module=mid,name=by[mid]['name'],selected_comparison_family=family_label,
            certain_ComputeInfo_execution_count=t['captured_resource_records'],optional_membership_execution_count=t['optional_resource_records'],
            comparison_family_is_analysis_classification_not_measured_bottleneck=True,
            hierarchical_parent_scope_needs_separate_denominator=mid in hierarchy,
            zero_primary_assignment_is_not_zero_cost=mid in zero,
            current_algorithm_shape_MAC_model=t['captured_shape_MAC_model'],minimum_evidence=need,
            numeric_time_lower_bound=None,physical_capacity_upper_bound=None,theory_status='INCOMPLETE',
            source_scope_not_completed_by_this_audit=True,whole_module_closed=False))
    assert len(rows)==26 and all(r['theory_status']=='INCOMPLETE'for r in rows)
    write(O/'whole-scope-resource-denominator-audit-v47.json',dict(generated_utc=stamp,modules=rows,
        canonical_count_units=dict(model_optimization_candidates=35,framework_optimization_candidates=7,module_surfaces=26,
            ComputeInfo_executions=286950,certain_single_module_assignment_executions=283814,
            ambiguous_module_assignment_executions=3136,ambiguous_module_sets=resource['ambiguous_sets'],
            shape_role_context_aggregate_regimes=2082,PMU_groups=286101,missing_PMU=849),
        terminology_correction='3136 is ambiguous module-membership ComputeInfo execution count,not optimization-candidate count or3136priced candidate resource rows.',
        model_candidate_era=dict(v20_model_ledger_count=33,current_model_ledger_count=35,later_ids=sorted({r['id']for r in current}-{r['id']for r in old}),
            inference='Functional resource assignment is by26module surfaces,not proof that33or35optimization candidates/variants are individually priced. Later candidates can reuse existing evidence,with explicit per-variant identity/boundary binding when asserting gain/rank.'),
        accounting_limits=['Certain/optional primary functional buckets are not full hierarchical module cost;optional sets overlap and are not additive.',
            'M19functional1560 is not entire14191native ancestry selected decoder scope;M21functional76943 is not209chain/84297ancestry navigation;M22primary0 is not zero132000graph-child work or zero host cost.',
            'No missingPMU,time,capacity,physical bandwidth or source-only observer proof is filled with zero.'],
        theory_comparisons_still_incomplete=26,all_modules_closed=False,overall_complete=False))

    sem=deepcopy(read(O/'M21-semantic-branch-inventory-v43.json'))
    # v44-v46 updated M21 progress history,not these stable semantic branch rows.
    current_matrix=read(O/'module-phase-branch-matrix-v46.json');m21=next(x for x in current_matrix['modules']if x['id']=='M21')
    assert len(m21['branches'])==18
    sem['branches']=deepcopy(m21['branches'])
    evidence=['M21-hardware-adaptor-and-DSACP-source-v47.json','M21-hardware-and-DSACP-dispatch-contract-v47.json','original-profile-access-status-v47.json','whole-scope-resource-denominator-audit-v47.json']
    for b in sem['branches']:
        if b['id']=='M21-S01':b.update(state='ADAPTOR_AND_CANONICAL_DSACP_DEFINITIONS_ANALYZED;LIVE_DEPENDENCIES_AND_PCP_SELECTORS_OPEN',
            gap='NormalA3packedNone/A5packed factory and canonicalDSACP getter definitions answered. Actual166generated build-info/dependency hashes and exact missing enable_pcp/decode-shard definitions still require original data/source access;constructor/group actual variants and graph/resource/process gates remain.',
            evidence=list(dict.fromkeys(b['evidence']+evidence)))
        if b['id']=='M21-S16':b['evidence']=list(dict.fromkeys(b['evidence']+['whole-scope-resource-denominator-audit-v47.json']))
    sem.update(generated_utc=stamp,parent_inventory='M21-semantic-branch-inventory-v43.json');assert len(sem['branches'])==18 and len(sem['static_branch_sites'])==268
    write(O/'M21-semantic-branch-inventory-v47.json',sem)
    audit=deepcopy(read(O/'M21-original-requirement-completion-audit-v43.json'));audit.update(generated_utc=stamp,parent_audit='M21-original-requirement-completion-audit-v43.json',result='NOT_COMPLETE')
    for r in audit['requirements']:
        r['evidence']=list(dict.fromkeys(r['evidence']+evidence+['M21-semantic-branch-inventory-v47.json']));r['completion_proved']=False
    write(O/'M21-original-requirement-completion-audit-v47.json',audit)
    print(json.dumps(dict(verified_login193=True,original166_and105_not_matched=True,original_profile_access_restored=False,
        normal_factory_hardware_cases=len(cases),resource_module_denominators=26,corrected_ambiguous_execution_count=3136,
        shape_role_context_regimes=2082,model_candidates=35,framework_candidates=7,new_variants=0,
        whole_modules_closed=0,overall_complete=False)))


if __name__=='__main__':main()
