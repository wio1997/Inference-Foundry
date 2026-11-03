"""Conserve actual regimes, keep planner/internal capacity and full model open."""
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    assert not (O/'operator-analysis-coverage-ledger-v69.json').exists()
    p=read(O/'remaining-field-regime-models-v69.json');assert(p['regime_count'],p['captured_execution_count'])==(161,6680)
    assert read(O/'remaining-field-fusion-variants-v69.json')['variant_count']==20
    assert read(O/'opaque-plan-TASK-type-audit-v69.json')['task_type_counts']=={'KERNEL_AICPU':529}
    c=deepcopy(read(O/'operator-analysis-coverage-ledger-v68.json'));rows={r['regime_id']:r for r in c['regimes']}
    e=['remaining-field-regime-models-v69.json','remaining-field-native-chains-v69.json','remaining-field-operator-contract-v69.json','remaining-field-fusion-costs-v69.json','remaining-field-fusion-variants-v69.json','remaining-field-math-validation-v69.json','opaque-plan-TASK-type-audit-v69.json','MC2-mask-typed-valuation-v69.json','remaining-field-source-binding-v69.json']
    for r in p['regimes']:
        x=rows[r['regime_id']];assert x['state'].startswith('REGIME_')and r['consumer_fusion_review_complete_in_this_turn']
        x['state']='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED';x['scoped_conditional_review_evidence']+=e
    rev=[r for r in rows.values()if r['state'].startswith('SCOPED_')];pending=[r for r in rows.values()if r['state'].startswith('REGIME_')]
    assert(len(rev),sum(r['executions']for r in rev))==(1992,282966)
    assert(len(pending),sum(r['executions']for r in pending))==(90,3984)
    assert Counter('|'.join(r['module_candidates'])for r in pending)=={'M21':81,'M19':4,'M12':3,'M05|M21':2}
    rem=[r for r in pending if r['module_candidates']==['M21']];assert sum(r['executions']for r in rem)==2632
    c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v68.json',scoped_conditional_review_regimes=1992,scoped_conditional_review_executions=282966,remaining_regime_reconciliation_count=90,remaining_execution_reconciliation_count=3984,newly_reconciled_regimes=161,newly_reconciled_executions=6680,new_pending_regimes_reconciled=161,new_pending_executions_reconciled=6680,prior_geometry_promoted_to_role_conditional_review=0,prior_geometry_executions_promoted=0,remaining_regimes_by_operator=dict(Counter(r['operator']for r in pending)),remaining_executions_by_operator={op:sum(r['executions']for r in pending if r['operator']==op)for op in sorted(set(r['operator']for r in pending))})
    assert(c['explicit_profile_insufficient_regimes'],c['explicit_profile_insufficient_executions'])==(7,1272)
    assert(c['explicit_unresolved_BOOL_field_regimes'],c['explicit_unresolved_BOOL_field_executions'])==(2,80)
    assert c['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    mask=[]
    for l in [230,231]:
        rs=[r for r in p['regimes']if r['family']=='MC2_ACTIVE_MASK'and f'({l})'in r['role']]
        counts={op:sum(r['executions']for r in rs if r['operator']==op)for op in set(r['operator']for r in rs)}
        written=sum(int(r['signature']['outputShapes_text'].strip('"'))*r['executions']for r in rs if r['operator']=='Cast'and r['signature']['outputDataTypes_text']=='BOOL')
        old=sum(r['known_logical_value_bytes_excluding_control']*r['executions']for r in rs);positive=counts['BroadcastTo'];extra=counts['Fill']-positive
        assert counts['Cast']==2*positive and old==9*written+10*positive+extra
        mask.append(dict(line=l,field='true_active_prefix'if l==230 else'false_padding_suffix',native_TASK_counts=counts,positive_range_3TASK_conversion_groups=positive,scalar_Fills_unpaired_by_trace=extra,known_value_bytes=old,materialized_BOOL_output_bytes=written,source_direct_full_mask_contract='All[0:P) BOOL rows must equal(i<A). ScalarFill1 is NOT one-byte mask slice. UnpairedFill not independently paired to a bank or assumed removable by time.'))
    assert sum(r['known_value_bytes']for r in mask)==1461368 and sum(r['materialized_BOOL_output_bytes']for r in mask)==161984
    contract=read(O/e[2]);pins={}
    for fn,key in [('M21-slot-rejection-state-and-packing-contract-v38.json','source_pins'),('M21-count-metadata-and-output-publication-contract-v37.json','source_files'),('M18-partition-bank-RoPE-and-slot-contract-v34.json','source_files')]:
        for remote,sha in read(O/fn)[key].items():pins[remote]=(sha,fn)
    pins['/vllm-workspace/vllm-ascend/vllm_ascend/models/common/ops/sequence_parallel.py']=('73bcf07050bce8b049dd0bbdb0a7504f6eba157a1e97a5033849ef31ef05fcb0','Previouslyboundv30/v62/v67SPmaskcontract')
    bindings=[]
    for r in contract['source_files']:
        assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
        remote='/vllm-workspace/vllm-ascend/'+r['path'].split('/vllm-ascend/')[-1];old=pins.get(remote)
        if old:assert old[0]==r['sha256']
        bindings.append(dict(**r,remote=remote,prior_historical_file_pin_matches=bool(old),prior_evidence=old[1]if old else'FrozenlocalfileSHA+matchingprofilecallsite;noindependent166filepinreused',historically_loaded_code_object_attested=False))
    assert sum(r['prior_historical_file_pin_matches']for r in bindings)==9
    assert [r['remote']for r in bindings if not r['prior_historical_file_pin_matches']]==['/vllm-workspace/vllm-ascend/vllm_ascend/ascend_forward_context.py']
    write(O/'operator-analysis-coverage-ledger-v69.json',c)
    write(O/e[7],dict(range_models=mask,source_known_value_bytes=1461368,known_materialized_BOOL_bytes=161984,logical_interface_budget_bytes=1299384,scope='KnownComputeInfovaluesonly. DirectsourceABIwholeP,notmeasuredoverallgain;80unresolvedBOOLprefixwritesand3136ambiguityunchanged.',overall_complete=False))
    write(O/e[8],dict(files=bindings,nine_prior_historical_SHA_matches=True,forward_context_current_file_same_revision_is_condition=True,scope='10 frozenlocaloperatorfiles;9 independentlyreusedpriorSHApins,forwardcontextlocalSHA/profilecallsiteonly. No newremote or loaded-binary attestation. ConditionaldirectMC2mask/dirtyproofdoesnotclaimunknown80storesidentity.',overall_complete=False))
    write(O/'operator-regime-refinement-v69.json',dict(refined_regimes=[dict(regime_id=r['regime_id'],family=r['family'],executions=r['executions'],scope='Typedoperatorwork/selectedIO/concreteconditionaloptions;notnativeperformance/wholemodule.')for r in p['regimes']],reviewed=[161,6680],remaining_M21=[81,2632],remaining_opaque_metadata=[18,529],remaining_opaque_family_clear=[3,79],remaining_M21_by_operator={op:dict(regimes=sum(r['operator']==op for r in rem),executions=sum(r['executions']for r in rem if r['operator']==op))for op in sorted(set(r['operator']for r in rem))},selected_missing_PMU=0,whole_modules_closed=0,overall_complete=False))
    old=next(m for m in read(O/'module-phase-branch-matrix-v68.json')['modules']if m['id']=='M21');bs=deepcopy(old['branches']);assert len(bs)==18
    details={
      'M21-S03':'Nextcount/select88cores13TASKexact38BK+94B→4BK+16B. Literalpredicateincludes<-1/count-1holes;K1/6FP32exact. ExistingseedINT64/GPUlength/reject/QSL/CPUauthoritativebanksretained.',
      'M21-S04':'88fusedtargetslotTASK13pointergroups,alreadyordinary/adaptive. Output52*C,Cunbound;dirtysuffix52*(N+max(oldN-N,0))onlyfullPADsuffixinvariant/perbankwriter/resetfences. FP32blockroundtrip2^24+1counterexample retained.',
      'M21-S06':'88SWAproducerTASKN10/20,B2/4,C256,alreadyindices+rawlengthsonekernel. ReqtilingcanremoveNscansofB;compactBrownewABIreducesoutput5xbutneedsSparseFlashMla/CP/allreader/mappingadaptation;rawlengthnotclipped.',
      'M21-S10':'Existingcopy-expand88writes20Tcontextpergroupnot8192fullbank;queryfields12BQ+4BD,28+4Rqueryreadreference. Querypositionaccepted-context/MMseparatelinearKVlength/DCP. Gdraftunknown,donotusetargetG13.72paddedkernelalreadydual-output;conditionaljointcountformulaquantified.',
      'M21-S11':'DummyPAD/positions127/indices0and936worker-tailFillhaveexactnativewriteextents. Coalescing/idempotenceonlysamebank/containedinterval/nointerveningwriter;realdummyresetrequiresfullinvalidate. Noall936deletionclaim.',
      'M21-S12':'288cos+288sincached3TASKand96+96advanced2TASKgroups. Cached4112N→1032N;advanced1056N→1032Nwithnegativewrap/validation. FLOATbanks512Nretained,position/config/stride/lastreader/capture/newconsumerABIexplicit.',
      'M21-S15':'Dirtyslot/thresholdchangescrossiterationsrequirefullestablishedbankinvariants,allwriters/readers/generations/resetandreplayparameterbinding;source-specificconditionsnotframeworkaudit. 16phaseunboundgraphsretained.',
      'M21-S16':'1992regimes282966execreviewed;90/3984pending=81M21/2632+7MemSet1272+2BOOL80. All529opaqueplansKERNEL_AICPU,notVector/Cube;internalworkstillpending. 849missingPMU/3136ambiguityunchanged.',
      'M21-S17':'MC2activeBOOLmask42regimes1448TASKknownlogical1461368→161984BOOLoutputbudget1299384;directthreshold/dirtyflipconditional.9sourcepriorSHApins,forwardcontextlocalfrozenonly/same166revisioncondition. SPNone24zeroscanmergeprefix/paddingproducer;suppliedmasksandambiguous80unchanged.'}
    for b in bs:
        if b['id']in details:b['evidence']+=e;b['typed_operator_review_v69']=details[b['id']]
        if b['id']=='M21-S16':b['state']='TYPED_OPERATOR_BOUNDS_PARTIAL;81_REGIMES_AND_PHYSICAL_CAPACITY_OPEN'
    write(O/'M21-remaining-field-branch-inventory-v69.json',dict(modules={'M21':bs},retained_semantic_families=18,extended_families=sorted(details),extended_family_count=len(details),scope='Retain18families/enrich9withoperatorcontracts.25othermodulesidentical. SMLA/QLIplaninternalworkandunobservedapplicablebranchesremain.',whole_modules_closed=0,overall_complete=False))
    print(dict(reviewed=[1992,282966],pending=[90,3984],M21_pending=[81,2632],families_extended=9,MC2logical=[1461368,161984],prior_source_pins=9))

if __name__=='__main__':main()
