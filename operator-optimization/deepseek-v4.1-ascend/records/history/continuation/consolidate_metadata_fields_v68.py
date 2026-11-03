"""Reconcile 586 metadata field regimes without claiming opaque plan/full-model closure."""
from collections import Counter
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    assert not (O/'operator-analysis-coverage-ledger-v68.json').exists()
    p=read(O/'metadata-field-regime-models-v68.json');assert p['regime_count']==586 and p['captured_execution_count']==67631
    assert len(read(O/'metadata-field-fusion-variants-v68.json')['variants'])==22
    assert read(O/'metadata-C2-header-native-check-v68.json')['calls']==104
    assert read(O/'metadata-field-operator-contract-v68.json')['existing_C2_fused_builder']['already_implemented']is True
    c=deepcopy(read(O/'operator-analysis-coverage-ledger-v67.json'));rows={r['regime_id']:r for r in c['regimes']}
    e=['metadata-field-regime-models-v68.json','metadata-field-native-chains-v68.json','metadata-field-fusion-costs-v68.json',
        'metadata-field-operator-contract-v68.json','metadata-field-fusion-variants-v68.json','metadata-C2-header-native-check-v68.json',
        'metadata-fixed-parameter-reconciliation-v68.json','metadata-existing-builder-reconciliation-v68.json']
    for r in p['regimes']:
        x=rows[r['regime_id']];assert x['state'].startswith('REGIME_') and x['executions']==r['executions'] and r['consumer_fusion_review_complete_in_this_turn']
        x.update(state='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED');x['scoped_conditional_review_evidence']+=e
    rev=[r for r in rows.values()if r['state'].startswith('SCOPED_')];pending=[r for r in rows.values()if r['state'].startswith('REGIME_')]
    assert (len(rev),sum(r['executions']for r in rev))==(1831,276286)
    assert (len(pending),sum(r['executions']for r in pending))==(251,10664)
    assert len(rev)+len(pending)==2082 and sum(r['executions']for r in rows.values())==286950
    assert Counter('|'.join(r['module_candidates'])for r in pending)=={'M21':242,'M19':4,'M12':3,'M05|M21':2}
    c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v67.json',scoped_conditional_review_regimes=1831,
        scoped_conditional_review_executions=276286,remaining_regime_reconciliation_count=251,remaining_execution_reconciliation_count=10664,
        remaining_regimes_by_operator=dict(Counter(r['operator']for r in pending)),
        remaining_executions_by_operator={op:sum(r['executions']for r in pending if r['operator']==op)for op in sorted(set(r['operator']for r in pending))},
        newly_reconciled_regimes=586,newly_reconciled_executions=67631,new_pending_regimes_reconciled=586,new_pending_executions_reconciled=67631,
        prior_geometry_promoted_to_role_conditional_review=0,prior_geometry_executions_promoted=0)
    assert (c['explicit_profile_insufficient_regimes'],c['explicit_profile_insufficient_executions'])==(7,1272)
    assert (c['explicit_unresolved_BOOL_field_regimes'],c['explicit_unresolved_BOOL_field_executions'])==(2,80)
    assert c['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    rem=[r for r in pending if r['module_candidates']==['M21']]
    opaque=[r for r in rem if r['operator']in ['SparseFlashMlaMetadata','VllmAscendQuantLightningIndexerV2Metadata']]
    plan_clear=[r for r in rem if r['operator']=='ZerosLike'and '(1174)'in r['role']]
    assert (len(rem),sum(r['executions']for r in rem))==(242,9312)
    assert (len(opaque),sum(r['executions']for r in opaque))==(18,529)
    assert (len(plan_clear),sum(r['executions']for r in plan_clear))==(3,79)
    write(O/'operator-analysis-coverage-ledger-v68.json',c)
    write(O/'operator-regime-refinement-v68.json',dict(refined_regimes=[dict(regime_id=r['regime_id'],field=r['field'],family=r['family'],
        executions=r['executions'],consumer_fusion_review_complete_in_this_turn=True,scope='Exacttypedfieldwork/selectedIO/publishedoutputABI/consumerconditions;notopaqueplan/nativephysicalcapacity.')for r in p['regimes']],
        reviewed=[586,67631],remaining_M21=[242,9312],remaining_opaque_metadata=[18,529],remaining_opaque_family_clear=[3,79],selected_missing_PMU=sum(r['missing_PMU']for r in p['regimes']),
        remaining_M21_by_operator={op:dict(regimes=sum(r['operator']==op for r in rem),executions=sum(r['executions']for r in rem if r['operator']==op))for op in sorted(set(r['operator']for r in rem))},
        whole_modules_closed=0,overall_complete=False))
    old=next(m for m in read(O/'module-phase-branch-matrix-v67.json')['modules']if m['id']=='M21')
    assert old['branch_schema']=='SEMANTIC_BRANCHES_V1'and len(old['branches'])==18
    bs=deepcopy(old['branches']);details={
        'M21-S01':'Pinnedconfigactualcompress_ratios0/1/2andwindow128;correctpriorgapwording0/4/32,notnumericcostinputs. Legacy910CvsA5packedguardandcache-kindmodesstillseparate.',
        'M21-S03':'Batchpositions/inactive-tailfieldsand208C2q/r5TASKgroups:56B→12Bknownvalues;CP/q/rmultiout94B+4→28Bsamegeneration/window/ready. Existinglengths:c2cachebaseline.',
        'M21-S04':'123physical-compressionslotregimes2276TASK+308coordinate/lengthregimes47642TASK:2601coordinate14TASKcores140N+3Fill1→20Ntyped(INT32pair,INT64flat). Legacyr1actualEndmaskabsent,r2physical+logical/skip/endgates,storageBsymbolic/power2conditional;existingpackedhelpernotdropin.',
        'M21-S05':'23CPregimes13960TASK=2792fiveTASKlengthgroups38B+4→20B. GPUauthoritativeL/suffix/nonempty/CPwindowconditionsandcache-hitdeadtemporarylemma;noinventedG/cachemisscountorCPUlengthreplacement.',
        'M21-S06':'69SWA4TASKC256recounts7684Nknownbytes;forwardglobalrawVonlyclamp/equalitydomain+orderedCPslice/epoch/fences. CurrentW128,Q<=5=>V<=133proofconditional;raw400/count256counterexample. Produceralreadyexists.',
        'M21-S09':'118C2fieldregimes3256TASK:104header14TASKgroups116B+8knowncore+40Bsource5copies;direct36Balternative.104cos+104sin3TASKgroups4112N→1044NrowCast+gatheror1032Njoint;preserveFLOATbanks/errors/strides. All-fieldC2kernelalreadyexistsv39underpackedguard;conditionallegacyreusepriced,notnewmissingimplementation.',
        'M21-S12':'Retaininput-ready/reuse/stage-group/capturedfrontiersandallC-S/header/lengthbanklastreaderswhenfusing/CSE/emptyfields. Some.copy_valuationsSOURCEsemanticoutsideComputeInfo,notinventedtracejoins.',
        'M21-S16':'Original76943M21functionalexecremainconsistent:586regimes67631nowtypedfieldreview,242regimes9312stillpendingincluding18opaqueplanregimes529and3plan-zero-regimes79. SelectedmissingPMU0;global849kept. Logicalbytes/scalarpartialsnotphysicalVector/INT64/memory/criticalcapacity.'}
    for b in bs:
        if b['id']in details:
            b['evidence']+=e;b['typed_operator_review_v68']=details[b['id']]
            if b['id']=='M21-S16':b['state']='TYPED_FIELD_BOUNDS_PARTIAL;242_REGIMES_AND_PHYSICAL_CAPACITY_OPEN'
    write(O/'M21-typed-field-branch-inventory-v68.json',dict(modules={'M21':bs},retained_semantic_families=18,
        extended_families=sorted(details),extended_family_count=len(details),remaining_opaque_plan_families=['M21-S07','M21-S08'],
        scope='Retain18families/provenance;extend8withtypednativework/IO/precision/fieldconditions. Allothersunchanged,notwholemoduleclosure.',whole_modules_closed=0,overall_complete=False))
    write(O/'metadata-field-local-validation-history-v68.json',dict(core_runs='Native586/67631extractionandalltypedmodelassertionsPASSfirstattempt;104headercross-checkPASS.',
        design_refinement='Reconciledexistingv39C2all-fieldfusedkernel/stride-domainproofagainstcurrentfallbackmodels;doNOTcallalreadyimplementedfusionanewmissingkernel. Receiptretainspriorwording;numericmodelsunchanged.',
        parameter_correction='v67gapwording0/4/32compressionincorrect;pinnedconfigactual0/1/2/window128;currentcanonicalscopefixedandoldimmutablehistoryretained.',
        resolved_local_assertion=dict(status='error',reason='Consolidationpreflightexpected21/608opaqueoperatorsbutthatfamilyincludes3/79ZerosLikeplan-clears;actual18/529custommetadataopsseparatelyretained. Uncheckpointedcoveragealreadywrittenwasnumericallycorrectandarchivedbeforefullregeneration;no previousauthority/sharedmastermutation. Movedscopeassertsbeforeoutputwrite.'),
        whole_modules_closed=0,overall_complete=False))
    print(dict(reviewed=[1831,276286],pending=[251,10664],new_review=[586,67631],remaining_M21=[242,9312],opaque_pending=[18,529],plan_clear_pending=[3,79],M21_families_retained=18,extended=8))

if __name__=='__main__':main()
