"""Bind M23 ComputeInfo regimes to all HCCL cost/consumer conditions."""
from collections import Counter,defaultdict
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    assert not (O/'operator-analysis-coverage-ledger-v66.json').exists()
    b=read(O/'communication-compute-regime-bindings-v66.json')
    pop=read(O/'communication-native-operator-population-v66.json')
    costs=read(O/'communication-work-payload-models-v66.json')
    assert len(costs['count_group_regimes'])==61 and costs['type_dtype_rank_count_regime_count']==42
    assert sum(r['calls']for r in costs['count_group_regimes'])==14720
    assert len(read(O/'communication-fusion-variants-v66.json')['variants'])==20
    opmap={(r['database_index'],r['COMM_OP_rowid']):r for r in pop['operations']}
    evidence=['communication-native-operator-population-v66.json','communication-compute-regime-bindings-v66.json',
        'communication-graph-role-bindings-v66.json','communication-work-payload-models-v66.json',
        'communication-operator-contract-v66.json','communication-fusion-variants-v66.json']
    models=[]
    for r in b['regimes']:
        ops=[opmap[tuple(k)]for k in r['exact_COMM_OP_keys']]
        assert len(ops)==r['executions'] and all(x['compute_regime_id']==r['regime_id']for x in ops)
        groups=Counter((x['semantic_role'],x['operator'],x['dtype'],x['rankSize'],x['count'],x['group_name'])for x in ops)
        models.append(dict(r,semantic_role_counts=dict(Counter(x['semantic_role']for x in ops)),
            typed_role_count_models=[dict(semantic_role=k[0],operator=k[1],dtype=k[2],P=k[3],count=k[4],group_name=k[5],calls=v)for k,v in sorted(groups.items())],
            work_payload_model='communication-work-payload-models-v66.json',consumer_contract='communication-operator-contract-v66.json',
            fusion_choices='communication-fusion-variants-v66.json',consumer_fusion_review_complete_in_this_turn=True,physical_numeric_ceiling_proved=False))
    assert len(models)==17 and sum(r['executions']for r in models)==11528
    write(O/'communication-operator-regime-models-v66.json',dict(regimes=models,regime_count=17,captured_execution_count=11528,
        broader_communication_operations_modeled=14720,nonComputeInfo_operations=3192,
        no_schedule_count_bijection_inference=True,whole_modules_closed=0,overall_complete=False))
    c=deepcopy(read(O/'operator-analysis-coverage-ledger-v65.json'));rs={r['regime_id']:r for r in c['regimes']}
    for r in models:
        x=rs[r['regime_id']];assert x['state'].startswith('REGIME_') and x['executions']==r['executions']
        x.update(state='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED')
        x['scoped_conditional_review_evidence']+=evidence+['communication-operator-regime-models-v66.json']
    rev=[r for r in rs.values()if r['state'].startswith('SCOPED_')];pending=[r for r in rs.values()if r['state'].startswith('REGIME_')]
    assert len(rev)==1015 and sum(r['executions']for r in rev)==201431
    assert len(pending)==1067 and sum(r['executions']for r in pending)==85519
    assert len(rev)+len(pending)==2082
    c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v65.json',scoped_conditional_review_regimes=1015,scoped_conditional_review_executions=201431,
        remaining_regime_reconciliation_count=1067,remaining_execution_reconciliation_count=85519,
        remaining_regimes_by_operator=dict(Counter(r['operator']for r in pending)),
        remaining_executions_by_operator={op:sum(r['executions']for r in pending if r['operator']==op)for op in sorted(set(r['operator']for r in pending))},
        newly_reconciled_regimes=17,newly_reconciled_executions=11528,new_pending_regimes_reconciled=17,new_pending_executions_reconciled=11528,
        prior_geometry_promoted_to_role_conditional_review=0,prior_geometry_executions_promoted=0)
    assert c['explicit_profile_insufficient_regimes']==3 and c['explicit_profile_insufficient_executions']==960
    assert c['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    write(O/'operator-analysis-coverage-ledger-v66.json',c)
    pending_by_module=defaultdict(lambda:dict(regimes=0,executions=0))
    for r in pending:
        x=pending_by_module['|'.join(r['module_candidates'])];x['regimes']+=1;x['executions']+=r['executions']
    write(O/'operator-regime-refinement-v66.json',dict(refined_regimes=[dict(regime_id=r['regime_id'],executions=r['executions'],
        consumer_fusion_review_complete_in_this_turn=True,scope='Exact14720COMMpopulation/61grouplabelcountmodels/13rolesand20conditionaloperatorchoices;notphysicalnetworkorfullmoduleclosure.')for r in models],
        reviewed_regimes=17,executions=11528,nonComputeInfo_extra_COMM_ops_modeled=3192,remaining_by_module=dict(pending_by_module),whole_modules_closed=0,overall_complete=False))
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v65.json')['modules']}
    def row(mid,i,s):return dict(id=f'{mid}-OP-{i:02d}',branch=s,
        state='SCOPED_OPERATOR_CONDITIONAL_WORK_IO_FUSION_REVIEWED;WHOLE_MODULE_OPEN',evidence=evidence+['communication-operator-regime-models-v66.json'],
        gap='Actualrows/alias/value/weightandstate-generation/allreaders/nativeprecision/loadedcollectivealgorithmandphysicaltopology/link/injection/cut/latencyremainconditions;unobservedapplicableoperatorbranchesopen.',
        scope='Operatorfusion/repartition/work/valueIO/communication-only;noframeworkcatalogorimplementationrequirement.')
    m23=[
        'All14720HCCLops=11528exactComputeconnection+interval+typejoins+3192withoutComputeInfo. Separate557688internalrecords/3192Scheduledefinitions;equalcountsnotSchedulebijectionproof.',
        '17M23functionalregimes11528TASKreviewed;61group-label/countcombinations,42type/dtype/P/countand76role/group/countmodels,13semanticroles. ComputeInfoshapesN/Anotmissingcommunicationcount.',
        'AllGathercountperrankinput,ReduceScattercountperrankoutput,inputPtimes;AlltoAllcountperpeer. Allreducecountwholearray. Declaredring/pairwisework/byte/roundmodelsnotactualwireorphysicallowerbounds.',
        'Native4639directfullchains21canonical+1reuseddecodedqueueproof+10080graphcalls. 80graphs126calls/40blocktriples;64root/16unboundretained,graphschemaisnottensor-edge/actualmetadataattestation.',
        'SPinputAG→projectionstreamandprojectbeforeAG:existingF32/F33typedSWA/C1/C2/indexallreaders/rowgeneration/quantnorm/cacheWARandnewcommunicationplanesretained.',
        'Attentionfull64headsperlocalquery→restoreTP8headsAlltoAll;directsendlayout/newoutputABIcanfoldpackbutnotpayload. CurrentM24send172032B,notcount*2=24576B.',
        'A2A→WOApanelpipeline:complete4096KdotbeforeBF16WA,GMqueue/tile/arrival/captureandVector-vs-Cuberesourceconstraints. Nozero-traffic/freeoverlapassumption.',
        'WA-beforeexchangeall8groupweights:4096→1024exchange,WAextra58.72MB/card/layer;M=AoutputA2A+RSpayload2/3,sameFbutnewG8localtiling/coldweights. A>MpaddingaddsFunlessproof.',
        'QueryownerallWA/WBweights:outputA2A/RSconditionaleliminationwith132.12MBextra/card/layer;all8BF16partialsandnativeSUMorder/rowowner/alias/readersstillrequired. Noone-accumulatoralgebra.',
        'WOBepilogue→RS→HcPosttilepipelinepreservesperrankBF16partialandBF16reduceoutput/fullHCresidual/coefficients/allreaders;input/outputandnetworkbudgetsnotadditive.',
        'EmbeddingAR→RSownershipprotocol:alignedsendhalfonlyalllocalreaders;currentfirstdecoder/preprocess/raw/CPfullsequenceexpectationmustrework. FullreaderAGrestoresARcostorworsepadding.',
        'FinalHC/BF16andpremix/FP32copygathers:bytepack2→1launch,payloadunchanged;P8ordinarypack/unpack18qextraaccess,newABI/alignment/bankprefix-tail/readytimeallreaders.',
        '3auxcheckpointgathersbatchonlyafterdeferral/allreadersallow;3distinctvaluesnotCSE. Retainlocalsources/layoutunpackorindirectGEMM/newgraphbanks/lostoverlap.',
        'Headglobalbestsummaryv65/contextfeatureRS-AGv64/Engramrow-headv62usecurrent61countmodels;fullprob/entropy/RNG/fullcontextbank/UVAandmetadatareadersremain.',
        'Uniformragged/padtrimneedsnewvariable-countprotocol/allrankmetadataandnewzero-fill/capture. LocalqueryemptycannotdeletetargetglobalKV/output/collectiveparticipation.',
        'Crosslayer/iterationreuseonlysamevalues/generation/request/acceptedprefix/parameters/positions/slot/group/lastreader;shape/addressaloneinsufficient. BF16→INT8losslessexactforallvaluesexcludedwithoutsideinformation.',
        'HCCLrankSize8groupsandHCCS/SIO/SDMAsavedlabelsnotMC2EP16physicaltopology. NativeMC2internalv56receiverwork/EPdispatchandCPU-Gloo/DPmetadataareseparatecoverageobligations.',
        'CurrentAIV/AI_CPU/otherprotocol/mixDP/CP/SP/nonSP/zeroempty/Nonebranchesonlypricedbyapplicability;actualVectorcorecontention/link/injection/cut/latency/HBM-L2/precisionunbound. IntervalunionnotcriticalpathorMFU.'
    ]
    inv={'M23':[row('M23',i+1,s)for i,s in enumerate(m23)]}
    for mid,extra in {
        'M11':[
            'NewWA-beforeAlltoAllfull8groupweightreplicationandqueryownerWA/WBallgroupalternatives;actualA_CPvsMprojectionpadding/work/coldweight/nativeBF16/HCCLsumconditions. Currentv61G1/twoGEMMcontractsretained.',
            'CurrentA2Aperpeer12288M24means172032Bremote/rank;WA1024exchange43008B. StreamedA2A→WA→WB→RSanddirectattentionpackcomposeoncewithnewABI/resource/readinessconstraints.'
        ],
        'M15':['MixedBF16HC/FP32premixgather2→1newbytepack/multioutputABI;payloadunchanged,ordinary18qpack/unpackextraaccessP8. KeepFP32premix/allrawsnapshot/readiness/capturebankprefix-tailconsumers.'],
        'M18':['Threecheckpointauxgatherscanconditionalbatchafterdeferralretainingalllocalsources/allreaders;payloadunchanged,rank/checkpointlayoutpack/unpackorindirectmainGEMMandlostearlyoverlappriced.']
    }.items():
        assert old[mid]['branch_schema']=='SEMANTIC_BRANCHES_V1'
        inv[mid]=deepcopy(old[mid]['branches']);n=len(inv[mid])
        inv[mid]+=[row(mid,n+i+1,s)for i,s in enumerate(extra)]
    assert old['M07']['branch_schema']=='SEMANTIC_BRANCHES_V1'
    write(O/'M23-communication-and-cross-operator-branch-inventory-v66.json',dict(modules=inv,
        M23_new_semantic_families=18,retained_M11_M15_M18_families=sum(len(old[x]['branches'])for x in ['M11','M15','M18']),
        new_cross_module_families=4,M07_existing_branches_history_review_only=True,
        branch_counts={k:len(v)for k,v in inv.items()},whole_modules_closed=0,overall_complete=False))
    write(O/'communication-local-validation-history-v66.json',dict(resolved_local_attempts=[
        dict(status='error',reason='Initialconnection-onlyComputejoinhad2graphreplaycandidates;qualifiedbyexactnativeinterval+operator,notnearesttime. Noartifactwritten.'),
        dict(status='error',reason='InitialnonComputeCOMMcount3192assumedone-to-oneScheduleTASKbyconnection+interval;op134hasonlyNOTIFY_RECORD/WAITwiththesameconnection. Removedinventedbijection;3192opsand3192Scheduledefinitionsseparate,all14720COMMstillretained. Noartifactwritten.'),
        dict(status='error',reason='Initial42assertwasappliedtogroup-awarekey;42onlytype/dtype/P/countwithoutgroup. Actual61grouplabelkeysand76semantic/grouplabelkeysretained,notmerged. Costassertfailedbeforeartifactwrite.')],
        resolved_by='32rawDBhashes,14720COMMops/557688internalrecords/11528Computejoins/3192otherops/61models42typedcombinationsand20variantsallconserved.',
        target_or_server_failure=False,overall_complete=False))
    print(dict(regimes=17,reviewed_total=[1015,201431],pending_total=[1067,85519],new_M23_families=18,new_cross_module_families=4,remaining_by_module=dict(pending_by_module)))

if __name__=='__main__':main()
