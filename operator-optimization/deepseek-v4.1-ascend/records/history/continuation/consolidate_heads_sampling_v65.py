"""Reconcile operator-reviewed regimes and retain explicit semantic branches/gaps."""
from copy import deepcopy
from collections import Counter
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    assert not (O/'operator-analysis-coverage-ledger-v65.json').exists()
    p=read(O/'heads-sampling-regime-models-v65.json')
    cost=read(O/'heads-sampling-fusion-costs-v65.json')
    variants=read(O/'heads-sampling-fusion-variants-v65.json')
    assert len(variants['variants'])==28
    assert sum(x['groups'] for x in cost['separate_draft_helper_collapse_models'])==104
    c=deepcopy(read(O/'operator-analysis-coverage-ledger-v64.json'))
    rows={r['regime_id']:r for r in c['regimes']}
    e=['heads-sampling-regime-models-v65.json','heads-sampling-native-chains-v65.json',
       'heads-sampling-operator-contract-v65.json','heads-sampling-fusion-costs-v65.json',
       'heads-sampling-fusion-variants-v65.json']
    for r in p['regimes']:
        x=rows[r['regime_id']]
        assert x['state'].startswith('REGIME_') and x['executions']==r['executions']
        assert x['module_candidates']==r['module_candidates'] and r['consumer_fusion_review_complete_in_this_turn']
        assert r['work_model'] and len(r['native_TASK_keys'])==r['executions']
        x.update(state='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED')
        x['scoped_conditional_review_evidence']+=e
    reviewed=[r for r in rows.values() if r['state'].startswith('SCOPED_')]
    pending=[r for r in rows.values() if r['state'].startswith('REGIME_')]
    assert len(reviewed)==998 and sum(r['executions'] for r in reviewed)==189903
    assert len(pending)==1084 and sum(r['executions'] for r in pending)==97047
    assert len(rows)==2082 and sum(r['executions'] for r in rows.values())==286950
    assert all(not r['state'].startswith('GEOMETRY_') for r in rows.values())
    bymodule=Counter('+'.join(r['module_candidates']) for r in p['regimes'])
    assert bymodule=={'M15':42,'M16':17,'M17':33,'M20':30,'M24':35,'M04+M16':9}
    c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v64.json',
        scoped_conditional_review_regimes=998,scoped_conditional_review_executions=189903,
        remaining_regime_reconciliation_count=1084,remaining_execution_reconciliation_count=97047,
        remaining_regimes_by_operator=dict(Counter(r['operator'] for r in pending)),
        remaining_executions_by_operator={op:sum(r['executions'] for r in pending if r['operator']==op) for op in sorted(set(r['operator'] for r in pending))},
        newly_reconciled_regimes=166,newly_reconciled_executions=9792,
        new_pending_regimes_reconciled=166,new_pending_executions_reconciled=9792,
        prior_geometry_promoted_to_role_conditional_review=0,prior_geometry_executions_promoted=0)
    assert c['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    assert c['explicit_profile_insufficient_regimes']==3 and c['explicit_profile_insufficient_executions']==960
    write(O/'operator-analysis-coverage-ledger-v65.json',c)
    write(O/'operator-regime-refinement-v65.json',dict(
        refined_regimes=[dict(regime_id=r['regime_id'],executions=r['executions'],module_candidates=r['module_candidates'],
            consumer_fusion_review_complete_in_this_turn=True,physical_numeric_ceiling_proved=False,
            scope='Operator-specificwork/valueIOandconsumer/rounding/dependency/partitionconditions;notwholemodelbranchclosure.') for r in p['regimes']],
        reviewed_regimes=166,executions=9792,regimes_by_module=dict(bymodule),
        ambiguity_retained=dict(module_candidates=['M04','M16'],regimes=9,executions=176,global_ambiguous_executions=3136),
        implementation_absence_alone_not_unfinished_conditional_analysis=True,whole_modules_closed=0,overall_complete=False))
    inventory={}
    def add(mid,items):
        inventory[mid]=[dict(id=f'{mid}-OP-{i:02d}',branch=s,
            state='SCOPED_OPERATOR_CONDITIONAL_WORK_IO_FUSION_REVIEWED;WHOLE_MODULE_OPEN',
            evidence=e+['module-phase-branch-matrix-v64.json'],
            gap='Actualvalidrows/indices/runtimeflags/aliases/values/parameterandstate-generations,allconsumers/nativeprecisionandphysicaltypedcapacityremainexplicitconditions. Thissemanticinventorydoesnotproveallunobservedapplicablemodelbranchescovered.',
            scope='Fixedcaseoperatorchoices;oldchronologicalrecordsretainedinhistory. Noindependentframeworkresearch.') for i,s in enumerate(items,1)]
    add('M15',[
        '104targettails:rawHCsnapshot→FLOATcast→weightedMul→HCsum→BF16round→RMSBF16. Preservetworoundingbarriers/gamma/eps/rstd;76NHprivateinterfacebudget.',
        '104separatedraft/helper4TASKcollapses:72NHprivateinterfacebudget;consumerrowdomainnotpairedtotargetRMSbyshape. Normalizedhead/rawconfidence/HCstateconsumersdistinct.',
        'RawMTPsnapshotlivevsdead:currentlypositive104copies,allsnapshotobserversrequired;directAllGatheroutnewABImayavoid16NHcopywithcapture/alias/prefix/tailgates.',
        'SPbufferNonealreadylocalcollapse→collapsedAG;localnormbeforeAGnewconditionalchoice. BufferlivepreserverawHCAGandpricemorecommunicationifalsoHAG.',
        'AllactivecontentreaderorderedrowunionUbeforetailvsheadselectionalreadybeforeLMGEMM;promptlogprobflagsunknown,duplicates/bounds/lateIandSP8max(U_p)padding.',
        'FinalRMS+headtiles:full5120statistic/BF16normalizedvaluesretained;privateoutputmayfoldbutrepeatedVtiles/inputloads/resourcepressureandothernormalizedreaderspriced.',
        'Target/draft/SP/nonSP/None/empty/packing/CPandauxiliaryconsumerbranches:knownv31/v35contractsretained,actualapplicabilityandfullreaderclosureopen.'
    ])
    add('M16',[
        '192TP8baseBF16GEMMs5120→16160perrank;8shape-specifichead/Markovreferencecostmodels. Cold165.48MBbaseweightpercardnotrow-proportional.',
        '520replicatedBF16MarkovGEMMs256→129280,B2/B4;66.19MBW2,5stepcold-firstcachelambda1..5/warmHBMlambda0,L2separate.',
        'BasevocabAG→rankmajorTranspose:4MVinterfacebudgetrequiresnewdirectlayoutconsumer;notordinary2Dviewornativecollectivefusionproof.',
        'Roundedheadlogits→softcapdivide/tanh/multiply→scale→validcolumns→globalfirstcolumnargmax→remap;monotonicroundingtiesprecluderawargmaxshortcut.',
        'GreedybestID-onlyprotocol:localprocessedbestandglobalkey/pair;49bitfinitekeypackingprovedmathematically,9.2modeINT64MAXunbound;twoINT32wordAGor2-stageARfallbackwithcost.',
        'Fullscore/probability/rawlogprob/entropy/recoveryreaders:distributednormalization/selectedscore/epsilonentropy/residualRNGprotocolseparatefrombestID.',
        '9ambiguousM04|M16regimes176Index/IndexCheckTASKs:orderedtargetheadrowselectionalreadyexists. Preserveambiguity,bounds/duplicates/currentrowmapandallsourcereaders.',
        'ShardedMarkovvsfullowner:compute/weightsand5serialglobalIDdependencies/newcollectives/fullscore/replicaconfidenceandstatehand-off;mutuallyexclusiveprotocolchoices.',
        'Combined(H+R)replicatedGEMM6xpercardwork;shardeddualaccumulator.75fromMarkovsharding,notoneunroundedaccumulator. PrebatchedBDbaseand5Bstepweightreusecomparison.',
        'Vocabpadding/additional/remapping/tiedweight/softcap/scale/LoRA/quantpostprocessor/runtimevalidityguards:physical129280/configmatchnotallflagsproof;nativechangedtilingprecisionopen.'
    ])
    add('M17',[
        '72positiveworkingraw-preservationclonesandgreedyrejectionchains;Tcapacity10/20/B2/4notactualdraftcuvalues/allgreedyproof.',
        'SourceIndexBF16→FLOATCast→workingclone→FLOATargmax:22TVvaluebaselinevs2TVpureIDread;20TVbudgetonlyifallraw/processors/entropy/probability/logprobreadersclosed.',
        'RawFLOATrecomputefromretainedBF16source/index:fullrawlaterneeded6TVregenerationv8TVclone=net2TV,not8TVsaved;selectedscalar/rowandentropyotherbranchesexplicit.',
        'Allgreedyearlyreturnvsrandom/mixedrows/zeroDraft/forcedbonus/rawlogprobandepsilonentropybranches;temp0doesnotdeactivateeveryinternalreader.',
        'CurrentgreedyTritonacceptedprefix/firstreject/bonuspartialwrite;FillB×6-1requiredforsuffixandnon-greedyrows. Newfullwritefusedwriterconditional.',
        'Syntheticacceptanceuniform/rates/negativeIDguardandBONUSrenew;preservecurrentdraworder,zeroDraftseededconsumptionandoutputdtypeimplicitcasts.',
        'Randomrejection/residualrecoveryfullV,globalRNGQindex/defaultvsseeded/streamconsumptionandprobabilitymasssemantics;independentrankRNGnotreplacement.',
        'Lazyargmax/recoveryprefixvsseparatelylazyheadGEMM:all-firstrejectF/6withonecoldweight,allaccept6dependencywavescanreload6x;cache/tiling/RNG/fullscoreconsumers.',
        'Penalty/temperature/topKtopP/minP/allowed/badword/processors/sentinel/logproberrorsandfinitetorchfallback:existingv28/v33/v43-v54evidenceretained;actualguardsneededforoperatorprune,notAPIcatalog.',
        'Selectedacceptedoutputs/INT64validcounts/INT32tokensandasyncconsumerpublication;literal(x!=-1)&(x<V),including<-1unlessvalid-domainproof. Statefusionrequiresowner/generationclosure.'
    ])
    add('M20',[
        '520Markovsteps:onepreembeddingIDSlice,Gather,fullW2GEMM,3scoreSlices,Add,2ViewCopy,Argmax,Cast. 10TASKpostembeddingchainincludingM16GEMM;4Sliceswholeposition.',
        'BF16baseandbias/postprocess→BF16Add→argmax→map→nextIDserialdependency;24BV+24Blogicalprivateinterfacebudgetnotfullscorebranch.',
        'Selectedw1embeddingread4BR+8BvalueIO,notwhole66MBtable. RetainT[:,:D]embeddingsincludingseedforrawhiddenconfidence,2BDRbuffer.',
        '104draftheadRMS10/20rowdomainsand208separateHCcollapsegroupsacrossM15;headnormalizedinputvsrawconfidence/HCstateallreadersdistinct.',
        'FullreplicatedvsW2vocabshardvsownerrecurrence:5serialIDresolutions,baseownership,newcollectives/scoreobservers/confidencebudgetstatebroadcastcost.',
        'VersionedpostprocessedBF16biasmemo:token/W1/W2/postprocessor/nativeaccum/rowposition/shapegeneration,missuniqueIDsandcapacity/hitunknown;B1cachefillnotautomaticallyB2/B4nativeequal.',
        'Fulltransitiontable33.43GBreplica/4.178GBshard,8.557Tprecomputeops/outputcost;notfreefit/amortizedgain. ExistinggatheredcandidateAPItruncatesproposal,notexactmemo.',
        'Exactcertifiedblockprune:505blocks/256vocab,2020Bnormboundmetadata;outwardnativeaccum/round/postprocesserrorandstrictupper<best,nonfinitefallback. BaseTopK-alonecounterexample.',
        'ConfidenceFP32(H5120+R256)→1sigmoidusesrawhiddenandorderedseedembeddings;sourceoptionalbranchand0profileinstancesnotfalseflagproof.',
        'GlobaldynamicconfidenceTopK/histogramandperiodicbudgetresolvebeforeprefixprune;orderedperrequestprefix/metadata/RNG;noncausaldraftdecoderfuturequeryrowsnotdead.',
        'Draft-vocabmapping/D1/parallelvsnonparallel/negativeorinvalidID/Noneandemptyhelperbranches:previousv32/v36contractsretained,unobservedapplicabilityandnativeprecisionremainopen.'
    ])
    add('M24',[
        '72batchcorrectioncalls17TASKs:clamp→3gathers+checks→participation→INT32validcountadd→computedSelect→accepted64Select→INT32dest;215Nconventionalvaluebaseline.',
        'WholeNsingleprogramsnapshotbeforeANYwrite40Nread/store;175NbudgetB2/4. Naiveper-rowinplacegather-maprace[1,0]gives[21,22]vs[21,11];allaliases/readers/streamgeneration.',
        'LargerNimmutableoldstate/doublebufferalternativecost/publication;INT32overflow/cast/bounds/error/currentprefixandoldacceptedelsevaluesretained.',
        'Acceptedoutput-1suffix/INT64validsummary/INT32desthandoff;producer-epiloguesummaryfusionandliteralvalidpredicatewithnegative<-1conditions.',
        'Requestremap/accepted-rejectedrollback/prevdraft/CPUoptimisticdriftandnewrequestdefaults;allold/newgenerationsandasyncconsumersretainedfromv38/v40.',
        'CrossiterationC2pool/overlap/pre-norm/KVcacheandSWAanchorstate:alreadyv62operatorcontracts,unchangedslotaddressnotvaluegeneration;allfutureobservers/acceptedprefixneeded.',
        'EngramINT64history/hash/cache/UVAhostrowownership/copyfullwriteandrequestgeneration:existingv62conditionalmodelsretained,actualbank/IDcapacity/nativealiasgatesopen.',
        'IndexerKeydualcache/RoPEposition/slot-layergroup/draftcontextbankgenerationandperlayerpublisher:existingv60/v64conditionalmodelsretained,metadata/duplicateorderandqueryweightreadersremain.',
        'RawHCsnapshot/fullnormalizedhidden/3auxcheckpointbanksandMarkovbiasmemo:physicaladdress/captureequalitynotsemanticreuse;lastreader/weight/config/rowgenerationinvalidate.',
        'Expertload/placementremap/MC2receiverstateandparameter-generation:existingv56receiver/ownershipcontractsretained;onlyoperatorinputmapping/collectivepublicationrelevant,frameworkcallbacksnotindependentgoal.',
        'Outputlifecycle/format/sentinelguardsareconsideredonlywhentheycontroloperatorfullwrite/prune/RNGorstatepublication. Oldchronological32branchhistoryarchived,notdeclaredallconditionsclosed.'
    ])
    write(O/'M15-M16-M17-M20-M24-operator-branch-inventory-v65.json',dict(modules=inventory,
        semantic_family_counts={k:len(v) for k,v in inventory.items()},total_families=sum(map(len,inventory.values())),
        retained_prior_scope='v64chronologicalinventoriesarchivedbyfinalizer;M24crossmodelstateandM17allalgorithmbranchesremainexplicit,notreplacedbyonlycurrent17TASK/greedykernel.',
        M04_M16_ambiguity_not_resolved=True,whole_modules_closed=0,overall_complete=False))
    write(O/'heads-sampling-local-validation-history-v65.json',dict(resolved_local_attempts=[dict(
        status='error',attempts=2,reason='InitialMarkovpostembeddingchainexpected2scoreSlices;native520groupscontain3scoreSlicesplusonepreembeddingIDSlice. Assertionfailedbeforeanymodelartifactwrite;diagnosticrerunconfirmedsamecount. Expected3corrected;all52010TASKchainsnowconserved.')],
        model_design_review='InitialdesignmathPASSthen104helpercollapsebudget/warmHBMlambda0clarifiedandunsupportedhiddenlayoutvariantremoved;before-designsnapshotretained. No device/code implementation.',
        target_or_server_failure=False,overall_complete=False))
    print(dict(new_regimes=166,new_exec=9792,reviewed_regimes=998,reviewed_exec=189903,pending_regimes=1084,pending_exec=97047,
        semantic_families=sum(map(len,inventory.values())),ambiguity_retained=3136,MemSet_pending=960))

if __name__=='__main__':main()
