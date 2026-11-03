"""Reconcile reviewed operator families while retaining genuine missing operands/readers."""
from copy import deepcopy
from collections import Counter
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    assert not (O/'operator-analysis-coverage-ledger-v67.json').exists()
    p=read(O/'embedding-initialization-regime-models-v67.json')
    done=[r for r in p['regimes']if r['consumer_fusion_review_complete_in_this_turn']]
    unknown=[r for r in p['regimes']if not r['consumer_fusion_review_complete_in_this_turn']]
    assert (len(done),sum(r['executions']for r in done))==(230,7224)
    assert (len(unknown),sum(r['executions']for r in unknown))==(6,392)
    c=deepcopy(read(O/'operator-analysis-coverage-ledger-v66.json'));rows={r['regime_id']:r for r in c['regimes']}
    e=['embedding-initialization-regime-models-v67.json','embedding-initialization-native-chains-v67.json',
        'embedding-initialization-fusion-costs-v67.json','embedding-initialization-operator-contract-v67.json',
        'embedding-initialization-fusion-variants-v67.json','initialization-profile-insufficiency-v67.json']
    for r in done:
        x=rows[r['regime_id']];assert x['state'].startswith('REGIME_') and x['executions']==r['executions']
        x.update(state='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED');x['scoped_conditional_review_evidence']+=e[:5]
    for r in unknown:
        x=rows[r['regime_id']];assert x['state'].startswith('REGIME_')
        if r['operator']=='MemSet':
            x['state']='REGIME_PROFILE_INSUFFICIENT_MEMSET_OPERANDS_AND_WORK_SCOPE_PENDING'
        else:x['state']='REGIME_BOOL_PREFIX_FIELD_AND_CONSUMER_IDENTITY_PENDING'
        x['explicit_pending_evidence']=e[0:2]+[e[3],e[5]]
    rev=[r for r in rows.values()if r['state'].startswith('SCOPED_')];pending=[r for r in rows.values()if r['state'].startswith('REGIME_')]
    assert len(rev)==1245 and sum(r['executions']for r in rev)==208655
    assert len(pending)==837 and sum(r['executions']for r in pending)==78295
    assert len(rev)+len(pending)==2082 and sum(r['executions']for r in rows.values())==286950
    assert Counter('|'.join(r['module_candidates'])for r in pending)=={'M21':828,'M19':4,'M12':3,'M05|M21':2}
    c.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v66.json',scoped_conditional_review_regimes=1245,
        scoped_conditional_review_executions=208655,remaining_regime_reconciliation_count=837,remaining_execution_reconciliation_count=78295,
        remaining_regimes_by_operator=dict(Counter(r['operator']for r in pending)),
        remaining_executions_by_operator={op:sum(r['executions']for r in pending if r['operator']==op)for op in sorted(set(r['operator']for r in pending))},
        newly_reconciled_regimes=230,newly_reconciled_executions=7224,new_pending_regimes_reconciled=230,new_pending_executions_reconciled=7224,
        prior_geometry_promoted_to_role_conditional_review=0,prior_geometry_executions_promoted=0,
        explicit_profile_insufficient_regimes=7,explicit_profile_insufficient_executions=1272,
        explicit_unresolved_BOOL_field_regimes=2,explicit_unresolved_BOOL_field_executions=80)
    assert c['unchanged_global_uncertainty']=={'ambiguous_module_assignment_executions':3136,'missing_PMU':849,'phase_unbound_graphs':16}
    write(O/'operator-analysis-coverage-ledger-v67.json',c)
    write(O/'operator-regime-refinement-v67.json',dict(refined_regimes=[dict(regime_id=r['regime_id'],executions=r['executions'],
        consumer_fusion_review_complete_in_this_turn=True,scope='Embedding/HCinit/draftSP/IDpositionselectedwork/valueIO/consumerconditions,notmodel/frameworkclosure.')for r in done],
        reviewed_regimes=230,executions=7224,pending_regimes=[dict(regime_id=r['regime_id'],executions=r['executions'],reason=r['work_model']['contract'])for r in unknown],
        new_MemSet_pending=[4,312],unresolved_prefix_BOOL=[2,80],total_MemSet_pending=[7,1272],whole_modules_closed=0,overall_complete=False))
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v66.json')['modules']}
    inv={}
    def row(mid,i,s,pending=False):return dict(id=f'{mid}-OP-{i:02d}',branch=s,
        state='PROFILE_INSUFFICIENT_OPERANDS_OR_CONSUMERS;PENDING'if pending else'SCOPED_OPERATOR_CONDITIONAL_WORK_IO_FUSION_REVIEWED;WHOLE_MODULE_OPEN',
        evidence=e+['module-phase-branch-matrix-v66.json'],
        gap='Actualvalidrows/indices/duplicates/aliases/allreaders/valueandstate-weight-configgeneration/nativeprecision/physicalVector-index-communication-memorycapacityandunobservedapplicablebranchesremainexplicit. Missingimplementationaloneisnotunfinishedconditionalanalysis.',
        scope='Operator-onlysemanticinventory;oldchronologicalhistoryarchived,notframeworkAPI/lifecycleaudit.')
    def add(mid,ss,pending_rows=()):inv[mid]=[row(mid,i+1,s,i+1 in pending_rows)for i,s in enumerate(ss)]
    add('M04',[
        'AsyncinputIDcommon-request0/directprefixcopy/permutedfallbackbranches:current72sample+draftselectedgather-scattergroups;source/oldIDbanksandrequestgeneration,untouchedrowsretained.',
        'SampleIndexCheck/Index/LinearIndex/Scatterfour-stage48K→24Kreference;sourceadvanced-indexnegativewrapvsdestscatternegativeerror/capacity8192,notwhole8192traffic.',
        'DraftSliceB×5fromBmax×6/Cast64→32/IndexCheck/Index/Linear/Scatter76K→28KwhenBD=K;logicalBDboundsandphysicalstride6/offset1/overflow/errorpreserved.',
        'Onecombinedsample/draftstoreonlywithdisjointdestprooforpriority/barrier;draftAFTERsamplecross-overlapwins,within-scatterduplicatepolicy/allsourcealiasoldvaluesretained.',
        'Sanitizeexactplaceholder-1thenclamp_min0fullprefix:22N→8Nonlysameprefix/nointerveningreaders. AllrawIDbankEngram/router/otherconsumersmustobservesamecleanedvalues.',
        'PartialnewIDscatterdoesnotcleanuntouchedprefix/pad;CPUinputcopy/common-prefix/oldvalidrowsandallcapturebanklastreadersmustparticipateinfullprefixcleaning.',
        'Positioncomputed32gather+cast64+query64add:88groups60N→28Nreference,sourcecapacity4/negativewrap/errors/intwidth/overflowandoldstateidentity.',
        'Seq_lensprefixINT32addandtail0/query_start_loctail-1/positionpadINT64zero:distinctB/N/capacitydomains;multioutputonlysamecomputedgeneration/readiness.',
        'Logitsindicesquery_start_loc[1:B+1]-1non-specbranch:16INT32Subpositivesretainnon-spec/samplingselectionandotherprefixreaders,notsharedspecmetadataformula.',
        '24persistentinputs_embedsN6400/6528/6668copies:4NHlogicalbudgetonlynewembedding/reductionoutABIwithMM/sentinel/is_token_ids/prompt/Captureprefix-tail/readers/lastreadercontracts.',
        'MROPE/ordinarypositions,firstPP/nonfirstPP/multimodal/prompt/precomputedembeddingandspec/non-specbranches:existingv31/v35/v37-v55contractsretained,onlyoperatorinput/readinessapplicabilityquestionsremain.',
        'Batchacceptancecorrection/oldcomputed/IDandpositioncrossiterationcache:priorv65snapshotprotocolandgeneration keysretained;shape/addressnotvalueidentity.',
        'Inputmetadata/slot/packingCP-DPownersandCPUpublicationremainingM21familiesarenotclosedbythese59M04regimes;oldframeworkchronologyarchivedwithoutclaimingindependentaudit.'
    ])
    add('M05',[
        '272standardBF16vocabmask/gather/maskedzero14TASKgroups:113regimes3832TASKincluding24SPclones. Currentcommenttorch.compilefusesnotobservednativefusionproof.',
        'No-addedrangeINT64validmask/offset/3INT32widen2BOOLpromotionscomposeswithlookup;sourceargumentreadonlyCSEnotshapeonly. Addedrangeoffset/ORbranchretainedseparately.',
        'InvalidvocabIDsmaprow0thenmaskedpositive0;newfullywritingmaskedlookupcanreadonlyUownedrowswhilekeepingerror/index/nonfinite/cast/outputzero/nativequantmethod.',
        'Table16160×5120BF16=165.48MBshardnotpercallIO;row0invalidN-Urefsatmostoneextrauniquerow10240B,cachemeansnotHBMP×savings.',
        'STANDARDvocabARvsnewowner-valuegather:balancedfixedAGpayload.5AR,one-ownerfixedpad4ARandraggedinjectionstraggler/metadata;SUMsignedzero/NaNprecisionnotautomaticcopy.',
        'Existingtoken_exchange/finegrainedDP/PCPxTPplans,capturecapacity/staticoutbanks/TP1replicatedandquantizedlookupbranches:currentstandardprotocolnotinterchangeablebygroupname.',
        'Embeddingdirectpersistentout/allreduceoutputbankcanavoid24inputs_embedsGPUcopiesonlyMM/is_token_ids/sentinel/prompt/firstPP/capture/lastreaders/prefix-tailcomplete.',
        'TargetSP2Dclone24copy-mapviews:currentfunctionalcustomopreturnsindependentstorage;newreadonlyview/directproducerneedsalias/lifetime/Tile-copy-before-parent-mutation,ID/maskviewsseparate.',
        'GraphprefixBOOL24InplaceZero80instancesM05|M21fieldidentity/readersunknown:24Bstoreworkknownbutfusiondecisionpending,ambiguitynotreassigned.',
        'Imagecondition/wrapper/inputs_embedsversusinputIDs/multimodalreplacements/dtypeplans:priorfixedcaseguardsretainedasoperatorconsumerboundaries,notindependentframeworkcatalog.'
    ],pending_rows=(9,))
    add('M06',[
        '104targetlocalHC4initializationsTile+4premixTASK:23regimes520TASK. Targetalreadyshards2DBEFORErepeat;draftreorderingisnotnewtargetbenefit.',
        'FP32onehotpremixzero+column0=1:28n+8→16nfullwrite;readonlyconstantbank/prefix/newPreconstantABIalternativeswithallinputmutation/capture/readinessgates.',
        'FirstTileandHcPreshare2DinputloadswhileemittingfullHCresidual:8nHreferencebudget,firstgeneration/nointerveningEngram/aliasmutationallreaders.',
        'VirtualHC4untilfirstPosttwo-consumer22nHreferencebudgetrequiresnewPre/Post/residualABIandallinitialreaders;full4Hnorm/HF32projection/nativeproductsandBF16outputsremain.',
        'Folding4channelweights/normandreplacingonehotweightedYbyx0notunconditional:FP32reassociationwitness0vs1,0*InfNaN/signedzero;actualHF32policyretained.',
        'Existingv58/v59HcPreABFLOAT/HF32coefficientsandA2Gram/stage2diagonal/reductiondomain;genericBF16GEMMcapacitydoesnotapply.',
        'AllobservedPrehaveexternalprevMix;currentBF16weightedYindependentcurrentnewcoeffbranchbutnewPre/Post/combremainlive. OtheroptionalpreMix/skipguardsconditional,notcapture-attested.',
        'Wholeoperatorstage1/stage2globalSync/currentcompletionbarrier:earlyYsplitneedsnewoutputhandoff/events/resourceprice;dataindependentdoesnotprovefreeoverlap.',
        'HcPreYBF16beforeRMS;existingdualRMSNormCastreturnsBF16andwidenedroundedBF16,notunroundedFLOATnorm. Expert/shared/index/attentionallreaderroundsretained.',
        'HcPostBF16x/residual→FP32increasing-iMul/Add→RINTBF16freshoutput;notFMA/reassociatedorunroundedproducerreplacement.',
        'PostoutputfeedsnextPreANDnextPostresidual/aux/Engram/tail;post-precompositioncannotdropallresidualmaterialization.',
        'Routing/shared/routedmerge/attention/FFNtwoHCboundarieskeepallcoefficients/scaling/streams/readiness;existingv23/v58/v59contractsnotreopenedbyinitializationreview.',
        'AuxHCmeanvsweightedYandHcPresharedinputat37/38/39:existingv64distinctmeanround/readiness/sourcegenerationremains,notonehotconstantreusebeyondfirstlayer.',
        'SP/nonSP/None/empty/dummyandconfigHCdimensionbranches:prefixonehotconstantshape/configversion/allinitialreadersneeded;allphysicalVector/HF32/memoryboundsstillopen.'
    ])
    add('M19',[
        '104draftinitializations15TASKeach:globalTile/3Pads/3Casts/2ZerosLike/3MemSet/premixFill-Broadcast-ViewCopy. Sourceorderedpopulationnotall312attentioncallsfirstentries.',
        'PurecopySP-before-HC4repeat:currentN10/20→A16/24→local2/3;new2Dpad/shardthenlocalTilecutsN20logicalinterface2222080B/cardwithoutalgebrachange.',
        'DirectlocalinitializerreadUownedrowsandemitHC4,ID0pad,BOOLpadTrueandFP32onehot;providedprefixmaskvalueskept. NewABI/capture/banklifetimeandunknownMemSetprotocolseparate.',
        '104sp_padding_mask149new_zerospositivesbindNonebranchfalseprefix;provided-maskbranchcancontainTrueprefixandmustnotreplacebyall-false.',
        '4ConstantPadNdMemSetregimes312callsN/Anoaddress/length/controlscope:caller/Padextentinsufficient;staypendingevenwhennewsemanticinitializerproposed.',
        'Initialpremix4TASKandreadonlyconstant/newPreABIalternativesretainFLOAT4channels/currentY/nonfinite/nativeHcPrenumerics;notalllaterpremixesconstant.',
        'Draft3decoderlayersSWA/CPranklocalqueriesandcanonicalglobalKV:existingv35v34contractsretainno-queryglobalcache-andoutputwork,notpruneallbylocalU0.',
        'Local2/3QAversusglobal10/20KVand5120normdomains,CPstart/SPowner/mask/slotpositions:naivesame-Mjointprojectionexcluded,existingtypedprojection/repartitionmodelsretained.',
        'OutputinverseRoPE/headrestore/WAG1/WB/RSincludingnoQrankcontributions:existingv61/v66allreader/padding/nativeBF16SUM/weightscostsremain.',
        'DraftfinalHCcollapsebeforefinalSPgatheralreadybaseline;rawhiddenforconfidencevsnormalizedheadandcontextKVinputsseparate,existingv31/v35/v65contracts.',
        'MetadataNonewarmupzero-returnvsmetadata-presentlocal-no-querybranchesdistinct;currentdummyoptimizedonlyguardedunusedcontent,notC8requestgain.',
        'Contextprecompute/querydecoderusecurrentlayerWKVwithdifferentcontent/positions/slotgeneration;conditionalrow-batchreusevsinvalidsame-slotvalueCSEretained.',
        'HCpre/post/FFN/routed/shared/sourceparameterandcrossiterationaccepted-statebranches:previousv35/v56-v65conditionsretained;initializationcoverageisnotdecoderallbranchclosure.'
    ],pending_rows=(5,))
    assert old['M10']['branch_schema']=='SEMANTIC_BRANCHES_V1'
    inv['M10']=deepcopy(old['M10']['branches']);n=len(inv['M10'])
    inv['M10']+=[row('M10',n+1,'48dummymetadataNoneDsaoutput.zero_positives20×5120:shape/fulloutputzeroobservercontract;virtualzeroHcPost/newABIcanavoidplaneonlywithnonfinite/round/allreaderandcollectivegates.'),
        row('M10',n+2,'UniformmetadataNonecanconditionalpruneunusedattentioninputnorm/AGcontentbutkeepHCcoeff/residual/FFN;metadata-presentlocal-no-querymuststillglobalKV/headrestore/output. DummyworknotC8decodegain.')]
    write(O/'M04-M05-M06-M19-M10-operator-branch-inventory-v67.json',dict(modules=inv,
        converted_legacy_modules=['M04','M05','M06','M19'],M10_retained_families=n,M10_new_families=2,
        new_or_reconciled_family_count=sum(len(v)for k,v in inv.items()if k!='M10')+2,
        total_current_family_count=sum(map(len,inv.values())),branch_counts={k:len(v)for k,v in inv.items()},
        explicit_pending_family_modules=['M05','M19'],whole_modules_closed=0,overall_complete=False))
    write(O/'initialization-analysis-validation-history-v67.json',dict(core_native_chain_and_model_attempts='AllcoreextractandcostgenerationassertionsPASSonfirstattempt;exploratorymissingfilename/hashsearchesdidnotmutateanalysisstate.',
        verified=dict(embedding_groups=272,target_init=104,draft_init=104,sample_ID=72,draft_ID=72,positions=88,
            selected_regimes=236,selected_exec=7616,reviewed=[230,7224],pending=[6,392]),
        pending_is_not_analysis_failure='Explicitlyretained4N/AMemSet312and2BOOLfield80,notpromotedbyreport/modelgeneration.',overall_complete=False))
    print(dict(reviewed=[1245,208655],pending=[837,78295],new_review=[230,7224],total_MemSet_pending=[7,1272],ambiguous_BOOL_pending=[2,80],
        semantic_families=sum(map(len,inv.values())),new_or_reconciled_families=sum(len(v)for k,v in inv.items()if k!='M10')+2))

if __name__=='__main__':main()
