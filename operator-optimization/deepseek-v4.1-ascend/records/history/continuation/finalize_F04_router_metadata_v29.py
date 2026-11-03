from copy import deepcopy
from datetime import datetime,timezone,timedelta
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl
from finalize_F33_consumers_v27 import graph_check


def packet(label,embedded=False):
    r=read(N/(label+'.json'));assert r['returncode']==0,label
    x=json.loads(r['stdout'])
    if embedded:
        b=(json.dumps(x['result'],ensure_ascii=False,separators=(',',':'))+'\n').encode();assert hashlib.sha256(b).hexdigest()==x['result_sha256']
    return x


def main():
    assert not(O/'REVIEW-CHECKPOINT-v29.json').exists(),'refuse duplicate finalization'
    now=datetime.now(timezone.utc);stamp=now.isoformat()
    p=packet('F04-profile166-v29');s=packet('F04-source166-v29',True);g=packet('F04-callers166-v29-v2')
    k=packet('F04-kernel-gate166-v29',True);a=packet('F04-ID-audit166-v29');mc=packet('F04-MC2-alignment166-v29',True)
    rb=packet('F04-router-branch166-v29');helper=read(O/'F04-router-input-selection-v29.json')
    assert rb['summary']['branch_API_counts']=={'DIRECT_PRECAST_FP32_LINEAR_408':1272}
    assert helper['sha256']=='2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4'
    assert read(N/'F04-callers166-v29.json')['returncode']==1
    assert p['summary']['RmsNormCast_TASKs']==4472 and p['summary']['source_link_states']=={'NO_NATIVE_API_LINK':3200,'EXACT_NATIVE_API':1272}
    assert p['summary']['missing_or_ambiguous_norm_blocks']==[]
    assert [r['TASKs']for r in p['native_chains']]==[48,264,960]
    assert g['summary']['gating_TASKs']==4472 and g['summary']['native_linked_gating_TASKs']==1272
    assert a['summary']['exact_ID_quads']==3200 and a['summary']['quad_TASKs']==12800 and a['summary']['native_ID_prepare_TASKs']==5088
    refs=[dict(label=label,receipt=str(N/(label+'.json')),full_result=x['full_result'],sha256=x['result_sha256'])for label,x in[
        ('F04-profile166-v29',p),('F04-source166-v29',s),('F04-callers166-v29-v2',g),('F04-kernel-gate166-v29',k),('F04-ID-audit166-v29',a),('F04-MC2-alignment166-v29',mc),('F04-router-branch166-v29',rb)]]
    write(O/'F04-actual-router-input-branch-v29.json',dict(generated_utc=stamp,summary=rb['summary'],native_headers=rb['canonical_headers'],helper_source=str(O/'F04-router-input-selection-v29.json'),scope='Native1272callspositivelyreach408hasweight_fp32branch;graphrolesremainSOURCE_CONSISTENT,notfabricatednativePythonstacks.',overall_complete=False))
    write(O/'F04-current-profile-and-fanout-v29.json',dict(generated_utc=stamp,evidence_grade='MEASURED_WITH_SOURCE_CONSISTENT_GRAPH_ROLES',
        norm=p['summary'],norm_native_chains=p['native_chains'],gating=g['summary'],gating_native_chains=g['native_chain_headers'],
        ID_preparation=a['summary'],evidence_refs=refs,whole_module_closed=False,overall_complete=False))
    kernel=next(f for f in k['result']['files']if f['path'].endswith('/op_kernel/rms_norm_cast.h'))
    assert kernel['sha256']=='5a05b11cce4a727b8c48aec4fe764d854c71c5a8ba072dad315a2acec342d472'
    source='\n'.join(kernel['source']);assert 'Cast(x_fp32, x_local, RoundMode::CAST_NONE, num_col_)'in source
    source_files={f['path']:f['sha256']for x in[s,k,mc]for f in x['result']['files']}
    source_files[a['target_model_input_ID_source']['path']]=a['target_model_input_ID_source']['sha256']
    source_files[helper['actual_server_path']]=helper['sha256']
    contract=dict(generated_utc=stamp,evidence_grades=['MEASURED','DOCUMENTED','DERIVED','UNKNOWN'],source_files=source_files,evidence_refs=refs,
        reviewed_facts=[
            'ActualnativeDeviceOperator.rms_norm_cast frame72 liesinsidehardwareRMS_NORM_CAST capability/custom-op-presence guards;1272nativecalls match thisexisting fusedboundary. Fallbackreturnsordinarynormalized andnormalized.float.',
            'ReviewedkernelBF16branch multipliesFP32normalizedinputbyFP32gamma,rounds toBF16 at:115,thenwidensroundedx_local toFP32at:122. WritesBF16yat:128 andFP32y_fp32at:129. The FP32plane isnotanunroundedRMSintermediate.',
            'Torchadapterallocatesbothsame-shapeplanes andcallsaclnnRmsNormCast;schema returns(y,y_fp32). Reviewedrepository/build/packagecopies sharethekernelSHA,whichisnotanexecution->loadedbinaryhashproof.',
            'Decoder912 passesbothplanes totheMLP;DeepseekV41MoE284-288 choosesFP32routerinputwhenpresentandpassesBF16hiddenstates toexperts. Currentnativecallers useinternalrouter285;externalrouter293 andlegacytuple/SP alternativesremain.',
            'Reusev21runner439-540 andreadexacthelpers389-410:onlygateswithweight_fp32selecttheprovidedFP32routerplane;othergatesreturnhidden_statesandtheirownforward. SavednativefullcallerspositivelyreachdirectF.linearat408forall1272selectedrouterAPIs,split960target/264draftproposal/48dummy. Do notinferconsumptionmerelybecauseanargumentwaspassed.',
            'Underthisactualdirectrouterbranch,FP32routerinputfeedsF.linearwithgate.weight_fp32,whileBF16hiddenstates feedsroutedandsharedpaths. CurrentMC2 disablesexistingsharedmultistream;thisrounddoesnotaddafreeoverlapgain.',
            'Currentgatingcaller204 selects sqrtsoftplus/vision-hash custombranch,withbias/optionalhashmap/image-sentinelparameters,topkweights andIDs. It ignoresathirdoutput andsetsout_flag=False;typedfullscoreoutputshapeisnotproof itwaswritten/readasHBMtraffic.',
            'RouterIDpath163 castsIDstoINT64;169doescomm-specificpad/split;175conditionallyTPchunks;176replacesonly-1by0. OthernegativeimageIDsandlayer-specifichash/biasparametersremain.',
            'Targetforwardstoresinput_ids onlyatpre-loopSPshard1549 andpassestheinput_IDexpression tolayer1592. Thisdoesnotaloneproveallcallee/asyncaliasesareimmutable;stable-input-generationremainsanexplicitCSEcondition.',
            'MC2prepare252-314 padsbothBF16hidden/FP32routerinput andFalsemasktotheDP-widephysicalrowextent;IDpad327-348 useszero paddingandTPselection. SPfinalize323-324trimsroutedoutputtooriginalself.num_tokens.',
            'SourceexampleM=800,P/TP=816 pads16rowsandtrimsbackto800. Savednorm/ID-cast800 andgating/where816regimes areconsistentwiththisprotocol;noequal-count/nearest-timeactualcallpairingorpadded_num_tokenssnapshotisfabricated.'],
        source_scope_closed=['Existingfusednorm/castoutputrelationunderreviewedkernelcontract','DirectBF16/FP32FFNfanout','RouterIDnormalizationandMC2rowtransformcontract'],
        remaining=['Noexecution-levelbinaryhash/numericalparitycertificate. Do notchaseallcompileroutputsmerelytoregisterconditionalvariants.',
            'Actualsame-callrowmap,maskvalues,captureaddresses/eventgenerations andallfutureconsumersstillnotfullyrecorded.',
            'MoE-only-SP sequence_parallel_chunk internalswerenotexpandedhere;applytheintegercommutationonlywhenitszero-pad/row-selectioncontractisprovedorrelevantbranchinactive. GraphIDcast3/norm3issource-consistent,notafabricatedflagvalue.',
            'F04furtherkerneltiling/precision/rowpruning variantsneedtheirownconditionalcontracts;wholeM12/othermodulesremainopen.'],
        no_target_implementation_change=True,overall_complete=False)
    write(O/'F04-precision-routing-and-row-contract-v29.json',contract)
    # Elementary exact integer identities support the graph rewrite, not native performance/parity.
    def f(ids):return [0 if x==-1 else x for x in ids]
    ids=[-3,-2,-1,0,1,129279];plan=[5,2,0,4]
    transformed=[ids[j]for j in plan]+[0,0]
    assert f(transformed)==[f(ids)[j]for j in plan]+[0,0]
    assert f([-2,-1])==[-2,0] and f([-2,-1])!=[max(0,x)for x in[-2,-1]]
    assert f([-1,7])!=f([9,7])
    assert f(ids+[-1])!=f(ids)+[-1]
    math=dict(generated_utc=stamp,evidence_grade='DERIVED_IN_STATED_PURE_INTEGER_DOMAIN',
        operator='f(x)_j = INT64(x_j) if x_j != -1 else 0. INT32->INT64 isexact;othernegativeimageIDsretained.',
        pointwise_commutation_proof='Foranyshape-onlyrowselectionandzero-paddingP:atavalidoutputposition,f(Px)_j=f(x_map(j))=(P f(x))_j;atapaddingposition,f(0)=0. Hencef(Px)=P(fx). Compositionoftheseplanshasthesameidentity.',
        applicability='NamedMC2IDtransform useszero-padding,reshape andTProwselection. Retain eachlayerplan/collective/controloperation;not a proofforarbitrarytoken-value remapping orunaltered-1padding.',
        finite_illustrations=dict(ids=ids,selected_plan=plan,normalization_after_alignment=f(transformed),alignment_after_normalization=[f(ids)[j]for j in plan]+[0,0],
            negative_image_example=dict(source=[-2,-1],required=[-2,0],wrong_clamp=[0,0]),
            stale_generation_example=dict(same_address_old=[-1,7],same_address_new=[9,7],old_result=f([-1,7]),new_required=f([9,7])),
            minus_one_padding_counterexample=dict(normalize_after=f(ids+[-1]),unchanged_padding_after=f(ids)+[-1])),
        numerical_norm_contract='ForreviewedBF16kernel,y_fp32=FLOAT32(y_BF16). FusingroutermustpreservetheroundedBF16boundary;itneednotretainaseparateFP32globalplaneifnewconsumerwideningisprovedequivalent.',
        forbidden_shortcuts=['DonotreplaceFP32routeinputbyunroundednormalizationaccumulator.',
            'DonotcancelarowRMSscaleusingonlytopkrankinvariance;currentnonlinear sqrtsoftplus/bias/weightsmatter,andBF16roundingisnotlinear.',
            'SameIDshape/addressdoesnotprovesamevalues/generation;sametokenIDsdoesnotmeanhiddenactivations/normweightsaresameacrosslayers.'],
        native_validation=False,hardware_or_request_gain=None,overall_complete=False)
    write(O/'F04-F20-ID-algebra-and-generation-review-v29.json',math)
    f04_variants=[
        dict(id='F04-A',status='ALREADY_IMPLEMENTED_CURRENT_FUSED_BOUNDARY',change='RMSNormplusBF16/FLOAToutputproduction.',scope='4472native/graphinstances;reviewedsourcewidensroundedBF16. Notanewfusionorspeedup.'),
        dict(id='F04-B',status='CONDITIONAL_COMPOSITION_WITH_F20',change='AvoidseparateFP32globalplane byanexactrouterconsumerthatwidensBF16internally whileBF16expert/sharedplane remains.',requires='SameBF16rounding,FP32routerweight/accumulationandallliveconsumers;currentF.linearFLOATinterfacecannotjustreceiveadifferentdtype.'),
        dict(id='F04-C',status='CONDITIONAL_LARGER_SUBGRAPH',change='JointlyreviewHC-FFNpre/RMS/router/gatingboundaries orsharedquantizationfanout.',requires='PreserveallHCcoefficients/residuals,normalizationparameters,FP32gateweights/IDs,bothrouted/sharedconsumers andexistingprecision/readiness;overlapsF01/F20/F22.'),
        dict(id='F04-D',status='CONDITIONAL_SPLIT_REORGANIZATION',change='Reviewrow/featuretilingorvalid-row-awareexecution forsmallM.',requires='Fullrowreduction/rounding,actualvalidmask/padded-rowvalues,allshared/residual/auxconsumers andcross-coretimingcost;nohardwarecapacityorbenefitproof.'),
        dict(id='F04-E',status='EXCLUDED_AS_UNCONDITIONAL_REWRITE',change='Removeoneplane,skipBF16rounding,cancelnormscalefromrouting,orreusehiddennormacrosslayersbysameshape.',reason='Liveexpert/shared/routerconsumers,nonlinear/biasweightedrouting andchanginghiddenvaluesdistinguishthesechanges.')]
    write(O/'F04-existing-fusion-and-extensions-v29.json',dict(generated_utc=stamp,candidate_id='F04',variants=f04_variants,source_contract='F04-precision-routing-and-row-contract-v29.json',overall_complete=False))
    # Explicit 40-layer source mapping and two separate conditional CSE graphs.
    baseline=dict(nodes=[dict(id='input_IDs'),dict(id='row_plans')],edges=[])
    cast_cse=dict(nodes=[dict(id='input_IDs'),dict(id='row_plans'),dict(id='shared_cast')],edges=[dict(a='input_IDs',b='shared_cast',kind='EXACT_INT64_PROMOTION')])
    pointwise_cse=dict(nodes=[dict(id='input_IDs'),dict(id='row_plans'),dict(id='shared_routing_IDs')],edges=[dict(a='input_IDs',b='shared_routing_IDs',kind='F_NONMUTATING_GENERATION_LOCAL')])
    for layer in range(40):
        names=[f'L{layer}_cast',f'L{layer}_align',f'L{layer}_sanitize',f'L{layer}_gate']
        baseline['nodes'] += [dict(id=x)for x in names]
        baseline['edges'] += [dict(a=x,b=y,kind='DATA_OR_PLAN')for x,y in[('input_IDs',names[0]),(names[0],names[1]),('row_plans',names[1]),(names[1],names[2]),(names[2],names[3])]]
        cast_cse['nodes'] += [dict(id=x)for x in names[1:]]
        cast_cse['edges'] += [dict(a=x,b=y,kind='DATA_OR_PLAN')for x,y in[('shared_cast',names[1]),('row_plans',names[1]),(names[1],names[2]),(names[2],names[3])]]
        pointwise_cse['nodes'] += [dict(id=names[1]),dict(id=names[3])]
        pointwise_cse['edges'] += [dict(a=x,b=y,kind='DATA_OR_PLAN')for x,y in[('shared_routing_IDs',names[1]),('row_plans',names[1]),(names[1],names[3])]]
    for graph in[baseline,cast_cse,pointwise_cse]:graph_check(graph)
    f36_variants=[
        dict(id='F20-ID-A',status='CONDITIONAL_CROSS_LAYER_CSE',change='OneINT64castforstableforwardinput_IDgeneration;retainper-layeralignment/sanitization.',requires='Sameinputvalues/dtype/layoutandnomutation/concurrentgenerationchange;pad/selectionconsumersmustnotmodifysharedcastsource.'),
        dict(id='F20-ID-B',status='CONDITIONAL_CROSS_LAYER_CSE_WITH_DERIVED_COMMUTATION',change='Computef onceforroutinginputgeneration,thenkeepeachlayerMC2zero-pad/selectplanP_l.',requires='f(P_l x)=P_l f(x)underreviewedzero-padding/selectioncontract;rawIDsremainseparate,allper-layergatingparamsandcommprotocolsremain.'),
        dict(id='F20-ID-C',status='CONDITIONAL_NEW_CONSUMER_INTERFACE',change='AbsorbIDnormalizationintoHashTopK oritsinputpublication.',requires='ExactINT32/INT64 ABI,-1-onlymapping,vision/hashsemantics/errors,allconsumersandgenerationupdates. CurrentAPIconstraintsarenotuniversalno-opportunityproof.'),
        dict(id='F20-ID-D',status='CONDITIONAL_CROSS_ITERATION_VARIANT',change='ReusepreparedIDsacrossinvocationsonlywithcompletevalue/generationandrow-planidentity.',requires='Stableaddress/shapealoneinsufficient;invalidateoninputwrite/requestremap/rollback/graphentry/DP-TPplanchange. Ordinaryper-forward CSEstillrerunseverygraphreplay.'),
        dict(id='F20-ID-E',status='EXCLUDED_IN_GENERAL_MODEL_DOMAIN',change='ClampallnegativeIDsto0,reusebyaddressorlengthalone,sharetarget/draftbanks,ordropalignment/collectives.',reason='ImageIDs,changedcontent,sentinelpolicy,rowmappingandparticipationaredistinguishable.')]
    id_reuse=dict(id='F20-ID',parent_candidate='F20',registry_status='REFINES_EXISTING_F20_VARIANT;NOT_NEW_MAIN_CANDIDATE',name='MoE 路由 token-ID 类型转换与占位修正的层间复用和代际重组',modules=['M04','M12','M21','M22','M23','M24'],
        profile_review_state='3200_EXACT_GRAPH_ID_QUADS_5088_NATIVE_PREP_TASKS_AND_MC2_COMMUTATION_REVIEWED;CONDITIONAL_UNPRICED',
        scoped_detail=dict(graphs=80,layers_per_graph=40,quad_records=3200,quad_TASKs=12800,native_prepare_TASKs=5088,
            quad_source=['fused_topk_router.py163','fused_topk_router.py176'],actual_same_value_or_generation_proved=False),
        source_graph=baseline,candidate_graphs=dict(shared_cast=cast_cse,shared_pointwise_routing_IDs=pointwise_cse),
        graph_scope='Sourceconsistent40layerIDproducer/consumerprojection,notfullpointer/event-completegraph. Per-layerrowplans/weights/vision/hashparamsremainindependent.',
        variants=f36_variants,algebra='F04-F20-ID-algebra-and-generation-review-v29.json',
        dependencies=['Stableinputvaluegenerationforoneforward;rawIDsdonotmutate','CurrentMC2zeropad/selectcontractorindependentlyprovedotherplan','Read-onlysharedpreparationinputs andcapture/update lifetimes','Separatetarget/draft/TPownership andallplaceholder/visionsemantics'],
        relation_to_existing=[dict(candidate='F20',relation='overlaps',scope='SameHashTopKfrontier;wholefusedroutercandidatecannotaddallmetadataremovalsagain.'),dict(candidate='F31',relation='interacts',scope='Metadata/rowpublicationgenerations;F20-IDisMoEIDnormalization,notrepeatAttention/DSparkmetadata.'),dict(candidate='H07',relation='interacts',scope='SameoriginaltokenIDlifetimes,butembeddingclampandrouter-1-onlysanitizationarenotinterchangeable.')],
        evidence_root=str(O),evidence_files=['F04-current-profile-and-fanout-v29.json','F04-precision-routing-and-row-contract-v29.json','F04-F20-ID-algebra-and-generation-review-v29.json','F20-routing-ID-condition-ledger-v29.json'],
        all_consumers_closed=False,legal_fusion_proved=False,candidate_closed=False,critical_path_gain_us=None,critical_path_gain_status='UNKNOWN_NOT_QUAD_SPAN_SUM',implementation=False,new_performance_result=False,overall_complete=False)
    f04_conditions=[
        ('existing_fusion','MEASURED_DOCUMENTED_LOCAL_QUESTION_CLOSED','4472existingRmsNormCastinstances=3200graph+960target+264draftproposal+48draftdummy;notnewfusion.'),
        ('output_values','DOCUMENTED_IN_PINNED_KERNEL_DOMAIN','BF16kernelroundsfirstandwidensroundedvalue;FP32planeisnotunroundedintermediate.'),
        ('binary_identity','UNKNOWN_RUNTIME_LINK','Repo/build/packagekernelSHAagreementdoesnotbindactualexecutionbinary;donotpretendnativeparitytest.'),
        ('live_planes','MEASURED_SOURCE_BRANCH_SUPPORTED','Native1272directrouter408callspositivelysupportweight_fp32path;BF16experts/shared andFP32routerbothlive inthatcontract. OthergatesmayignoretheprovidedFP32plane;graphrolesremainconditional.'),
        ('FP16_alternative','DOCUMENTED_DISTINCT_BRANCH','Halfkernelcastsbeforescalegamma;do notgeneralizeBF16arithmetictoalltypes.'),
        ('router_output','DOCUMENTED_RETAINED','sqrtsoftplus/vision/hashbias,weightsandIDsrequired;notargmax-onlyrankcontract.'),
        ('aux_output_shape','DOCUMENTED_LIMIT','Thirdgateoutputignoredandout_flagFalse;shapedoesnotprove fullscoreHBMwrite.'),
        ('row_padding','DOCUMENTED_PROFILE_CONSISTENT','MC2padsbothplanes/falsemaskandIDs,andSPfinalizetrims;800->816->800sourceexample,notguessedactualcallpair.'),
        ('layout_SP','DOCUMENTED_CONDITIONAL','KeepmodelSP/alreadySP/repl-allreduce andexternalrouter/legacytuplebranches.'),
        ('residual_shared','REUSED_SOURCE_CONTRACT','Routed/shared/HCconsumers andMC2serial-overlapmutexremain;noadditivefreeoverlap.'),
        ('fusion_domain','CONDITIONAL','RMS/router/HC/sharedquantreorganizationspreserveBF16roundingandFP32math/readiness.'),
        ('row_tiling_pruning','CONDITIONAL','Actualvalidmasks/paddedvalues/auxconsumers andreductionorderrequired;nopeak/gainclaim.')]
    id_conditions=[
        ('observed_repetition','MEASURED','80graphs*40layers*4TASK=12800;6nativefullchains5088additionalTASKs.'),
        ('quad_identity','MEASURED_NOT_VALUE_PROOF','Exactrowids/order/source sites;equalshape/optypeisnotsamevalue/pointer.'),
        ('root_generation','PARTIAL_SOURCE_CONTRACT','TargetSPshardonceandpassesinput_IDexpression;allcallee/asyncimmutabilityandruntimegenerationremaincondition.'),
        ('functional_mapping','DOCUMENTED_DERIVED','Only-1->0andINT64promotion;preserveothernegativeimageIDs andoriginalrawIDs.'),
        ('commutation','DERIVED_IN_STATED_DOMAIN','Pointwisefcommuteswithreviewedzeropad/rowselection;retainper-layerplans,notrequireidenticalplanforalllayers.'),
        ('pad_counterexample','DERIVED_EXCLUSION','Unchanged-1paddingdoesnotcommutewithf;otherprotocol/value-mapsrequireownproof.'),
        ('context_key','CONDITIONAL','Model/entry/inputgeneration,TPsize/rank,replaceSPflag,paddedlength,num_tokens,rowmap,dtype/sentinelpolicy matter.'),
        ('per_layer_params','RETAINED','Hashmap/tid2eidconversion,visionbias,expertscaling/topk remainperlayer;notsharebyshape.'),
        ('collectives','RETAINED_CONDITIONAL','CurrentMC2IDalignmentlocal;otherALLGATHER/EPprotocolcallsnotremovedbythisrewrite.'),
        ('bank_lifetime','UNKNOWN_RUNTIME_GENERATIONS','OriginalIDsnotinplacemodified;sharedbankreadonly/fixedcapturedaddresses/updates/lastreadmustbevalid.'),
        ('across_iteration','DERIVED_INVALIDATION_REQUIREMENT','Sameaddresswith[-1,7]->[9,7]distinguishesstale bank;per-forwardCSErerunseachreplay.'),
        ('phase_and_gain','UNPRICED_PARTIAL','16graphphasesunbound;quadspanmedian7500nsisobservedorderonly,notlatencygain orsumacrossranks/layers.')]
    for name,cid,conditions in[('F04-FFN-norm-condition-ledger-v29.json','F04',f04_conditions),('F20-routing-ID-condition-ledger-v29.json','F20-ID',id_conditions)]:
        write(O/name,dict(generated_utc=stamp,candidate_id=cid,conditions=[dict(id=f'{cid}-v29-{n+1:02d}',branch=x,state=y,evidence=z)for n,(x,y,z)in enumerate(conditions)],all_conditions_closed=False,overall_complete=False))
    write(O/'F20-routing-ID-reuse-review-v29.json',dict(generated_utc=stamp,candidate=id_reuse,source_contract='F04-precision-routing-and-row-contract-v29.json',evidence_refs=refs,overall_complete=False))
    assert not(R/'candidates/F20-ID-derived-v29.json').exists();write(R/'candidates/F20-ID-derived-v29.json',id_reuse)
    write(O/'F20-ID-deduplication-review-v29.json',dict(generated_utc=stamp,existing_parent='F20',existing_evidence=str(O/'candidate-profile-ledger-v28.json')+'#F20',overlap='ExistingF20alreadycoversIDpreparationfusionandcrosslayerreuse;currentworkaddsallrankprofile/MC2commutation/generationproofconditions.',new_main_candidate_created=False,model_candidate_count=35,subvariant_record='F20-routing-ID-reuse-review-v29.json',overall_complete=False))
    model=deepcopy(read(O/'candidate-profile-ledger-v28.json'));fw=deepcopy(read(O/'framework-candidate-ledger-v28.json'));mods=deepcopy(read(O/'module-profile-ledger-v28.json'));matrix=deepcopy(read(O/'module-phase-branch-matrix-v28.json'))
    assert len(model['candidates'])==35 and len(fw['candidates'])==7
    f04=next(c for c in model['candidates']if c['id']=='F04');f04.update(profile_review_state='EXISTING_FUSED_BOUNDARY_AND_ROUNDED_WIDENING_CONTRACT_REVIEWED;FURTHER_FANOUT_REORGANIZATION_CONDITIONAL',scoped_detail=dict(current_TASKs=4472,graph=3200,target_native=960,draft_proposal_native=264,draft_dummy_native=48,dual_outputs='BF16roundedanditsFP32wideningunderreviewedkernel',existing_fusion=True),
        conclusion_changing_gap='Existingfusion/valuecontractansweredunderpinnedsource;actualbinary/generationandnewrouterconsumerprecision/rowtiling remainconditional.',next_profile_or_contract_action='Do notreportexistingnorm/castasanewoptimizationorrepeatprecisioncomment;advanceunreviewedmodulefrontier.',variants_v29=f04_variants,
        evidence_files=['F04-current-profile-and-fanout-v29.json','F04-precision-routing-and-row-contract-v29.json','F04-existing-fusion-and-extensions-v29.json','F04-FFN-norm-condition-ledger-v29.json'])
    f20=next(c for c in model['candidates']if c['id']=='F20');f20['norm_ID_frontier_v29']='F04-precision-routing-and-row-contract-v29.json;F20-routing-ID-reuse-review-v29.json;preservesqrtsoftplus/bias/weights,notrank-only;out_flagFalsemeansdo notinferfullscorewritefromshape.'
    f20['profile_review_state']='EXISTING_SHARED_SERIAL_CONFIG_REVIEW_RETAINED;ALLRANK_ROUTER_ID_PREP_AND_MC2_COMMUTATION_REVIEWED;CONDITIONAL'
    f20['conclusion_changing_gap']='Current4472norm/gatepopulations,3200graphquads and5088nativeIDprepTASKsreviewed;actualinputgeneration/read-onlyalias,rowcontext/mask,loadedbinary andper-replaygain stillnotallattested. Historical320sourcegroupsremainseparatelyscoped.'
    f20['next_profile_or_contract_action']='DirectIDsemantic/sourcequestionanswered;advanceotherwholemodulecoveragefrontier. Queryspecificgeneration/protocolgaponlyifstrongerlegalityconclusionneedsit.'
    f20.setdefault('scoped_detail',{})['current166_v29']=dict(norm_gate_TASKs=4472,graph_ID_quads=3200,graph_ID_TASKs=12800,native_ID_prepare_TASKs=5088,source_contract='F04-precision-routing-and-row-contract-v29.json',algebra='F04-F20-ID-algebra-and-generation-review-v29.json',new_main_candidate=False)
    f23=next(c for c in model['candidates']if c['id']=='F23');f23['row_domain_protocol_v29']='MC2preparepadsBF16hiddenandFP32routertoDP-wideextentandfalsemask,IDszeropad/select;SPfinalizetrimsoriginalN. Supportsconditional800/816/800mechanism,butactualsamecallcontext/maskbindingremains,notneartimepairing.'
    f20['ID_preparation_reuse_v29']=dict(parent_candidate='F20',review='F20-routing-ID-reuse-review-v29.json',scope='ExistingF20crosslayerIDpreparationvariantrefined;notanewmaincandidate.');model['count']=35
    for cid in['F04','F20']:
        q=R/'candidates'/(cid+'.json');old=read(q);write(O/(cid+'-master-before-v29.json'),old);old['review_v29']=dict(source_contract=str(O/'F04-precision-routing-and-row-contract-v29.json'),model_ledger=str(O/'candidate-profile-ledger-v29.json')+'#'+cid,scope='Existingboundary/currentfanoutrefinement;notimplementationorgain.')
        if cid=='F04':old['extensions_v29']=f04_variants
        if cid=='F20':old['ID_reuse_v29']=id_reuse
        write(q,old)
    mm={m['id']:m for m in mods['modules']};bm={m['id']:m for m in matrix['modules']}
    for key,cid in[('M06','F04'),('M13','F04'),('M22','F04')]+[(key,'F20')for key in id_reuse['modules']]:
        if cid not in mm[key]['candidate_links']:mm[key]['candidate_links'].append(cid)
    notes={'M00':'PositiveRMSdevicecustom-capabilitybranchrecorded;fallbackreviewed,notallimplementationsclosed.',
        'M04':'F20-IDsameinputgeneration/asyncupdateguard;router-1-onlymapnotembeddingclamp.',
        'M06':'HCFFNpre->existingdualoutputnorm->MLP->HCpostfanoutbound;nohiddennormreuseacrosslayers.',
        'M12':'4472norm/gatepopulationssplit;existingfusedroundedBF16/FLOATcontract andF20-IDIDquadsource/commutationbound.',
        'M13':'BF16sharedconsumer retained;FP32routerplaneisroundedwidening,notunroundedmath.',
        'M21':'F20-IDrowplansretainMC2padding/TPslice,IDs/sentinels andmetadataepochs.',
        'M22':'Sharedproducergraphsconditionalonentry/address/valuegeneration;80graphs40quadsnotpointerproof.',
        'M23':'MC2preparealignsbothnormplanes/IDsmask,finalizetrimsSPN;actualcallcontext/maskvaluesstillopen.',
        'M24':'F20-IDnormalizationCSErerunseachforward/replay;address-onlycrossiterationcacheexcluded.'}
    for key,text in notes.items():
        mm[key]['review_v29']=dict(finding=text,evidence=['F04-current-profile-and-fanout-v29.json','F04-precision-routing-and-row-contract-v29.json','F20-routing-ID-reuse-review-v29.json'],whole_module_closed=False)
        bm[key]['branches'].append(dict(branch='CurrentFFNnorm/casttypedfanout/routerIDpreparation/MC2rowprotocol',state='SCOPED_PROFILE_SOURCE_ALGEBRA_REVIEWED;WHOLE_MODULE_OPEN',evidence=text,gap='Fullmodule/consumer/generation/nativebinary/precision/resourcequestions remainseparate.'))
    for m in mods['modules']:m['phase_branch_record']='module-phase-branch-matrix-v29.json#'+m['id'];bm[m['id']]['candidate_links']=list(m['candidate_links'])
    for obj,parent,dest in[(model,'candidate-profile-ledger-v28.json','candidate-profile-ledger-v29.json'),(fw,'framework-candidate-ledger-v28.json','framework-candidate-ledger-v29.json'),(mods,'module-profile-ledger-v28.json','module-profile-ledger-v29.json')]:
        obj.update(generated_utc=stamp,parent_ledger=parent,whole_modules_closed=0,overall_complete=False,new_source_read=True,new_target_source_read=True,repository_wide_audit=False,new_device_work=False)
        if obj is model:obj['framework_candidate_ledger']='framework-candidate-ledger-v29.json'
        if obj is mods:obj.update(existing_model_candidates=35,framework_candidates=7)
        write(O/dest,obj)
    matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v28.json');write(O/'module-phase-branch-matrix-v29.json',matrix)
    registry=[]
    for c in model['candidates']:
        q=R/'candidates'/(c['id']+'.json');assert q.exists();registry.append(dict(id=c['id'],master_record=str(q),master_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),current_authority=str(O/'candidate-profile-ledger-v29.json')+'#'+c['id'],modules=[m['id']for m in mods['modules']if c['id']in m['candidate_links']]))
    assert all(c['modules']for c in registry)
    write(O/'candidate-record-index-v29.json',dict(generated_utc=stamp,model_candidates=registry,framework_authority=str(O/'framework-candidate-ledger-v29.json'),unlinked_candidate_ids=[],whole_modules_closed=0,overall_complete=False))
    remaining=deepcopy(read(O/'module-coverage-gaps-v28.json')['remaining'])
    remaining=[x for x in remaining if not x.startswith('NextprofilefrontierF04:')]
    remaining.insert(1,'F04existingdualoutputfusionnowreviewed4472instances;reviewedkernelFP32planeiswidenedroundedBF16,notunroundedaccumulator. Norm/gate/router/sharedfanoutsourceandMC2pad/finalizemechanismanswered;actualbinary/samecallmask/generationandnewtilingprecision remainconditional. Do notrepeatoldF04scoped_detailNonegap.')
    remaining.insert(2,'NewF20-ID:3200graphIDquads/12800TASK plus5088nativeIDprepTASK;f=-1to0/int64commuteswithMC2zero-pad/select,per-layerrowplansretained. PureCSErequiresstableinputgeneration/read-onlybank,perreplayrefresh;actualvalue/alias/eventattestationnot inferredfromsameSSAshapes. Cross-iterationaddress-onlyreuseexcluded.')
    remaining.insert(3,'NextwholecoveragefrontierM14/F24-F25 Engram:existinghash/cachehelpersalreadyimplemented,actual hash->host/UVA gather->dequant/gate publication fullconsumer/phase/state mapping notwholemoduleclosed. Useexisting nativefullcallers/graphoutersegments first;onlypointreadunansweredselectedhash/gather/gateconsumer,notentiremodelorSDK.')
    write(O/'module-coverage-gaps-v29.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
    summary=dict(generated_utc=stamp,status='PROGRESS',previous_turn_classification='PROGRESS:v28F35/rawentropyconsumer/wholecoverageindexrepair',
        RmsNormCast_TASKs=4472,graph_norms=3200,native_target_norms=960,native_draft_proposal_norms=264,native_dummy_norms=48,
        rounded_BF16_then_FP32_source_contract=True,actual_execution_binary_hash_bound=False,direct_router_FP32_branch_native_APIs=1272,ID_graph_quads=3200,ID_graph_TASKs=12800,native_ID_prep_TASKs=5088,
        MC2_row_prepare_finalize_reviewed=True,new_model_candidates=[],refined_model_candidates=['F04','F20','F23'],deduplicated_variant_parent='F20',F04_conditions=len(f04_conditions),F20_ID_conditions=len(id_conditions),
        model_candidates=35,framework_candidates=7,whole_modules_closed=0,overall_complete=False,evidence_refs=refs,
        all_jobs_terminal=True,no_new_model_NPU_service_or_health_work=True,remaining=remaining)
    write(O/'FFN-norm-and-routing-ID-summary-v29.json',summary)
    gate=read(O/'PROFILE-CODE-QUESTION-GATE-v29.json');gate.update(status='PROFILE_AND_NECESSARY_POINT_CONTRACTS_REVIEWED',answered_by=['F04-current-profile-and-fanout-v29.json','F04-precision-routing-and-row-contract-v29.json','F04-F20-ID-algebra-and-generation-review-v29.json','F20-routing-ID-reuse-review-v29.json'],
        closed_scope='Existingnorm/casttypedpopulation/sourcevaluecontract,directFFNfanout,routerIDpuremapandMC2zero-pad/select/trimprotocol.',not_closed='Actualbinary/mask/pointer/generationproof,newkernelparity,allmodulescan/otherconsumersandphysicalgain.',stop_rule='F04localprecision/fanoutquestionsanswered;do notchaseallcompiler/kernelinternals. AdvanceM14orothergenuinecoveragefrontier.');write(O/'PROFILE-CODE-QUESTION-GATE-v29.json',gate)
    report='''本轮从v28发现的F04覆盖缺口推进，先读保存profile。全部32份原生分片及80张图中，共4472次RmsNormCast：3200次图内，960次target原生，264次draft proposal，48次draft dummy。dummy仅是调用祖先，不等于启动或可删除。每次有BF16/FLOAT双输出；图内3200个FFN块各有对应FP32 router、HashTopK、BF16 routed/shared专家后继。没有按static definition去重，也没有把相同次数配成同一次调用。

原生帧把问题收敛到device_op72、model846/912、MoE285及router204。定点源码确认，现有RMS+Cast已经融合；不能再当作新实现。关键精度问题也在指定算子源码合同内回答：BF16分支先做FP32归一化/乘gamma，在115行舍入为BF16，再在122行将该舍入结果扩成FP32，128/129行分别写两路。因此FP32路不是未舍入中间值。新router消费者可研究边读BF16边精确扩展、避免独立FP32全局平面，但还必须保留舍入边界、FP32权重/累加、专家/shared分支和其他读者。不能把unrounded accumulator直接给router，也不能只凭topk排序不变删除norm缩放；当前sqrtsoftplus、bias、路由权重同样属于结果。

kernel、build及打包源码的SHA相同只支持源代码一致性，不证明每个执行已绑定到相同binary hash；本轮没有新NPU数值实验。进一步核对已验SHA的两个router helper发现，只有gate带weight_fp32时才选传入FP32路由平面，其他gate可能忽略它并走自身forward。因此再查原生完整栈，1272个router API全部明确命中408行直接F.linear分支，分为target960、draft proposal264、dummy48；没有仅凭参数出现就声称实际消费，graph部分仍分级为source-consistent。FP16另有先低精度cast再乘gamma的分支，未推广BF16算术合同。F04保留已有融合、FP32平面/路由联合边界、HC/共享quant更大子图和行/feature/有效行重组三类后续条件；没有直接drop输出、跨层复用norm值或精度放宽结论。

本轮又从3200个FFN块识别出新的跨层机会。每张图40层都出现Cast、Equal、Fill、SelectV2四步ID准备，80×40×4共12800条TASK；另有5088条原生TASK由router163/176完整caller确认。四步区间中位7500ns只是观察排程，不能相加当作收益。源代码处理的是INT64转换以及仅-1→0，其他负数图像sentinel必须保留。HashTopK还接收逐层的hash表、vision/text bias、scaling等；第三输出被忽略且out_flag=False，不能从profile仍显示全score shape便推断大矩阵写流量。

补强已有F20的ID复用变体（本轮内部标识F20-ID，不新增主编号）：路由ID转换/占位修正的层间复用与代际重组。target forward在循环前SP切分input_ids，并把该表达式传给各层。MC2方法对IDs做零填充和行切分，取决于每层num_tokens、replace-allreduce、TP rank/size和DP padded长度。对逐元素f(x)=(-1则0，否则INT64(x))，有效行上f(Px)=P(fx)，填充行上f(0)=0，所以它与这类零填充/行选择可交换。由此可以在稳定input代际内只算一次路由用f，再保留各层独立行对齐；不要求层间所有P都相同。这里只改分析图，没有实现共享bank。

保留三项明确反例：把其他负数一并clamp会改变图像token；不变地址里的[-1,7]更新为[9,7]后旧bank不再有效；若别的协议用-1填充而不调整填充值，则直接交换也不等价。F20-ID不修改原始IDs，不省略collective，不把target/draft银行混用，不按地址/shape跨迭代复用。每次forward/graph replay需要匹配新的输入generation、捕获地址/就绪协议，callee/异步别名不可变性仍是条件。

MC2行域也补齐了源码机制：prepare对BF16 hidden、FP32 router输入和false mask按DP统一长度padding，IDs使用相同行轴零填充/TP选择；SP finalize返回原始num_tokens前缀。因此M=800、P/TP=816时会补16行再截回800，解释了规范上的norm800/where或gating816/残差800关系。保存的这些形状与源码机制一致，但没有靠相同次数或最近时间造出实际same-call padded_num_tokens/mask证据。旧F23的行域问题从“机制未定位”缩到“实际上下文/有效行/代际绑定”仍待证。

首版op文件名定位把生成物也计入候选，触发数量保护；失败保留。修正版只列16个C++源码metadata，并保留其余路径；后续只读已定位的四个小op文件、直接consumer和MC2方法，没有扫整个SDK。一个MC2 subclass没有自己的__init__，原样记录，未当作没有prepare实例化。RMS/ID/kernel源码合同、source/build线索、profile与未知级别分开。

F04新增12项条件，F20-ID新增12项条件；模型候选仍为35项，框架仍7项。模块表和阶段表链接同步，候选主记录与索引更新。26个整模块仍未关闭，目标继续active。下一批优先转M14 Engram hash/cache、host/UVA gather-dequant与gate边界，用已有fullcaller和图外段继续补覆盖，不重复追查已回答的F04舍入问题。本轮没有模型导入、NPU负载、服务或健康状态变更，最后健康证据仍v12post。
'''
    (O/'PROFILE-FINDINGS-v29.md').write_text(report)
    pub=R/'publication-draft/FUSION-INVENTORY.md';(O/'FUSION-INVENTORY-before-v29.md').write_bytes(pub.read_bytes());lines=pub.read_text().splitlines();assert not any(x.startswith('| F20-ID |')for x in lines)
    lines=[x.replace('当前持续分析台账（v28）','当前持续分析台账（v29）').replace('为 **34项模型候选、7项框架候选**','为 **35项模型候选、7项框架候选**').replace('新增F33/F34见表','F33–F35及F20的后续变体见表').replace(str(O/'candidate-profile-ledger-v28.json'),str(O/'candidate-profile-ledger-v29.json')).replace(str(O/'PROFILE-FINDINGS-v28.md'),str(O/'PROFILE-FINDINGS-v29.md'))for x in lines]
    lines=[('| F20 | MoE router→TopK→dispatch；ID准备融合与跨层复用 | 条件 | v29补齐4472路由人口、80图×40层ID四步处理及5088原生ID准备TASK；仅-1→0与MC2零pad/行选择可交换，保留图像负数、逐层参数/行对齐、原始IDs和每次input generation；F20-ID是原项变体，不新增主号，无提速结论。 |')if x.startswith('| F20 |')else x for x in lines]
    lines += ['','v29持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 '+str(O/'FFN-norm-and-routing-ID-summary-v29.json')+'；历史枚举统计不代表当前全范围闭合。'];pub.write_text('\n'.join(lines)+'\n')
    action=('继续Loop004，v29实质进展。先读FFN-norm-and-routing-ID-summary-v29.json、F04-precision-routing-and-row-contract-v29.json、F04-F20-ID-algebra-and-generation-review-v29.json、F20-routing-ID-reuse-review-v29.json、module-coverage-gaps-v29.json。'
        'F04已由familyhit/scopedNone变为4472RmsNormCast=3200graph+960target+264draftpropose+48dummy。现有融合非新实现。源码BF16kernel115round、122widen、128/129双写：FP32plane是roundedBF16扩展，不是unroundedRMS。'
        'kernelh SHA5a05b11cce4a727b8c48aec4fe764d854c71c5a8ba072dad315a2acec342d472；adapterSHA6f810db3a1e565ebcd2f012ad04a476964fbb616695eb42c67f15dc5d9bb57ff。Source/build/archive同hash不等于executionbinarybinding，别为条件分析扫全编译器。'
        'model912双plane->MoE285internalrouter，BF16expert/shared和FP32router都live；进一步router可内联widen避免globalFP32plane但必须保留BF16round/FP32math，当前F.linearABI不可随便改dtype。'
        'router_compute140–241 SHA06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838，sqrtsoftplus204 vision/hash+bias+weights，thirdoutput ignored/out_flagFalse不是fullscorewrite证据。'
        '既有F20-ID变体补强：80graph×40层Cast/Eq/Fill/Select=3200quads/12800TASK，另native5088，site163/176。f仅-1->0且I64，f(Px)=Pfx对MC2零pad/行select成立，保留每层P不需P全相同。目标forward1549preloopSP IDs,1592传相同表达式；实际callee/asyncgeneration/alias不全证。'
        'MC2 prepare252–314:BF16/FP32router一起pad及falsemask，ID327–348零pad/TPselect，SPfinalize323–324trim原numtokens。sourceexampleM800,P/TP816->补16->trim800；profilecast800/where816一致但不按counts/neartime配actualcall。commmethodSHA7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7；preparefinalSHA dff4fac292200fc06f60aef48505f1ee38314ea5adb53560b325901db308eb24。F23gap现在是actualcontext/mask/validrows，不再机制全未知。'
        'F20-ID原始IDs保留、image负数非全部clamp、每forward/replay重算匹配inputepoch、不能同地址/shape跨步缓存；-1padding未改值不满足交换，其他collective/参数/tid2eid不被CSE省略。'
        'remoteM结果：F04profile28656617d1c2900c2b1fdacb390dabed59267575c28d525aa53a0ece7dc8ba5f；source e0fea22beaa88f8244077693d6a09bc2616da5645055779f1c5930eaf287dc9e；callersv2 f114017a44db7d2bf37c3ad59aae4a57da3a332e4c3fca5d59e90d01b7246b17；kernel55a7e88c918d9f92f08221db3063d811799d9f274fd9a4623c33329d0fc73547；IDauditce446f9d9588bb20f919698add9a2986e62d3f2e7f50356479c2b855b88226f1；MC2align1a3d6046fe9581dc831e23e7ddb87d9592b1827c27ac50d16380491e068d06be。'
        '下一批M14/F24-F25 Engram：先全nativefullcaller/outergraph段定位hash/cache、host/UVA gather-dequant、gate/投影/位置代际实际路径；既有缓存helpers不重复报新，必要再点唯一缺消费者，不整库审计。'
        '35model/7framework/0wholemodule；F04/F20-ID各12conditions，F35entropyraw消费者/v28greedy!=all_greedy保留；3136candidate资源/849missingPMU/16phaseunboundgraphs/26theoryINCOMPLETE不变。'
        'Routerhelper389–410 SHA2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4：无weight_fp32会忽略providedFP32；额外savednativequery1272全positive408 directF.linear，960target/264draft/48dummy，结果SHA048009ba595e3ea402d72c00509e3ccdd6e3db04fd534150707dd3182271f117；不是仅凭传参证明消费。'
        '8remoteCPUjobs中1原文件inventoryguard失败保留、7pass；mainreview含2个SHA-verifiedlocalhelper读取，全部terminal，无newmodel/NPU/service/health，lastv12post，Goalactive。当前CPU关键缺口若要有限补采仍freshhealth/ownership/frozenworkload/clone/restore，不用旧PID猜idle。')
    jobs=[('profile','pass','Saved32partRmsNormCastcensus and80x40FFNsuccessor surfaces retain4472executions/dualoutputs and3nativecallercontexts.',[N/'query_F04_norm_cast_fanout_v29_166.py',N/'F04-profile166-v29.json']),
        ('review','pass','ActualnativeRMS/devicewrapper anddecoderFFNfanout methods extracted,source SHA pinned;no wholemodelread.',[N/'read_F04_norm_dispatch_and_fanout_v29_166.py',N/'F04-source166-v29.json']),
        ('profile','error','Originalexact-opfilenameinventoryhitscandidate-countguardbeforefilteringgeneratedmatches;source resultnot emitted,script/error retained.',[N/'query_F04_gate_callers_and_op_files_v29_166.py',N/'F04-callers166-v29.json']),
        ('profile','pass','CorrectedboundedC++metadata inventory plus4472gatingTASKs/3nativecallercontexts;unlistedsourcepaths retained,notabsence.',[N/'query_F04_gate_callers_and_op_files_v29_166_v2.py',N/'F04-callers166-v29-v2.json']),
        ('review','pass','Fourexactopfiles andnativeMoE/routermethods bindrounded-wideningsemantic/directconsumers/scoring/ID-normalization contracts.',[N/'read_F04_kernel_and_gate_contract_v29_166.py',N/'F04-kernel-gate166-v29.json']),
        ('review','pass','Exact3200IDquads/12800graphTASKs and5088nativesourceTASKs audited;targetinput_IDidentifiercontexts readonly.',[N/'audit_F04_router_ID_preparation_v29_166.py',N/'F04-ID-audit166-v29.json']),
        ('review','pass','MC2namedprepare/IDpad/finalizemechanism reviewed;conditional800->816->800rowdomain andretainedmask/rowplan stated.',[N/'read_F04_MC2_row_alignment_v29_166.py',N/'F04-MC2-alignment166-v29.json']),
        ('profile','pass','Savednativecallerspositivelyreachrouter408directFP32weightbranchin1272APIs;localSHA-verifiedhelpersdistinguishignore-provided-FP32alternatives.',[N/'query_F04_actual_router_linear_branch_v29_166.py',N/'F04-router-branch166-v29.json',O/'F04-router-input-selection-v29.json',O/'F04-actual-router-input-branch-v29.json']),
        ('review','pass','MainreconcilesF04existingfusion/roundingandexistingF20-IDcrosslayerintegerCSErefinementwithgenerationcounterexamples,MC2commutation and24conditionrows;no implementation/gain.',[O/'F04-precision-routing-and-row-contract-v29.json',O/'F04-F20-ID-algebra-and-generation-review-v29.json',O/'F20-routing-ID-reuse-review-v29.json',R/'candidates/F20-ID-derived-v29.json',O/'PROFILE-FINDINGS-v29.md'])]
    runs=[]
    for j,(kind,status,desc,arts)in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-j)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind',kind,'--command','Preparedv29retrospectiveregistration: '+desc,'--code-ref','Savedprofile/exactFFNsource;targetunchanged;noNPU/model/servicework','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',desc]
        for q in arts:assert q.exists(),q;args+=['--artifact',str(q)]
        ctl(*args);runs.append(dict(run_dir=run,status=status,description=desc))
    cp=dict(status='PROGRESS',previous_turn_classification=summary['previous_turn_classification'],current_verified_changes=['4472norm/gate populations andsource rounded-BF16->FP32contract/directFFNfanout bound;existingfusionnotnewgain.',
        '3200IDquads/5088nativeprepTASKs plusMC2zero-pad/selectcommutation andtargetinputIDroot supportexistingconditionalF20-IDrefinement.',
        'MC2prepare/finalizemechanism refinesF23rowdomain gap;F04/F20-IDeach12conditionrows/masterindexpersisted.'],
        summary=str(O/'FFN-norm-and-routing-ID-summary-v29.json'),candidate_ledger=str(O/'candidate-profile-ledger-v29.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v29.json'),module_ledger=str(O/'module-profile-ledger-v29.json'),branch_matrix=str(O/'module-phase-branch-matrix-v29.json'),report=str(O/'PROFILE-FINDINGS-v29.md'),source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v29.json'),coverage_gaps=str(O/'module-coverage-gaps-v29.json'),pending_full_scope_domains=remaining,
        remote_job_status='8CPU-onlyremotejobs terminal:7pass,1inventoryguarderrorretained;mainreviewwithlocalSHA-verifiedhelperpass.NoNPU/model/service/healthwork.',model_candidates=35,framework_candidates=7,next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp)
    write(O/'REVIEW-CHECKPOINT-v29.json',cp)
    for q in[L/'state.json',T/'index.json']:
        d=read(q);assert d['status']in['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(q,d)
    for q in[L/'status.md',T/'status.md']:
        q.write_text('\n'.join(('- 下一步：'+action)if x.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if x.startswith('- 更新时间：')else x for x in q.read_text().splitlines())+'\n')
    (O/'profile-summary-before-v29.json').write_bytes((R/'profile-summary.json').read_bytes());prof=read(R/'profile-summary.json');prof['supplementary_FFN_routing_ID_v29']=dict(summary=cp['summary'],model_candidates=35,framework_candidates=7,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
    for q in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);d.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action'if q==R/'STATE.json'else'next_action']=action
        if q==W/'STATE.json':d.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
        if q in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[cp['summary'],str(O/'F04-precision-routing-and-row-contract-v29.json'),str(O/'F20-routing-ID-reuse-review-v29.json'),cp['coverage_gaps']]
        write(q,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v29.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    arts=[q for q in O.glob('*v29*')if q.is_file()and not q.name.startswith('checkpoint-manifest')]
    arts += [N/(label+'.json')for label in['F04-profile166-v29','F04-source166-v29','F04-callers166-v29','F04-callers166-v29-v2','F04-kernel-gate166-v29','F04-ID-audit166-v29','F04-MC2-alignment166-v29','F04-router-branch166-v29']]
    arts += [N/x for x in['query_F04_norm_cast_fanout_v29_166.py','read_F04_norm_dispatch_and_fanout_v29_166.py','query_F04_gate_callers_and_op_files_v29_166.py','query_F04_gate_callers_and_op_files_v29_166_v2.py','read_F04_kernel_and_gate_contract_v29_166.py','audit_F04_router_ID_preparation_v29_166.py','read_F04_MC2_row_alignment_v29_166.py','query_F04_actual_router_linear_branch_v29_166.py','finalize_F04_router_metadata_v29.py']]
    arts += [q for q in N.glob('zcode-F04-*v29*20261031*')if q.is_file()]
    arts += [R/'candidates/F04.json',R/'candidates/F20.json',R/'candidates/F20-ID-derived-v29.json',pub,N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
    arts=sorted(set(arts));write(O/'checkpoint-manifest-v29.json',dict(generated_utc=stamp,artifacts=[dict(path=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())for q in arts],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v29.json'),manifest_files=len(arts),runs=len(runs),pass_runs=8,retained_error_runs=1,model_candidates=35,framework_candidates=7,validation=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
