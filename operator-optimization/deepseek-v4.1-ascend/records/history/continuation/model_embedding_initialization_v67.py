"""Conditional embedding/initialization/index fusion costs with exact data domains."""
from pathlib import Path
from collections import Counter
import hashlib,math,struct
from finalize_origin_coverage_v9 import N,O,W,read,write
from model_router_vector_v63 import f32
S=W/'frozen-source-snapshot'
H,C,P=5120,4,8

def main():
    assert not (O/'embedding-initialization-fusion-costs-v67.json').exists()
    regs=read(O/'embedding-initialization-regime-models-v67.json')['regimes']
    g=read(O/'embedding-initialization-native-chains-v67.json')
    taskmodel={tuple(k):r for r in regs for k in r['native_TASK_keys']}
    embed=[]
    for n in sorted({r['N']for r in g['embedding_groups']}):
        group=next(r for r in g['embedding_groups']if r['N']==n)
        base=sum(taskmodel[(group['database_index'],t['TASK_rowid'])]['one_read_write_logical_value_bytes']for t in group['stages'])
        assert base==8*n*H+150*n
        cases=[]
        for u in sorted({0,n,n/P}):
            new=2*u*H+2*n*H+4*n
            cases.append(dict(U_rank_owned_valid_rows=u,source_14TASK_value_bytes=base,new_masked_lookup_value_bytes=new,
                interface_and_invalid_reference_budget_bytes=base-new,contract='U=N/8illustrativebalancedexpectedrows,notnativeownerhistogram. CompareconventionalGather+MaskedFillone-read-writevaluevaluation,newkernelreadsvalidweightrowsandwritesfullNincludingpositivezero.'))
        ar=2*(P-1)/P*(2*n*H)
        ownership=[]
        for counts,label in [([n/P]*P,'balanced_illustration'),([n]+[0]*(P-1),'one_vocab_owner_illustration')]:
            umax=max(counts)
            ownership.append(dict(case=label,valid_owner_counts=counts,
                fixed_padded_owner_AllGather_send_bytes_per_rank=(P-1)*2*umax*H,
                baseline_AllReduce_send_bytes_per_rank=ar,fixed_AG_to_AR_payload_ratio=((P-1)*2*umax*H)/ar,
                ragged_all_group_minimal_broadcast_content_bytes=2*n*H*(P-1),
                AR_declared_ring_all_group_send_bytes=P*ar,
                contract='NewSTANDARDowner-valuegatherprotocol,notexistingtoken_exchangeplan. Needownerrowpermutation/countmetadata/dynamiccapture/allreaderandnativeSUM-equivalence. Raggedaggregatehalfcanstillhaveoneownerinjectionbottleneck;fixedpadbalanced.5versusconcentrated4payloadratio.'))
        embed.append(dict(N=n,H=H,weight_shard_bytes=165478400,cases=cases,owner_protocol_cases=ownership,
            logical_invalid_row0_references='N-U',extra_unique_invalid_weight_rows_at_most=1,
            extra_unique_invalid_weight_bytes_at_most=2*H,
            contract='CurrentinvalidIDsmaprow0thenmaskfill+0. EliminatingN-UreferencesdoesNOTsave(N-U)*2HcompulsoryHBM:allsamerow0canhitcache. Current272groups14TASKsnot14provenhostlaunches;3INT32→64castsamereadonlyinputand2BOOL→64castscancompose.'))
    draft=[]
    for n in sorted({r['N']for r in g['draft_initializations']}):
        a=math.ceil(n/P)*P;l=a//P
        old=(18*n+8*a)*H
        late=(2*n+2*a+10*l)*H
        local=[]
        for rank in range(P):
            u=min(l,max(0,n-rank*l));new=(2*u+8*l)*H
            local.append(dict(rank=rank,valid_owned_rows=u,new_direct_HC_init_value_bytes=new,
                old_HC_Tile_plus_global_Pad_value_bytes=old,HC_interface_budget_bytes=old-new,
                new_HC_ID_mask_premix_output_and_read_value_bytes=new+4*u+21*l))
        assert sum(r['valid_owned_rows']for r in local)==n
        draft.append(dict(N=n,A_padded=a,local_rows=l,HC=4,
            current_global_Tile_plus_HC_Pad_value_bytes=old,
            delayed_Tile_after_2D_Pad_SP_value_bytes=late,delayed_Tile_interface_budget_bytes=old-late,
            current_ID_and_mask_pad_value_bytes=8*n+7*a,
            current_premix_4TASK_value_bytes=28*l+8,new_premix_onehot_fullwrite_bytes=16*l,
            current_typed_value_total_excluding_3MemSets_and_scalar_cast=old+8*n+7*a+28*l+8,
            rank_models=local,
            contract='OriginalTileglobalN4H→PadA4H→viewrankl;newPad2D→rankview→localTilecommutesexactcopy/zero-fillandcanreduceworking-set. StrongerdirectlocalinitializerreadsUownedBF16rows,emitsHC4copies/ID0forpad/maskTrueforpad/FP32onehot. PrefixmaskFalseonlyforpositiveNone/new_zerosbranch;suppliedTrueprefixretained. No skippingdecoder/cache/outputcollectivesonzero-queryranks.3N/AMemSetcontrol/trafficNOTpricedorclaimeddeleted.'))
    assert next(r for r in draft if r['N']==20)['delayed_Tile_interface_budget_bytes']==2222080
    premix=[]
    for n in sorted({r['local_rows']for r in g['target_HC_initializations']+g['draft_initializations']}):
        premix.append(dict(N_local=n,source_zero_Fill_Broadcast_columncopy_value_bytes=28*n+8,
            new_onehot_fullwrite_value_bytes=16*n,materialization_interface_budget_bytes=12*n+8,
            readonly_FP32_onehot_bank_bytes=16*n,
            target_Tile_plus_Pre_one_full_HC_read_value_bytes=18*n*H,
            Tile_with_Pre_shared_load_but_materialized_residual_value_bytes=10*n*H,
            Tile_Pre_fused_shared_input_budget_bytes=8*n*H,
            Tile_Pre_Post_two_read_reference_value_bytes=26*n*H,
            virtual_HC_readonce_each_Pre_Post_value_bytes=4*n*H,
            virtual_HC_two_consumer_interface_budget_bytes=22*n*H,
            contract='MaterializedHCpreservesfirstPostresidual/readers;virtualrepeatuntilfirstPostneedsnewABIandallinitialHCconsumers/nointerveningEngram/aliasmutations. BudgetdeclaresonefullHCinputreadperPre/Post,notactualHBM/nativeinternalpasses. Firstcoeffprojection/full4Hnorm/HF32nativeorderandBF16outputsremain;onehotbankreadonlyprefixcanpersistonlyifnoinputmutation/aliasescape/capturelifetime.'))
    positions=[]
    for n in sorted({r['N']for r in g['position_groups']}):
        positions.append(dict(N=n,source_IndexCheck_Index_Cast_Add_value_bytes=60*n,
            fused_index64_computed32_query64_position64_value_bytes=28*n,interface_budget_bytes=32*n,
            contract='Gathercapacity4/sourcevalidnegativewrap/error→widen64→INT64add. Preserveoldcomputedgeneration/INT64overflowandpositions/tailreaders;IndexCheckerrorflagsandscalarGMunpriced. Positionsandseq_lensmultioutputonlysamecomputedgeneration/readytime.'))
    scatter=[]
    for b in sorted({r['K']for r in g['sampled_ID_groups']}):
        k=5*b
        scatter.append(dict(sample_K=b,draft_selected_K=k,draft_source_BD=k,
            source_sample_four_stage_value_bytes=48*b,new_sample_gather_to_scatter_value_bytes=24*b,
            sample_interface_budget_bytes=24*b,
            source_draft_Slice_Cast_IndexCheck_Index_Linear_Scatter_value_bytes=28*k+48*k,
            new_draft_strided64_gather_narrow32_to_scatter_value_bytes=28*k,
            draft_interface_budget_bytes=48*k,
            input_ids_storage_capacity=8192,semantic_sample_plus_draft_stores=b+k,
            contract='SelectedKonlywrites;8192baseextentnottraffic. SourceB×5viewofBmax×6,flatjnormalizeagainstBDthenphysicalrow=j//5,col=1+j%5. KeepdestINT64ABI/capacity/negative-error,sourceadvanced-indexwrap/IndexCheck,INT64→32narrowpolicy;alloldsourcevalues/aliasandduplicatehazards. ExistingdraftscatterAFTERsamplewinsifcross-setoverlap;fuseoneparallelstoreonlydisjointprooforpriority/barrier. UnwrittenoldIDprefix/tailmustremain.'))
    sanitize=[dict(N=n,source_equal_maskfill_then_clamp_value_bytes=22*n,new_clamp_only_value_bytes=8*n,interface_budget_bytes=14*n,
        contract='f=where(x==-1,0,x),g=max(x,0):g(f(x))==g(x)ONLYsameprefixgeneration/orderandnoobserverbetween. Embed-onlysanitizewithoutmutatingpersistentIDbankchangesEngram/router/otherrawreaders;scatter-onlynewstoresdoesnotcleanuntouchednegativeprefix/pad.')for n in [24,6400,6528,6672]]
    # Pure reference witnesses, not claims about unrecorded actual values/native accumulation.
    separate=f32(f32(f32(2**24+1.)-2**24)+0.)
    folded=f32(f32(2**24-2**24)+1.)
    assert separate==0 and folded==1
    assert math.isnan(0.*float('inf'))
    negzero=struct.unpack('<I',struct.pack('<f',-0.))[0]
    pluszero=struct.unpack('<I',struct.pack('<f',f32(-0.+0.)))[0]
    assert negzero!=pluszero
    for x in [-2**31,-2,-1,0,1,129279,2**31-1]:assert max(0 if x==-1 else x,0)==max(x,0)
    # Synthetic permutation verifies exact source-view addressing, including legal negative indices.
    buf=[[100*r+c for c in range(6)]for r in range(4)]
    for b in [2,4]:
        logical=[x for row in buf[:b]for x in row[1:]]
        for j in list(range(5*b))+[-1,-5*b]:
            jj=j if j>=0 else j+5*b
            assert logical[j]==buf[jj//5][1+jj%5]
    write(O/'embedding-initialization-fusion-costs-v67.json',dict(embedding_models=embed,draft_initialization_models=draft,
        initial_premix_and_virtual_HC_models=premix,position_gather_cast_add_models=positions,ID_gather_scatter_models=scatter,
        placeholder_clamp_models=sanitize,persistent_target_embedding_copy_models=[dict(N=n,H=H,copy_value_interface_budget_bytes=4*n*H,
            contract='24positivecopiesintoinputs_embedsGPUbank;newembedding/collectiveoutABIrequiresallmultimodal/is_token_ids/sentinel/prompt/capturebanks/readiness/lastreader/prefix-tailcontracts,notordinarytemporarycopy.')for n in [6400,6528,6668]],
        witnesses=dict(HC_pre_folded_weight_not_universal=dict(repeated_x=[1.,1.],four_channel_weights=[[2**24,1.],[-2**24,0.],[0.,0.],[0.,0.]],reference_flat_accum=separate,reference_folded=folded,actual_HF32_native_mode_not_replaced=True),
            onehot_not_drop_zero_products=dict(x0=1.,other_channel='Inf',weighted_zero_term='NaN'),
            owner_copy_vs_zero_sum_signed_zero=dict(copy_bits=negzero,add_positive_zero_bits=pluszero,actual_native_HCCL_policy_unbound=True)),
        roofline='Typedcopy/index/predicate/INT64/VectorworknotBF16CubeFLOPs. LogicalvaluebudgetsnotmeasuredHBM. T>=max(requiredtierbytes/Btier,requiredtypedwork/Ptyped,latency,dependencies)withdeclaredmemory/cache/error/aliasdomain;actualVector/index/latencycapacityunbound.',
        overall_complete=False))
    pins=[]
    old65=read(O/'heads-sampling-operator-contract-v65.json')['source_files']
    for r in old65:
        if r['path'].endswith(('models/deepseek_v41/model.py','models/deepseek_v41/dspark.py','ops/vocab_parallel_embedding.py')):pins.append(r)
    seq=next(r for r in read(O/'state-engram-operator-contract-v62.json')['existing_pinned_operator_callers']if r['path'].endswith('models/common/ops/sequence_parallel.py'));pins.append(seq)
    old31=read(O/'M15-normalized-content-consumer-and-precision-contract-v31.json')['source_files']
    for rel in ['vllm-ascend/vllm_ascend/worker/model_runner_v1.py','vllm/vllm/v1/worker/gpu_model_runner.py']:
        pins.append(dict(path=str(S/rel),sha256=old31['/vllm-workspace/'+rel],original166_hash_binding=True))
    attn=S/'vllm-ascend/vllm_ascend/attention/dsa_v41.py'
    pins.append(dict(path=str(attn),sha256=read(O/'M19-embedding-HC-CP-row-and-consumer-contract-v35.json')['source_files']['/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py'],original166_hash_binding=True))
    for r in pins:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
    write(O/'embedding-initialization-operator-contract-v67.json',dict(source_files=pins,new_remote_source_read=False,
        source_scope='Sevenold166SHA-boundlocaloperatorclausesonly. Worker1859-1924/3530-3532/3575-3584/3630and1599-1606/1752readsolelyforIDcast-gather-scatter/positions/zero-tail/sanitizeandpersistentembedbankconsumers. No framework/API/lifecycleaudit.',
        facts=[
            '272embeddinggroups14TASK=3INT32→64casts+2BOOL→64casts+range/predicate/INT64offset+Gather+MaskedFill. Commenttorch.compilewillfusenotprofileproof;retainednativeTASKsremain14.',
            'Current9ma<REDACTED_TOKEN>;addedrange/TP1/token_exchange/PCPxTP/quantizedandMM/promptbranchesremainconditional. InvalidIDsloadsafeweightrow0thenmaskedpositive0;skipinvalidloadsneedsnewfullywritingkernel.',
            'Targetshards2DhiddenBEFOREHC4Tile,whileDSparkrepeatsglobalHC4BEFORESPpad/split. Copy-mapcommutationprovesdelayingdraftrepeatwithoutchangingnumerics;all104nativeN10/20havepadding.',
            '104draftZerosLikeatsp_padding_mask149bindsNonebranch→falseprefix;F.padTrueformspaddingmask. SuppliedmaskpathpreservesanyTrueprefix,notallfalsebyN.',
            'InitialpremixFP32[Nlocal,4]zeros+column0=1,4TASK. ReadonlyconstantbankornewHCvirtualconstantABIpossibleonlyallinputmutations/aliases/captureconsumersclosed. HcPreYandcoefficientpaths/currentHF32/full4Hnorm/BF16roundnotalgebraicallyremoved.',
            'M04isworkerinputID/positionpreparation,notnorm. Scatter_8192descriptorisbankcapacity;144nativeScattercallswriteKselectedIDs. Sourceadvanced-indexnegativewrapanddestscatternegative-errorareNOTsame.',
            'SampleIDscatterprecedesdraftIDscatter;cross-overlapnewmergedparallelwriterrequiresdisjoint/priority/barrier. B×5logicaldraftviewfromBmax×6stridedtablehasownbounds;fullbaseextentnotvalid-indexdomain.',
            'SanitizeexactPLACEHOLDER-1andlaterclamp_min0composeonlysameprefix/nointerveningobservers. RawpersistentIDbankfeedsothermodelconsumers;embedding-privatecleaningnotdropin.',
            '24positivefullN×Hcopiespublishpersistentinputs_embeds;directout/aliasviewneedsMM/sentinel/is_token_ids/prompt/captureandlastreader/prefix-tailprotocol. TargetSPclonefunctionalcustomopcannotbechangedtoviewwithoutnewaliascontract.',
            '48dummyDsaZerosLikeatmetadataNoneguard698returnszerooutputwithoutQKV/attention. Differentfrommetadata-presentlocal-query-emptywhichstillneedsKV/globalheadrestore/outputcollectives. DummyoptimizationisnotC8decodebenefit.',
            '4M19ConstantPadNdMemSetregimes312and2M05|M21BOOL24prefixregimes80remainpending. NewsemanticinitializerdoesnotpriceunknownMemSetcontroltrafficorresolveunrelatedprefixfield.' ],
        evidence_refs=['M19-embedding-HC-CP-row-and-consumer-contract-v35.json','HC-operator-numerical-and-dependency-contract-v58.json','HC-operator-numerical-and-dependency-contract-v59.json','router-MemSet-profile-insufficiency-v63.json','initialization-profile-insufficiency-v67.json'],
        native_binary_or_actual_tensor_values_proved=False,whole_modules_closed=0,overall_complete=False))
    vs=[]
    def v(i,c,s,e,gate,d='CONDITIONAL'):
        vs.append(dict(id=i,candidate=c,choice=s,work_IO_effect=e,conditions=gate,decision=d,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,critical_path_gain_us=None))
    v('EMBED-MASK-LOOKUP-ZERO','F02','FuseIDwiden/range/offset/lookup/rowzero','14TASKtyped8NH+150N→2H(N+U)+4Nvalue-reference','CurrentBF16table/no-addedrangevsaddedmapping,allpositivezero/NaN/error/cast/IDgeneration/allreader/nativealiasandquantmethodgates. U/uniqueDunknown;row0cacheprecludesHBMP×claim.')
    v('EMBED-SHARED-ID-WIDEN','F02','Reuse3castsfromsameimmutableINT32argument','Avoid2INT64planesandtheirreaderswhereallpromotionrulesmatch','Source_maskargumentunchanged,nativeINT64offset/BOOLpromotion/asyncmutation/lifetime;notCSEbyshapealone.')
    v('EMBED-OWNER-VALUE-GATHER','F02','Broadcastowner-selectedembeddingrowsinsteadzero-filledAR','BalancedfixedAGpayload.5AR;alloneownerfixedpad4AR,raggedaggregatehalfbutskew','NewSTANDARDrowowner/count/permutation/captureprotocol;nativeSUMzero/NaN/signedzeroequivalence,allranksinputIDidentityandallMM/readers. Differentfromexistingtoken_exchangeplans.')
    v('EMBED-DIRECT-PERSISTENT-OUT','F02','Lookup/reducepublishespersistentinputs_embedsbank','24copies4NHvaluebudget','Newout/collectiveABI,sentinel/MM/is_token_ids/promptreaderandcapturebankprefix-tail/lastreader/readiness;inputsprecomputedbranchnotmodelreembedduplication.')
    v('TARGET-SP-CLONE-VIEW','F02','Replaceprivate2DSPclonewithnewreadonlyview/producerdirectslice','24clones4nHvaluebudget','Functionalcustomopaliascontractmustchange;Tilecopiesbeforeparentreuse,noinplaceobserver/Engram/tokenbankaliasescape. ID/maskshardsotherconsumersseparate.')
    v('DRAFT-SP-BEFORE-REPEAT','F28','Pad/shard2DembeddingthenrepeatHC4locally','N20/A24ordinaryinterfacebudget2222080B/card','SameTPcopy/rowowners/prefixvalues/zero-pad/suppliedpaddingmask;reorderpureTileonlyandkeepallglobalotherreaders/dtype/capture;decoderandglobalKV/outputworkremain.')
    v('DRAFT-DIRECT-LOCAL-INIT','F28','OneownerinitializeremitsHC4/paddedIDs/mask/onehot','ReadownedUrows,writelocal4H+ID+BOOL+FP32premix;avoidglobalTile/Padplanes','MaskFalseonlyNonebranch;providedTrueprefixkept,alllastreaders/functionalalias/captureandunknownMemSetbackendprotocolsseparate. No native15→1speedupclaim.')
    v('PREMIX-ONEHOT-FULLWRITE','F01','OnekernelwritesallFP32N×4onehot','4TASK28n+8→16nvaluebudget','Fullprefix/writeanddtype1/0/capture/otherreaders;noBF16mixconversion.')
    v('PREMIX-READONLY-BANK','F01','ReuseimmutableonehotbankbyHC/dtype/capacity','Removepercallinit;16nbytesbankandprefixview','ActualHcPreinputreadonly/nooutputalias,inflight/capturelifetime/bankinitreadybeforecapture,allobservers;constantdatadoesnotneedrequestgenerationbutconfigdoes.')
    v('INIT-TILE-HCPRE-SHARED-LOAD','F02','FirstHCprefetches2DwhileemittingfullHCresidual','8nHlogicalshared-inputbudget','Firstgeneration/nointerveningEngramorrawstatechange,keepfull4Hnorm/HF32coefficientarithmetic/BF16outputs/residualforfirstPost/allreaders.')
    v('INIT-VIRTUAL-HC-UNTIL-POST','F02','Virtual4-wayrepeatuntilfirstHcPostpublishesfullstate','22nHtwo-consumerreferencebudget','NewPre/Post/residualABIwithallinitialHCreaders/alias/Engramguards;broadcastloadsbutkeeporiginalproducts/reductionorder/onehot0*InfNaN,notfoldweight/normalgebra.')
    v('INIT-FOLDED-WEIGHTS','F02','Sum4channelweightsandreplacefull4HcoeffGEMM','Reference2^24weightswitness0vs1;HF32orderfurtherunbound','Finiteactualweight/toleranceproofneeded;BF16/HF32norm/accumulationnotalgebraicallyexact.', 'EXCLUDED_AS_UNCONDITIONAL_EXACT_REWRITE')
    v('ID-SAMPLED-GATHER-STORE','F31','FuseIndexCheck/gather/linearize/scatterselectedsampleIDs','48K→24Ktypedvalue-reference','Sourceboundsnegativewrap,destbounds/no-negativeerror,oldsourcealiases/duplicates/sample-draftorder,untouchedbankrowsretained.')
    v('ID-DRAFT-STRIDED-GATHER-STORE','F31','GatherselectedINT64stridedB×5thennarrowandstore','28BD+48K→28K;BD=K10/20budget48K','LogicalB×5boundsnotBmax×6storage,physicalstride6/offset1,INT32narrow/nativeerror/alias/orderedwrites;allconverted-planeotherreadersabsent.')
    v('ID-MERGED-SAMPLE-DRAFT','F31','Oneorderedmulti-sourceIDstore','Sharedsetup/indicesbutnostorecountreduction','Proveddisjointdestinationsorpriority/barrierandallsourcepreload;originaldraft-after-samplewinscross-overlap. Within-scatterduplicatepolicyalsopreserved.')
    v('POSITION-GATHER-CAST-ADD','F31','DirectINT32computedlookup+INT64queryoffset→INT64position','60N→28Ntypedvalue-reference','Originalsourcecapacity4/error/negativewrap/intwidth/overflow,oldstateandallposition/tailreaders/generation/capturealias.')
    v('POSITION-SEQ-TAIL-MULTIOUT','F31','Generatepositions/seq_lens/query-tailinoneproducer','Avoidprivatecounts/gatherplanesandtinyfilllaunches','Ntoken/Brequest/Qcapacitydistinctdomains,querytail-1vsseqtail0/positionpad0,actualoldcomputedgeneration/CPUandattentionreadiness/errorsemantics;notallzeroCSE.')
    v('ID-SANITIZE-CLAMP-COMPOSE','F31','Composesentinel-1replaceandclamp_min0','22N→8Nsame-prefixreference','NointerveningrawIDreaders/sameprefix/valuegeneration;persistentbankmuststillbeupdatedforallconsumers. FusingonlynewscatterKcannotcleanuntouchedNprefix.')
    v('DUMMY-ZERO-HCPOST','F03','UsevirtualzeroattentionoutputunderexplicitmetadataNone','Avoid48dummyN20Hzeroplaneproducer-consumerinterface','NewHcPostzeroinputABIpreserves0*postcoeff/nonfinite/round/allreadersandcollectiveprotocol;notmetadata-presentno-querybranchandnotC8requestgain.')
    v('DUMMY-DEAD-ATTN-INPUT','F28','Pruneonlyunusednorm/attentioninputAGunderuniformmetadataNone','Inputcontentunusedbyearlyzero-return;HCcoeff/residual/FFNremain','AllTPuniformNone/allreaders/sideeffects+captureprotocolproof,shape/readiness/zeroSUMkeep;notdropglobalKV/outputwhenmetadataexists.')
    v('INPUT-CROSS-ITERATION-CACHE','F31','Reusepuremask/map/positionsonlysameID/stategeneration','Cachelookup/version/maintenanceandcontentcost','BatchID/reorder/acceptedprefix/positions/weights/shardbounds/CP-SP/capturelastreaderallmatch;samephysicalbank/shapealoneinsufficient.')
    v('PENDING-MEMSET-BOOL-DELETE','F31','DeleteN/AMemSetorambiguousBOOLclearbyparentname','Unresolvedcontrolbytes/readers,6regimes392keptpending','Existingrawprofilehasnomemory/op_args;name/adjacentPadsize/callerdoesnotprovewritefootprintorfieldidentity.', 'UNRESOLVED_EVIDENCE_RETAINED')
    write(O/'embedding-initialization-fusion-variants-v67.json',dict(variants=vs,variant_count=len(vs),interaction_rules=[
        'Maskedlookup/IDcastCSE/directoutandowner-gatherbudgetscomposeonce;row0referencesarenotcompulsoryHBMtraffic.',
        'DraftSP-before-repeatanddirectlocalinitializerarealternativegranularity;doNOTaddtheirfullbudgets. TargetalreadySP-before-Tile,notnewbenefitthere.',
        'Premixfullwrite/readonlybank/virtualPreconstantarealternativeinterfaces;sameinputbanklifetime/gatesonce.',
        'MaterializedresidualTile-PrevsvirtualuntilPostdifferentall-readerrequirements;keeporiginalHF32norm/coefficients/BF16/postorder.',
        'SelectedIDstoresandfullprefixclampworkdifferentdomains;fusedparallelstoresneedcollision/aliasproofandoldbankrowsremain.',
        'DummyNonebranchseparatefrommetadata-presentno-queryglobalKV/outputwork;dummyimprovementnotrequestspeedup.',
        '230reviewedregimes7224exec;4N/AMemSet312+2ambiguousBOOL80remainpending. No physicalcapacity/nativeimplementation/wholemoduleclosure.'],whole_modules_closed=0,overall_complete=False))
    print(dict(embedding_shapes=len(embed),draft_shapes=len(draft),premix_shapes=len(premix),position_shapes=len(positions),scatter_shapes=len(scatter),source_pins=len(pins),variants=len(vs),draft_N20_late_Tile_budget=2222080,HF32_fold_counterexample=[separate,folded]))

if __name__=='__main__':main()
