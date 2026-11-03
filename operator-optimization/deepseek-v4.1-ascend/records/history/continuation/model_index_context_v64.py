"""Source-conditioned index/cache and context partition bounds, including added communication."""
from pathlib import Path
import hashlib,math,struct
from fractions import Fraction
from finalize_origin_coverage_v9 import N,O,W,read,write
from model_router_vector_v63 import f32,bf16
S=W/'frozen-source-snapshot'
def main():
 assert not (O/'index-context-fusion-costs-v64.json').exists()
 key=[]
 for m in [24,6400,6528,6668]:
  d=128;r=64;baseline=m*(11*d+12*r+34)+2*d;fused=m*(d+8*r+14)+2*d
  # includesWKoutput,RMSauxrstd,bothcachewrites;allslotsassumedvalid.
  assert baseline-fused==m*(10*d+4*r+20)
  key.append(dict(M=m,D=d,rope_tail=r,WK_ops=2*m*512*d,WK_weight_bytes=2*512*d,post_WK_including_WK_output_logical_bytes=baseline,fully_composed_WK_to_dual_cache_value_bytes=fused,full_interface_budget_bytes=baseline-fused,standalone_WK_then_fused_consumer_budget_bytes=m*(6*d+4*r+20),required_cache_payload_bytes_per_valid_row=130,source_two_indices_read_bytes_per_row=16,fused_one_indices_read_bytes_per_row=8,RMS_aux_rstd_retained_bytes=4*m,scope='Idealallvalidrows/retainedrstd/privateintermediates,onchipfull128norm+amax,preserveBF16WK/norm/RoPE,FLOATscalequantthenFP16scale. Cos/sinFLOATloadsandrequiredcachewritesremain. NoactualHBM/newkerneltimeclaim.'))
 weight=[]
 for m in [3,800,816,830,834]:weight.append(dict(M=m,heads=32,scale_FP32=1/64,CastMulCast_only_interface_budget_bytes=16*m*32,projection_epilogue_composed_budget_bytes=20*m*32,contract='BF16projectionround→FLOATwiden→multiply2^-6→FP16round. NeverFP16castbeforedivide:BF16value2^20wouldcastInfbutoriginalfinal16384finite.'))
 joint=[]
 for m in [12,20,24,6400,6528,6668]:
  h=5120;d=512;f=3*2*m*h*d;w=3*2*h*d;base=3*2*m*h+w+3*2*m*d;new=2*m*h+w+3*2*m*d
  joint.append(dict(M=m,three_KV_ops=f,three_weight_bytes=w,original_three_GEMM_value_bytes=base,joint1536_output_GEMM_value_bytes=new,shared_input_interface_budget_bytes=4*m*h,original_aggregate_reference_floor_us=max(f/376e12,base/1.6e12)*1e6,joint_aggregate_reference_floor_us=max(f/376e12,new/1.6e12)*1e6,contract='ThreeBF16KVoutputsandthree512Dnormsremain,notone1536norm. NewjointGEMMtiling/nativeprecision/firstlayerreadinessconditions.'))
 assert next(x for x in joint if x['M']==6400)['original_aggregate_reference_floor_us']==next(x for x in joint if x['M']==6400)['joint_aggregate_reference_floor_us']
 partitions=[]
 for n,m in [(24,12),(24,24),(6400,6400),(6528,6528),(6672,6668)]:
  main=2*m*15360*640;kv=3*2*m*5120*512;owner_main=2*(n/8)*15360*5120;owner_kv=3*2*(n/8)*5120*512;feature_kv=3*2*m*640*512
  baseline_aux=3*n*5120*2*7/8;baseline_feature=m*5120*2*7/8
  p=((m+7)//8)*8
  normstats=4*m*2*7/8;kv_allreduce=3*m*512*4*2*7/8;kv_RS=3*p*512*4*7/8;kv_AG=3*p*512*2*7/8
  partitions.append(dict(N_physical_aux=n,M_selected=m,feature_RS_uniform_padded_rows=p,feature_partial_required_FP32_padding_bytes_per_rank=4*(p-m)*1536,baseline_main_plus_KV_ops_per_rank=main+kv,complete_weight_row_owner_main_plus_KV_ops_per_rank=owner_main+owner_kv,owner_work_ratio=(owner_main+owner_kv)/(main+kv),feature_shard_RS_main_plus_KV_ops_per_rank=main+feature_kv,feature_shard_work_ratio=(main+feature_kv)/(main+kv),baseline_main_weight_bytes=19660800,complete_main_weight_bytes=157286400,extra_main_weight_bytes=137625600,baseline_context_KV_weight_working_set_bytes=15728640,feature_shard_KV_weight_working_set_bytes=1966080,query_KV_full_weights_still_live=True,baseline_aux_ring_send_bytes_per_rank=baseline_aux,baseline_feature_AG_ring_send_bytes_per_rank=baseline_feature,owner_full_context_AG_ring_send_bytes_per_rank=n*5120*2*7/8,owner_three_KV_AG_ring_send_bytes_per_rank=3*n*512*2*7/8,feature_norm_stat_AllReduce_ring_send_bytes_per_rank=normstats,feature_three_KV_FP32_AllReduce_ring_send_bytes_per_rank=kv_allreduce,feature_KV_FP32_RS_then_BF16_AG_ring_send_bytes_per_rank=kv_RS+kv_AG+normstats,feature_RS_AG_to_baseline_feature_AG_ratio=(kv_RS+kv_AG+normstats)/baseline_feature,feature_AR_to_baseline_feature_AG_ratio=(kv_allreduce+normstats)/baseline_feature,feature_RS_AG_if_full_context_bank_also_retained_send_bytes=baseline_feature+kv_RS+kv_AG+normstats,baseline_compute_only_reference_us=(main+kv)/376e12*1e6,owner_compute_only_reference_us=(owner_main+owner_kv)/376e12*1e6,feature_compute_only_reference_us=(main+feature_kv)/376e12*1e6,scope='Ringlogicalsendmodel,notHCCLphysicalwire/time/overlapmeasurement. Existing3auxgathersunchangedunlesslateselectivevariantcomposed. Feature-shardmainnormglobalstat→localBF16gamma/round→3partialKVFLOAT→rowRS→BF16KVround→3full512norm/RoPE→BF16KVGather. UniformrowRS/AGrequiresP=ceil(M/8)*8;zeroFP32partialsandBF16suffix,posttrim/slots/activepadmaskmustremain. ExtraM→PVectorworknotpricedasGEMMops. PreserveallfullcontextbankreadersoradditsAGback.'))
 for model in partitions:
  base=model['baseline_aux_ring_send_bytes_per_rank']+model['baseline_feature_AG_ring_send_bytes_per_rank']
  new=model['baseline_aux_ring_send_bytes_per_rank']+model['feature_KV_FP32_RS_then_BF16_AG_ring_send_bytes_per_rank']
  model.update(baseline_aux_plus_feature_ring_send_bytes_per_rank=base,new_feature_variant_aux_plus_RS_AG_ring_send_bytes_per_rank=new,feature_variant_whole_context_comm_payload_ratio=new/base)
 assert abs(partitions[1]['feature_shard_work_ratio']-11/18)<1e-14
 assert abs(partitions[0]['owner_work_ratio']-11/9)<1e-14
 assert abs(partitions[1]['feature_RS_AG_to_baseline_feature_AG_ratio']-8071/8960)<1e-14
 assert partitions[0]['feature_RS_AG_to_baseline_feature_AG_ratio']>1
 selectors=[]
 for n,u in [(24,12),(24,24),(6672,6668)]:
  balanced=[u//8+(i<u%8) for i in range(8)]
  # Concentrate unique selected rows subject to the original owner capacity N/8.
  cap=n//8;remain=u;concentrated=[]
  for i in range(8):q=min(cap,remain);concentrated.append(q);remain-=q
  assert remain==0
  selectors.append(dict(N=n,U_unique=u,baseline_aux_send_bytes=3*n*5120*2*7/8,ragged_selected_aux_send_bytes=3*u*5120*2*7/8,balanced_owner_counts=balanced,balanced_fixed_AG_padded_rows=8*max(balanced),balanced_fixed_AG_aux_send_bytes=3*(8*max(balanced))*5120*2*7/8,concentrated_owner_counts=concentrated,concentrated_fixed_AG_padded_rows=8*max(concentrated),concentrated_fixed_AG_aux_send_bytes=3*(8*max(concentrated))*5120*2*7/8,scope='IllustrativeuniquevalidI,U<=M;actualduplicates/order/Iunrecorded. Keepcurrent640columnWshardsbutdeferauxgatheruntilselectoravailable,reconstructorderedIwithduplicates. FixedAGcost8max(U_p),raggedprotocoladdsmetadata/latency;notassumeU=MorsavesN/U.'))
 slots=[]
 for m in [12,20,24,6400,6528,6668]:slots.append(dict(M=m,source_slot_prep_logical_value_bytes=120*m,source_plus_scatter_pair_reads=128*m,fused_raw_INT32_slot_read_bytes=4*m,one_layer_slot_interface_budget_bytes=124*m,three_layers_same_source_bank_CSE_budget_bytes=240*m,three_layers_individually_inline_raw_slots_budget_bytes=372*m,scope='Cast/compare/clamp/div/mod/broadcast/Pack/Not/MaskedFillconventionalvaluemodel.Scalars/metadataunpriced,notHBM. Samegroupbank/B/epochrequiredforCSE;negativeALLsmapinvalidpair(-1,-1),notF20tokenIDs.'))
 lookup=[]
 for m in [12,20,24,6400,6528,6668]:
  lookup.append(dict(M=m,current_three_cycles_Index_tasks=12,current_three_cycles_IndexCheck_tasks=12,baseline_Index_value_bytes=12*516*m,union_one_config_Index_value_bytes=2*516*m,union_two_configs_Index_value_bytes=4*516*m,one_config_lookup_interface_budget_bytes=10*516*m,two_configs_lookup_interface_budget_bytes=8*516*m,baseline_native_C_S_cast_tasks=6,one_config_new_explicit_BF16_bank_cast_tasks=2,two_config_new_explicit_BF16_bank_cast_tasks=4,scope='Eachcycleobserved4Index/4IndexCheck,Tensor/defaultposmapimplies2configC/Spairs;actuallayerconfigunionC=1or2unrecorded. FP32lookuphoistdoesNOTautomaticallyremove6nativecasts. ExplicitBF16bank/newRotaryABIrequiresnativecastpolicyandallreaders.'))
 cos=257/512;x=129/128;out_old=bf16(f32(x*bf16(cos)));out_new=bf16(f32(x*cos));assert out_old!=out_new
 qw=read(O/'attention-indexer-native-numerical-contract-v60.json')['witnesses']['rounded_scale_vs_unrounded_scale'];assert qw['quant_with_FP32_scale']==95 and qw['quant_with_FP16_scale']==96
 mean_models=[dict(local_rows=m,H=5120,HC=4,one_checkpoint_separate_mean_read_bytes=8*m*5120,one_checkpoint_required_mean_BF16_store_bytes=2*m*5120,three_checkpoints_shared_HcPre_input_read_budget=24*m*5120,scope='OnlysameHCinputgeneration/nointerveningmutation;fixedcheckpoints37/38/39havenoEngram(layer1/14). Mean3adds+scaleandBF16storestillrequired.HcPrefusedkernelcompletionmaydelaymeanreadiness/AG;currentcontrolAG→HcPreisnotRAWbutresources/eventsneedprice.') for m in [3,800,816,834]]
 write(O/'index-context-fusion-costs-v64.json',dict(IndexKey_models=key,IndexWeight_models=weight,context_three_KV_joint_models=joint,context_partition_models=partitions,late_selective_aux_gather_models=selectors,slot_metadata_models=slots,context_RoPE_lookup_models=lookup,aux_mean_HcPre_models=mean_models,
  crosslayer_main_partial_model=dict(ordinary_main_ops_per_rank='19660800*Nifallphysicalrowsprocessed',current_selected_main_ops='19660800*M',work_ratio='N/MunlessIavailableearly',FP32_partial_plane_bytes_per_checkpoint_per_rank='4*N*640',baseline_main_BF16_output_bytes='2*M*640',contract='ThreecheckpointmeansalreadyBF16. Streaming3K-slicesintoFP32partial/accumulatorcanavoidconcat/stateplanesbutmustnotroundpartialsBF16;preservefull15360nativeaccumulationorvalidatedtolerance. SelectedIonlycomputedaftertargetsample/rejection,soearlypartialworkmaydoubleatN24/M12. Holdrow/weightgeneration,lifetimeandCubecompetition.'),
  M18_norm_layout_budget='4*M*5120Transposewrite/readifglobalRMSreadsrank-major[8,M,640]directly. Full5120reduce/gamma/BF16outputandNc-prefixbanktailreadersretained;notindependent640norm.',
  crossiteration_incremental_contract='Existingcachealreadybaseline. Reuseonlywhensame(request/tokenvalue,positions,auxhidden3checkpointgeneration,main/KVweights,norm/Ropeconfig,accepted/rejectedprefix,group/page/slotgeneration,lastreader)matches. WithVvalidandDunchangedrows,delta=V-DcanpruneprivateKVprepandretainwritesonlyunderallobservers/duplicate-slotproof. Sameaddress/slot/shapealoneinsufficient;metadata274stilllive.',
  roofline='DeclaredtypedF/P,HMB/L2traffic/B,communicationpayload/latencyandcriticaldependencymax;addworksharingCube/Vector. Logicaltraffic/coldreferenceisnotmeasuredactualphysicalceiling orcausalgain.',overall_complete=False))
 old=read(O/'M18-partition-bank-RoPE-and-slot-contract-v34.json')['source_files'];sourcepins=[]
 refs=[('models/deepseek_v41/dspark.py',old['/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py']),('models/deepseek_v41/model.py',old['/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py']),('ops/rope_dsv4.py',old['/vllm-workspace/vllm-ascend/vllm_ascend/ops/rope_dsv4.py']),('spec_decode/dspark_proposer.py',old['/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py']),('models/deepseek_v41/indexer.py',next(p['sha256'] for p in read(O/'attention-indexer-native-numerical-contract-v60.json')['parts'] if p['path'].endswith('/indexer.py')))]
 refs.append(('spec_decode/llm_base_proposer.py',old['/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py']))
 worker_ref=read(O/'M21-slot-rejection-state-and-packing-contract-v38.json')['source_pins']['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'];refs.append(('worker/model_runner_v1.py',worker_ref))
 for rel,expected in refs:
  path=S/'vllm-ascend/vllm_ascend'/rel;sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==expected,rel;sourcepins.append(dict(path=str(path),sha256=sha,prior166_hash_binding=True))
 write(O/'index-context-operator-contract-v64.json',dict(source_files=sourcepins,source_scope='Existingv34/v38/v60SHA-boundoperatorclausesonly;worker2058-2094readsolelyfortokenIavailabilityaffectinglate/earlyprojection. No newremote/source/frameworkcatalogaudit.',
  facts=['IndexKeyDynamicQuantusesFLOATscaleforINT8thenFP16cachecast;QueryheadcustomquantusesFP16scaleBEFOREquant. Witness95vs96precludesunifyingthembyFP16scaleonly.',
   'Indexweightscalewidth128^-0.5*heads32^-0.5 roundsFP32to2^-6;keepBF16projectionround→FP32multiply→FP16cast. Float16firstcanoverflowtooearly.',
   'Indexmode1candidate_outfreshoutputthenbankcopy24nativecalls;currentAPIcandidateargumentonlyinputforfiltermode2. DirectbankoutputneedsoutABI;notprepare-indicesexistingaliasedoutput/selfcopy.',
   'M18main15360→640perrank/featureAG5120norm,KV3distinct5120→512+3norms. HCmeanBF16/mainBF16/normBF16/KVBF16/normBF16/RopeBF16stagesremain.',
   'Native240auxcheckpointsboundbeforeblockordinals37/38/39,notguessedfromconfigzero/onebasedIDs. 312contextcycles30TASKseachand88mainchainsstrictfunctionalpopulation9936;ancestry/metadataothermodulesretainedseparately.',
   'Slotnegativeall→invalidpairandintwidth,3layersmayusedifferentgroupbanks. None/emptylayerSlotstore returnsaftercurrentprojection;movingguardearlierneedsnormalvalidmetadata/pureprivatecalc/allobservers.',
   'M18C/SisFLOATtablegatherthenBF16castsintobuiltinRotary. ViewCopyreportsbaseM512storagebutactuallycopiesM64tail. ReuseM09FLOATcoskernelwithoutC/Sroundchangesresults.',
   'Targetauxmeansgatherduringtargetforward;tokenIinworker2080/2085depends onsampling/rejectionmetadataafterforward. Late-selectivegathercanretainWshardsbutrequiresdeferral/liveness/metadataandraggedorstaticpaddingprotocol. EarlypartialswithoutIcomputeNnotM.',
   'Feature-shardedKVRS→rowfullKVnorm→BF16KVGathercanavoidfullmainweightreplicationandreducecontextKVcompute,butchangedFP32sum/normprecision andfullcontextbankallreaders/metadata274mustbeaccounted. Attn.wkvfullweightsalsoquerypathconsumers;working-setviewnotVRAMreclamation.',
   'Atfixedcheckpointlayers37/38/39(noEngram1/14),targetHCmeanandfollowingattentionHcPresourceinputarethe samehidden_statesexpressionwithoutinterveningmutation. Theycanconditionalshareloads,butmeanisnotpremixweightedY;reusingsamevaluebyshapewrong. PureauxAllGatherandHcPrebranchesdata-independent,currentcontrolnotfreeoverlap.',
   'llm_base_proposer1048-1055returnsearlyatK0beforecombine. SkippingtargetauxproductionrequiresK0knownbeforeforwardandnootherauxreaders/allTPcollectiveparticipants;currentK5capturedoesnotattestK0active.' ],
  witnesses=dict(KeyScale_not_QueryScale=qw,context_cos_BF16_rounding=dict(BF16_x=x,FLOAT_cos=cos,BF16_cos=bf16(cos),with_required_cos_round=out_old,without_cos_round=out_new,domain='Firstinterleavedcoordinatewithpartner0andfinitevalidsin;referencearithmeticnotnativeRotarytest.'),wrong_Float16_first=dict(BF16_value=2**20,required_scaled_FP16=16384,FP16_first='Inf'),HC_mean_not_weighted_Y=dict(HC_values=[1,2,3,4],mean=2.5,pre_mix=[1,0,0,0],weighted_Y=1)),
  new_model_Torch_NPU_service_capture_or_native_test=False,current_loaded_native_binary_identity_proved=False,overall_complete=False))
 vs=[]
 def v(i,c,s,b,g,d='CONDITIONAL'):vs.append(dict(id=i,candidate=c,choice=s,work_IO_effect=b,conditions=g,decision=d,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,physical_gain_proved=False))
 v('INDEX-KEY-FUSE','F11','WK→norm→RoPE→FLOATscalequant→dualcache','Allvalidfullproducerbudget1556Mbytes,M24=37344B','Keep3BF16boundaries/full128reduce+amax,FLOATscaleBEFOREFP16cachecast,cachepairpublication/validslots/duplicates/allreaders.')
 v('INDEX-KEY-CONSUMER','F11','StandaloneWKthenoneVectornorm/rope/quant/store','1044Mbytesinterfacebudget,WKoutputstillmaterialized','Fullrowstate,tail64,C/SFLOATpolicy,rstdreaders/nativeDQpolicyanddualcachecoherency.')
 v('INDEX-KEY-MASK','F11','Computeprivatekeyonlyvalidslotrows','M→Vkeyprepwork,newgather/packcost','Onlykeybranchpruned;C1/C2latentlongKV/Indexerotherreadersremain. Slotsbounds/duplicateorder/nativeinvalidskipanderrorspreserved.')
 v('INDEX-KEY-TP-OWNER','F11','CanonicalTProwownerkeyprepthenINT8+FP16cachebroadcast','PerTPkeycomputecan/8,130B/rownewpayload','Sameweight/norm/latent/positiongenerationacrossTP,nocrossDPdedup;eachrankappliesownphysicalslots,pad/cachegeneration/allreadyreaders. Queryrowsremainsharded.')
 v('INDEX-SCALE-UNIFY','F11','UseFP16cachescaleasKeyquantdivisor','WitnessFLOAT95vsFP1696','CurrentFLOATquant→FP16metadataordercannotreplacebyQueryheadscaleorder.','EXCLUDED_AS_UNCONDITIONAL')
 v('INDEX-WEIGHT-EPILOGUE','F10','BF16round→FLOAT*2^-6→FP16inepilogue','20M*32Bwithprojection,16M*32BstandaloneCastMulCast','OtherFLOATweightreaders/capturelifetime/nativecasttiesandnonfinites;donotcastFP16beforedivide.')
 v('INDEX-QUERY-COMPOSE','F09','InputDQ/QueryRoPE/Headquantcomposition','Existingv60/v61/v63budgetsreconciled,608DQincludes416keys+192Qinput','Ordinary1280FLOATscaleand128headFP16-beforequantrulesdistinct;QRsource/rowownership+BF16/RINTandheadamaxallpreserved.')
 v('INDEX-CANDIDATE-BANK','F12','QLImode1writesexistingcandidatebank','24nativecopyinterface8M*2048bytes','NeedoutbufferABI(notexistingmode2input),captureaddresses/lastreader/banktail. Source1→4consumersalreadyshared;nocrossiterationreusewithoutquerygeneration.')
 v('CONTEXT-INDIRECT','F27','MainGEMMreads3selectedauxsourceswithoutmaterializedConcat','12MHConcatplusselectedIndexpayloadbudgetconditional','OrderedI/duplicates/negativeboundsand3checkpointBF16means;no3separatelyBF16roundedpartialsum.')
 v('CONTEXT-LATE-SELECT','F27','DeferauxgatheruntilIknown,gatheronlyUselectedrows','Retain640Wshards;raggedUorstatic8max(U_p),notfreeN/Msaving','Iaftertargetsample,retaintargetlocalstates/allauxobservers,globalownerrowplan/orderduplicates;deferral/metadata/protocol/lostoverlappriced.')
 v('CONTEXT-STRIDED-NORM','F27','RMSreadsrankmajor[8,M,640]andpublishesbank','4MHTransposebudget,fullHnormstillrequired','Gatherreadiness/full5120reduce/nativeprecision,gamma,BF16round,Ncsourceprefix/unchangedtail/allbankreaders.')
 v('CONTEXT-ROW-OWNER','F27','ReplicatecompleteWmain,project/normrowowners','Mainweights+137625600B,mainN/Mwork;KVownercompute8xfewerrows','Loadedparamvalues/cardrowmap,allselectedI/positions/weights/metadata,communication/allcontextbankreaders;N24/M12combinedops11/9worse.')
 v('CONTEXT-FEATURE-AR','F29','Shardnormalizedfeatures,3KVFP32AllReduce','Mainnormstat+3KVARsend10759MvsfeatureAG8960M','ChangedFP32sum/nativeRMSround,3full512norms/cache;smaller512dimdoesnotguaranteelesscomm.','CONDITIONAL_WITH_PAYLOAD_PENALTY')
 v('CONTEXT-FEATURE-RS-AG','F29','Globalnormstat→partialKV→rowRS→norm/RoPE→KVGather','Ops11/18baseline;ring8064P+7Mvs8960MfeatureAG,P=ceil(M/8)*8;M12P16communicationworse,mainWnotreplicated','3collectivesvscurrent1,FP32partial/reductionorder/BF16KVround,rowowners/zeroPadding/trimandallbankreaders;retainedfullcontextbankadds8960Mback;fullwkvweightsstillquery-live.')
 v('CONTEXT-CROSS-LAYER-PARTIAL','F27','Compute3mainK-slicepartialsatcheckpointsandFP32accumulate','Potentialremoveconcat/earlylivenessbutN/McomputeifIunknown','PreserveBF16mean,full15360accumulation/tolerance,noBF16partialround;targetCubecontention/lifetime/rowgenerationandfinalreadiness.')
 v('CONTEXT-THREE-KV-JOINT','F29','One1536-outputGEMMfor3distinctKVmaps','Save4M*5120Binputreads;M6400compute-referencefloorunchanged','3weights/3BF16outputs/3separate512norms/cache;jointandstreamingbenefitsnotadditive,firstlayerwaitmatters.')
 v('CONTEXT-KV-CACHE-EPILOGUE','F29','KVnorm→BF16C/SRoPE→cachewithdirect448+64stores','TailSlice/ViewCopy/Concatselectedpayloads,notbaseM512copy','KeepBF16KV/norm/C/S/Ropepolicy,tailmutationallobservers,perlayergroupslots/duplicates/cachepublication.')
 v('CONTEXT-ROPE-HOIST','F29','OnepureFP32C/Sunionqueryfor3layers','12Index/12check→2U/2U,Uconfigunion1or2','Tables/registry/positionsgenerationreadonly,layerfilter/bounds/errors;native6C/ScastsNOTautomaticallyremoved.')
 v('CONTEXT-ROPE-BF16-BANK','F29','ExplicitsharedBF16C/Sbank/newRotaryABI','Ifsameconfigpositionsgeneration,6casts→2Ucasts','KeepactualnativeCastpolicy/BF16round,allC/Sotherreaders/capturebanks;use_cacheTrueexistingmutableAPIisnotdropin.')
 v('CONTEXT-SLOT-INLINE','F29','ComputeINT32pairsinsideperlayerKVstore','124MmetadatainterfaceB/layer;3layerinlinebudget372M','Allnegativeinvalid,int64addresswidth,B/blockgroup/slotgeneration/duplicates;3differentgroupbanksnotsharedbyshape.')
 v('CONTEXT-SLOT-CSE','F31','Reusepairspergroup/B/valuegeneration','ThreeeligiblelayerCSEsave240MmetadataB','Actual_layer_group_idxandbankidentityunknown;pairpreparationonly,cachewrites/metadata274stilllive.')
 v('CONTEXT-NONE-GUARD','F29','CheckperlayerNone/emptyslotsbeforeprivateKVprojection','SkipfullprivateKVcycleifnoreader/store','CurrentglobalNone/emptyhidden/positionsguardsalreadybaseline;newperlayerguardvalidnormalmetadatadomain,pureRopequeryandallobservers.')
 v('CONTEXT-INCREMENTAL','F29','ReuseunchangedcontextKVbysemanticgeneration','Vvalid→delta=V-Drowspluscomparison/keymaintenance','ExistingpersistentKVcachebaseline,accepted/rejectedtargetaux/pos/request/weight/slotgenerationandfutureobservers;samephysicalslotnotenough.')
 v('CONTEXT-LOCAL-NORM','F27','8independent640normsinstead5120norm','Differentdenominators','Global5120statisticrequired;v34toyproofretained.','EXCLUDED')
 v('AUX-HCPRE-SHARED-LOAD','F27','EmitBF16HCmeanalongsideHcPreweightedY/coefficientwork','8MHsharedinputreadbudget/checkpoint,3meansandstoresremain','Sameinputgeneration/currentfixedcheckpointnointerveningEngram,allHcPreoutputs/numericalorder;meanAGreadinessmaywaitfusedkernelcompletion,Vector/Cube/BWcompetition. NotreuseweightedYasmean.')
 v('AUX-ZERO-SPEC','F27','Skipauxmean/gatheronlywhenK0andallauxconsumersinactive','3mean+3AGbranchworkconditional,notcurrentcapturegain','K0availablebeforetargetforward,otheraux/prompt/readers/capturedgraphs/TP-DPcollectiveparticipation;sourceproposeearlyreturndoesnotaloneprovethesegates.','GUARD_BRANCH_NOT_ATTESTED_ACTIVE')
 write(O/'index-context-fusion-variants-v64.json',dict(variants=vs,interaction_rules=['Keyproducerandconsumerfusionbudgetsarealternatives,not1556M+1044M;KeyFP16scalemetadataisnotquantizationscale.',
  'M18rowownerfullW,late-selectiveshardW,featureRS/AGandcrosslayerpartialsareseparatepartition/schedulechoices;priceextraweights,N/M,selectiontiming,collectivecount/retainedbank.',
  'FeatureRS/AGcompute/bytesnotwhole-modelgain;fullwkvweightquerypeerconsumersprecludeclaimVRAM/8.',
  'Lookupfilter/hoist/BF16bank/fusedRoPEstoreandSlice/ViewCopy/Concatbudgetscomposeonce;native4Index/cycleis2configpairs,not4whole268MBtablesread.',
  'SlotCSEandinlinealternatives;metadata274/rejection/anchor/DCP/query-slotpublicationlive. SourceKVcachingnotnewcrossiterationgain.',
  '83types/2082regimesobservedonly;actualtypedcapacity/cachephysicaltraffic/fullbranchcoverageandnewnativeprecisionremainexplicit.'],overall_complete=False))
 print(dict(key_models=len(key),partitions=len(partitions),selective_models=len(selectors),variants=len(vs),source_pins=len(sourcepins),KeyScale_witness=(qw['quant_with_FP32_scale'],qw['quant_with_FP16_scale']),RopeCos_witness=(out_old,out_new)))

if __name__=='__main__':main()
