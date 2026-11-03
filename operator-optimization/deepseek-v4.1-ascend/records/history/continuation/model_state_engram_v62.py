"""Source-conditioned C2/Engram fusion, splitting and numeric reference models."""
from fractions import Fraction
from finalize_origin_coverage_v9 import N,O,read,write

def main():
 assert not (O/'C2-state-and-pool-costs-v62.json').exists()
 H=512;cases=[]
 for label,ss,nn in [('T24_four_even_chunks',[0]*4,[6]*4),('T24_four_odd_chunks',[1]*4,[6]*4),('T6400_two_even_chunks',[0]*2,[3200]*2),('T6528_two_even_chunks',[0]*2,[3264]*2),('T6668_two_even_chunks',[0]*2,[3334]*2)]:
  T=sum(nn);E=sum((s+n)//2-s//2 for s,n in zip(ss,nn));B=sum(s%2 and (s+n)//2>s//2 for s,n in zip(ss,nn));U=sum(min(n,32) for n in nn)
  qpool=16*H*(E+B);qold=16*H*B;update=16*H*U
  cases.append(dict(case=label,T=T,start_positions=ss,used_lengths=nn,all_pages_assumed_valid=True,completed_pairs=E,residual_pairs=B,ring_updated_rows=U,pool_input_logical_bytes=qpool,old_ring_reads_logical_bytes=qold,pool_output_BF16_write_bytes=2*H*E,zero_output_write_bytes=2*H*T,norm_input_read_bytes=2*H*T,norm_output_write_bytes=2*H*T,norm_weight_read_bytes=2*H,norm_auxiliary_rstd_write_bytes=4*T,ring_update_read_plus_write_bytes=update,ring_update_write_only_bytes=8*H*U,residual_singleload_save_logical_bytes=16*H*B,residual_duplicate_exp_elements_removable=2*H*B,pool_norm_intermediate_budget_bytes=2*H*(E+T),additional_zero_write_budget_if_fused_full_final_output=2*H*T,total_pool_norm_zero_interface_budget_bytes=2*H*E+4*H*T,ring_update_all_T_naive_overcount_factor=T/U if U else None))
 assert cases[2]['ring_update_read_plus_write_bytes']==524288 and cases[2]['ring_update_all_T_naive_overcount_factor']==100
 c2=dict(H=H,ratio=2,ring_rows=32,state_descriptor_bytes=17306*32*1024*4,
  definitions={'T':'physicalprojectedrows;mayincludepad. Eachrequestsegmenthasstart s,usedlength n,physicalrowbase andpage p.',
   'E':'sum(valid_page * (floor((s+n)/2)-floor(s/2))). OnlycompletiontokenrowiswrittenbyTOKEN_ALIGNEDpool.',
   'B':'count(valid_page and s odd and at least one completedpair). Atmostonehistorypairperrequest,notE.',
   'U':'sum(valid_page * min(n,32)). Currentkernelmaintainsonlytail32,notallT.',
   'valid_page':'0<p<NUM_CACHE_BLOCKS. Actuals,n,p,skip/generationandownershipnotinthecapturedshapes.'},
  current_pool_work_per_nonresidual_pair={'max_comparisons':H,'ordinary_score_subtractions':2*H,'exp_elements':2*H,'denominator_additions':H,'divisions':2*H,'weighted_multiplies':2*H,'weighted_sum_additions':H,'BF16_round_elements':H},
  residual_three_pass_extra={'score_reread_logical_bytes':16*H,'exp_recomputation_elements':2*H,'score_subtractions_recomputed':2*H,'qualification':'Conventionalpairalgorithmcount;sourceincludesneutralinitializations/reductiontreeandaddresswork. Doesnotstateexactcompiledinstructioncount.'},
  traffic_formulas={'pool_values_read':'16H(E+B)','pool_old_ring_read':'16HB','pool_BF16_write':'2HE','zero_write':'2HT','external_norm_read_write':'4HT+2H+4T includingrstd','ring_update_read_write':'16HU','ring_update_write_only':'8HU','pool_norm_interface_elision_budget':'2H(E+T)','with_full_final_initialization_in_same_fusion':'2HE+4HT total,notaddintermediatebudgettwice'},
  examples=cases,scope='Illustrativevalidallpagesmetadata,notattesteddevicevalidrows. State2.268GBisallocationfootprint;pooldoesnotstreamit. HBMcachetraffic/cachetier/Vectorandspecialfunctioncapacityunbound.',
  numerical_contract=['Keepstablemax/exp/sum/divweightedpool andFP32score/KV. PreserveFP32→BF16pooledroundBEFOREnorm; nonTOKEN_ALIGNEDsourcealreadydoes this.',
   'CurrentTOKEN_ALIGNEDtrue,eps0 andplaceholdernorm_w=out disablefusednorm. Existingfalsebranchcannotbedirectlyselected:compactrowmap,wrongweights/eps. Newindependentnormflagandrealweights/epsneeded.',
   'Incomplete/unwrittenrowsstart+0,thenexternalRMScomputeswithactualgamma/eps. Signedzero/Inf/NaNandallconsumersmatter; poolonlycompletedrows cannotblindskipremainingnorm.',
   'Ringmustreadoldhistorybeforeoverwritingitsphysicalslots. Pool+updatefusionneedsrequestpageownershipandcrossprogramordering;Vectorworkisnotfreeoverlap.',
   'Even-onlystatewrites underRATIO2observabilityproofremainconditionalonrollback/allreaders; doesnotshrinkcurrentSWAanchoredallocation. Keepint64pageoffsetmath.',
   'NormBF16latentfansouttoIndexerupdate_keysbeforeinplace64tailRoPEandlongKVpublication. Do not overwritepreRoPElatentbeforeallIndexerreads; preserveownercard/slots/padding.'],
  boundary_reference='Tref >= max(Q_HBM/B_HBM,Q_L2/B_L2,F_vector/P_vector,N_exp/P_exp,N_div/P_div,dependencychain) in a namedalgorithm/cache/capacitydomain. QlogicalisnotQ_HBM; noactualnumericphysicalceilings.',overall_complete=False)
 write(O/'C2-state-and-pool-costs-v62.json',c2)
 h=4;D=5120;b=32;eng=[]
 for t in [2,3,800,816,830,834]:
  f=2*t*6144*25600;q=2*(t*6144+6144*25600+t*25600)
  ideal=2*t*(3*h+1)*D+4*h*D+4*b*b+t
  two=(8*h+2)*t*D+8*t*h+4*h*D+4*b*b+t
  repeated=10*t*h*D+8*t*h+4*h*D+4*b*b+t
  eng.append(dict(local_tokens=t,WKV_ops_2perMAC=f,WKV_BF16_weights_bytes=2*6144*25600,WKV_one_read_write_value_bytes=q,WKV_reference_compute_us=f/376e12*1e6,WKV_cold_value_io_us=q/1.6e12*1e6,WKV_conditional_reference_floor_us=max(f/376e12,q/1.6e12)*1e6,rotation_M=640*t,rotation_K=b,rotation_N=b,rotation_ops_2perMAC=2*t*h*D*b,FLOAT_execution_mode_and_capacity_unbound=True,gate_ordinary_ops_excluding_rotation=t*h*(9*D-3+7),gate_special_elements={'rsqrt':2*t*h,'sqrt':t*h,'sigmoid':t*h,'abs_clamp_sign_select_each':t*h},gate_ideal_prepared_parameter_IO_bytes=ideal,gate_two_pass_shared_value_IO_bytes=two,gate_two_pass_repeated_value_IO_bytes=repeated,WKV_gate_BF16_interface_budget_bytes=4*t*(h+1)*D,gate_final_copy_interface_budget_bytes=4*t*h*D))
 assert eng[1]['gate_ideal_prepared_parameter_IO_bytes']==485379
 assert eng[2]['gate_two_pass_shared_value_IO_bytes']==139376416
 host=[]
 for T in [24,6400,6528,6672]:
  A=3*T
  host.append(dict(global_tokens=T,local_head_rows=A,hash_INT64_mul_and_XOR_updates=8*T,hash_INT64_remainders=48*T,hash_output_INT32_bytes=192*T,request_ID_intermediate_write_plus_two_layer_reads_bytes=12*T,lookup_format_payloads_per_layer={'BF16_host_unique_payload_bytes':512*A,'INT8_host_unique_payload_bytes':288*A,'MXFP8_host_unique_payload_bytes':264*A},lookup_INT8_repeated_logical_FP32_scale_refs=256*A,lookup_INT8_group_unique_scale_refs=8*A,lookup_INT8_unique_scale_payload_bytes=32*A,lookup_device_BF16_full_output_bytes=512*A,lookup_prezero_write_budget_bytes=512*A,lookup_owned_false_load_elision_benefit_unbound=True,TP_full_head_global_content_per_rank_bytes=12288*T,TP_head_to_row_exchange_content_per_rank_bytes=1536*T,content_ratio=8,TP_transpose_materialized_read_write_bytes=24576*T,global_bank_copy_materialized_read_write_bytes=24576*T,SP_clone_materialized_read_write_bytes=3072*T,partition_partial_WKV_FP32_plane_bytes=4*T*25600,partition_partial_plane_to_head_row_input_content_ratio=Fraction(4*25600,1536).numerator/Fraction(4*25600,1536).denominator))
 qkprep=24*h*D;rotcast=6*b*b
 assert qkprep==491520 and rotcast==6144
 assert host[0]['partition_partial_WKV_FP32_plane_bytes']==2457600
 write(O/'Engram-hash-UVA-and-gate-costs-v62.json',dict(config=dict(layers=2,ngram_lengths=[2,3,4],hash_heads_each_length=8,hash_IDs_per_layer=24,TP=8,local_heads=3,width=256,hc=h,hidden=D,WKV_input=6144,WKV_output=(h+1)*D,rotation_block=b),
  hash_model=dict(rolling_updates='8T INT64mul/XOR pairs across2layers',remainders='48T INT64 % prime',output='192T INT32bytes',request_search='Alreadyoncepertoken,separatekernel;backendabortreasonexplicitlydocumentedifmergedwithhash. Boundbyrequestcount/whileiterations,notBF16TFLOPS.',history='Two layerprogramsrepeatresolution. Sharedhistoryprogramconditionalonregisters/parallelism;hashremaining48Tmods unchanged.',request_intermediate='4Twrite+8Tread=12Tbytes iflaterelided;compilerbarriermeansnotfreedropin.',cachewrite='4VvalidINT32slotswrite plusrawID/map/dead/slotreads underactualstrides. Cachewritebeforehash maynotreorderwithoutphysicalfallback-slotdisjointness.',remainder_specialization='Registeredprime/offsetgenerationfixed:exactintegerreciprocal/mulhi/correction canbestudied; proveTLsignedremainder+64bitwrapoverflow. Runtimeprimedescriptordoesnotattestcompileconstant.',low32_counterexample=dict(prime=16000003,a=1,b=2**32+1,a_mod=1,b_mod=(2**32+1)%16000003)),
  lookup_model=dict(formats={'BF16':dict(code_bytes=512,scale_bytes=0),'INT8_GROUP32':dict(code_bytes=256,unique_scale_bytes=32),'MXFP8_GROUP32':dict(code_bytes=256,unique_scale_bytes=8)},format_constexpr_not_attested=True,
   host_vs_device='Hostmappedcode/scalerows pricedbyB_UVAunknown; IDs/pointertables/BF16outputdeviceIOseparate. No1.6TB/sHBMpricingofhosttables.',
   int8_scale_broadcast='256logicalFP32scalerefs→8grouprefs perrow; unique32Bpayloadsame. ExistingMXFP8already8groups; nofake992Bwiregain.',
   owned_false='Kernelusesrow0fallbackloads beforezerooutput. Maskloadscanconditionallyskipfallbackwhen!owned,butmustfullwritezero. DEAD_IDhashrows canstillowned/padhashed;deadmasknotownership.',
   row_cache='ForA_ownedrequestsandUdistinctownedIDs,minimumhostpayload288U vs288A forINT8underimmutableversion. Actualduplicate/hitrateunknown;BF16decodedcache512B/entry vsencoded288B. Head/shard/table/scalegenerationandcollectiveparticipationpreserved.',
   full_write='LOCAL_HEADS=PAD_HEADS=3,all256outputcolumnsownedvaluesorzeros:176lookup-localprezerosconditionallyremovable. Dummybankandmasktailzerosnotcovered.'),
  parameter_prep=dict(per_body_qk_cast_mul_tasks=3,bodies=208,total_tasks=624,one_body_logical_value_IO_bytes=qkprep,cached_product_bytes=4*h*D,rotation_cast_tasks=208,rotation_cast_one_body_IO_bytes=rotcast,cached_rotation_bytes=4*b*b,scope='Onceperparameterdevice/dtype/versionpreparation;notacrosshiddenactivationlayers. Existingnativeprep13.24usmedianisonlyexposure,notcriticalpathgain.'),
  gate_models=eng,hash_lookup_partition_models=host,
  partition_contract='HeadshardedpartialWKV(global8Tlocal,Klocal768,N25600)hasunchangedpercardFLOPs comparedlocalrows/fullK. Weight314.5728MB→39.3216MB,partialFP32plane4Tglobal*25600 addsreduction/comm;ratio200/3vshead→rowinputcontent,notoldfullAllGather. BF16baselineWKVroundandreductionorderchange mustvalidate.',
  gate_IO_contract='Idealonevaluepassrequiresonchipcapacityforrotation+Hwide2norm/dotbarrier+residualandsharedV. 2passmodelsaddcoeff4Thwrite/readandhiddenreread; allarelogicalvaluemodelsnotmeasuredHBM. WKVfusedinterfacebudgetsnonadditivewithgatecopy/layoutbudgets.',
  numerical_contract=['WKV isalreadycombined4keys+1sharedvalue;keepBF16projectedboundarybeforekeynorm/dot.',
   'Rotationapplieshiddenforgate;residualusesoriginalhidden. Block32FLOATnotdense5120,HF32/fullFP32activepolicyunbound. Noorthogonalitycancellationofnormassumed.',
   'Gate2RMSoverfullH,weightedFP32dot,eps,clamp1e-6,sqrt,signbit(-0),sigmoid,mask,FP32g*V+BF16residualthenBF16round. NoFMA/reassociationclaimedexactwithouttolerance.',
   'Deadmaskgate=0 doesnotpermitreturnhiddenforNaN/Inf/signedzero. 0*Inf=NaN;valuefanouttoall4HCstreamsremains.',
   'Globalfixedbankpublication,TPtranspose/clone/sourceviewaliases/paddingandreadyallreadersneededforcopyelision. DPexchangealreadyfeedsbothlayers. Empty/localdeadcannotunilaterallyskipcollectives.',
   'Rotatedcurrentprofileordinarygate208bodies distinctfromsourceunrotatedhardwarefusedbranch,CPU/deviceint8/TP1/slotless/dummy/validcountguardfamilies;conditionalcontractslisted,wholemoduleopen.'],
  roofline='Tref>=max(Qhost/B_UVA,Qdevice_HBM/B_HBM,Q_L2/B_L2,Ftyped/Ptyped,Nmod/P_INT64mod,Nspecial/Pspecial,dependencies). Sharedresources/countsaddnotfreelymaxallstages; absenttypednumericcapacitystaysparameter.',overall_complete=False))
 variants=[]
 def v(i,c,choice,decision,budget,gate):variants.append(dict(id=i,candidate=c,choice=choice,decision=decision,work_IO_or_partition_effect=budget,conditions=gate,conditional_operator_analysis_deliverable=True,new_implementation_or_benchmark=False,physical_gain_unproved=True))
 v('C2-NORM','F18','Pool→BF16round→RMS in token-alignedrows','CONDITIONAL','2H(E+T)intermediatebytes;additional2HTzeroonlyifallfinalrowswritten','Realnormweight/eps,newnormflag,keepBF16intermediateandnativeprecision/auxrstd;incompleterowssemantics/allreaders.')
 v('C2-RES2','F18','Specializeresidualratio2 toone scoreload/exp','CONDITIONAL','16HBlogicalread+2HBexp budget;B≤requests','Keephistory/new-rowselection,max/sub/exp/sum/divorder;registerpressureandactualB.')
 v('C2-SPLIT','F18','Pool completion forks norm and ringmaintenance','CONDITIONAL','Update→normiscontrolnotRAW;nofree5.38ussaving','Oldringreadcompletion,resources/streams,cacheownership/consumerwait;bothVectorandmemory.')
 v('C2-UPDATE-FUSE','F34','Pool+ringupdateperrequestonekernel','CONDITIONAL','Canreuseprojectedtailtilesonchip;16HUlogicalupdateIOceiling','Readalloldneededringbeforewrite,requestphysicalpagedisjointness/crossprogramsync;noautomaticgridbarrier.')
 v('C2-EVEN','F34','OnlyobservableevenpositionsinRATIO2ring','CONDITIONAL','~halfupdatewritesundermetadata;stateallocatorunchanged','Rollback/allreaders/nativeABIandpageownership;keepolderobservableevenhistory,notjustlatest.')
 v('C2-ALLOC','F34','Halfarray=>halfcurrentpoolcapacity','EXCLUDED_IN_UNCHANGED_ALLOCATION','2.268GBdescriptorisnotpercalltrafficorindependentlyallocatedsaving','Priorv25SWAanchoredtupleallocationanswerretained;noframeworkreaudit.')
 v('C2-FANOUT','F19','Norm/RoPE/cachewrite withIndexKeyfork','CONDITIONAL','Tail64only+selectedscatterrows;wholecacheextentnotIO','PreRoPElatentneededbyIndexKeymatmul;retainsBF16normboundary,allslotmasksandpublication;neverinplacerewritebeforeforkreads.')
 v('C1-COMPOSE','F33','C1linear norm tailRoPE selectedscatter','CONDITIONAL','v61projectioncost+norm/RoPEselectedpayload;nomandatewholenormalizedGMplane','Full512normbarrier,preRoPEIndexerconsumer,BF16roundand128slotlongcachevsSWAphysicalrows;1600ambiguousassignmentsnotrewritten.')
 v('ENG-PARAM','F25','Prepareq*kFP32androtationFP32oncepergeneration','CONDITIONAL','491520+6144BprepIO/body;624+208TASKsexposure','Actualweightdevice/dtype/version/staticlifetimeandcapturedreaderbanks;noactivationreuseacrosslayers.')
 v('ENG-GATE','F25','FuseordinaryrotatedgateafterWKV','CONDITIONAL','FullHnorm/dot+sharedVidealvs2passIOmodels','PreserveBF16WKVround,rotationresidualdistinction,specialfunctions/signbit/reductionorderandNaNmasksemantics;capacity/tilingexplicit.')
 v('ENG-WKV-GATE','F25','WKVpanelsfeedBF16gatepartials','CONDITIONAL','4T(h+1)Hintermediatebudget,notallguaranteedremovable','FullHkeynorm/dotrequiresbarrier/partials;sharedvaluefanout/BF16round;mode/resourcejointcost.')
 v('ENG-EARLY-HIDDEN','F25','Rotation+hiddennorm independentofWKV','CONDITIONAL','NoRAWfromWKVtohiddenprep;bothbranchescomputeadditive','HiddenBF16ready+residualretained;bothrotationandWKVmaycompeteCube;nofree7usgain.')
 v('ENG-HEAD-WKV','F25','HeadshardedpartialWKV thenreduce-scatter','CONDITIONAL_WITH_COST_PENALTY','PercardFunchanged,weight/8;partialFP32plane200/3timeshead-rowinputcontent','ComparebestF24rowexchange,collectiveprotocol,pad/rowownership anddifferentFP32sum→BF16roundorder.')
 v('ENG-SCALE','F24','INT8group32scalesload8+broadcast','CONDITIONAL','256→8logicalrefs;unique32Bunchanged','ActiveINT8format/strides/groupgen;MXFP8alreadybroadcast;coalescing/cachetrafficnotinferred.')
 v('ENG-OWNED','F24','Skiprow0fallbackhostloadsfor!owned','CONDITIONAL','288*(A-Aowned)INT8uniquevaluebudget;couldzero','Actualownedcounts,fulloutputzeros;deadtokenandunownednotidentical;collectivesretainparticipation.')
 v('ENG-DEDUP','F24','SameIDlookupdedup orversioneddevicerowcache','CONDITIONAL','INT8host288Aowned→288Uowned;hitratio/capacityunknown','Immutabletable/code/scalegeneration,head/shard,decodedroundidentical;nocrosslayerreuseofnonidenticalIDs.')
 v('ENG-HISTORY','F24','Resolvehistoryonceforbothhashlayers','CONDITIONAL','Samein_batch/window/fallbacksourcesreused;48Tmodsstillremain','Keepsseparatereqsearchbaseline,INT64wrap/sourcepriority/blockedprefix;registeroccupancy/parallelismtradeoff.')
 v('ENG-MOD','F24','Exactconstant-primeINT64remainder','CONDITIONAL','48T%replacebyexactmulhi/correctiononlyunderproof','Primes/offsetsgeneration,TLsignedsemantics,64bitoverflowandspecializationcodecapacity;noFP32reciprocal/low32truncate.')
 v('ENG-REQ-MERGE','F24','Directmergeexistingbinarysearch+hash','REJECT_DROPIN_ON_CURRENT_DOCUMENTED_BACKEND','Reqsearchalreadyoncepertoken,not2;12Tinterfaceonlytheoretical','CurrentbishengirsourceexplicitlydocumentsmergedkernelLLVMreleasedbufferabort. Newalgorithm/compilerfixwouldbeaseparatevalidatedchoice.')
 v('ENG-HASH-CACHE-ORDER','F24','Writecacheafterhashorblindcurrentdeadskip','NOT_UNCONDITIONALLY_LEGAL','No48Tmodsorcollectivesdeclareddead','Physicalfallbackslotread/writealias,negativehistory/pad/DEAD_IDpropagationandfutureaccepted/rollbackstate.')
 v('ENG-ZERO','F24','Elidelookup-localoutzeros','CONDITIONAL_SOURCE_DOMAIN_PROVED','176prezeros*2Tglobal*3*256Bwrite;otherzerosretained','Kernelalllocalhead/widthfullwrite,allocationnotaliasinflightreaders;dummybank/masktaildifferentsemanticwrites.')
 v('ENG-ROW-EXCHANGE','F24','TPhead→rowalltoall anddirectrowlayout','CONDITIONAL','12288T→1536Tperrankcontent;transpose/bank/SPcopybudgetsnotadditive','DPpad/chunk/headorder/TPparticipation,clonelifetimeandfixedglobalbankallreaders;contentnotwiretimegain.')
 v('ENG-FINAL-STORE','F25','Gatewritehiddenbankdirectly','CONDITIONAL','4ThHBF16copyinterfacebudget','Readoriginalhiddenbeforewrite,nootheraliasreaderandcapturedbankmutation/lifetimeobservers;shortgraph38TASKbodydoesnotcontainnativeTensorMove.')
 v('ENG-NORM-REUSE','F25','Cancelrotationnorm/reuseacrosslayeroriteration','EXCLUDED_WITHOUT_VALUE_PROOF','SameH/blockshapeisnotorthogonalityorvalueidentity','Currentweights/hiddenchange;fullHnormandsourceBF16/FP32contractsretained.')
 v('ENG-ID-VIEW','F24','PassperlayerstridedhashIDviewdirectlytoUVAlookup','CONDITIONAL_SOURCE_STRIDE_DOMAIN','176Slice/contiguouscopies,192TselectedINT32read+writebytesperlayer','Hashoutput[T,2,24]perlayerstride48/inner1;UVAkernelacceptsids_stride_t. Keepallotherlookup/collectivebranchcontiguityrequirementsandbanklifetime.')
 assert len(variants)==25
 write(O/'state-engram-fusion-variants-v62.json',dict(variants=variants,interaction_rules=['C2NORMinterface/zeroandC1/C2projection/v61budgetsnotoverlapaddtwice. Fusionnorm+RopekeeppreRoPEfork.',
  'ENGPARAM+GATEjointcostexcludesrepeatparamprep,whilepreparedparamreadsremain. WKVpanelsneedfullHbarrier;noallintermediateelisionpromise.',
  'Lookupscale8refsdonotchangeunique32Bhostpayload. Dedupcache/ownedskipbenefitsjointlyuseUowned,notindependentpercentssummed.',
  'TProwexchange andheadpartialWKVarealternativesinpartitioncost; compareagainstbestrowexchange,notclaimboth8x.',
  'Vector/GEMM/UVA/commsharefinitecapacity;separatestagesumsandexposureare notcausale2egain.'],overall_complete=False))
 print(dict(C2_examples=len(cases),Engram_local_models=len(eng),Engram_global_models=len(host),variants=len(variants),state_bytes=c2['state_descriptor_bytes']))

if __name__=='__main__':main()
