"""Price extra candidate/partial/pad work, rather than assuming fusion only deletes work."""
import math,struct,hashlib
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,W,read,write
S=W/'frozen-source-snapshot'
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def bf16(x):
 b=struct.unpack('I',struct.pack('f',x))[0];b=(b+0x7fff+((b>>16)&1))&0xffff0000
 return struct.unpack('f',struct.pack('I',b))[0]
def main():
 assert not (O/'router-vector-fusion-costs-v63.json').exists()
 top=[]
 for e,k in [(384,6),(128,3)]:
  for width in [16,32,64,128,e]:
   counts=[min(k,min(width,e-j)) for j in range(0,e,width)];C=sum(counts);r=len(counts)
   top.append(dict(E=e,k=k,expert_tile_width=width,column_programs_per_row=r,candidate_count_per_row=C,baseline_logit_write_read_bytes_per_row=8*e,candidate_write_read_bytes_12B_per_row=24*C,candidate_write_read_bytes_8B_per_row=16*C,merge_recompute_bias_adds=C,merge_recompute_bias_logical_read_bytes=4*C,candidate_8B_plus_merge_bias_to_logits_ratio=2.5*C/e,candidate_to_logits_12B_ratio=3*C/e,candidate_to_logits_8B_ratio=2*C/e,source_nonlinear_elements_per_row=e,final_selected_weight_count=k,qualification='12B=(biasedrankscore,unbiasedweightscore,INT32id). 8B=(unbiasedscore,INT32id),recomputeidenticalFP32biasaddwithsameimmutabletext/visionbias;extra4Cmerge-biaslogicalreadspricedseparately. Workspaceonlyifcrossprogrammaterialized;singleprogram/onchipmergecanavoiditbutparallelism/UB/syncdifferent. DenseGEMMworkstill2MHE.'))
 assert next(x for x in top if x['E']==384 and x['expert_tile_width']==16)['candidate_to_logits_12B_ratio']==1.125
 qr=[]
 for m in [2,3,800,816,830,834]:
  h=1280;qr.append(dict(M=m,H=h,norm_to_one_quant_BF16_interface_budget=4*m*h,duplicate_quant_producer_budget=3*m*h+4*m,joint_QB_IndexQ_quant_input_read_budget=m*h+4*m,fully_composed_norm_quant_reuse_joint_budget=8*m*h+8*m,scope='Atindexsources2originalquantizersand2GEMMconsumers. Atotherlayersonlysinglequantizer,noduplicatereusebudget. Allremoveinterfacesonce;allroundedBF16QRvaluesstillmathematicallyproduced. FLOATrowscaleandINT8outputsremain.'))
 assert next(x for x in qr if x['M']==3)['fully_composed_norm_quant_reuse_joint_budget']==30744
 pads=[]
 for m,p in [(800,816),(10,16),(20,24)]:
  h=5120;e=384
  pads.append(dict(M=m,P=p,H=h,MC2_hidden_pad_one_read_write_bytes=2*h*(m+p),MC2_logits_pad_one_read_write_bytes=4*e*(m+p),MC2_hidden_valid_prefix_copy_budget_bytes=4*m*h,MC2_logits_valid_prefix_copy_budget_bytes=8*m*e,MC2_hidden_required_zero_suffix_bytes=2*(p-m)*h,MC2_logits_required_zero_suffix_bytes=4*(p-m)*e,MC2_TopK_pad_rows_nonlinear_elements=(p-m)*e,RS_concat_read_write_budget_bytes=4*p*h,RS_required_zero_suffix_bytes=2*(p-m)*h,scope='Typedillustrativemodels,notclaimall3MC2padpairsobserved;actualobservedMC2only800→816. RS10→16and20→24capturedseparately;preserveallcollectivecounts/bankmasks. N/AMemSetunpriced.'))
 output=[]
 for t in [10,20,24,6400,6528,6672]:output.append(dict(global_output_rows=t,H=5120,WO_B_output_bank_copy_interface_budget_bytes=4*t*5120,scope='960directtargetTensorMoves6400/6528/6672 observed;smallcaseshereformulasnotclaimcopyexists. Existingv61WOA→WOB4MRbudgetdistinct,butfinalbankandRSfusionmustavoidduplicatecount.'))
 cp=dict(actual_rows=830,padded_rows=834,heads=64,D=512,full_zero_write_bytes=2*834*64*512,active_prefix_zero_overwritten_bytes=2*830*64*512,active_prefix_copy_read_write_bytes=4*830*64*512,combined_copy_plus_suffix_zero_value_bytes=2*(830+834)*64*512,scope='40native830prefixTensorMovesand40full834zeroscurrentfunctionalpopulation;sourceboundconditionalpair,notperrootaliasproof. Empty/dummy2/3zerosdistinct. Attentionproducerdirectpadded/sendlayoutbudgetcomposedwithv60RoPElayout;do notsumoverlappingelisions.')
 assert cp['full_zero_write_bytes']+cp['active_prefix_copy_read_write_bytes']-cp['combined_copy_plus_suffix_zero_value_bytes']==cp['active_prefix_zero_overwritten_bytes']
 x=[1.,129/128];inv=f32(1/math.sqrt(f32((f32(x[0]*x[0])+f32(x[1]*x[1]))/2)));z=[f32(t*inv) for t in x];zb=[bf16(t) for t in z];q=[round(t/max(z)*127) for t in z];qb=[round(t/max(zb)*127) for t in zb];assert q==[126,127] and qb==[127,127]
 bias=[-2.,0.];logits=[4.,0.];s=[math.sqrt(math.log1p(math.exp(t))) for t in logits];assert logits[0]+bias[0]>logits[1]+bias[1] and s[0]+bias[0]<s[1]+bias[1]
 idmodel=dict(M_graph=3,layers=40,cast_read_write_per_layer=36,equal_read_write_per_layer=27,fill_store_per_layer=8,select_read_write_per_layer=51,TopK_INT64_read_per_layer=24,baseline_all_layers_bytes=5840,one_CSE_producer_plus_40_INT64_consumers_bytes=1082,one_CSE_interface_budget_bytes=4758,inline_raw_INT32_in_each_TopK_bytes=480,inline_budget_bytes=5360,scope='GraphINT32→INT64-1sanitizerpurevalueIO;38M+8prep,8Mconsumer;fullbaseline46M+8. ScalarsnotassumedextraGMloads. Samegeneration/rowplan40layersconditional;notmeasuredHBM/timegain.')
 assert idmodel['baseline_all_layers_bytes']-idmodel['one_CSE_producer_plus_40_INT64_consumers_bytes']==idmodel['one_CSE_interface_budget_bytes']
 write(O/'router-vector-fusion-costs-v63.json',dict(TopK_column_partition_models=top,QR_norm_quant_joint_models=qr,padding_models=pads,output_bank_models=output,CP_padding_model=cp,graph_ID_value_IO_model=idmodel,
  local_TopK_merge_proof='ForanequivalentTOTALorder,anelementoutsideitsownpartitiontopkhasatleastkpartitionelementsbeforeit,soitcannotbeinglobaltopk. Unionofpartitiontopkcontainsglobaltopk. MatchingnativeSortties/NaN/InfandFP32score/visionbias orderremainrequired;predicatealoneisnotnativecorrectness.',
  router_compute_split_K='AnyKsplitaddspartialFP32M×Eplanes/reductionandchangesFP32sumorder;same2MHEplusreduce,notfree24coreparallelism. NormHsplitlikewiseaddsMrsqrt/reductionmetadataandoffchippartials;3blocksnotproofphysicalunderutilization.',
  reference='Maxofdeclaredtypedcompute/HBM/L2/sort/special/dependencybounds,withsharedresourcesadditive. FLOATrouterpeakunbound;Exp/Ln/Sqrt/SortcannotbepricedbyBF16CubeTFLOPS. LogicalvaluebytesnotactualHBM. No measuredgain.',overall_complete=False))
 # Narrow current source facts with exact hashes and line bands.
 parts=[]
 bands={'csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash_without_group.h':[(144,216),(244,311),(349,420)],'csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_tiling.cpp':[(128,151),(190,214),(519,539),(618,645)],'csrc/attention/inplace_partial_rotary_mul/op_kernel/inplace_partial_rotary_mul.h':[(125,143),(169,226),(230,299)]}
 for rel,bs in bands.items():
  path=N/'router-vector-operator-source-v63'/rel;lines=path.read_text().splitlines()
  for lo,hi in bs:parts.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),first_line=lo,last_line=hi,source=lines[lo-1:hi]))
 write(O/'router-vector-numerical-contract-v63.json',dict(parts=parts,prior_contracts=['F04-precision-routing-and-row-contract-v29.json','F04-F20-ID-algebra-and-generation-review-v29.json','projection-operator-numerical-contract-v61.json','attention-indexer-native-numerical-contract-v60.json','state-engram-operator-contract-v62.json'],
  current_source_facts=['TopKtilingdefaulttid2eidDtypeINT32;hashFlagrequiresbothinputIdsandtid2eid. Currenttypedtid2eidDT_UNDEFINEDisnotEngramhashwork. ConfigG1andnormType2callerselectwithout-groupconsistentpath;entry/tilingexecutionnotattested.',
   'CurrentTopKnorm2computesFP32Exp(x),+1,Ln,Sqrt,THENvision/textbiasforordering. SelectedweightsGATHERUNBIASEDscores,ReduceSum+eps,Div,thenMuls. renorm0doesnotdisablethisnorm2normalization.',
   'out_flagFalseguardsCopuOutXNorm;fullFLOATthirdoutputallocated/describedbutnotwritteninreviewedpath. outOutQueueE4allocatedinUBregardlessflag,notGMtraffic.',
   'Sourcewithout-groupfullSort(E)withinUBalreadyfusednonlinear/bias/topK/weights. Candidatepartialselection/newepilogueisalternativealgorithmwithtie/NaN/nativeordergates,notremovingmanyexistingkernellaunches.',
   'TopKSelectTopKExpertIdxcontainsDataCopyextentexpertCountAlignwhilek-queuesuseAlign(k). Thissourceallocation/copyextentisnotusedtoinferactualGMtrafficorcurrentnativefault;loadedbinaryidentitynotproved. Newkernelmustestablishitsownbufferbounds.',
   'RoPEcurrentABAinterleavetouchesonly64tailwithstridedGMcopy,FP32cos/sinbroadcastacrossheads,partnerGather,twoMul/paritysign/Add,CAST_RINTBF16. NegateSinimplementedinparitysign;do notinventseparateNegGMkernelinflagpath.',
   'NormCastalreadydualoutputfusedandFP32iswidenedroundedBF16. OrdinaryRMS→DQretainsBF16QRround;sourceordinaryquant1280FLOATscaleisnotIndexer128FP16headquant.',
   'RMS/quantprojection/ID/copyelisionrequiresallotherconsumers,immutableparamgeneration,exactrowmap/slot/mask,banklifetimeandcurrentcollectiveparticipation. Shapes/timeonlydonotprovealias.' ],
  witnesses={'RMS_rounding_cannot_cancel_before_quant':dict(input_BF16=x,FP32_reference_normalized=z,BF16_RNE_normalized=zb,quantize_unrounded=q,quantize_rounded=qb,domain='PositiveH2RMSwithgamma1,FP32illustration,quantround(127*z/maxabs). RepeatpaircoversH1280idealbalancedreduction;notnativecorrectness/scale-policytest.'),
   'nonlinear_bias_rank_counterexample':dict(logits=logits,bias=bias,raw_plus_bias=[logits[i]+bias[i] for i in range(2)],sqrtsoftplus_plus_bias=[s[i]+bias[i] for i in range(2)],raw_choice=0,required_score_choice=1),
   'stable_softplus_not_identical_to_source':dict(x=100,reviewed_FP32_Exp='overflowsto+Inf',source_sqrt_log1pexp='+Inf',stable_sqrtsoftplus_approx=10,implication='Newstablefunctionmaychangeweights/ties/nonfinitebehavior;actualcapturevaluesunrecorded.')},
  published_reference_screen=dict(search_performed=True,used_for_fixed_version_semantics=False,reason='PubliclatestunifiedTopKdocs/forks/AIGCwerenotboundtothis9.2customhash/visionoperator;actual166sourceandpinnedcallerusedinstead.'),
  current_loaded_native_binary_identity_proved=False,new_Torch_NPU_model_or_native_test=False,overall_complete=False))
 vs=[]
 def v(i,c,s,b,g,d='CONDITIONAL'):vs.append(dict(id=i,candidate=c,choice=s,work_IO_effect=b,conditions=g,decision=d,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,physical_gain_proved=False))
 v('ROUTER-WIDEN','F04','ConsumeBF16normalizedplaneandwideninsideFP32router','8MHFP32interfacewrite/readbudget','KeepBF16roundedvaluesandliveBF16expert/sharedplane;actualweight_fp32branch,otherFP32readers/nativeprecision.')
 v('ROUTER-ROW-EPILOGUE','F20','OnefullErowGEMM→source-nonlinear/bias/topK','8MElogitinterfacebudget,FloatGEMMworkunchanged','FullrowEavailableonchip,sourcefunctionorder/visionIDs/bias/unbiasedweights/eps/div/order/ties;Cube-Vectorhandoffcapacity.')
 v('ROUTER-COLUMN-TOPK','F20','Expert-columnprogramslocalTopKthenmerge','12Bvs8Bcandidatespriced;E384tile16/12Bworkspace1.125×oldlogitinterface','Equivalentscoretotalorder/nativeSortties/NaN,biasedrankvsunbiasedweights,parameterreads/newmerge/sync/parallelism.')
 v('ROUTER-PARTIAL-SELECT','F20','ReplacefullUBsortwithpartialTopK','AllEscorefunctionsstillrequiredunlessknownexpertset;comparisonalgorithmnotmeasuredSortinstructiongain','k3/6,exactnativeorder/duplicateIDsandnonfinites;Vectorselectorimplementationandlaunch/UBcost.')
 v('ROUTER-NORM-SPLIT','F04','SplitfeatureRMSorGEMM-KtoparallelizeM2/3','Addreduction/FP32partialplanes,sharedcapacity','Preserveoriginalnorm/RMSBF16roundandFP32GEMMsumtolerance;3blocksnotproofphysicalutilization.')
 v('ROUTER-NONLINEAR-CANCEL','F04','DropRMSscale/BF16roundorselectrawlogitswithbias','RMSquantandnonlinear-biascounterexamples','CurrentroundedBF16/nonlinearandbiaspreventunconditionalrewrite. Uniformbias/exacttie/knownvalueswouldbeaseparatecase.','EXCLUDED_AS_UNCONDITIONAL')
 v('ROUTER-STABLE-FUNC','F20','SubstitutestableSoftplusforExp+1+Ln','Canchangeoverflow/underflow/ties/weights','CurrentFP32primitiveorderpreservedunlessexplicitnewnumericalcontractapproved;sourcex100counterexample.','NOT_EQUIVALENT_BY_DEFAULT')
 v('ROUTER-UB-OUT','F20','Do notallocateunusedoutOutQueuewhenoutflagfalse','4EUBbytes/program,NOnewHBMoutelisiongain','AllCopuOutXNormusesguarded,tiling/queueinterfaceandnativepath;capacitybenefitnotautomaticlatencygain.')
 v('ID-ONCE','F20','SanitizeINT64IDsonceperinputgenerationandrowplan','Cast/Equal/Fill/Selectinterfacesandrepeatlaunches','fonly-1→0,rawIDsandnegativevisionIDsretained,fcommuteszero-pad/select;alllayerplans/bankupdatesandgenerationinvalidation.')
 v('ID-IN-TOPK','F20','LoadrawINT32IDsandinline-1→0inTopK','Graphbaselineprep38M+8plusTopKIDread8M,fusedrawID4M:budget42M+8/layer;M3/40layers5360Blogical','Inputdtype/rowplan/adaptorABI/otherIDreadersandvisionpredicate;bias/hashgenerationnotsharedbyshape.')
 v('MC2-DIRECT-PAD','F20','Norm/routerproducerswritepaddedBF16/logitsbanks','Validprefixcopy4MH+8MEbudget;zeroSuffixstillrequired','M800→P816validprefixdifferentfromM816source;keepunrounded/roundedplanes,sharedconsumers/masks,bankaliasesandPcollectivecount.')
 v('MC2-MASK-PAD','F20','BOOLpad/INT8casts/IDpadlocalpack','Canremoveconversionintermediates/launches,notmandatefewerrequiredpayloadbytes','KeepBOOL0/1semantics,IDzero-padding,per-rankrowmapandallinputs. ScalarCastemptyextentunknown;N/AMemSetnotdeclareddead.')
 v('MC2-PAD-TOPK','F20','Avoidnonlinear/sortonexplicitfalsepaddingrows','ForM800P816,16*384functionelementspercallconditional','OutputsafeexpertIDs/weightsandallconsumersdispatchmaskprotocol;do notskipothercardscollectivesorassumeall640M816callswerepadded800.')
 v('MC2-MEMSET','F21','DeleteunknownMemSetasredundantPadclear','3regimes960execNOwork/bytevalueassigned','Saved32DBshavenodestination/length;workspace/controlvsoverwrittenpayloadunresolved. Remainpending.','PROFILE_INSUFFICIENT')
 v('QA-RMS','F05','Q_Aepilogue→full1280RMS','4M*1280BF16interfacebudget','Full1280rowbarrier;QAoutputBF16roundBEFOREnorm,auxrstdandQRotherreaders/nativeFP32reduce.')
 v('QR-NORM-QUANT','F05','RMS1280→BF16round→ordinaryrowDynamicQuant','4MHsingleinterface;fullycomposedQRreuse/jointbudget8MH+8M','KeepBF16round,globalrowamax/scale/round/nonfinitepolicy;atindexsourcesbothQB/IndexQconsumersshare,otherlayerssingleconsumer.')
 v('QR-COMMON-QUANT','F05','ReusequantproducerthenjointQB/IndexQreads','v61reuse3MH+4M,jointreadMH+4M;nonadditivewithNormQuant','SameQRrows/quantpolicy/currentparameters,peroutputscales/NZweights;distinctFP16headquantnotreplaced.')
 v('KV-NORM-ROPE-CACHE','F08','KVfull512norm→BF16round→64tailRoPE→selectedscatter','4M*64tailwrite/readandselectedcachepayload,notwholecacheextent','Full512RMS/readers,preRoPEIndexerforkatcompressors,actualslotvalidity/duplicates/alias/generation;RINTboundary.')
 v('QB-ROPE','F07','QBepilogueemitsBF16roundedthenrotated64tail','4M*64heads*64tailinterfacebudget','KeepQBoutputBF16roundbeforeFP32RoPE,RINTresult,head/cosbroadcastandattentioninputlayout.')
 v('O-ROPE-LAYOUT','F16','AttentionBF16output→inverse64tailRoPE→TPsendlayout','v60tail+transposejointbudget,notfull512RoPEtraffic','ExistingBF16attentionround,samehead/cardrowmap,CPpadding andAlltoAllprotocol;notdoublecountproducercopy.')
 v('WO-B-BANK','F16','WriteWO_Bdirectlyintoprojectedoutputbank','4THfinalTensorMovecopybudget;M6400=131072000B','Alloutputalias/readiness/capturedbankcontracts;BF16localpartialbeforeReduceScatter. Combinev61panelbudgetonce.')
 v('RS-PAD-CONCAT','F16','ProjectionproducerwritesPbankprefixandsuffixzeros','4PHConcatread/writebudget;zeropayloadstillrequired','CurrentP=16/24RSshape/countandperrankBF16partialround;activeMview/otherreaders/padprotocol.')
 v('CP-PAD-SEND','F16','Emitactualattentionrowsandzeroonlypaddedsuffixinfinalsendbank','CP830→834overwrittenprefixzero54.39488MBconditional;jointRoPE/layoutcountonce','Mactual/perrank/Pglobalroworder,localemptyquery/globaloutputs,allbankreaders;sourcefullzeroisnotuniversallydead.')
 write(O/'router-vector-fusion-variants-v63.json',dict(variants=vs,interaction_rules=['QRreuse+normquant+jointinputbytescombined8MH+8M,indexsourcesonly;notindependentpercentssummed.',
  'Routerone-rowepiloguevscolumnlocalTopKvsKsplitaredifferenttilings;pricedcandidate/partial/mergeworkmustnotbeomitted.',
  'TopKalreadyfusednonlinear/bias/selection/selectednorm;thirdoutputGMwritenotpresentinreviewedoutfalsepath.',
  'Directpaddedbank+outputRope/sendlayout+WO_Bcopy/RSconcatcomposeoverlappinginterfacesonce;requiredzeros/collectivecountsremain.',
  'N/AMemSet3regimesstayspending;scalarCastemptyshapeandactualmask/rowmetadataunpriced,notfilledwith0.',
  'Alltimeexposureisnotmeasuredgain;FLOAT/Vector/sort/special/UVA/HCCLcapacitiescannotbereplacedbyBF16Cubepeak.'],overall_complete=False))
 print(dict(partition_models=len(top),QR_models=len(qr),padding_models=len(pads),variants=len(vs),rounding_witness=(q,qb)))

if __name__=='__main__':main()
