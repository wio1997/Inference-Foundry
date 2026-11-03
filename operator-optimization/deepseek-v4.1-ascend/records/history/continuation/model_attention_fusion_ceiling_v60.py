"""Operator-specific work, traffic, numerical witnesses and conditional reorganizations."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,math,struct
from finalize_origin_coverage_v9 import N,O,read,write

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def f16(x):return struct.unpack('e',struct.pack('e',x))[0]
def bf(bits):return struct.unpack('>f',struct.pack('>I',bits<<16))[0]
def binprefix(a):
 lo=0;hi=len(a)
 while lo<hi:
  mid=(lo+hi)//2
  if a[mid]<0:hi=mid
  else:lo=mid+1
 return lo

def main():
 assert not (O/'attention-indexer-work-traffic-models-v60.json').exists()
 stamp=datetime.now(timezone.utc).isoformat();inputs=read(O/'attention-operator-inputs-v60.json');src={}
 for r in inputs['files']:
  if r['status']=='READ':
   p=Path(r['local_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];src[p.name]=(r,p.read_text())
 assert len(src)==17
 cr=next(r for r in read(O/'sampling-guard-inputs-v55.json')['files'] if r['path'].endswith('/config.json'));cfg=read(Path(cr['local_path']))['text_config']
 assert [cfg['num_attention_heads'],cfg['head_dim'],cfg['index_n_heads'],cfg['index_head_dim'],cfg['index_topk'],cfg['sliding_window'],cfg['candidate_block_size'],cfg['candidate_topk_blocks']]==[64,512,32,128,512,128,8,2048]
 h,d,hi,di,k,w,cb,cc=64,512,32,128,512,128,8,2048;B=1.6e12;Pb=376e12;Pi=752e12
 prof=read(O/'attention-indexer-profile-and-edges-v60.json');models=[]
 # Fixed full-valid examples are not alleged runtime tensor values or measured HBM.
 for m in sorted({r['M'] for r in prof['regimes'] if isinstance(r['M'],int)}):
  att=[];idx=[]
  for name,n in [('SWA_FULL128',128),('CSA_FULL128_PLUS512',640),('DRAFT_ORI_SPARSE_CAPACITY256',256)]:
   f=4*h*d*m*n;qbase=4*m*h*d+4*h
   qrange=[qbase+2*d*n,qbase+2*d*m*n]
   att.append(dict(example=name,valid_query_rows=m,keys_per_query_assumption=n,
    dense_BF16_QK_PV_ops_2perMAC=f,reference_BF16_compute_us=f/Pb*1e6,
    ideal_semantic_value_bytes_if_key_union_min_max=qrange,
    ideal_cold_reference_max_of_compute_and_IO_us=[max(f/Pb,q/B)*1e6 for q in qrange],
    source_score_FP32_and_exp_BF16_GM_interfaces_logical_bytes=12*h*m*n,
    qualification='Fullvalid fixedn example,not boundonobservedruntime unlessactualn/validqueries/reuse proven. KVlatent servesKandV;Uunion ∈[n,m*n]. SourceGMqueues/softmaxstats/gathers/padding/repeatedloads/metadatanotpriced. Draft256 isdescriptorcapacity,not asserted visiblewindow.'))
  for name,L,mode in [('C2_UNFILTERED_32K_CONTEXT',16384,3),('C1_SOURCE_32K_CONTEXT',32768,1),('C1_CONSUMER_32K_CONTEXT',32768,2)]:
   candpos=min(L,cc*cb) if mode==2 else L
   integer=2*m*hi*di*L;head=2*m*hi*L;head16=16*head
   qfixed=m*hi*di+4*m*hi+4*m*k+(4*m*cc if mode==1 else 0)+(4*m*cc if mode==2 else 0)
   qinterval=[qfixed+(di+2)*L,qfixed+(di+2)*m*L]
   idx.append(dict(example=name,valid_query_rows=m,key_length_assumption=L,candidate_mode=mode,
    current_dense_INT8_QK_ops=integer,useful_FP16_head_reduce_ops=head,
    source_requested_FP16_head_reduce_ops_if_M16_per_query=head16,
    source_QK_relu_to_L1_DEQF16_factor=struct.unpack('>f',bytes.fromhex('3a800000'))[0],
    current_fullscan_INT8_reference_us=integer/Pi*1e6,
    current_useful_FP16_reference_us=head/Pb*1e6,source_requested_FP16_reference_us=head16/Pb*1e6,
    ideal_semantic_value_bytes_if_key_union_min_max=qinterval,
    source_weight_expansion_GM_interface_bytes=64*m*hi,
    source_scalar_score_GM_write_read_bytes=8*m*L,
    candidate_eligible_position_capacity=candpos,
    hypothetical_prefilter_INT8_QK_ops=2*m*hi*di*candpos,
    qualification='Current mode2 stillscansfullL inCube;candidate isVectorpost-scoremembership. Eligiblecapacity isnot actualcount,not currentworkreduction. NativeM16headreduce andINT8QK use differentpeaks;do notcombine intooneINT8MFU. ColduniqueKeybytes conditional,notwholeallocatedpool/HBM.'))
  models.append(dict(M=m,attention_examples=att,indexer_examples=idx,
   interface_models=dict(SMLA_to_inverse_RoPE_tail_read_write_bytes=4*m*h*64,
    materialized_send_layout_read_write_bytes=4*m*h*d,
    SMLA_inverse_RoPE_layout_combined_removed_bytes=4*m*h*(d+64),
    quantized_query_and_scale_write_read_bytes=2*m*hi*(di+2),
    QLI_raw_selected_to_prepare_write_read_bytes=8*m*k,
    stable_prepared_indices_bytes_must_remain=4*m*k,
    candidate_INT32_list_bytes=4*m*cc,
    candidate_membership_bitmap_bytes_at_maxmodel34816=m*544),
   no_numeric_Vector_topK_sort_or_special_function_ceiling=True))
 assert len(models)==6
 write(O/'attention-indexer-work-traffic-models-v60.json',dict(generated_utc=stamp,models=models,
  symbols=dict(T='actualvalidqueryrows,not automaticallyallocatedT',nq='visibleoriginal+selectedcompressedpositions forqueryq',
   U='uniquelogicalcachedvectors requested byallqueries',Lq='keypositionsvisitedinfullscanbeforecandidatepenalty',
   Eq='visiblecandidate-memberpositions inmode2',Hi=32,Di=128,H=64,D=512),
  formulas=['SMLAdensealgorithmGEMMops=4HD*sum_q(nq);SWA/CSAsoftmax/sink/rowmaskwork andtilepadding additional.',
   'IdealSMLAvalueIO=4THD+2DU+4H + metadata/indices;onecachedlatentactsasbothKandV. 2reads/modeldoesnotimplyphysicalHBM.',
   'CurrentQLI INT8QKops=2HiDi*sum_q(Lq),includingmode2;headreduceuseful2Hi*sum_q(Lq),sourceComputeWsrequestsM16 perquery.',
   'QLIcube outputsFP32scalar-scoreGMthenVectorappliesFP16keyscale/membership/topK;source scalarinterface8*sum_q(Lq)bytes,notHcorebigscoreplane.',
   'SourceSMLA FP32score GMwrite/read8H*sum_q(nq) +BF16exponentialGMwrite/read4H*sum_q(nq),excludingotherworkspaces/padding;onefusedkernelnotzeroGM.',
   'ReferenceT>=max(F_INT8/P_INT8,F_BF16_or_FP16/P_half,Q_HBM/B_HBM,Q_L2/B_L2,Vector/TopK/online-softmax/dependencybounds),onlyunderdeclaredactualwork/capacity assumptions.'],
  no_actual_perquery_lengths_or_selected_cache_union=True,no_endtoend_gain_or_MFU=True,
  hardware_reference='Previously verified publishedper-die376FP16/BF16TF,752INT8TOPS,1.6TB/s;not currenttypedphysicalresourceattestation. Vector/TopK/specialfunctions notnumericallybound.',overall_complete=False))
 # Independent semantic witnesses, no target/native test or implementation.
 witness=None
 for ab in range(0x3c00,0x4400):
  a=bf(ab);sf=f32(a/127);sh=max(f16(sf),2**-24)
  for kk in [15,31,47,63,79,95,111]:
   target=a*(kk+.5)/127;b16=struct.unpack('>I',struct.pack('>f',target))[0]>>16
   for xb in range(max(0,b16-1),b16+2):
    x=bf(xb)
    if x>a:continue
    qf=round(f32(x/sf));qh=round(f32(x/sh))
    if qf!=qh:witness=dict(BF16_amax=a,BF16_value=x,FP32_scale=sf,rounded_FP16_scale=sh,quant_with_FP32_scale=qf,quant_with_FP16_scale=qh);break
   if witness:break
  if witness:break
 assert witness is not None
 raw=[7,0,9,-1];visible=6;filtered=[x if 0<=x<visible else -1 for x in raw];prepared=sorted(x for x in raw if 0<=x<visible)+[-1]*3
 assert prepared==[0,-1,-1,-1] and binprefix(filtered)!=binprefix(prepared)
 neg=f32(-1e30);bound=2**49
 for s in [-bound,-1,0,1,bound]:assert f32(f32(neg-f32(s))+f32(s))==neg
 universe=4352;sets=[[],[0,1,5,4351,-1],[3,3,-1,25,4352]];checked=0
 for values in sets:
  bitmap=bytearray(544)
  for j in values:
   if 0<=j<universe:bitmap[j//8]|=1<<(j%8)
  for j in range(universe):assert bool(bitmap[j//8]&(1<<(j%8)))==(j in values);checked+=1
 ranges={
  'sparse_flash_mla_torch_adpt.h':[(141,155),(188,208)],
  'quant_lightning_indexer_v2_torch_adpt.h':[(101,154)],
  'quant_lightning_indexer_v2_kernel_arch22.h':[(474,526),(701,762)],
  'quant_lightning_indexer_v2_service_cube_arch22.h':[(500,579)],
  'quant_lightning_indexer_v2_service_vector_arch22.h':[(448,523),(545,552),(622,627),(714,759),(828,881)],
  'sparse_flash_mla_csa_kernel.h':[(433,461),(540,576),(814,850)],
  'sparse_flash_mla_csa_block_vector.h':[(454,489),(844,857)],
  'sparse_flash_mla_swa_block_vector.h':[(825,857),(1031,1044)],
 }
 parts=[]
 for name,rs in ranges.items():
  r,t=src[name];lines=t.splitlines()
  for lo,hi in rs:parts.append(dict(path=r['local_path'],sha256=r['sha256'],first_line=lo,last_line=hi,source=lines[lo-1:hi]))
 for name,names in [('indexer.py',['select_projected','quantize_query','project_weights','update_keys']),('dsa_v41.py',['_forward_attention','_select_sparse_indices','_apply_rotary']),
                    ('quantize_indexer_query.py',['_quantize_indexer_query_kernel']),('prepare_indexer_indices.py',['_prepare_indexer_indices_kernel','prepare_indexer_indices'])]:
  r,t=src[name];lines=t.splitlines()
  for node in ast.walk(ast.parse(t)):
   if isinstance(node,ast.FunctionDef) and node.name in names:parts.append(dict(path=r['local_path'],sha256=r['sha256'],first_line=node.lineno,last_line=node.end_lineno,source=lines[node.lineno-1:node.end_lineno]))
 write(O/'attention-indexer-native-numerical-contract-v60.json',dict(generated_utc=stamp,parts=parts,
  model_config=dict(path=cr['local_path'],sha256=cr['sha256'],attention_heads=64,D=512,rope_dim=64,index_heads=32,index_width=128,topk=512,window=128,candidate_blocks=2048,block_size=8,max_model_len_reference=34816),
  current_native_QLI_pipeline=['QINT8xKINT8→INT32dot;FixpDEQF16 withconstant0x3a800000=1/1024 andReLU,keptinL1.',
   'Vector0computesFP16(weightsFP16*queryScaleFP16),broadcasts16wideandwritesGM;CubeFP16headreduce requestsM16 andstoresoneFP32scalar-score row.',
   'Vector1multipliesFP32scalar-score byFP16keyScale,thenmode2membershippenalty,thenpositionTopK. Mode1additionallyblockamaxTopKandpinslatestvisible-tokenblockusing+inf.',
   'Mode2doesnotremoveout-of-candidateQK/Cubecomputation. OutsidefinitepositionsaredemotedtowardNEG_HUGE butremainvalidTopKfillers whenneeded;neverreplaceallwith-1.',
   'Queryquantizer computesperheadamax over128values;scale=RN_FP16(div_rn_FP32(amax,127)),thenclampmin2^-24;quant uses thisroundedscale,div_rn,nearbyint/clamp[-127,127].',
   'CurrentindexKcallerusesDynamicQuant→separateFP16scaleCast→twoScatterSKwrites. Do notassumequeryandkeyquantizersinterchangeable;internalkeyquantizerroundingnotinferredfromdtype.'],
  current_native_SMLA_pipeline=['OnekernelalreadyhandlesSWA+selectedcompressedKV+onlineSoftmax/sinks+PV;sourcehasFP32score/BF16exponential/FP32PV/updateGMworkspaces.',
   'FP32onlineSoftmax tile casttoBF16 withCAST_ROUNDbeforePV;finalBF16output usesRINTinBF16branch. Changingtiling/normalization/order canchangeoutput evenifreal-valuedformula agrees.',
   'LSEreturnFalse constructs0-elementFLOATtensor;profileremptyshape isnot4byte scalar. QLIunrequestedvalues/candidates likewise notuniversalscalars.',
   'CSAcountsvalidsortedpositiveprefixvia binarysearch forfirstnegative. Preparehelperfiltersfuture/negativepositions,chronologicallysortsvalidentriesandput-1suffix;cannot justdrophelperorforwardscoreorder.',
   'inverseRoPEappliesonly64-dimtail(448..512). Directlayoutfusion mustretainSMLABF16round thenrotaryBF16round,allaliasreaders/padding/collectiveownership.'],
  source_resolved_F12_gap='prepare_indexer_indices returnsprovidedindices_output object onlegacy path. dsa passesstabletopkbank then copiesselectedbackonto thesameviewrange. Conditionalselfcopyelision nowhasasourcealiasproof;historicalnativeaddresses/version/captureobservers notasserted.',
  witnesses=dict(rounded_scale_vs_unrounded_scale=witness,
   chronological_prefix=dict(raw_score_order=raw,visible=visible,filter_only=filtered,prepared=prepared,filter_only_binary_prefix=binprefix(filtered),prepared_binary_prefix=binprefix(prepared)),
   mode2_filler=dict(scores_after_penalty=[5.0,neg,neg],topk=3,positions=[1,2,3],hypothetical_drop_outside=[1,-1,-1],current_fillers_must_remain=True),
   outside_constant_under_finite_domain=dict(conservative_abs_score_bound=bound,negative_constant=neg,
    derivation='INT8dot<=2^21;after/1024+halfReLU<=2^11. finitehalfweightProduct<2^16,32headFP32sum<2^33,finitehalfKscale<2^16→|score|<2^49. This isfarbelowFP32ULPat1e30;penchainroundstoexactNEG_HUGE.',
    actual_weight_product_and_keyscale_finiteness_proved=False,
    implication='A compatibleprefiltervariantcancomputeonlyeligiblekeys andsynthesizeexcludedconstant-score positionsfororiginalsort/merge. Preservefillers/tiepolicy;nonfinite/intermediateoverflow requiresfallback orseparatecontract.'),
   bitmap_membership=dict(universe_blocks=4352,bitmap_bytes=544,INT32_candidate_list_bytes=8192,checked_memberships=checked,
    domain='LogicalcandidateblockIDswithinmaxmodel34816/block8;validIDsor-1,preserveperforwardrow/generation. Duplicatesdonotchangemembership;notphysicalKVpages.')),
  current_loaded_binary_identity_proved=False,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,whole_module_closed=False,overall_complete=False))
 variants=[
  dict(id='F12-BITMAP',proposal='Publishcandidate membershipbitmap oncepersource-query andshare4consumer layers, replacingrepeated2048blocksort/membershipsearch.',decision='CONDITIONAL_PRIORITY',
   conditions='Currentlogicaluniverse≤4352blocks→544B vs8192Blist/query. Preserveoriginalscoreorderedcandidateoutput ifotherreadersneed;newbitsetsameisOut;row/generation/clearinvalidsuffix/producerreadinessmustmatch. Existingcandidatebanksharingisbaseline.',work_delta='RemoveconsumerlocalSortAll2048 anddistance-membershiploops;bitmapbuild/readcosts andnativeS2partitionrepetitionnotmeasured. NoQKworkreductionfrombitmapalone.'),
  dict(id='F12-PREFILTER-FILL',proposal='Gathercandidate-memberkeys beforeCube andsynthesizeexcludedNEG_HUGEscorepositionsforunchangedsort/merge.',decision='CONDITIONAL_PRIORITY',
   conditions='FiniteFP16weightproduct/keyscale andsameINT8/DEQF16/head-reduce path;fixednativefillers/ties/masks/outputoffset/pin preserved. Insufficienteligiblekeys needoriginaloutsidefillerpositions;nonfinite fallback/separatecontract. Physicalpage/scale stride and64bitaddressing retained.',work_delta='CurrentconsumerfullLQK→eligibleE,withE≤min(L,2048*8),notautomatic2xspeedup. Retainingoriginalfullscoresortcanpreservefiller/tiecontractandcost.'),
  dict(id='F12-PREPARE',proposal='QLIfinalTopKepilogue performvisibilityfilter/chronologicalsort/lengths andwritestablepreparedindexbank.',decision='CONDITIONAL',
   conditions='Keepcandidate_outforlaterlayers andallvalidity/sentinel/int32ordering/outputbankreaders. CurrentallT≤834 areunderfusedlengththreshold1024whenlengthoutputprovided. Stablebankwrite4T512 remains.',work_delta='Eliminate8T512 rawselectedwrite/read,notallpublishedindices. TopKglobalmerge completionandsortcostremain.'),
  dict(id='F12-SELF-COPY',proposal='Elidepostpreparepublicationcopyontoidenticalstableoutputrange.',decision='SOURCE_ALIAS_RESOLVED_CONDITIONAL',
   conditions='Exactcurrenthelperreturnsindices_output. Keepstorageversion/debug/capture/autogradhookobserversifpresent. NoextraTensorMovekernelcountorHBMgainasserted.',work_delta='Potentialsubmission/selfcopyoverheadonly;sourceidentitydoesnotmeasuredevicecopywork.'),
  dict(id='F12-HEAD-REDUCE',proposal='Reduce16duplicatedFP16head-reductionrows oruseequivalentVectorreduction.',decision='CONDITIONAL_NUMERICAL_VARIANT',
   conditions='PreserveINT32→DEQF16ReLU/1024,FP16(weight*queryScale) andFP32Cubeheadreductionnumerics;Vectorordernotautomaticallysame. RepriceCube/VectorandweightGMbroadcast.',work_delta='CurrentComputeWs requestsM16 butFixpoutputsoneusefulrow;16xrequest/usefulFP16headops isnot16xwholeoperatorpotential.'),
  dict(id='F09-ROPE-QUANT',proposal='FuseindexquerypartialRoPE andperheadINT8quantization.',decision='CONDITIONAL',
   conditions='RoundrotatedvaluesasBF16beforerowamax;full128headscale,roundedFP16scale/div_rn/nearbyint/min2^-24 andquant/scaleownershippreserved. Preexistingquantizekernelalreadyfusesamax/quant.',work_delta='RemoveonlyintermediateRotatedqueryinterfaces/launches;scalereductionandquantremain.'),
  dict(id='F10-WEIGHT-EPILOGUE',proposal='Projectweightsepilogue emitsrequiredFP16scaledweights.',decision='CONDITIONAL',
   conditions='PreserveBF16projectionround→FP32cast*weights_scale→FP16round andallreaders;separateQLIFP16weight*queryScale round stilllive.',work_delta='RemoveCast/Mul/recastexternalinterfaces onlyifnootherFP32reader;notfoldunroundedaccumulator.'),
  dict(id='F11-KEY-CACHE',proposal='FuseindexkeyNorm/RoPE/DynamicQuant/FP16scalepublication withkey+scaleScatter.',decision='CONDITIONAL',
   conditions='PreserveBF16norm/RoPE,inputquantizerandfloat→halfscaleorder,slotskip/physicalpage/stridesandcoherentkey-scalevisibility. Queryquantizerisnotassumedkeydrop-in.',work_delta='Reducekey/scaleintermediatesandlaunches;cachewrites remainforfutureiterations/readers.'),
  dict(id='F13-INTERNAL-QUEUE',proposal='RetileSMLA score/exponential/PV/gatherGMqueues forcachelocality orsupportedonchip handoff.',decision='CONDITIONAL_HARDWARE_AND_NUMERICS',
   conditions='Architecture22legalhandoff/capacity/ownershipandflags required;wholematrixdoesnotfitautomatically. PreserveBF16exptilecast,onlineSoftmax/sink/max/sum/PVorder. Ifnodirecthandoff,GMqueue/L2reuse remains.',work_delta='Source12Hsum(n) score+expGMinterface islogicalnotHBM;neverclaimall12Hsum(n) eliminatedwithoutlegalhandoff.'),
  dict(id='F13-SWA-CMP-SPLIT',proposal='RunSWApartialattentionbeforeIndexerfinishes,thenmergecompressedattention.',decision='CONDITIONAL_NEW_STATE_AND_NUMERICS',
   conditions='Expose(m,l,a)partialstatesorcompatiblecontract,includeSinkexactlyonce;keeporiginalonlineBF16tilerounding/orderifexactfidelityrequired. ExistingnormalizedBF16outputs alone+disabledLSE areinsufficient. SWAKVready/resourcecompetition/extralaunches priced.',work_delta='SWAdoesnotdependonselectedcompressedindices,sooffersdependencyreordering. NeitherSWAworknorcompressionworkisdeleted.'),
  dict(id='F14-SMLA-ROPE-LAYOUT',proposal='SMLAepilogue producesinverse-RoPEandsend-layout outputdirectly.',decision='CONDITIONAL',
   conditions='PreserveSMLABF16round→64tailrotaryBF16round;unchangedunrotated448dims,valid/paddingrows,aliasconsumers/row-headcollectiveprotocol.',work_delta='Tailonly4TH64 plusmaterializedlayout4TH512 bytes;M3combined442368B. SMLAfinaloutput/collectiveinputnotdeleted.'),
  dict(id='F12-CROSSCARD-KEYPARTITION',proposal='RepartitionkeyscansacrossTPandmergelocalTopK.',decision='CONDITIONAL_REPARTITION_NOT_FREE_TP_FACTOR',
   conditions='Fullquery/headscoreidentity andkey/cacheownership/candidateblock8groups first. TopK(localcomplete-key-scores)merge needscommontiecomparator;headpartialTopKunionisnotgenerallyvalid. Numerics/communication/rowownership preserved.',work_delta='IfcurrentrowsareTPsharded,changingfromMlocal*LtoTP*Mlocal*(L/TP) keepspercardarithmetic unchanged,plusexchange. Shape/full32heads doesnotprove8xduplicatequerywork.')]
 write(O/'attention-indexer-fusion-reorganization-variants-v60.json',dict(generated_utc=stamp,variants=variants,
  interactions=['BITMAPandPREFILTER composebutmembershipwork countedonce;bitmapalonekeepsfullCube scan.',
   'PREPAREplusQLI-SMLAcompositecannoterasestableindices/candidatesbecauseotherlayersreadthem.',
   'SWA-CMP-SPLIT competeswithfullSMLAfusion/queueelimination andchangesonlineorder/newstate,soitsgaincannotbeaddedfreely.',
   'SMLA-ROPEandlayoutinterfacebytescountedonce;RoPEtailonlyandfulltransposearetwodistinctoperations.',
   'Crosscardkeypartition isanalternativeworkpartition,notadditionalTPdivisiononexistingquerysharding.'],
  conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,measured_speedup=None,overall_complete=False))
 refined=[]
 for r in prof['regimes']:
  refined.append(dict(regime_id=r['regime_id'],operator=r['operator'],executions=r['executions'],M=r['M'],
   model='attention-indexer-work-traffic-models-v60.json',contract='attention-indexer-native-numerical-contract-v60.json',variants='attention-indexer-fusion-reorganization-variants-v60.json',
   status='CAPTURED_NATIVE_COUNTS_SOURCE_CONDITIONED_FORMULA_WORK_IO_AND_CONDITIONAL_FUSION_REVIEWED',full_numeric_actual_hardware_ceiling=False,
   remaining='Actualperquerylength/selectedunion/candidatevalues/nativephase/physicalcachecapacity/topK-Vectorrates andnewkernelallconsumer/readiness precisionconditions remainparameterized. Conditionalanalysisdoesnotrequireimplementation.'))
 assert len(refined)==51 and sum(r['executions'] for r in refined)==6863
 write(O/'operator-regime-refinement-v60.json',dict(parent='operator-shape-ceiling-ledger-v55.json',refined_regimes=refined,refined_regime_count=51,
  refined_executions=6863,disabled_SMLA_LSE_semantic_bytes=0,all_new_numeric_hardware_ceilings_bound=False,whole_modules_closed=0,overall_complete=False))
 print(json.dumps(dict(source_files=17,models=len(models),variants=len(variants),regimes=len(refined),semantic_witness=witness,bitmap_bytes=544,overall_complete=False)))

if __name__=='__main__':main()
