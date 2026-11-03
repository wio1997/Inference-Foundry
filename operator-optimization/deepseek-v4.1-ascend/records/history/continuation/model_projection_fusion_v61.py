"""Projection fusion/specialization models, rounding counterexamples and communication alternatives."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,struct,math
from finalize_origin_coverage_v9 import N,O,read,write

def bf16(x):
 u=struct.unpack('I',struct.pack('f',x))[0];u=(u+0x7fff+((u>>16)&1))&0xffff0000
 return struct.unpack('f',struct.pack('I',u))[0]

def main():
 assert not (O/'projection-fusion-and-partition-costs-v61.json').exists()
 stamp=datetime.now(timezone.utc).isoformat();p=read(O/'projection-work-and-reference-bounds-v61.json');g=read(O/'projection-graph-chains-v61.json')
 H=5120;R=1280;QA=32768;IQ=4096;K=4096;rank=1024;Nn=5120;P=376e12;B=1.6e12
 shared=[]
 for M in sorted({r['M'] for r in p['regimes'] if r['role']=='INDEX_QUERY_B'}):
  shared.append(dict(M=M,qr_width=R,attention_out=QA,index_out=IQ,
   weights_INT8_bytes=R*(QA+IQ),
   common_quant_producer_removed_value_bytes=3*M*R+4*M,
   common_quant_plus_joint_GEMM_removed_value_bytes=4*M*R+8*M,
   shared_QA_index_weight_BF16_input_read_bytes=2*M*H,
   qualification='QRsameBF16generation andsameW8dynamicquant contract. Reuseproducerkeeps2GEMMreads;combiningGEMMs savesoneadditionalINT8read+scale. Packedoutputstrides,scales/NZweights/allreader/lifetime/readiness constraints remain;bytebudgets nottimegains.'))
 out=[]
 for M in sorted({r['M'] for r in p['regimes'] if r['role']=='WO_A_GROUP1'}):
  f=2*M*rank*(K+Nn);fq=2*M*K*Nn
  q=2*rank*(K+Nn)+2*M*(K+Nn)+4*M*rank
  fused=q-4*M*rank;fold=2*K*Nn+2*M*(K+Nn)
  out.append(dict(M=M,K=K,R=rank,N=Nn,
   factorized_GEMM_ops=f,precomposed_GEMM_ops=fq,compute_and_weight_ratio=fq/f,
   factorized_weight_bytes=2*rank*(K+Nn),precomposed_weight_bytes=2*K*Nn,
   original_logical_value_bytes=q,preserve_round_fused_value_bytes=fused,precomposed_value_bytes=fold,
   panel_fusion_removed_intermediate_write_read_bytes=4*M*rank,
   original_aggregate_reference_floor_us=max(f/P,q/B)*1e6,
   preserve_round_fused_aggregate_reference_floor_us=max(f/P,fused/B)*1e6,
   precomposed_aggregate_reference_floor_us=max(fq/P,fold/B)*1e6,
   qualification='Samehalfprecision/reference376TF/1.6TB/s conditionaldensealgorithm,notactualcache/capacity/phasegain. TwoGEMMs shareCube resource;do notmax(individualcompute) toinventfreeoverlap. Precomposedpathnotnumericallyequivalent evenwhenitsIOfallsatlargeM.'))
 c2=[]
 for M in sorted({r['M'] for r in p['regimes'] if r['role']=='C2_KV_OR_GATE'}):
  K2=5120;N2=512;two=8*(M*K2+K2*N2+M*N2);joint=4*M*K2+8*K2*N2+8*M*N2
  assert two-joint==4*M*K2
  c2.append(dict(M=M,K=K2,N_each=N2,FLOAT_dense_ops=4*M*K2*N2,
   two_GEMM_value_bytes=two,joint_GEMM_value_bytes=joint,joint_removed_duplicate_FLOAT_input_read_bytes=4*M*K2,
   inputcast_already_shared=True,input_and_cast_value_bytes_before_fusion=14*M*K2,
   with_existing_cast_and_joint_GEMM_value_bytes=10*M*K2,ideal_onchip_cast_joint_input_bytes=2*M*K2,
   combined_cast_plus_joint_input_interface_reduction_bytes=12*M*K2,
   numeric_compute_reference_unbound=True,
   qualification='ExistingoneBF16→FLOATcast feedsbothFP32parameterGEMMs;not2casts. Jointalone savesoneFLOATread. Idealcast+join saving12MK isnot4MK+12MK. FLOATmode/capacity,GMcachetier/nativehandoff/outputstride/reductionnumerics conditional.'))
 comm=[dict(layer_role='36non_compressor_source_layers',BF16_hidden_bytes_per_row=10240,required_projected_payload_bytes_per_row=1024,feature_payload_ratio=10,
   live_planes='SWA KV512BF16;Q_A remains local;otherglobalrawreadersmustbeexcludedbyexistingF32contracts.'),
  dict(layer_role='1C1_source_layer',BF16_hidden_bytes_per_row=10240,required_projected_payload_bytes_per_row=2048,feature_payload_ratio=5,
   live_planes='DistinctSWA512BF16 +C1latent512BF16;independentnorms/indexlatentbeforelongRoPE retained.'),
  dict(layer_role='3C2_source_layers',BF16_hidden_bytes_per_row=10240,required_projected_payload_bytes_per_row=5120,feature_payload_ratio=2,
   live_planes='SWA512BF16 +C2 WKV512FLOAT +WGATE512FLOAT. Gatherprojectedglobalrows before ringpool;C2statepool isnotrowlocal andcannotblindlymovebeforegather.')]
 x=[129/128,1];a=[129/128,65/64];mid=[bf16(x[0]*a[0]),bf16(x[1]*a[1])]
 fact=bf16(mid[0]-mid[1]);direct=bf16(x[0]*a[0]-x[1]*a[1]);assert fact==0 and direct==1/16384
 write(O/'projection-fusion-and-partition-costs-v61.json',dict(generated_utc=stamp,QR_and_QA_shared_input_models=shared,
  output_two_GEMM_models=out,C2_joint_projection_models=c2,project_before_gather_feature_payload_models=comm,
  WO_precompose_IO_break_even_rows=5632,WO_precompose_compute_weight_ratio=20/9,
  rounding_counterexample=dict(x=x,A_diagonal=a,B_column=[1,-1],BF16_intermediate=mid,factorized_BF16_output=fact,precomposed_output=direct,
   implication='EvenexactBF16precomposedweights andallfinitevaluesdo noteraseintermediateBF16round. Canembed2x2casewithzerosinactuallargermatrices;notnativeexecution.'),
  row_domain_countermodel=dict(query_rows=3,KV_rows=24,QA_width=1280,KV_width=512,
   separate_QA_KV_dense_ops=2*3*5120*1280+2*24*5120*512,
   blindly_compute_both_on24rows_ops=2*24*5120*(1280+512),excess_compute_ratio=8/3,
   qualification='Excludesnaivecurrent-domainconcat;localproject-then-redistribute isseparatevalidconditionalproposal,notexcluded.'),
  qnorm_quant_dependency='QA→BF16v→RMS1280fullrow→BF16qr→fullrowDynamicQuantscale→INT8Q_B. RMSrstd androundedqrrowamax formdependencies;cannotindependentlyquantizeKtiles andstartQBunchanged. Priorrowquantcounterexample applies.',
  KV_composite_interface_formula='BF16KV GEMM→RMS512→RoPEtail64→cacheScatter:conditionalremoved8*M*512+4*M*64 bytes,cachewrite2*M*512 remains;preservebothBF16roundsandactualslotmask.',
  zero_local_query_scope='QA/QB4367 versusWO_A/WO_B4424 doesnotprove57redundantoutputs. Existingv18cache-onlylocalzero branches canstillconsumeothercardqueryrows aftercollective. Allrankemptiness,zero/bias/ownershiprequiredforanywholeoutputskip.',
  no_performance_gain_or_physical_traffic_claim=True,overall_complete=False))
 # Source excerpts stay local and limited to operator dataflow/numerical methods.
 files=read(O/'projection-operator-inputs-v61.json')['files'];parts=[]
 for r in files:
  path=Path(r['local_path']);text=path.read_text();assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256'];lines=text.splitlines();tree=ast.parse(text)
  names=['_apply_wo_b','_get_batched_wo_a_weight','_forward_o_proj'] if path.name=='dsa_cp.py' else ['prepare_input_for_overlap','apply']
  if path.name=='w8a8_dynamic.py':
   cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='prepare_input_for_overlap' for m in n.body));nodes=cls.body
  else:nodes=ast.walk(tree)
  for node in nodes:
   if isinstance(node,ast.FunctionDef) and node.name in names:parts.append(dict(path=str(path),sha256=r['sha256'],first_line=node.lineno,last_line=node.end_lineno,source=lines[node.lineno-1:node.end_lineno]))
 dsa=Path(next(r['local_path'] for r in read(O/'attention-operator-inputs-v60.json')['files'] if r['status']=='READ' and r['path'].endswith('/attention/dsa_v41.py')))
 txt=dsa.read_text();ls=txt.splitlines()
 for lo,hi in [(350,382),(394,415),(450,495),(521,540)]:parts.append(dict(path=str(dsa),sha256=hashlib.sha256(dsa.read_bytes()).hexdigest(),first_line=lo,last_line=hi,source=ls[lo-1:hi]))
 frozen=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/whole-scope-20261030/frozen-source-snapshot/vllm-ascend/vllm_ascend/models/deepseek_v41/compressor.py')
 ls=frozen.read_text().splitlines();parts.append(dict(path=str(frozen),sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),first_line=15,last_line=25,source=ls[14:25],source_scope='HistoricalfrozenC2dtype/parametercontract,notnewnativebinaryattestation.'))
 write(O/'projection-operator-numerical-contract-v61.json',dict(generated_utc=stamp,parts=parts,
  facts=['RecordedQ_A/KV andoutputprojectionsBF16→BF16;RMSconsumesroundedBF16,notunroundedGEMMaccumulator.',
   'W8linearmethodacceptsPreparedLinearInput atquant_methodapply boundary;bareTensorcallsDynamicQuant thenQuantMatmul withBF16channelscale/FLOATrowscale andoutput_dtypeinput. WholeLinearwrapperacceptance/TP/bias/outputcontractnotprovedmerelybymethodsupport.',
   'Q_BandIndexQbothconsumeexactsourceqr;640graphblocks have2DynamicQuant3x1280 records. This supportscommonquantproducer aftersamequantizer/generation/ownership checks. It doesnotmakecurrentQA3rows equalKV24rows.',
   'Indexweightprojectionusesquery_hidden independentofqr/IndexQ;sourcecurrentlyrecordsweights_startafterIndexQsubmission. EarlierorjointQA+weightprojection retainsseparateoutputwidths1280and32 andindependentuses.',
   'C2hidden_states.float() alreadyonce sharedbydistinctwkv/wgate FLOATweights;do notcountcastsharingasnew. FLOATMatMulV2/V3doesnotproveactiveHF32/fullFP32mode.',
   'TransposeBatchMatmul alreadyfuses itspermutations;all7capturedregimes/4424calls haveG1. LogicalM=T,notM1. GenericAPIlabeldoesnotprovethatnativebackendlacksaG1fastpath.',
   'WO_A→WO_B retainsBF16intermediate;fusionkeepsbothweights andcompute,whileprecompositionincreasesweights/ops20/9 andfailsroundingcounterexample.',
   'WO_B→ReduceScatter mustroundperrankpartialtoBF16 beforecurrentcollectivereduction;newtiledhandoffneedssamepanelorder/ownership/collectiveprotocol andpaddedrowsemantics.',
   '264same-geometryMatMulV2 executionsbelongto5M18shared-contextregimes,notSWA. Costmodels includethemseparately;theirconsumerfusionnotprovedbyshape.'],
  conditional_analysis_deliverable=True,loaded_native_binary_identity_proved=False,implementation=False,native_test=False,overall_complete=False))
 V=[]
 def variant(i,proposal,conditions,cost,decision='CONDITIONAL'):
  V.append(dict(id=i,proposal=proposal,decision=decision,conditions=conditions,cost=cost))
 variant('F05-QA-NORM-QUANT','QAepilogue+RMS1280+rowDynamicQuant','PreserveQA BF16round,RMSroundedqr/gamma/eps,fullrowrstdandquantamax barriers. IndexQ/liveqrreaders retained;priceNtilepartialreductions/buffering/Cubeparallelism.','Reducesintermediates/launches,notfullrowdependencies orQBweightreads.')
 variant('F06-QR-QUANT-SHARE','ReuseQRINT8+rowscale forQ_BandIndexQ','Sameqr/currentgeneration/quantizer/rowmap andlifetime;existingPreparedLinearInputmethodcontractcandidate,notprovenwholewrapperdrop-in.','M3removeonequantproducer11532Blogicalbudget;832IndexQcalls,640graphpairs;QB/indexweights47.18592MBremain.')
 variant('F09-QB-INDEX-JOINT','JointQ_B+IndexQGEMM withcommonINT8QR','Keepdistinct32768/4096outputs/scales/bias/NZawarepacking andBF16outputs. Packedrowstride36864needssupportedstrides orseparateoutputbuffers/copies;notblindweightaxisconcat.','M3totalsharedquant+jointinputbudget15384B;doesnoterase47.18592MBweights.')
 variant('F06-F10-QA-WEIGHT','ComputeIndexweightprojectionearlierorjointwithQA','SamequeryhiddenBF16generation;keepseparate1280QA/32weightoutputs,QA1280norm andweightBF16→FP32scale→FP16round. Do notRMS1312combinedoutputs.','M3oneinputread30720B;mostlylaunch/readinessproposal;newtiling/overlapcostunmeasured.')
 variant('F06-NAIVE-GLOBAL-CONCAT','ComputeQAandKVtogetheroncurrentglobalKVrows','LocalQA3versusglobalKV24 aredifferentdomains. Ragged/repartitionedvariantseparate.','Naiveall24rows raisesdensework165.15072M→440.40192M,8/3ratioforcase3/24.',decision='REJECT_UNCONDITIONAL_CURRENT_DOMAIN_MERGE')
 variant('F07-QB-ROPE','QBepilogue+partialRoPE','RoundQBtoBF16beforeRope,correct64tailpair/positions/cos-sin/headlayout;unrotated448dimsandallreaders preserved.','Onlyrotatedtailintermediate access/launchbudget;notfull512Qtraffic.')
 variant('F08-KV-COMPOSITE','KVprojection+RMS512+tailRoPE+cacheScatter','PreserveGEMMBF16,normBF16,RopeBF16 andfull512norm;invalidslots/actualrows/cacheownership andpublication complete. Existingmultistreamoverlapisbaseline.','Conditionally8M512+4M64 removedvaluebytes;cachewrite2M512 remains.')
 variant('F11-KEY-PROJ-CACHE','Indexkeyprojectionepilogue+norm/Rope/quant/cache','Preserve512→128projectionBF16 andnorm/Rope/quantizer/scale rounds,oldlatentreaderbeforelongRoPE(WAR),physicalslots/key-scalecoherence.','Fuseonlycompatibleproducer-to-cachechain;futurecachewrites/oldlatentreadersretained.')
 variant('F17-C2-JOINT','JointdistinctFLOATC2WKVandWGATEinputtiles','OneFLOATinputcastalreadyshared;keepboth512FLOAToutputs/weights/separatebufferstrides andpoolsemantics. FLOATprecisionpolicy/rankweights/tiling retained.','M24save491520BduplicateFLOATread;M6400save131072000Blogical,notactualHBM.')
 variant('F17-CAST-JOINT','JointC2projectionwithcompatibleonchipBF16→FLOATcast','KeepFLOATparameter/arithmeticmode;nativeonchiphandoff/capacityrequired. Do notconvertweights toBF16.','Baselinecast+2GEMMinput14MK→cast+joint10MK→idealonchipjoint2MK;total12MKsavingnot4MK+12MK.')
 variant('F32-PROJECT-THEN-COMM','Projectlocalrowsbeforegather,exchangeonlyneededroleoutputs','Replicatedweight/valueprecision/rowmap/cardownership/inputreadersbindfirst. SourceC1/C2needadditionaloutputs;C2poolstateafterglobalprojectedrows,notlocalrowpool. Padding6668/6672retained.','Featurecontentratios10(no compressor)/5(C1)/2(C2) underdeclaredplanes,notHCCLwiretime;no freeTPfactorwherequeryalreadysharded.')
 variant('F16-G1-GEMM','SpecializeallcapturedG1TransposeBatchMatmulintoordinaryGEMM/views','Sameweightversion/stride/orientation,BF16accum/round/output contract. G>1/full-gatherpathkeptseparate. Noexistingnativefastpathabsenceclaim.','SameFLOPs/bytes;possibletiling/submissionimprovement. No same-shapepairedGEMMbenchmark or40usgainclaim.',decision='CONDITIONAL_PRIORITY')
 variant('F16-TWO-GEMM-PANEL','Panel-pipelinedWO_A→WO_Bkeepingrank1024BF16','KeepBF16intermediateandR1024reductionorder;onchipcapacity/handoff/intercorebroadcastorcachequeueprice. BothstagescompeteCube.','M24interface98304B;M6400=26.2144MBbutaggregatecompute-limitedreferencefloorunchanged321.26584us.')
 variant('F16-PRECOMPOSE','Precomputefull4096×5120weight','FailsBF16intermediateroundingcounterexample;changesfactorization/reduction.','Weights18.874368→41.94304MB,compute20/9;IOonlybreak-evenM5632notperformanceendorsement.',decision='REJECT_UNCONDITIONAL;COST_DISFAVORED')
 variant('F16-WOB-REDUCESCATTER','TiledWO_BepilogueandReduceScatter','RoundperrankpartialBF16beforecurrentreduce,sameTPpanels/order/rowownership/padding/events. No genericring-wireassumption.','MaypipelineCube/network,notdeletepartialresults orsumallkernel/commtimesasgains.')
 variant('F16-ZERO-LOCAL-SKIP','Skipwholeoutputprojectionswhenlocalqueryempty','Localemptinessdoesnotproveglobal/alltoallresultempty;othercardsmaycontributequeryrows. PreserveKV/cache/collectives.','QA/QB4367 versusWOA/WOB4424 difference57notredundantcount.',decision='REJECT_UNCONDITIONAL_LOCAL_PREDICATE')
 assert len(V)==16
 write(O/'projection-fusion-variants-and-interactions-v61.json',dict(generated_utc=stamp,variants=V,
  interactions=['QRproducerreuseandjointQB/IndexQ shareinputbudgets,countonce;cannotadd11532and15384asindependentM3savings.',
   'QA+weight canjoinQA+norm onlywithindependent1280/32outputs andunchangednormdomain;packedstrides mayaddcopies.',
   'KVfullchainfusionandproject-before-gather composeonlywithrealglobalcache/rowreadiness;do notdiscardexistingCube/Vectoroverlap.',
   'C2cast+joint budgets overlap;C1/SWAexistingF33distinctweights/normsandWARcontractsretained.',
   'G1specialization,panelWOA/WOBfusion andWOB/RSoverlapmaycompose butsameCube/queues/interfaces cannot yieldindependentlyadditivegain.',
   '5M18lookalikeroles havegeometryanalysis only;do notapplySWAconsumerfusionbasedonshape.'],
  implementation=False,native_test=False,measured_speedup=None,whole_module_closed=False,overall_complete=False))
 refined=[]
 for r in p['regimes']:
  refined.append(dict(regime_id=r['regime_id'],operator=r['operator'],role=r['role'],M=r['M'],K=r['K'],N=r['N'],executions=r['executions'],
   cost_model='projection-work-and-reference-bounds-v61.json',fusion_contract='projection-operator-numerical-contract-v61.json',variants='projection-fusion-variants-and-interactions-v61.json',
   status='DENSE_GEOMETRY_WORK_AND_LOGICAL_IO_ONLY_OTHER_ROLE' if r['role']=='DRAFT_SHARED_CONTEXT_KV' else 'CAPTURED_PROJECTION_WORK_IO_AND_SOURCE_CONDITIONAL_FUSION_REVIEWED',
   full_numeric_hardware_ceiling=False,consumer_fusion_review_complete_in_this_turn=r['role']!='DRAFT_SHARED_CONTEXT_KV'))
 write(O/'operator-regime-refinement-v61.json',dict(parent='operator-shape-ceiling-ledger-v55.json',refined_regimes=refined,refined_regimes_count=74,
  refined_executions=25078,core_fusion_reviewed_regimes=69,other_role_geometry_only_regimes=5,
  fullscope_remaining='FLOATactiveprecision/typedcapacity/cachetier/Vector/dequant/internaltiling/actualrowmasks/newimplementationprecision/readiness/allremainingmodeloperators notfullybound.',whole_modules_closed=0,overall_complete=False))
 print('74regimes(69core+5shared-context),16conditional/rejectiondecisions;20/9WOweight/compute ratioandBF16roundingwitness;QR/C2/gatherbudgets recorded.')

if __name__=='__main__':main()
