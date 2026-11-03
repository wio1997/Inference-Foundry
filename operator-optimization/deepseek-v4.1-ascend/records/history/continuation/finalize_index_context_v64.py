"""Persist IndexKey and draft context operator analysis; whole scope remains active."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
 assert not (O/'REVIEW-CHECKPOINT-v64.json').exists()
 p=read(O/'index-context-regime-models-v64.json');g=read(O/'index-context-native-chains-v64.json');cost=read(O/'index-context-fusion-costs-v64.json');v=read(O/'index-context-fusion-variants-v64.json');cov=read(O/'operator-analysis-coverage-ledger-v64.json');inv=read(O/'M09-M18-operator-branch-inventory-v64.json');contract=read(O/'index-context-operator-contract-v64.json')
 assert p['regime_count']==242 and p['captured_execution_count']==15976
 assert len(g['IndexKey_graph_chains'])==320 and len(g['context_KV_source_order_cycles'])==312 and len(g['main_projection_source_order_chains'])==88
 assert len(g['aux_checkpoint_graph_chains'])==240 and len(v['variants'])==26 and inv['total_families']==28
 assert cov['scoped_conditional_review_regimes']==832 and cov['scoped_conditional_review_executions']==180111 and cov['remaining_regime_reconciliation_count']==1250
 assert cov['geometry_only_other_role_regimes']==0 and cov['explicit_profile_insufficient_regimes']==3
 assert cost['context_partition_models'][0]['feature_RS_uniform_padded_rows']==16 and cost['context_partition_models'][0]['feature_RS_AG_to_baseline_feature_AG_ratio']>1
 e=['index-context-regime-models-v64.json','index-context-native-chains-v64.json','index-context-operator-contract-v64.json','index-context-fusion-costs-v64.json','index-context-fusion-variants-v64.json','operator-regime-refinement-v64.json','operator-analysis-coverage-ledger-v64.json','M09-M18-operator-branch-inventory-v64.json','index-context-local-validation-history-v64.json']
 write(O/'PROFILE-CODE-QUESTION-GATE-v64.json',dict(status='INDEX_KEY_QUANT_CACHE_AND_DRAFT_CONTEXT_TYPED_PARTITION_WORK_BOUNDS',previous_turn_classification='PROGRESS:v63router_vector_operator_models_and_MemSet_gap_retention',
  questions=['CanKeyquant/cachefusewithoutchangingFLOAT-before-FP16scaleorBF16RMS/RoPEboundaries?',
   'Doorderedrows,feature640shards,N/M,weightreplicationandlateIavailabilitypermitcontextreorganization?',
   'WhatadditionalFP32partials/rowpadding/collectives/fullbankreadersdoKVowner/RS-AGalternativescost?',
   'Whichlookup/cast/tail/slot/HCmeaninterfacescanfoldwhilepreservingallper-layer/crossiterationgenerations?'],
  allowed_source='7previously166SHA-boundlocaloperator/callerfilesreused;newlocalclausereviewonly. Worker2058-2094solelyanswersnamedselectoravailability;noindependentframework/HTTP/SSE/loader/lifecycle/SDKcatalogaudit.',new_remote_source_read=False,local_pinned_operator_clause_review=True,no_new_model_Torch_NPU_capture_health_process_kill_or_implementation=True,loaded_native_binary_identity_not_assumed=True,overall_complete=False))
 scopes={
  'F01':'SharedinputHCmean/HcPremultioutputatfixedno-Engramcheckpoints;weightedYnotmean,outputreadinessandresourcecostconditions. No newHCregimeclosure.',
  'F09':'RemainingordinaryIndexQinputDQandqueryRoPEtypedcostsreconcileexistingFP16headscalecontract;notrepeatQLIsemanticstudy.',
  'F10':'BF16projectionround→FLOAT*2^-6→FP16conversion/epilogueworkandcast-ordercounterexample.',
  'F11':'320graphIndexKeychains+directregimes,FLOATquantbeforeFP16scale,dualcache/coherency/validslot/TPcanonicalKeyprepconditionalmodels.',
  'F12':'24nativefreshcandidateoutputcopiesmode1→bank;newoutABI/capture/readerconditions,existingprepareindicesaliasnotanewcopyclaim.',
  'F27':'Main15360→640featurecombine/full5120norm,orderedI/lateauxselection/fullWrowowner/crosslayerpartials/HCmeanload/K0guardmodels.',
  'F29':'3distinctsharedKVmaps/joint1536gated3norms,featureFP32RS→rowBF16norm/RoPE→AGwithPpaddingandallbank/queryweightconsumers;tail/lookup/slot/None/deltaKVvariants.',
  'F31':'Per-layergroup/B/slotgenerationCSE/inlineintegerinterfacecostandactualDSparkmetadata274retained,notframeworkcatalog.'}
 updates={cid:dict(review_v64=dict(regime_models=str(O/e[0]),native_chains=str(O/e[1]),operator_contract=str(O/e[2]),work_costs=str(O/e[3]),variants=str(O/e[4]),coverage=str(O/e[6]),candidate_specific_scope=scope,
  relation='242regimes15976nativeexecutions:237previouspending/15712plus5M18geometry/264nowwithownconsumercontracts. 26operatorchoicesandtypedtraffic/compute/padding/precisionconditions;noimplementation/nativebenchmark.',conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
  profile_review_state='INDEX_CONTEXT_OPERATOR_WORK_IO_TYPED_PARTITION_AND_NUMERICAL_CONDITIONS_REVIEWED',
  conclusion_changing_gap='ActualI/validslots/positions/groups/weights/bankgeneration/aliases/allreaders,changednativeGEMM/RMS/RoPE/collectiveprecisionandtypedphysicalcapacityremainconditions;unobservedapplicablemodelbranchesopen. Missingimplementationaloneisnotincompleteconditionalanalysis.',
  next_profile_or_contract_action='NextremainingM15-17/M20/M24head/Markov/sampler/acceptanceoperatorregimes;thenM23commandM21modelinput/indexoperatorwork. Do notrepeatansweredKeyFLOATscale/C-SBF16round/M18mainpartitionorDSparkpublisherquestions.') for cid,scope in scopes.items()}
 mods={
  'M09':dict(finding='72remainingregimes6040execconditionallyreviewed,320exactKeygraphchains/256root. KeyINT8usesFLOATscalethenFP16metadata,unlikeQueryFP16-beforequant;weight2^-6castfusion,Keydualcache/TProwprep and24mode1candidateoutcopiesmodeled. Existing10semanticfamiliesretained/extended.',evidence=e[:8],candidates=['F09','F10','F11','F12'],gap='Actualmasks/slots/duplicateorder/newcachecoherency,table/weight/bankgeneration/aliases andnativequant/physicalVector/commcapacity;otherapplicablebranchesnotclosed.'),
  'M18':dict(finding='All170strictfunctionalregimes9936execmodeled,including5oldgeometry264nowconsumer-specificF27/F29. 31230TASKKVcycles,88mainchains,240auxcheckpoints. 18semanticfamiliespricefullWrowowner/late-selectedgather/featureRS-AGpadding/3KVjoint/crosslayerpartials/lookup-C-Sround/tail-view/slots/generations.',evidence=e[:8],candidates=['F01','F27','F29','F31'],gap='ActualI/position/group/slot/parameterbank/allconsumeridentitiesandnewnorm/GEMM/RoPE/reductionprecision;modelbranchguards/nativephysicalcapacityopen. Fullquerywkvweightsstilllive,fullbankretentionmustbepriced.'),
  'M06':dict(finding='Atnativeauxcheckpointsbefore37/38/39,HCmeanandnextattentionHcPresourceinputcanshareloadsconditionally(fixedEngram1/14notintervening). Mean≠weightedY;preservemeanBF16,allHcPreoutputs/coefficients andAG/fusedkernelreadiness. Existingv58HCsemanticbranchesunchanged.',evidence=[e[1],e[2],e[3],e[4]],candidates=['F01','F27'],gap='Sameinputgeneration/allreaders/native4-streammeanpolicyandnewmultioutputtile/readiness/resourcecost;notanewHcPrephysicalceilingorwholemoduleclosure.')}
 remaining=deepcopy(read(O/'module-coverage-gaps-v63.json')['remaining'])
 remaining.append('v64Index/Context242regimes15976TASK=72M09pending6040+170M18strict9936;237oldpending15712+5geometry264nowownconsumerconditionalreview. 320Keychains256root,240auxchains192rootbeforeblocks37/38/39,31230TASKKVcycles,88mainprojectionchains. KeyFLOATscalequant→FP16cachevsQueryFP16-beforequantwitness95/96;M18C/SBF16-beforeRotarywitness. Keyfusedallvalidbudget1556M,consumer1044M;weightsBF16→FLOAT*2^-6→FP16. M18mainWshard19.66MB/full157.29MB;N24/M12rowownercombinedmatrixops11/9worse. Featureshardglobalnormstats/KVFP32RS→owner512norm/RoPE/BF16AGmatrixops11/18 butcollectives3vs1/Pceil8: M12P16send129108vs107520worse,M24send193704vs215040;3auxAGunchangedtotalpayloadratio.9752forN=Mlarge,fullcontextbankreadersaddoldAGback. 3KVjointM6400saves131072000Blogicalinputbutaggregate376TFcoldreferencefloorboth267.72153us. LateIaftertargetsample:fixedselectedgather8max(U_p)couldsavenothingevenU<N. 26choices/28M09M18semanticfamilies,HCmean+HcPresharedloadnotweightedYreuse,K0notattestedactive. Fullledger832/180111scopedreview,0geometry,1250/106839pendingincluding3MemSet960. No newremote/health/model/NPU/implementation;35candidates0wholemodules,allphysicalnumericceilings/globaluncertaintiesunchanged. NextM15-17/M20/M24headMarkovsampleracceptanceoperatoranalysis,thencommandmodelinput/indexoperatorsnotframeworkaudit.')
 action=(
  '继续原Goal最新v64 PROGRESS。上一轮v63也是PROGRESS。坚持profile优先，代码只为具体算子融合/拆分/重组/理论上限，禁止独立framework/HTTP/SSE/loader/lifecycle/SDKcatalog审计。'
  '先读operator-analysis-coverage-ledger-v64、index-context-regime-models-v64、index-context-native-chains-v64、index-context-operator-contract-v64、index-context-fusion-costs-v64、index-context-fusion-variants-v64、M09-M18-operator-branch-inventory-v64。本轮242regimes15976exec=72M09pending6040+170M18strict9936，237pending15712+5geometry-only264全部补自身consumer条件；全2082/286950现在832/180111scopedreview、0geometry-only、1250/106839pending。不是Goal百分比，不等于所有模型分支完整。'
  '320IndexKeygraphchains在2/8/14/20，256root/64unbound；240auxmean+M23gather链192root，原生checkpoint在block37/38/39前，初次按36/37/38猜编号已assert拒绝并修正。312M18source-orderedKVcycles每30TASK，88mainprojectionchains；functional9936不同v34ancestry10760+firstpass480，不强行合并。Keyrootmedianus WK6.660,RMS9.7005,RoPE7.480,DQ2.820,Kscatter12.210,scaleCast1.340,scaleScatter12.281；auxMean14.921/AG16.120，曝光不是e2egain。'
  'F09是IndexQuery、F10是Weight、F11是Key、F12是indices/candidatepublication，别误归父候选。KeyordinaryDQ用FLOATscale生成INT8再castFP16写缓存；Queryhead专用quant先roundFP16scale，已有BF16amax.00787353515625/x.00592041015625反例FLOAT95/FP1696，不能共用FP16除数。Key融合保WK BF16→RMS BF16→RoPE BF16、full128amax/nativeDQ零/非有限/round、同槽双cache发布/重复槽/有效行，fullproducer理想budget1556M，standaloneWKconsumer1044M，rstd4M和FLOATC/S读取/cache130B每validrow保留。M24fullbudget37344B，仅logical不是HBM/微秒gain。'
  'Weights_scale=128^-0.5*32^-0.5→FP32 2^-6，F10 keepBF16projectround→FLOATmultiply→FP16round，单独CastMulCast省16M32B/producer合20M32B；不能FP16cast再缩放(2^20先castInf而原最终16384)。M09remaining608ordinaryDQ=416KeyH128+192QueryinputH1280，不同customHeadQuant。TProwownerKeyprep可conditional计算/8+130B/row新通信，需TP同参数/latent/positiongen，各rank物理slot仍各自应用；queryrows分片不能当重复QLI，DP不同请求不共享。24candidateTensorMove是mode1 fresh candidate_out→bank，现APIcandidates只mode2input，directout需新outABI，区别v60prepare-indices已同bank/selfcopy。'
  'M18main projectionBF16[M,15360]×[640,15360]每卡640，BF16featureAG[8,M,640]→Transpose→full5120RMS，不能640独立norm。MainW当前19660800B/full157286400B/额外137625600B。3targetHCmean BF16边界、single15360GEMM BF16、mainnormBF16、3不同KV512 GEMM/normBF16、M18C/S BF16和Rotary BF16都保留，不能3 BF16 partial加或1536一起norm。'
  '新的feature方案：mainnorm global FP32stats AR→本地640 gamma/BF16→3KVpartial FLOAT→按row ReduceScatter→KV BF16round→各512norm/RoPE→BF16KV AllGather。每卡矩阵ops baseline35389440M→21626880M=11/18，不需复制mainW；KV context用weightK640视图working-set15728640→1966080B，但全wkv还queryKV消费者，不能声称VRAM/8。通信ring send baselinefeatureAG8960M，new8064P+7M,P=ceil(M/8)*8(需FP32partial0/BF16suffix/trim/validrow)，M12P16:129108>107520；M24:193704<215040；M6400:51654400<57344000。这是3collectives vs1，还有FP32sum/norm数值改变/native支持/资源等待。保留3auxgathers时N=M总comm payload new/base=.9751953125，不是整流程少10%；若fullcontextbank其他读者仍需AG，再加8960M。'
  '完整Wrowowner方案是另一选择：投影Nphysical而非Mselected，mainops N/M、KVrows/8，combinedratio=(11/18)(N/M)；N24M12为11/9反更大。Rowowner上下文KVgather2688Nbytes/rank vsfullcontext8960N，但所有rawaux/fullcontextbank读者/位置、真实参数复制/rowmap要保护。三KVjoint1536输出保3norm，省4M5120input：M6400省131072000B，aggregateBF16参考floor原/新都267.7215319us(Cube项已主导)。No hardwarebenchmark或gain。'
  'Late selective aux gather保当前640主Wshard：targetforward已经3mean/3AG，ordered tokenI在worker2080/2085依sampling/rejection之后才产生。可延迟gather并保localmeans直到I，但raggedU=unique I需新protocol/latency，固定AG按8max(U_p)，N24U12均匀P16/concentratedP24，N6672U6668固定仍6672不省。不能只按U/N报通信。Crosslayer main partial可在3checkpoint提前算FP32累积，但I未知会N/M额外work/partialbuffers/Cube争用；不能BF16 partial后相加。'
  'M18Rope每KVcycle4Index/4IndexCheck=2config C/S pairs(Tensor defaultposmap)，table[1048576,1,1,64]268MB分配不等percall，eachIndex只selected M64 read/write+INT32indices516M。3层12lookup可union Ucfg1/2→2U lookup/check；FP32hoist不自动删6内部Cast，显式BF16bank需原nativecastpolicy/所有读者与新ABI。M18 C/S FLOAT→BF16 BEFORE builtinRotary，不同M09FLOATcosABA；反例x129/128,c257/512，保C round输出.50390625，去掉round .5078125。ViewCopy descriptorM512base，实际tail M64 sourcecopy4M64，不全512；Nope448/tail64 Slice/Concat/private mutation融合条件保留。use_cacheTrue是mutablebufferAPI不是翻flag即等价。'
  'Slot source30TASKcycle里3 INTcasts/GEqual/Clip/Div/Broadcast/Mod/Pack/Not/MaskedFill，valueIO120M+scatter读pairs8M；inline原INT32slot读4M，省124M/layer；3同group-bank/B/epoch CSE省240M，3各inline省372M，互斥不加。所有负slot→(-1,-1)，不是F20tokenID仅-1规则。_layer_group_idx真实values未知，各层group不同不能按shape共享。None/emptyperlayer可条件移guard到privateKV前但保validnormaldomain/原errors，metadata274 rejection/anchor/DCP/query slots/tokenindices仍活。跨迭代KV仅语义(request,targetaux三checkpoint/接受拒绝、position、weights/norm/Rope、group/page/slotgen、lastreader)一致复用，已有KV缓存不是新收益。'
  'Fixedcheckpoint37/38/39无Engram1/14介入，HCmean与attentionHcPre同hidden输入可co-load省8MH/checkpoint，mean不是weightedY[1,2,3,4]mean2.5 vs premix[1,0,0,0]Y1。融合多输出可能延迟mean AG直到HcPrekernel结束，不能免费重叠/删HCcoeff。K0_proposeearlyreturn有source，skiptargetaux还需K0先于target/all其他aux读者与TP/DPcollectives inactive，当前K5profile不证明K0 active。'
  '26条件/排除方案与28semanticfamilies(10M09保持扩展+18M18新inventory)完成局部分析交付，无需先native实现。M18旧history archived；M06只review_history和F27关联，HCbranches保持；M01/M21/M22/M25、M07/M11/M12、M08/M14、M10保护。35model+7legacyframework、0wholemodules，global3136归属歧义/849缺PMU/16unbound、3MemSet960pending都保持。所有actual physical numeric ceiling未证明，Goalactive。'
  '下一步M15/M16/M17/M20/M24最终norm/head/Markov/sampler/acceptance算子，对齐v31/v32/v33/v36/F35和当前regime，补dense/argmax/rowselect/scoredtype/跨卡head sharding与Markov成本；之后M23通信和M21模型输入/索引发布算子(不能独立框架catalog)。优先local5GBcache。v64无新remote读取，7来源全部旧166SHA匹配本地点查；无Torch/model/NPU/newcapture/health/process/kill/实现/background。若后续必要remote仅已有xinyufeng/socket ssh166-v56，凭据授权持续不重复问，不能因socket历史存活就当verifiedwait。')
 report='''本轮补齐242个观察组合、15,976次执行的算子条件分析：72组属于Indexer剩余组合，170组属于草稿共享上下文。后者包含此前仅有几何分析的5组、264次执行，现在已接上其自身消费者约束。没有运行模型、NPU或新性能测试。

Indexer Key已绑定320条图内链，其中256条root绑定。其WK、RMS、RoPE、DynamicQuant、Key Scatter、scale Cast、scale Scatter中位设备耗时分别为6.660、9.701、7.480、2.820、12.210、1.340、12.281μs。这些是工作暴露，不是可删除的端到端时间。

Key量化必须先用FLOAT scale生成INT8，再把scale转换为FP16写缓存。Query的专用head量化则先舍入scale；已有反例中，两种顺序得到95与96。融合Key norm、RoPE、量化和两路缓存写入时，需保留WK、norm、RoPE的BF16边界、FLOAT量化规则、有效slot与两路缓存发布关系。M=24、全部slot有效且中间值私有的理想模型，可减少37,344B逻辑接口读写；这不能直接换算为微秒收益。

Index weight的转换链可保留BF16投影舍入后，在epilogue内执行FP32乘2^-6再转换为FP16。不能先转换FP16再缩放：BF16值2^20在前一种顺序下最终为16384，提前转换则已溢出。

草稿上下文已绑定88条主投影链和312个KV段，每段30个算子。主投影每卡产生640维，汇集后才做完整5120维RMS。保留列分片权重、复制完整权重到行所属卡、以及保留特征分片再重组KV，是不同方案，成本分别核算。

| 条件模型 | M=12 | M=24 | M=6400 |
|---|---:|---:|---:|
| 当前feature AllGather的ring发送字节/卡 | 107,520 | 215,040 | 57,344,000 |
| norm统计＋KV FP32 ReduceScatter＋BF16 KV AllGather | 129,108 | 193,704 | 51,654,400 |
| 新方案按行补齐后的行数 | 16 | 24 | 6400 |

新方案的常规矩阵乘计算量是原main＋KV计算量的11/18，不需要复制完整main权重。但它增加collective次数，改变FP32归约顺序，并要保留BF16 KV舍入和完整512维norm。12行补到16行后，通信量反而更大。大批量下，表内发送量约为原feature阶段的90.08%；若把未改变的三路aux gather也计入，整段条件payload约为原来的97.52%。仍有完整context bank读者时，还须加回原feature gather。以上是ring逻辑发送模型，不是实测线速、时延或加速。

复制完整main权重的另一方案，会让每卡该权重从19.6608MB增至157.2864MB。它若先处理N个物理行、再选M行，会多做N/M倍的main投影。N=24、M=12时，即使后续KV按行分工，合计矩阵乘工作量仍增至原来的11/9。若保持列分片权重而改为按需收集选定行，selector在采样/拒绝处理之后才可用，需要延迟gather并保留局部状态；固定AllGather还按8×最大owner选中行数补齐，不能直接按唯一行数比例宣称收益。

三路KV联合为1536列投影时，三套权重、三个BF16输出和三个512维norm仍然独立。M=6400可减少131.072MB逻辑输入读写，但计算量不变；按此前声明的BF16算力与冷流量参考，原方案与联合方案的聚合参考下界都约为267.72μs，计算项已占主导。

M18的RoPE还需保留FLOAT cos/sin转BF16的顺序，不能直接替换成Indexer的FLOAT cos/sin路径。ViewCopy的描述符虽显示M×512底层存储，实际拷贝的是64维尾部。查询表、转换、尾部组装和slot元数据融合都按被访问的数据与真实消费者计算，没有把整个表、cache或底层存储体积算作单次流量。

共记录26个融合、拆分、跨层/跨卡重组、按代际复用或排除方案。当前统一观察台账为832组、180,111次执行已有显式条件分析；1,250组、106,839次仍待核对，包含原有3个MemSet待定项。此前5组几何项已补齐消费者分析，但观察样本、条件方案和报告成稿仍不证明全部模型分支完成；实际硬件容量、缓存流量和新方案数值条件继续保留。
'''
 jobs=[
  dict(kind='profile',status='error',description='Retain2resolvedlocalattempts:auxcheckpointnumbering36/37/38rejectedbyactualTASK/markerbounds(correct37/38/39);sourcepinfieldsource_filescorrectedtosource_pins. No target/serverfailure.',artifacts=[O/e[8]]),
  dict(kind='profile',status='pass',description='242strictfunctionalregimes15976TASKs/PMUcountsconserved;320Keychains256root,240auxchains192root,31230TASKKVcyclesand88mainchains. Distinctancestry/metadata/M23populationsnotmerged.',artifacts=[O/e[0],O/e[1],N/'analyze_index_context_v64.py']),
  dict(kind='review',status='pass',description='7previously166SHA-boundlocalfilesreusedfornamedoperatorclausesonly:KeyFLOATscale/weights2^-6/M18C-SBF16/tailview/slot/stateandselectorlateavailability. No newremote/frameworkaudit.',artifacts=[O/e[2]]+[Path(r['path']) for r in contract['source_files']]),
  dict(kind='design-check',status='pass',description='26operatorchoices;fullKey/weight/threeKVjoint/5typedpartitions/3selectivegather/slots/lookup/meanmodelswithroundingwitnesses. RSrowpaddingM12P16andunchangedauxAGcostincluded;weightsquery-live/newsumprecision/fullbankreadersretained.',artifacts=[O/e[3],O/e[4],N/'model_index_context_v64.py']),
  dict(kind='review',status='pass',description='832review180111exec+1250pending106839conserve2082/286950;237pendingand5geometryM18rolesreconciled. 28M09/M18semanticfamiliesandM06sharedinputreview;MemSet/globaluncertaintyretained,wholemodules0.',artifacts=[O/e[5],O/e[6],O/e[7],N/'consolidate_index_context_v64.py'])]
 persist_refinement(version=64,previous=63,title='Indexer Key/cache and draft context operator work, typed partitions and generation constraints',
  summary=dict(filename='index-context-ceiling-summary-v64.json',previous_turn_classification='PROGRESS:v63router_vector_operator_models_and_MemSet_gap_retention',reviewed_regimes=242,reviewed_executions=15976,previous_pending_regimes=237,previous_pending_executions=15712,previous_geometry_regimes=5,previous_geometry_executions=264,IndexKey_graph_chains=320,IndexKey_root_chains=256,aux_graph_chains=240,aux_root_chains=192,context_KV_cycles=312,main_projection_chains=88,operator_decisions=26,M09_M18_semantic_families=28,
   observed_regimes=2082,scoped_reviewed_regimes=832,scoped_reviewed_executions=180111,geometry_only_regimes=0,geometry_only_executions=0,pending_regime_reconciliation=1250,pending_execution_reconciliation=106839,MemSet_explicit_pending_regimes=3,MemSet_explicit_pending_executions=960,new_remote_source_read=False,local_pinned_operator_clause_review=True,native_test=False,implementation=False,new_performance_result=False,all_actual_physical_numeric_ceilings_proved=False,
   remote_job_status='No new remote job. All work local saved-profile/nativeindices/pinnedsource/math. No model/Torch/NPU/newcapture/health/process/kill/implementation/backgroundanalysis.'),
  changes=['242regimes15976nativeexecutionsscopedconditionalreview:237pending15712plus5M18geometry264nowwithownconsumercontracts.',
   '26operatorchoicesquantifyKeyscaleorder,typedcontextpartitions/RSrowpadding/extraweights/selectionreadiness,lookup/tail/slots/generationandcrosslayermeanconstraints.',
   '832regimes180111execwithscopedreview,1250pending106839,0geometry-only;28M09/M18semanticfamiliesandHCsharedinputreviewwithoutwholemoduleclosure.'],
  candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,jobs=jobs,report=report,read_first=[e[6]]+e[:5]+[e[7]],publication_rows={},gate_updates=dict(overall_complete=False),
  scripts=[N/x for x in ['analyze_index_context_v64.py','model_index_context_v64.py','consolidate_index_context_v64.py']],executor_prefix='zcode-index-context',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
