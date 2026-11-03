"""Persist attention/indexer operator-focused coverage, fusion decisions and bounds."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
 assert not (O/'REVIEW-CHECKPOINT-v60.json').exists()
 p=read(O/'attention-indexer-profile-and-edges-v60.json');m=read(O/'attention-indexer-work-traffic-models-v60.json');c=read(O/'attention-indexer-native-numerical-contract-v60.json')
 v=read(O/'attention-indexer-fusion-reorganization-variants-v60.json');inv=read(O/'M09-M10-operator-branch-inventory-v60.json');ins=read(O/'attention-operator-inputs-v60.json')
 assert p['regime_count']==51 and sum(p['native_execution_counts'].values())==6863
 assert len(v['variants'])==13 and sum(len(a) for a in inv['modules'].values())==19
 assert len(p['graph_SMLA_chains'])==3200 and len(p['graph_QLI_chains'])==640
 # Explicitly separate example shapes from shapes observed for each primary operator.
 domains={op:sorted({r['M'] for r in p['regimes'] if r['operator']==op}) for op in ['SparseFlashMla','VllmAscendQuantLightningIndexerV2']}
 m['observed_M_domains_by_primary_operator']=domains
 for r in m['models']:
  r['observed_primary_operators']=[op for op,ds in domains.items() if r['M'] in ds]
  if r['M']==2:
   for a in r['indexer_examples']:a['observed_indexer_shape_in_capture']=False
 write(O/'attention-indexer-work-traffic-models-v60.json',m)
 e=['attention-indexer-profile-and-edges-v60.json','attention-indexer-work-traffic-models-v60.json',
    'attention-indexer-native-numerical-contract-v60.json','attention-indexer-fusion-reorganization-variants-v60.json',
    'operator-regime-refinement-v60.json','M09-M10-operator-branch-inventory-v60.json','attention-operator-inputs-v60.json']
 write(O/'PROFILE-CODE-QUESTION-GATE-v60.json',dict(status='SMLA_QLI_OPERATOR_FUSION_WORK_NUMERICS_AND_CANDIDATE_REORGANIZATION',
  previous_turn_classification='PROGRESS:v58HCoperator_analysis_with_v59denominator_clarification',
  profile_evidence=[e[0],'operator-shape-ceiling-ledger-v55.json'],
  questions=['Does candidate_mode2 actuallyskipCubeQKwork oronlychangeVectorTopKscores?',
   'Which INT8/FP16/FP32/BF16 boundaries and internalGMqueues constrainfusion/ceilings?',
   'Can QLIfinalprepare/publication andSMLAinverseRoPE/layoutinterfacesbefused whilekeepingotherlayerreaders?',
   'Whatexactsemanticconditionspermitbitmap/pre-gather/crosscard orcrossiterationreuse?'],
  allowed_source='17exactoperatorcaller/adapter/A3kernel/directnumericalhelperfiles only. Pinmodelconfig+existingserveargument. No independentframework/lifecycle/loader/SDKcatalogaudit.',
  source_loaded_native_binary_identity_not_assumed=True,opaque_profile_attrs_not_mapped_by_API_ordinal=True,
  no_new_model_Torch_NPU_capture_health_process_or_implementation=True,overall_complete=False))
 updates={}
 for cid in ['F09','F10','F11','F12','F13','F14']:
  gap=('Namedfusion/quantizationcontracts reviewed;remainingquery/weight/keyprojectionregime-specificwork/weight/cachecapacity models stillneedseparateanalysis.' if cid in ['F09','F10','F11'] else
       'Capturedconditionaloperatoranalysis delivered;actualperquerylengths/keyunion/candidatevalues/phasecost/cachetiers andtypedVectorTopKcapacity remainparameters. Newnativeallconsumer/readiness/numericalproof notassumed. No implementationrequiredtoacceptconditionalanalysis.')
  updates[cid]=dict(review_v60=dict(profile=str(O/e[0]),models=str(O/e[1]),contract=str(O/e[2]),variants=str(O/e[3]),semantic_inventory=str(O/e[5]),
   relation='51regimes/6863nativeexecutions reconciled. SMLAalreadyfusedwithGMscore/exp/PVqueues;QLImode2fullQKthenVectorpenalty. 13conditional/rejectiondecisions includebitmap544Bvs8192Blist,prefilterwithfillers,preparedindexepilogue,source-resolvedselfcopy,quant/cache/weight/SMLAepilogueandcrosscard/iterationlimits.',
   conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
   profile_review_state='OPERATOR_FORMULA_NATIVE_NUMERICS_WORK_IO_AND_CONDITIONAL_FUSION_RECONCILED_v60',
   conclusion_changing_gap=gap,
   next_profile_or_contract_action='AdvanceQ/KV/outputprojection dense/quantizedoperators andotherunreviewedshape-role-context ceilings. Do notre-auditknownQLIfullscan,FP16quant,helperreturnalias orSMLAexistingfusion asunknown.')
 mods={
  'M09':dict(finding='832eachQLI/queryquant/prepare executions,18QLI+12helperregimes;640graphchains/512rootbound. 10semanticoperatorfamilies nowexplicit. QLImode2fullscan→postscorecandidatepenalty,INT32→FP16ReLU/1024→FP16headreduce→FP32scalarGM path;bitmap/prefilter-filler andprepare/publication contracts bound.',
   evidence=e,candidates=['F09','F10','F11','F12'],gap='Projection-specificceilings andactualvalues/cachetiers/typedVectorTopK/phasecost/newkernelcontracts remainopen;10families are scopedinventory notallguards/fullmoduleclosure.'),
  'M10':dict(finding='All4367SMLAexecutions/21regimes reconciled;3200exactinverseRoPE/layoutgraphchains. 9semanticfamilies includingSWA/C2/C1/draftoriginalsparse/emptyquery/GMqueues/epilogue/SWAsplit/reuse. ConventionalworkandlogicalIOparameterizedbyactualvisiblekeys/union,notpoolcapacity;disabledLSE0elements clarified.',
   evidence=e,candidates=['F13','F14'],gap='Actualvisiblequery/keycounts/union/runningsoftmax/cachedphysicalqueues/resourcecapacity/newfusionprecision andallremainingapplicableoperatorbranches notclosed.'),
  'M11':dict(finding='Current1663200SMLA→inverseRoPE→Transpose rolechains tiedtoexactTASKIDs. RoPEtouches64tail;full512layoutmovementseparate. M3conditionallogicalinterfacedeletion49152+393216=442368B whilefinaloutput/collectiveinput remain.',
   evidence=e[:4],candidates=['F14'],gap='Alloutputprojection MatMul/QuantBatchMatmul/TransposeBMM regimes andCP/collectivecost/consumerprecision/ownership requirefurtherquantitativeoperatoranalysis.')}
 remaining=deepcopy(read(O/'module-coverage-gaps-v59.json')['remaining'])
 remaining.append('M09/M10v60:51regimes/6863executions(SMLA4367,QLI/queryquant/prepare832each),39primary+12N/Ahelperregimes;3200SMLAand640QLIgraphchains,64root/16unboundgraphs kept. 13operatorconditional/rejectiondecisions and19semanticfamilies reviewed. QLImode2doesfullCube scanthenVectorpenalty;bitmap544B/query atserve34816 versus8192Blist mayshare4consumers. PrefiltermustpreserveNEG_HUGEoutsidefillers/ties/nonfinitefallback;finiteFP16boundproof available,notactualvalueattestation. QueryscaleFP16-beforequant,QLI DEQF16ReLU/1024/halfweightproduct/M16headreduce andSMLABF16exptile/GMs known. Actualnq/Lq/U/Eq/cache/phase/typedVectorTopK/crossrankcriticalpath stillparameterized. Nextprojection/QKV/output/otherunreviewedoperatorceilings;do notloopframework/nativeAPIcatalog/knownlocalcontracts.')
 action=(
  '继续原operator Goal，最新v60。上一轮v58/v59为PROGRESS，当前v60又完成attention/Indexer算子取证与条件分析；整体未完成。用户要算子融合/拆分/重组和理论上限，profile优先，定点代码只解决具体算子问题；不独立研究framework/HTTP/SSE/lifecycle/loader/SDKcatalog。'
  '先读attention-indexer-profile-and-edges-v60、attention-indexer-work-traffic-models-v60、attention-indexer-native-numerical-contract-v60、attention-indexer-fusion-reorganization-variants-v60、M09-M10-operator-branch-inventory-v60及operator-regime-refinement-v60。51regimes=21SMLA+18QLI+6queryquant+6prepare，完整32DBnative计数4367/832/832/832=6863；39primary和12helper区别，helperComputeInfo是N/A，只有精确graphchains可bindM3不能按最近时间补全部runtimeShape。'
  '80图3200层SMLA→inverseRoPE→Transpose，640QLI→prepare链(512root/128unbound)；每图indexer层2/8/14mode3、20mode1producer、24/28/32/36mode2consumer。SMLAroot SWA128次median37.9105us、C2 1152次44.6505us、C1 1280次44.681us；QLIroot mode3/1/2=192/64/256次median94.232/212.364/497.125us，prepare约12.7..12.91us。跨context/unbound同形不能当pairedbenchmark；64rootgraphQLI曝光median3.92854%不是严格端到端收益上限/MFU。'
  '最关键新事实：原QLI mode2候选过滤在Vector ProcessVec1、全CubeQK之后，现有candidate2048×block8不代表提前减少dot。QK INT8→INT32；Fixp DEQF16 ReLU并用0x3a800000=1/1024；wq=FP16(weightsFP16*queryScaleFP16)，Brcb16写GM；Cube FP16headreduce请求M16只存1行FP32score；Vector乘FP16keyscale，mode2penalty，再TopK。SourceQK/FP16/GMem/Vector分开计，不套一个INT8MFU。'
  'F12-BITMAP优先：同forward1producer→4consumer，原bank已共享baseline；consumer还反复Sort2048和membership距离循环。maxmodel34816由历史serveSHA c297绑定，逻辑candidate块universe4352，位图544B/query versus2048INT32 list8192B；membership证明有效ID/-1域、duplicate集合等价，需新bank完整清零/写入/row-generation/readiness并保其它原list消费者。bitmap本身不减QK。'
  'F12-PREFILTER-FILL：候选先gather可减少QK，但不能把候选外全部写-1。原mode2 outside分数NEG_HUGE而真实position仍可做TopK填充；有限FP16wq/keyscale时保守|score|<2^49，FP32pen链结果exactNEG_HUGE，可只算candidate并生成原outsideconstant-score位置给原sort/merge保tie。非finite/halfproductoverflow需fallback或单独合同，实际capture值未证。不要再把“candidate不足TopK必须保fillers”当未知。'
  'F12prepare原函数源码已读：过滤selected>=0且<(position+1)//ratio，chronological sort，-1suffix，optionalvalidlength；CSA native用binarysearch找firstnegative，不能直接删prepare/送scoreorder。helper返回provided indices_output，dsa当前同bank再copy回；source域alias问题已答，version/capture/debug等观察者仍保留，不报多出832devicecopy。QLIfinalepilogue可条件融合prepare但stablebank/candidateout给后续层读仍需写。'
  'Queryquant源码perhead128：amax/127 FP32 div_rn先roundFP16再clampmin2^-24，normalized用该roundedscale/div_rn/nearbyint/clamp127。明确BF16反例amax.00787353515625,value.00592041015625，FP32scale模型q95 versusroundedFP16scaleq96；不能换成任意DQ。Keypath DynamicQuant后才castscale16再twoScatter，内部query/keyquantizer不假设同一。保RoPE先BF16round和完整128amax。'
  'SMLA已融合SWA+compressed+onlineSoftmax/sinks/PV，但A3源码有FP32scoreGM/BF16expGM/PV/update/gatherworkspaces；Softmaxexp tile CAST_ROUND BF16后PV，finalBF16RINT。LSEreturnFalse实际0elements，v55descriptor空shape当scalar多算4B不适用于此op；新semantic输出校正，不能全局把emptyshape都当0或1。'
  '公式SMLA GEMMwork4HDΣnq，idealValueIO4THD+2DU+4H+metadata，H64D512，latent同作K/V。QLI INT8work2HiDiΣLq，Hi32Di128；usefulhalfheadsum2HiΣLq，M16requested16倍；scalar-score GMinterface8ΣLq，wqpack64THi/pack会按S2corepartition重复，非总HBM。SMLA score+expGM接口12HΣnq，非实HBM。6M示例含2/3/800/816/830/834，但QLIM2未捕获已标注；actualnq/Lq/U/Eq/有效T没有opargs，使用参数/假设fullvalid示例，不能拿上界work报实际time下界。'
  '13变体覆盖bitmap,prefilter-fill,prepare,selfcopy,headreduce,RopeQuant,weightEpilogue,keyCache,SMLAqueue,earlySWAsplit,SMLARopeLayout,keypartition,crossiterationdelta。SWA可在Indexer之前起部分attention，但新(m,l,a)state/sink只计一次/online数值保序与资源必需；归一BF16out+disabledLSE不足做精确merge。SMLA→inverseRope仅64tail(448..512)，融合sendlayout条件M3省logical49152+393216=442368B，保持SMLABF16与RopeBF16两round、输出/collectiveownership/alias/pad。'
  '跨卡不要因所有32indexheads在每rank就报TP8重复：若query行已分片，Mlocal*L换成(TP*Mlocal)*(L/TP)每卡工作不减还多通信；实际rowidentity/ownership仍需绑定。跨迭代旧Key不变不代表旧score可复用：q(1,0)→(0,1)、keys(1,0)/(0,1)使TopK反转；仅全query/wq/scale/key/mask/candidate/pin/generation一致时可条件delta。'
  'M09 10/M10 9个operator语义family替换旧chronological台账，旧记录archived；并非全guards或wholemodule闭环。M01/M21/M22/M25语义不改。17个exact原166stage代码只读docker cp已完成(2个.cpp精确路径不存在，header内实现存在)，dsa3505/indexerd5d6历史hash匹配，loadednativebinary不假定attested。无模型/Torch/NPU/newprofile/health/process/kill/实现。'
  '下一步转向Q/KV投影、RMSNorm/Quant/RoPE、attention输出GEMM/TransposeBMM等尚未定量的全部shape-role-context算子，包括MatMulV2/QuantBatchMatmulV3/TransposeBatchMatMul及资源/精度/跨卡分区；不要再重复已答HC/MC2/shared/QLI本地合同。完整83类型/2082regimes和35模型候选仍未全部定案，849PMU缺、16unbound、3136module歧义保留，0wholemodulesclosed。条件分析不需要先native实现benchmark。Goalactive，优先本地缓存；必要exact文件只读socket ssh166-v56-xinyufeng.sock显式override，授权已给不重复问。')
 report='''本轮完成 attention/Indexer 的51个捕获组合分析：SparseFlashMla 21组、QLI 18组、query quant与prepare各6组。与32份原始执行索引逐条核对，执行次数分别为4,367、832、832、832，共6,863次；模型和服务器运行状态未改动。

最值得优先研究的是候选位图与提前取Key。当前QLI的候选过滤发生在完整QK和head归约之后，不能把2048候选块理解为已提前减少矩阵乘法。每个候选生产层把结果共享给4个消费层，但消费层仍会排序2048个块号并计算membership。

```mermaid
flowchart LR
 Q[INT8 Query与Key] --> MM[完整QK扫描]
 MM --> H[FP16 ReLU中间值\nFP16加权head归约]
 H --> S[FP32标量分数]
 C[候选块号] --> P[候选外分数降级]
 S --> P
 P --> K[TopK]
 K --> I[可见性过滤与时间顺序排序]
 I --> A[SparseFlashMla及后续层]
```

| root绑定、M=3图中的QLI模式 | 次数 | 中位设备耗时 |
|---|---:|---:|
| C2普通扫描 | 192 | 94.232μs |
| C1候选生产 | 64 | 212.364μs |
| C1候选消费 | 256 | 497.125μs |

各模式的prepare约12.7–12.91μs。模式间的差值没有隔离成单项成本，不能直接报候选过滤收益；16个阶段未绑定图也没有混作同负载基准。64个root图中QLI设备任务时段曝光占比中位3.9285%，不是端到端可实现收益。

当前case的max-model-len为34,816，候选块大小8，逻辑块域最多4,352个，因此membership位图为544B/query，原2048个INT32块号为8,192B/query。原始候选列表可继续供需要它的消费者使用，同时发布位图以减少消费层重复排序与查找。已验证有效块号/-1域的membership等价；仍须保持每行、每次forward的完整写入、清零、代际和就绪关系。位图本身不减少QK。

提前按候选取Key可以减少QK，但必须保留候选外的补位语义：当前候选外位置被降为NEG_HUGE，候选不足TopK时仍可输出真实位置。有限FP16权重乘积与key scale条件下，可以证明原分数远小于NEG_HUGE的FP32间隔，从而生成同一常量分数，再沿用原sort/merge保留补位和tie规则。直接丢弃候选外位置或改成-1不等价；非有限中间值需要回退或另立数值合同。

代码定点读取还明确了三个精度边界。Query量化先把scale舍入到FP16，再用这个scale量化；独立BF16反例的量化结果为95与96，证明不能任意改成用未舍入FP32 scale。QLI先将INT32 QK经ReLU和1/1024转换到FP16，再做FP16权重乘积和head归约，不能当作一次FP32数学求和。SMLA的在线Softmax指数tile会先舍入BF16再参与PV，改变tile或归约顺序需要核对精度。

SMLA本身已经融合SWA、压缩KV、在线Softmax/sink和PV，内部仍有FP32 score、BF16指数、PV及更新工作区。它在root图中的中位耗时为：SWA 37.9105μs、C2 44.6505μs、C1 44.681μs。共3,200条精确SMLA→inverse RoPE→Transpose图链已绑定。

理论模型按实际有效键数和缓存向量并集建模，不按整个KV池容量计流量。令nq为每query实际参与的原始窗口及压缩位置数，U为所有query请求的独立缓存向量数，则SMLA两次GEMM的普通算法工作量为4HDΣnq，理想值读写为4THD＋2DU＋4H，再加索引、metadata和内部队列；H=64、D=512。QLI当前INT8 QK工作为2×32×128×ΣLq，候选消费模式仍是完整Lq。

| 明确假设：M=3且每行有效 | 普通算法工作量 | 参考计算下界 |
|---|---:|---:|
| SMLA 128个SWA位置＋512个压缩位置 | 251.658M BF16操作 | 0.6693μs |
| QLI扫描32,768个Key | 805.306M INT8操作 | 1.0709μs |

参考沿用已注明的每die 376 BF16/FP16 TFLOPS、752 INT8 TOPS、1.6TB/s，仅用于声明条件下的模型。[硬件参考来源](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1) 上表不是捕获输入值的断言，也没有计全Vector、TopK、在线归约、tile填充、缓存层级和依赖，不能据此报几十倍加速或MFU。

共记录13个方案/决策：候选位图、提前取Key并保补位、QLI融合prepare、源代码已明确别名的self-copy特化、head归约重组、Query RoPE＋量化、weight epilogue、Key量化＋cache写入、SMLA内部队列、SWA提前执行、SMLA＋inverse RoPE＋布局、跨卡Key分区，以及跨迭代增量复用的条件与反例。SMLA的LSE关闭时输出为0元素；没有把profile空shape额外计为4B标量。

跨层融合仍须发布后续层读取的indices/candidate bank。跨卡分析不能仅凭每卡拥有全部32个index heads就声称8倍重复工作：若Query行已分片，改成Key分片可能保持每卡计算量并增加交换。跨迭代也不能只因旧Key不变就复用旧分数；新Query可以让TopK排序反转。

M09、M10已分别建立10和9个算子语义分支族，旧记录保留为历史证据。整体仍未完成：实际逐query长度、缓存并集、候选值、内部阶段成本及Vector/TopK能力仍有参数缺口，Q/KV与输出投影等其它算子也尚未全部完成定量分析。
'''
 src=[Path(r['local_path']) for r in ins['files'] if r['status']=='READ']
 receipts=[N/('attention-operator-source166-v60-'+x+'.json') for x in ['entry','direct-dependencies','math-and-workspace']]
 jobs=[
  dict(kind='profile',status='pass',description='51regimes/6863executions conservedagainst32nativeDBindices;3200SMLAand640QLIgraphrolechains,64root/16unbound scopedwithouttensoridentity/gaininference.',artifacts=[O/e[0],O/e[4],N/'analyze_attention_profile_v60.py']),
  dict(kind='review',status='pass',description='17exactoriginal166operatorfiles readwithoutcontainerexec;twoadaptercpp exactpathsabsentbutheaderimplementationspresent. HF16/FP32/INT8/BF16contracts andfullscan-postcandidate boundary bound;nativebinarynotassumed.',artifacts=[O/e[2],O/e[6],N/'read_attention_operator_sources_v60.py']+receipts+src),
  dict(kind='design-check',status='pass',description='6Mconditionaldense/trafficmodels,FP16scale andprefix/filler/bitmap/query-changewitnesses;finite2^49outsideconstantproof;13fusion/reorganization decisions withcrosscard/nonadditivebounds. No wrong-peak/MFU/actualkeycountclaim.',artifacts=[O/e[1],O/e[2],O/e[3],N/'model_attention_fusion_ceiling_v60.py',N/'refine_attention_coverage_v60.py',Path(c['model_config']['path']),Path(c['model_config']['max_model_len_case_binding']['path'])]),
  dict(kind='review',status='pass',description='19semanticoperatorfamilies forM09/M10,priorchronologicalhistoryarchived;51regimeincrementalevidencelinks. NoM01/M21/M22/M25frameworkinventory orwholemoduleclosure.',artifacts=[O/e[4],O/e[5]])]
 persist_refinement(version=60,previous=59,title='Attention and Indexer fusion, candidate reorganization and operator ceilings',
  summary=dict(filename='attention-indexer-ceiling-summary-v60.json',previous_turn_classification='PROGRESS:v58HCoperator_analysis_with_v59denominator_clarification',
   captured_regimes=51,primary_regimes=39,helper_NA_regimes=12,native_execution_counts=p['native_execution_counts'],captured_executions=6863,
   SMLA_graph_chains=3200,QLI_graph_chains=640,conditional_or_rejection_variants=13,semantic_operator_families=19,
   exact_operator_source_files=17,current_candidate_mode2_skips_Cube_scan=False,bitmap_bytes_per_query_reference=544,candidate_list_bytes_per_query=8192,
   target_implementation_changed=False,native_test=False,new_performance_result=False,
   remote_job_status='Onlyexact166stageoperatorfiledocker-cpreadscompleted;localprofile/math analysis only. No model/Torch/NPU/newcapture/health/process/kill/implementation/backgroundanalysis.'),
  changes=['51operatorregimes/6863nativeexecutions andexactattention/indexergraphchains reconciled.',
   'Mode2post-Cubecandidatefilter,roundedFP16quant/QLIheadreduce/SMLAonlineBF16queues/preparestableoutput contracts bound.',
   '13conditional/rejectiondecisions,6Mwork/IOmodels and19semanticoperatorfamilies addedwithactual-value/typed-capacity/fullscopegaps retained.'],
  candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,jobs=jobs,
  report=report,read_first=e,publication_rows={},gate_updates=dict(overall_complete=False),
  scripts=[N/x for x in ['read_attention_operator_sources_v60.py','analyze_attention_profile_v60.py','model_attention_fusion_ceiling_v60.py','refine_attention_coverage_v60.py']],
  executor_prefix='zcode-attention-ceiling',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
