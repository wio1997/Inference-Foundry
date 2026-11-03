"""Persist projection operator coverage and the complete observed-regime reconciliation ledger."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
 assert not (O/'REVIEW-CHECKPOINT-v61.json').exists()
 p=read(O/'projection-work-and-reference-bounds-v61.json');g=read(O/'projection-graph-chains-v61.json');m=read(O/'projection-fusion-and-partition-costs-v61.json')
 v=read(O/'projection-fusion-variants-and-interactions-v61.json');inv=read(O/'M07-M11-operator-branch-inventory-v61.json');coverage=read(O/'operator-analysis-coverage-ledger-v61.json');inputs=read(O/'projection-operator-inputs-v61.json')
 assert p['regime_count']==74 and p['captured_execution_count']==25078 and p['core_projection_regimes']==69
 assert len(v['variants'])==16 and len(g['QR_shared_quant_candidates'])==640 and len(g['C2_FP32_projection_pairs'])==240
 assert coverage['observed_regimes']==2082 and coverage['scoped_conditional_review_regimes']==185 and coverage['geometry_only_other_role_regimes']==5
 assert sum(len(x) for x in inv['modules'].values())==19
 z=next(r for r in m['output_two_GEMM_models'] if r['M']==6400)
 assert z['original_aggregate_reference_floor_us']==z['preserve_round_fused_aggregate_reference_floor_us']
 e=['projection-work-and-reference-bounds-v61.json','projection-graph-chains-v61.json','projection-fusion-and-partition-costs-v61.json',
    'projection-operator-numerical-contract-v61.json','projection-fusion-variants-and-interactions-v61.json',
    'operator-regime-refinement-v61.json','operator-analysis-coverage-ledger-v61.json','M07-M11-operator-branch-inventory-v61.json','projection-operator-inputs-v61.json','projection-geometry-role-collision-v61.json']
 write(O/'PROFILE-CODE-QUESTION-GATE-v61.json',dict(status='Q_KV_OUTPUT_PROJECTION_OPERATOR_GEOMETRY_FUSION_PRECISION_AND_PARTITION',
  previous_turn_classification='PROGRESS:v60attention_indexer_operator_models_and_semantic_coverage',
  questions=['WhattrueGEMMgeometrydoesTransposeBatchMatmulG1executeafterpermutations?',
   'Whichprojectionscanreuseinputquantizationorinputtileswithoutchangingrowdomains/norms/rounding?',
   'Do2GEMMpanel/precomposition/project-before-gather choicesreducecomputeandIOunderactualprecisiondomains?',
   'Whichsame-geometryrecordsbelongtodraftshared-contextratherthanSWA,andwhichregimeshaveexplicitreviewevidence?'],
  allowed_source='2exactoriginal166quantization/outputcallerfiles plusalreadypinneddsa/indexer andnarrowhistoricalC2parameterclauses;noindependentframeworkresearch.',
  no_new_model_Torch_NPU_capture_health_process_or_implementation=True,source_loaded_native_binary_identity_not_assumed=True,overall_complete=False))
 updates={}
 for cid in ['F05','F06','F07','F08','F09','F10','F11','F16','F17','F32','F33']:
  updates[cid]=dict(review_v61=dict(profile=str(O/e[1]),work_bounds=str(O/e[0]),fusion_costs=str(O/e[2]),numerics=str(O/e[3]),variants=str(O/e[4]),full_observed_coverage=str(O/e[6]),
   relation='69coreprojectionregimes/24814execplus5distinctM18lookalikes/264exec havecostmodels. 16conditional/rejectionchoices includeG1GEMM,QRquantreuse,QA+indexweight,C2joint/cast,typedproject-before-gather andBF16-preservingoutputpanels/RS. Precompositionfailsroundingandraiseswork/weights20/9.',
   conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
   profile_review_state='PROJECTION_GEOMETRY_WORK_IO_NUMERICS_AND_CONDITIONAL_REORGANIZATION_REVIEWED',
   conclusion_changing_gap='Capturedprojectionconditionalanalysis delivered. FLOATactiveprecision/typedcapacity/cachetiers/Vector/dequant/internaltiling/actualmasks/newkernelallconsumer/readiness/precision andremainingnonprojectionoperatorfamilies open. 5M18lookalikes havegeometryonly;do notinheritSWAconsumerfusion.',
   next_profile_or_contract_action='Reconcileotheroperatorfamiliesagainstfull2082regimeledger,nextC2pool/state andEngram/remainingGEMM/elementwise-semanticwork. Do not repeatG1geometry,QRsame-source,C2onecast orstaticWOprecompose asunknown.')
 mods={
  'M07':dict(finding='CapturedQ_A/Q_B/KVdensework/IOmodels and3200exactgraphchains;Q_A3vsKV24currentdomains,640QRduplicatequantpairs andindependentIndexweightbranch identified. 10semanticoperatorfamilies;fullrowRMS/quant/BF16barriers andexistingmultistreamoverlap retained.',evidence=e[:8],candidates=['F05','F06','F07','F08','F32'],gap='Actualtypedcapacity/cache/internaltiling/Vector/masks/newfusionreadiness andunobservedapplicableQKVbranches notclosed.'),
  'M11':dict(finding='All7WO_Aregimes/4424calls areG1,logicalM=T notM1;ordinaryGEMMspecialization prioritized. WO_BsmallMatMulV2 andlargeMatMulV3included. 9semanticfamilies;BF16-preservingpanels/RSprice,precompose20/9workweightpenalty androundingwitness.',evidence=e[:8],candidates=['F16'],gap='NativeexistingG1fastpathnotknownabsent;actualnewGEMMprecision/stride/handoff/capacity andcollectivepanel/rowownership costs remainconditional.'),
  'M08':dict(finding='BothC2FLOATinputprojections shareoneexistingcast;240graphpairs +directregimes modeled. Jointtilesremove4MKduplicateFLOATread;cast+jointbudgetsnonadditive. Projectlocal→gather2FLOAToutputs+SWA reducesfeaturecontent2xconditionally beforeglobalringpool.',evidence=e[:5],candidates=['F17','F32','F33'],gap='FLOATactiveprecision/capacity andC2ringpool/state/update/norm operatorregimes notfullyquantified;newrow/globalstateconsumers andallotherapplicablebranches open.'),
  'M09':dict(finding='AllcapturedIndexQ/weight/keyprojectioncosts included;commonQRquant andQA+Indexweight optionsretain32x128headquantseparatefrom1280-wideFLOATscaleDQ. v60QLI/prepare/bitmapsemanticbranches retained.',evidence=e[:5],candidates=['F09','F10','F11'],gap='Actualquantizer/readiness/weights/precisionofnewjointvariants andtypedphysicalcapacity/otherremainingIndexeroperatorparameters remainopen.'),
  'M18':dict(finding='Conservationidentified5same-signaturedraftshared-contextprojectionregimes/264execdistinctfromSWA. Includedgeometrywork/IOonly;do notassignSWAconsumerchainbyshapematch.',evidence=[e[0],e[5],e[9]],candidates=['F27'],gap='Shared-contextprojectionotherregimes/consumer-specificfusionnotnewlyreviewed;priorF27contractsretainedandexplicitgeometry-onlyscope.'),
 }
 remaining=deepcopy(read(O/'module-coverage-gaps-v60.json')['remaining'])
 remaining.append('v61projection:69core/24814exec+5M18lookalikegeometry/264exec=74/25078;nativeconservationfoundandresolved264typed-shapecollision,notserverfailure. 3200projectionchains,640QRduplicateDQpairs,240C2pairs;16operatorconditional/rejectiondecisions,19M07/M11semanticfamilies. G1BMMall4424logicalM=T;WOprecomposeBF16witness+20/9work/weightpenalty;M6400paneldelta26.2144MBbutaggregate376TFreferencefloorunchanged321.26584us. Completeobserved2082ledgerindexes185scopedreview/76397exec,5geometry-only/264exec,1892pendingexplicitregime-familyreconciliation/210289exec. Pendingisnotclaimpriorcandidateevidenceabsent;allphysicalnumericceilings/wholemodelbranches stillunproved. NextC2state/pool andEngram/remainingoperators,notrepeatansweredprojection/frameworkcontracts.')
 action=(
  '继续原operator Goal，最新v61。上一轮v60为PROGRESS，当前v61也有新算子模型/条件方案/完整observed台账。profile优先，代码只围绕算子融合/拆分/重组和理论上限，禁止独立framework/HTTP/SSE/loader/lifecycle/SDKcatalog研究。'
  '先读operator-analysis-coverage-ledger-v61、projection-work-and-reference-bounds-v61、projection-graph-chains-v61、projection-fusion-and-partition-costs-v61、projection-operator-numerical-contract-v61、projection-fusion-variants-and-interactions-v61及M07-M11-operator-branch-inventory-v61。74regimes=69Q/KV/Indexer/output核心(24814exec)+5M18共享上下文同形(264exec)；4类nativecount14015MatMulV2/5199QuantBatchMatmulV3/4424TransposeBMM/1440MatMulV3=25078。初始signature核对少264触发assert，已保留collision证据并按M18单列，只分析几何不继承SWA融合。'
  '69核心全部有densework、dtype与logicalIO/cold-reference模型，FLOAT真实执行模式/typedpeak仍参数；不能套BF16峰值。3200精确graph投影链、640QR重复DQ候选、240C2双投影对。rootmedianus QA17.501,Qnorm5.04,QDQ3.06,QB42.261,KV11.221,KVnorm13.08,WO_A54.98,WO_B14.961,RS9.9905,IndexQ14.06,IndexWeight10.821。时间/曝光不是e2egain。'
  'Q_B和IndexQ从同source qr走两次DQ3x1280；832IndexQ calls、640graphpairs。QR DQ为1280-wide/FLOATrowscale，完全不同于IndexQ后32heads×128的FP16-roundedheadscale量化。共同producer可条件复用现有quant_method层PreparedLinearInput合同(不是证明wholeLinearwrapperdrop-in)；M3省11532Blogical。再jointQB+IndexQ总预算15384B(不要与11532相加)，47.18592MBINT8weights仍全要；保NZawarepack、BF16round、独立32768/4096输出及rowstride/scale/bias/readers。'
  'Indexweights从query_hidden独立于QR/IndexQ，可提前或与QA做1280+32输出；保QA1280norm与weightBF16→FP32scale→FP16，不得改RMS1312。现有QA3/KV24不同rowdomain，盲目都算24行使165.15072M→440.40192Mops(8/3)，不排除localproject再交换：条件replicatedweights时localjoint55.05024M，原每卡3倍，但权重流量/通信/精度必须另算。'
  '全部7TransposeBatchMatmulregimes/4424calls只有group1，[T,1,4096]perm→[1,T,4096]，实为M=T，不是T个M1GEMM。优先F16-G1普通GEMM/views特化，保stride/activeweights/精度；不声称native一定无G1fastpath，也无同shapebenchmark/40usgain。WO_B大M6400/6528/6672是MatMulV3，已纳入不可漏。'
  'WO_A→WO_B秩1024 BF16中间值不可删round。面板融合省4MR：M24=98304B,M6400=26.2144MB；M6400 aggregate参考原/融合都321.265838us，Cube算力占主导。预乘4096x5120权重18.874368→41.94304MB，compute/weights20/9；IO break-even5632不代表performance。BF16明确反例x=(129/128,1),A=diag(129/128,65/64),B=(1,-1)：有中间round输出0，预乘输出1/16384。RS必须保每rankpartial BF16再原归约，不能把未舍入FP32 accumulator当等价。'
  'C2同输入FLOAT WKV/WGATE已有一次hidden.float共用，不把sharecast算新优化。联合投影仅省4MK重复FLOAT输入读：M24=.49152MB,M6400=131.072MB；原cast+2read输入预算14MK→cast+joint10MK→理想onchipjoint2MK，合计省12MK不是4+12。保两个不同参数和512FLOAT输出/stride/后续pool，FLOAT activeHF32/fullFP32与capacity仍无证明。'
  'Project-before-gather内容模型：36noncompressor source仅SWA512BF16→10x；1C1source需SWA512BF16+C1latent512BF16→5x；3C2source需SWA512BF16+WKV/WGATE各512FLOAT→2x。均相对5120BF16 rawhidden内容，不是HCCL线速/latencygain，保replicatedweight/row/card/pad/其他rawreaders。C2pool是跨row/state操作，必须收到global projectedrows后运行，不能blindlocalpool。'
  'QA/QB4367对WOA/WOB4424差57不等于deadoutput；已有v18localzero/cache-only证据保留，alltoall后可能有别卡query必须算输出。160M07|M08原功能归属歧义仍保留，GraphSWA/C1 source-order role不等于物理tensorproof。'
  '全2082observedregimes/83types/286950exec已统一每项ledger：185regimes/76397exec有显式scopedconditional融合+工作IO review，5/264只geometry，1892/210289需与已有family证据逐项对齐。87genericdense参考不提升为fusion已分析；pending不代表以前所有候选证据没做。observed universe也不等于所有model分支，任何实际physicalnumericceilings/wholemodule仍未全部证明。'
  'M07新10/M11新9个operatorsemanticfamilies，旧记录archived；M09/M10v60semanticbranches不改，仅review_history，M01/M21/M22/M25完整不改。35model+7legacyframework候选、0wholemodulesclosed。16条件/排除方案已交付，不需先native实现benchmark。下一步从全ledger推进C2state/pool/norm、Engram与其它尚未定量算子(包括剩余GEMM角色、Vector/elementwise/索引)。别重复已答HC/MC2/shared/attention/QLI/projection本地合同。'
  '无模型/Torch/NPU/newcapture/health/process/kill/实现，Goalactive。原166只读docker cp2个文件hash与frozen一致完成，其他代码复用已绑定数据；优先本地5GBcache，必要exact文件显式socket ssh166-v56-xinyufeng.sock授权已给不重复问。')
 report='''本轮分析69个Q/KV、Indexer及输出投影组合，共24,814次执行；另有5个同形的草稿共享上下文组合、264次执行，仅补齐其成本模型。与32份native索引核对后，总计74组、25,078次，避免把草稿共享上下文投影归入SWA。模型和设备运行状态未改动。

| root绑定图中的投影阶段 | 中位设备耗时 |
|---|---:|
| Q_A / Q Norm / DynamicQuant / Q_B | 17.501 / 5.040 / 3.060 / 42.261μs |
| SWA KV / KV Norm | 11.221 / 13.080μs |
| WO_A / WO_B / ReduceScatter | 54.980 / 14.961 / 9.991μs |
| Index Query / Index Weight | 14.060 / 10.821μs |

优先候选是group=1输出投影特化。全部7个TransposeBatchMatmul组合、4,424次调用，输入[T,1,4096]经过指定perm后是[1,T,4096]，逻辑GEMM的M为T，不是T个M=1的GEMM。可条件改走普通GEMM和视图变换，保留当前权重、stride、累加精度及BF16输出。API标签不能证明底层没有G1优化，也没有同shape对照测量，因此未报告具体提速。

另一候选是复用Q_B和Index Query对同一qr的动态量化，已绑定640对图内记录。M=3时，仅复用量化生产者的逻辑字节预算为11,532B；进一步联合两个GEMM，总预算为15,384B，不能将两项相加。两套INT8权重合计47.18592MB仍需保留。这里的qr量化是1280维、FLOAT行scale，和Index Query之后32×128的FP16舍入scale量化不同。

Index Weight只依赖query_hidden，可以研究提前执行，或与Q_A联合输出1280＋32列；两部分的舍入、归一化域和读者必须分别保留。当前Q_A为3行、KV为24行，盲目把两路都扩到24行联合计算，普通算法工作量会从165.151M增加到440.402M。先在本地投影再交换结果是另一种方案，需要单独核对跨卡权重、行序与所有消费者。

两级输出投影不能靠预乘权重无条件合并：4096→1024→5120的权重体积会从18.874368MB变为41.94304MB，计算量也增至20/9，约2.22倍。中间BF16舍入还有明确反例：分级结果为0，预乘结果为1/16384，即使预乘权重可精确表示也不等价。

| 保留中间BF16舍入的两级融合模型 | M=24 | M=6400 |
|---|---:|---:|
| 可减少的逻辑中间写读 | 98,304B | 26,214,400B |
| 原聚合参考下界 | 12.1344μs | 321.2658μs |
| 融合后聚合参考下界 | 12.0730μs | 321.2658μs |

参考使用此前注明的376 BF16 TFLOPS和1.6TB/s，仅适用于声明的计算量、冷访存和容量假设。M=6400时计算项已占主导，省26.2MB不直接降低该下界。两个GEMM共享Cube资源，不能把各自计算时间取max当成免费重叠；实际缓存、向量工作和调度仍需定案。

C2两路FLOAT投影已经共用一次hidden.float()。联合输入tile可条件减少一次FLOAT输入读取，M=6400时为131.072MB逻辑数据；若同时研究cast融合，必须按联合后的接口计账，不能重复领取同一读写。FLOAT输入标签不足以证明实际采用完整FP32还是HF32，未套用BF16峰值计算其上限。

跨卡投影前移也按所需输出区分：只需SWA KV时，每行输出内容相对5120维BF16 hidden可缩至1/10；C1源层需两路BF16，为1/5；C2源层还需WKV和WGATE两路FLOAT，为1/2。这些是条件内容比，不是网络或端到端加速。C2池化依赖全局行和持久状态，仍须在收到完整投影后执行。

共记录16个融合、特化、重组或排除决策。WO_B接ReduceScatter保留每rank部分结果的BF16舍入；局部query为空也不能直接跳过全局输出投影，其他卡仍可能贡献数据。Q_A/Q_B与输出投影调用数相差57，不是冗余调用证明。

完整观察台账已覆盖2,082个组合、286,950次执行：185组有显式条件融合与工作量/逻辑IO分析，5组仅几何成本，1,892组仍需与已有family证据逐项对齐。该计数不代表全部模型分支或总任务进度；通用形状公式和报告成稿没有提升为“全部完成”。M07、M11新增19个算子语义分支族，整体继续保持进行中。
'''
 jobs=[
  dict(kind='profile',status='error',description='Retainedinitialconservationaudit:signaturematchingfound264M18shared-contextlookalikesoutside69coreregimes. Incorrectshape-onlypopulationassumption rejected;subsequentprofilejobresolvesroles/counts. No server/nativefailure.',artifacts=[O/e[9]]),
  dict(kind='profile',status='pass',description='69core+5M18lookalikes/25078nativeexecutions conserved;3200graphprojectionchains,640sharedQRpairs,240C2pairs. AllG1BMMgeometry andV2/V3largeoutputbranches included.',artifacts=[O/e[0],O/e[1],O/e[5],N/'analyze_projection_ceiling_v61.py']),
  dict(kind='review',status='pass',description='2exact166quant/outputcallerfiles historicalhashmatches;existingpinnedDSA/Indexer/C2operatorclauses reused. QRFLOATscale versusIndexerFP16scale,BF16intermediates,oneC2cast andG1APIperm contracts bound.',artifacts=[O/e[3],O/e[8],N/'read_projection_contract_v61.py',N/'projection-operator-source166-v61.json']+[Path(r['local_path']) for r in inputs['files']]),
  dict(kind='design-check',status='pass',description='16conditional/rejectionchoices;allprojectionwork/IOmodels,20/9precomposecost andBF16counterexample,nonadditiveQR/C2/interfaces,typedproject-before-gather andcompute-limitedM6400floor checked.',artifacts=[O/e[2],O/e[3],O/e[4],N/'model_projection_fusion_v61.py']),
  dict(kind='review',status='pass',description='Complete2082regime/286950executionledger conserves185explicitreview,5geometry-only,1892pendingreconciliation;19M07/M11semanticfamilies. Oldmodulebranchesarchived;noframeworksemanticclosure/wholemodelcompletion.',artifacts=[O/e[6],O/e[7],N/'consolidate_operator_coverage_v61.py'])]
 persist_refinement(version=61,previous=60,title='Projection fusion, group-one GEMM specialization and complete observed-regime coverage ledger',
  summary=dict(filename='projection-ceiling-summary-v61.json',previous_turn_classification='PROGRESS:v60attention_indexer_operator_models_and_semantic_coverage',
   core_projection_regimes=69,core_projection_executions=24814,other_role_geometry_only_regimes=5,other_role_geometry_only_executions=264,
   total_modeled_regimes=74,total_modeled_executions=25078,operator_decisions=16,M07_M11_semantic_families=19,
   observed_regime_ledger_rows=2082,scoped_reviewed_regimes=185,geometry_only_regimes=5,pending_regime_reconciliation=1892,
   group1_WO_A_executions=4424,shared_QR_graph_pairs=640,C2_graph_pairs=240,WO_precompose_work_weight_ratio=20/9,
   native_test=False,implementation=False,new_performance_result=False,
   remote_job_status='2exactstagefiledocker-cpreadscompleted;allotherworklocalimmutableprofile/math/source. No model/Torch/NPU/newcapture/health/process/kill/implementation/backgroundanalysis.'),
  changes=['69coreprojectionregimesand5same-geometrydifferent-roleM18regimes conservativelyreconciledwith32nativeindices.',
   '16operatorchoices quantifiedwithG1geometry/QRinputreuse/C2jointandtypedcommunication/BF16rounding/nonadditiveceilings.',
   'Complete2082observed-regimeledger and19M07/M11semanticfamilies addedwithoutwholemodule/fullscopecompletion.'],
  candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,jobs=jobs,
  report=report,read_first=[e[6]]+e[:6]+[e[7]],publication_rows={},gate_updates=dict(overall_complete=False),
  scripts=[N/x for x in ['read_projection_contract_v61.py','analyze_projection_ceiling_v61.py','model_projection_fusion_v61.py','consolidate_operator_coverage_v61.py']],
  executor_prefix='zcode-projection-ceiling',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
