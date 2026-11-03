"""Persist operator-focused C2/Engram analysis without declaring whole goal complete."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
 assert not (O/'REVIEW-CHECKPOINT-v62.json').exists()
 p=read(O/'state-engram-regime-models-v62.json');g=read(O/'state-engram-graph-operator-chains-v62.json');c=read(O/'C2-state-and-pool-costs-v62.json');a=read(O/'Engram-hash-UVA-and-gate-costs-v62.json');v=read(O/'state-engram-fusion-variants-v62.json');inv=read(O/'M08-M14-operator-branch-inventory-v62.json');cov=read(O/'operator-analysis-coverage-ledger-v62.json');inputs=read(O/'state-engram-operator-inputs-v62.json')
 assert p['regime_count']==276 and p['captured_execution_count']==13704
 assert len(g['C2_chains'])==240 and len(g['Engram_bodies'])==160 and g['C2_root_bound_chains']==192
 assert len(v['variants'])==25 and inv['semantic_families']==31
 assert cov['observed_regimes']==2082 and cov['scoped_conditional_review_regimes']==448 and cov['remaining_regime_reconciliation_count']==1629
 assert c['examples'][2]['ring_update_read_plus_write_bytes']==524288
 assert a['parameter_prep']['total_tasks']==624 and a['parameter_prep']['one_body_logical_value_IO_bytes']==491520
 assert a['hash_lookup_partition_models'][0]['partition_partial_WKV_FP32_plane_bytes']==2457600
 e=['state-engram-regime-models-v62.json','state-engram-graph-operator-chains-v62.json','C2-state-and-pool-costs-v62.json','Engram-hash-UVA-and-gate-costs-v62.json','state-engram-operator-contract-v62.json','state-engram-fusion-variants-v62.json','operator-regime-refinement-v62.json','operator-analysis-coverage-ledger-v62.json','M08-M14-operator-branch-inventory-v62.json','state-engram-operator-inputs-v62.json','state-engram-local-validation-history-v62.json']
 write(O/'PROFILE-CODE-QUESTION-GATE-v62.json',dict(status='C2_POOL_STATE_NORM_AND_ENGRAM_HASH_LOOKUP_GATE_OPERATOR_MODELS',previous_turn_classification='PROGRESS:v61projection_operator_models_and_observed_regime_coverage',
  questions=['WhichC2pair/history/updatedrowcountsdetermineactualoperatorwork,notstateallocationextent?',
   'CanC2pool/norm/ring/forkfuseorsplitwhilepreservingBF16round,historyread-before-writeandIndexKeyconsumer?',
   'WhichEngramINT64hash/UVAlookup/IDlayout/staticparameter/gatematrixinterfacesareconditionallyremovable?',
   'Whatworkandtypedmemory/computeconstraintssetreferenceceilings,andwhichunknownscannotbepricedwithBF16peak?'],
  allowed_source='3exactpreviouslyhashboundoperatorfiles;reusepinnednarrowoperatorcallersandpreviouscontracts. No independentframework/lifecycle/loader/HTTP/SSE/SDKcatalogaudit.',
  no_new_Torch_NPU_model_capture_health_process_kill_implementation=True,loaded_native_binary_identity_not_assumed=True,overall_complete=False))
 updates={}
 for cid in ['F17','F18','F19','F24','F25','F33','F34']:
  updates[cid]=dict(review_v62=dict(regime_models=str(O/e[0]),graph_chains=str(O/e[1]),C2_costs=str(O/e[2]),Engram_costs=str(O/e[3]),operator_contracts=str(O/e[4]),variants=str(O/e[5]),coverage=str(O/e[7]),
   relation='276M08member/M14strictregimes13704exactnativeexecutions,13projectionoverlap808;263newregimes12896exec. 25conditional/rejectiondecisionswithsourcework/logicalIOandtypedcapacitygates,31semanticfamilies. No wholemodelcompletion.',conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
   profile_review_state='OPERATOR_SEMANTIC_WORK_IO_FUSION_SPLIT_PARTITION_AND_REFERENCE_CONSTRAINTS_REVIEWED',
   conclusion_changing_gap='Conditionaloperatoranalysisdelivered;actuals/n/page/ownership/hashIDs/activeUVAformat/parameterandbankgeneration/cachetiertraffic/FLOAT/Vector/INT64/UVAtypedcapacities remainexplicitparameters. Unobservedapplicablebranchesnotclosed. Nativeimplementationabsencealoneisnotincompleteconditionalanalysis.',
   next_profile_or_contract_action='ContinuefullobservedledgerM12routerandM07norm/quant/RoPE/cache/commregimesusingpriorcontracts;remainingdense/sampling/indexingoperatorroles. Do not repeatansweredC2BF16round/Utail32/allocator orEngramfullwrite/once-reqsearch/currentbackendmergefailurecontracts.')
 mods={
  'M08':dict(finding='56memberregimes4024executions,including8M07|M08ambiguous1600retained. C1/C2projectionmodelsreused;240C2graphchainsplus72directpool/update. C2E/B/Uwork/traffic,normBF16andtokenalignedzero/rowmap,pool-updatefork/onekernel,pageint64andpreRoPEIndexerforkreviewed. 12semanticfamilies.',evidence=e[:9],candidates=['F17','F18','F19','F33','F34'],gap='Actualsegmentmetadata/validpages/rollback/allreaders/newkernelnumerics/typedcacheandVectorcapacity;allunobservedapplicablecompressorbranchesnotclosed.'),
  'M14':dict(finding='220strictregimes9680executions,160graph38TASKbodiesand48directWKV/gatebodies. INT64hash8Trolling/48Tmod,UVA3formats/scale8group/fullwrite/stridedID,bank/TProwexchange,combinedWKV/referenceworkandrotatedgate/staticparams/2passIOmodeled. 19semanticfamilies.',evidence=e[:9],candidates=['F24','F25'],gap='ActualIDs/ownership/UVAformat/hitratio/B_UVA/INT64andFLOATcapacity/weight-bankgenerations/precision;unrotated/TP1/CPU/device/slotless/dummyotherapplicablebranchesremainconditional.'),
  'M09':dict(finding='C1/C2normalizedBF16latentfirstfeedsIndexKeybeforeinplace64tailRoPElongcache. F19fusionmustretainpreRoPEconsumerforkandallslot/validrowguards;v60attentionIndexersemanticsunchanged.',evidence=[e[2],e[4],e[5]],candidates=['F19'],gap='RemainingIndexeroperatorregimes/physicalcapacitynotclosed;nochangev60semanticbranches.')}
 remaining=deepcopy(read(O/'module-coverage-gaps-v61.json')['remaining'])
 remaining.append('v62operator-onlyC2/Engram:276regimes13704exec(56M08member4024incl1600ambiguous+220M14strict9680);13projectionoverlap808so263new/12896. Exactfunctionalkey/nativeTASKjoinnotshapecollision. 240C2chains/192rootand160Engram38TASKbodies/128root. Epaircount,Boddstartresidual,U=summin(n,32)makeC2IOparameterized;T6400two3200validexampleupdate524288Bnot52.4MB. ExistingnonTOKENfusednormalreadyBF16roundbutcurrentdummyweight/eps0/tokenmapnotdropin. Engram8TINT64rolling/48Tmods,onealreadyreqsearch/backendmergeabort;UVAINT8scale256refs→8butunique32Bsame;hostB_UVAunbound. 176lookupzeros/176IDcopiesconditional,DPsharealreadybaseline. WKV314.5728MBweights andblock32FLOATrotation;624qkprep+208rotationcasts versioncache,gate2pass/capacity/round/NaNguards. PartialheadWKVFunchanged,2.4576MBpartialatglobal24vs36.864KBheadrowinput=200/3. 25choices31semanticfamilies. Wholeobservedledger448scopedreview89293exec,5geometry264,1629pending197393;notwholegoalpercentage. 3136ambiguities/849missingPMU/16unboundunchanged. ContinueM12router/M07normquantropecachecommremainingoperators,notframeworkaudit;allphysicalnumericceilings/wholemodulesopen.')
 action=(
  '继续原Goal，最新v62 PROGRESS。用户强调代码仅为具体算子融合/拆分/重组与理论上限服务；profile优先，无法定案才定点补代码。禁止独立framework/HTTP/SSE/loader/SDK/lifecycle研究。'
  '先读operator-analysis-coverage-ledger-v62、state-engram-regime-models-v62、state-engram-graph-operator-chains-v62、C2-state-and-pool-costs-v62、Engram-hash-UVA-and-gate-costs-v62、state-engram-operator-contract-v62、state-engram-fusion-variants-v62、M08-M14-operator-branch-inventory-v62。276regimes13704exec=56M08member4024+220M14strict9680，含8M07|M08歧义1600不改归属；13projectionoverlap808，263新12896。按functionalmodule+op+role+origin+context+7ComputeInfo字段join原v20assignment到32nativeTASK，不shape-only。'
  '240C2图链(192root/48unbound)在blocks2/8/14，另72directpool/update。rootmedianus zero3.4,pool5.161,update5.38,norm9.821；时间不可相加报gain。E=sumvalid(floor((s+n)/2)-floor(s/2))，B=valid oddstart首residual数≤requests，U=sumvalid min(n,32)。pool读16H(E+B),oldring16HB,update读写16HU；T6400两3200偶起点example更新524288B，不是2.268GBstate或52.4MB全T。residualratio2可条件减16HB重读+2HBexp；真实s/n/page/skip未捕获不得把hostschedule当native行证明。'
  'Pool旧nonTOKEN_ALIGNED分支已pooled.toBF16.toFLOAT再norm，不得误说未round；当前TOKEN_ALIGNEDtrue用eps0/placeholderoutweight/完成tokenrow，不能直接翻flag。融合需realweight/eps+独立normflag+保rowmap/auxrstd；未写行zero经外部RMS的signedzero/Inf/NaN和所有消费者保留。Poolnorm与ringupdate可在oldringreads后fork但不是免费5.38us，onekernel需pageownership/crossprogramread-before-write。Even-onlystate/rollback条件和v25SWAanchor不缩allocator结论保留，int64地址。Normlatent先IndexKey再inplace64tailRoPE/cache，F19保preRoPEfork，Scatter按validrowpayload不全cacheextent。'
  '160Engram图body每个38TASK(128root/32unbound)，另48directWKV/gatebody；functionalM14总9680不同v30ancestry9712不得合并。rootmedianWKV276.3255us,rotation7.13us,3paramprep和13.24us,其余gateTASKsum148.0925us；不是e2egain。WKV6144→25600已4key+共享V，BF16weight314572800B，M3完整cold-reference196.72704us/376TF compute2.5099us，physical实际未证明；rotationblock32,M=640T,F=1310720T FLOAT模式/capacity仍unbound。'
  'Engram q/kBF16每body两Cast+Mul=624TASK/208body，prepIO491520B/body；rotationBF16→FLOAT cast208,6144B/body可按generationcache。Gate用rotatedhidden做2norm/dot，残差原hidden，V共享4HC，BF16WKVround与fullHbarrier/signbit(-0)/NaN/0*Inf保留。理想preparedparam gateIO133120T+86016+T，M3=485379B；2pass共享V M800139376416B/重复V163952416B，onchip容量/额外partial显式。WKVgate界面budget20TH，不承诺全省。Hiddenprep独立WKV可提前但可能争Cube。HeadshardpartialWKV每卡F不变，weights/8；global24 FP32partial2457600B对headrow输入36864B=200/3，不免费8xcompute；比较F24最佳rowexchange和新sum/round。'
  'Hash两layerprogram每shift4：8T INT64mul/XOR，48T INT64%prime，INT32192T；reqsearch已一次/token不是两次，源码明示并回hash的bishengir LLVM releasedbuffer编译失败，不能dropin。Sharedhistory或exactconstantprime整数reciprocal/mulhi需TLsigned/64overflow/generation，不FP32/低32替代。Cachewrite先于hash，物理fallbackslotalias未证明不能盲移；inbatch>knownwindow>fallback和DEAD/negativeprefix。'
  'UVA三格式每headrowBF16512B/INT8code256+FP32scale32=288/MXFP8code256+8=264；actualconstexpr未证明，hostB_UVA未知不能HBM1.6TB定价。INT8scale256logicalrefs→8 groupref但unique32B不变，MXFP8已broadcast。不owned可maskrow0load但全写0；dead token不等于unowned，actualbenefit可0。ID去重cache按Aowned/Uowned、table/head/shard/code/scalegeneration，hit未知。176lookup-localprezeros由3head×256全写source合同条件可删，dummyglobalbank/masktail不可一起删；176perlayerID Slice contiguous可在UVAids_stride_t支持stride48/inner1下直接view。'
  'TPAllGather→Transpose/bankcopy/SPclone可head→rowAlltoAll，perrank内容12288T→1536T=8x而非wire/latencygain，所有globalbank读者/pad/chunk/clone生命周期仍需；DPgather已两层共享。Gate末nativehiddenTensorMove可directstore但graph38TASKbody无此copy，不claim图收益。25conditional/rejectionchoices与31semanticfamilies(12M08/19M14)已分析交付；无实现/nativebench不能仅以此说局部未分析。'
  '完整2082/83types/286950 ledger现448/89293显式scopedconditional工作/IO/fusionreview，5/264M18geometry-only，1629/197393 pending逐项对齐已有family evidence，不代表原候选全没做，更不是全Goal百分比。3136归属歧义/849缺PMU/16unbound保留，35模型+7legacyframework候选、0wholemodulesclosed。M08/M14新语义分支旧archive；M01/M21/M22/M25及M07/M09/M10/M11旧branches不改。'
  '下一步优先从pending明确算子推进M12 router/RMSNormCast/MatMul/TopK/inputFillEqualSelect，及M07剩余RMS/DynamicQuant/RoPE/Scatter/collective接口、M11通信/布局；用已有v55-v62合同作regime逐项对齐/补工作量，别重查已答source。M21剩余算子作为模型输入/索引/采样工作，不独立框架研究。FLOAT/Vector/INT64mod/UVA实际capacity/缓存traffic仍参数，不以PMU比值造MFU。'
  '本轮仅三exactdocker cp源文件SHA全匹配历史，其余本地cache/profile/math。无Torch/model/NPU/newcapture/health/process/kill/实现/background。原166socket ssh166-v56-xinyufeng.sock本轮可用，授权已给不重复问，优先本地5GB缓存。Goalactive，不因报告成稿完成。')
 report='''本轮继续围绕算子融合、拆分、重组及理论约束分析。覆盖276个观察组合、13,704次执行，其中13组、808次投影执行已有v61分析；本轮新增263组、12,896次执行的显式证据。未运行模型、NPU或新性能测试。

| C2图内阶段 | root绑定记录数 | 中位设备耗时 |
|---|---:|---:|
| 输出清零 | 192 | 3.400μs |
| 池化 | 192 | 5.161μs |
| 状态更新 | 192 | 5.380μs |
| 外部RMSNorm | 192 | 9.821μs |

C2已绑定240条图内链，另外72次直接调用覆盖6400、6528、6668行。池化只产生完成token对的行；每个奇数起点的请求最多有一个历史残留组。状态更新仅维护每段最后min(段长,32)行。以6400行、两段各3200行且页有效的条件例子计算，更新读写为524,288B，而非按全部6400行计算的52.4MB；2.268GB状态数组体积也不是单次流量。实际请求段、有效页和回滚状态仍须作为模型参数。

C2主要候选是池化→BF16舍入→RMS融合、残留两行组的单次score读取，以及在历史读取完成后拆开norm与状态维护。现有非token-aligned分支已经保留BF16舍入，不能误作新发现；当前调用的占位weight、eps=0和行映射使它不能直接启用。融合还要保留未写行的归一化语义、rstd读者，以及归一化latent先供Indexer Key读取、再做64维尾部RoPE和cache写入的分叉。状态与池化合并需要证明物理页读写顺序，不能用一次kernel launch替代跨程序同步。

Engram已绑定160个图内算子段，每段38项；另有48个直接WKV/gate段。root绑定图中，WKV中位276.326μs，32维分块旋转中位7.130μs，三项q/k参数准备的设备时间和中位13.240μs。这些是观察到的耗时与工作暴露，不是可删除的端到端时间。

WKV已经把4路Key和1路共享Value合在6144→25600投影内。BF16权重314.5728MB；M=3时，常规算法计算量943.7184M ops。按此前注明的每die 376 BF16 TFLOPS与1.6TB/s参考，计算项约2.510μs，完整单次冷读写项约196.727μs。该参考依赖声明的流量和容量假设，不能作为实际物理上限或用来反推MFU。FLOAT旋转、Vector归约和特殊函数另设容量参数。[硬件参考](https://arxiv.org/html/2506.12708v1#S3.SS3.SSS1)

可优先研究按参数版本缓存q*k的FP32结果和FP32旋转块：208个段共有624项q/k转换与乘法，以及208项旋转块转换。它与动态gate融合是不同候选。动态gate需保留WKV的BF16边界、两次完整5120维RMS、FP32加权点积、signbit、NaN/Inf和共享Value的4路消费。已给出一次读取的理想模型与两遍模型；M=800时，两遍并共享Value的逻辑读写为139.376416MB，实际缓存流量与片上容量不能由此直接推定。

哈希按两层、4个shift计算，为每token 8次INT64滚动乘法/XOR和48次INT64余数。请求查找已只做一次；源码明确记录直接并入哈希的当前编译器失败。可研究两层共享历史解析和精确整数余数特化，不能用FP32倒数或截断低32位代替。

查表分开建模BF16、INT8和MXFP8；INT8每行256B code加32B唯一scale，主机UVA带宽尚无实测约束，不能套HBM带宽。INT8 scale可从256个逻辑引用改为8组广播，但唯一32B负载不变；MXFP8已做分组广播。176次查表局部预清零在完整写入合同下可消除，176次分层ID连续化拷贝可在UVA的行stride合同下改用视图。dummy全局bank和mask尾部清零有独立语义。

跨卡重组比较了head到row交换与head分片WKV。前者每卡内容模型可从12288T缩至1536T，但保留全局bank读者、padding和clone生命周期；后者每卡FLOPs不变，权重减少8倍，却增加FP32部分结果归约。global T=24时，部分结果平面2.4576MB，是head到row输入36.864KB的200/3倍。二者不能叠加成免费8倍收益。

本轮共25个条件方案或排除结论，新增31个算子语义分支族。全部2,082个观察组合仍在统一台账中：448组、89,293次执行已显式对齐条件融合与工作/逻辑IO分析；5组、264次仅有几何分析；1,629组、197,393次仍需与已有family证据逐项对齐。该计数不是总Goal百分比，观察样本也不覆盖全部模型分支。3,136项归属歧义、849项缺PMU、16个阶段未绑定图均保留；整体尚未完成。
'''
 jobs=[
  dict(kind='profile',status='error',description='Retain3resolvedlocalattempts:missing2exactcustomkerneldispatchnamesandwronghelperSimport. No model/serverfailure;finalexplicitpopulationmodelsandconsolidationpass.',artifacts=[O/e[10]]),
  dict(kind='profile',status='pass',description='Exactfunctionalassignment/nativeTASKjoin276regimes13704exec;240C2chainsand160Engram38TASKbodies,192/128rootbound. No shape-onlycollisionorancestrypopulationmerge.',artifacts=[O/e[0],O/e[1],N/'analyze_state_engram_v62.py']),
  dict(kind='review',status='pass',description='3exactoriginal166kernelfilesmatchhistoricalSHA;5pinnednarrowcallersmatcholdcontracts. C2normBF16/rowmap/update/fork andEngramhashUVA/gateboundariesreviewed,notframeworkaudit.',artifacts=[O/e[4],O/e[9],N/'read_state_engram_sources_v62.py',N/'state-engram-source166-v62.json']+[Path(r['local_path']) for r in inputs['files']]),
  dict(kind='design-check',status='pass',description='25fusion/split/partition/rejectionchoices;C2E/B/UandEngram3format/INT64/WKV/rotation/2passgate/referencecostmodels. Nonadditivebudgets/unchangedpartialWKVFLOPs/backendfailureguardspreserved.',artifacts=[O/e[2],O/e[3],O/e[5],N/'model_state_engram_v62.py']),
  dict(kind='review',status='pass',description='Conserve2082regimes286950exec:448explicitreview89293exec,5geometry264,1629pending197393. 263newregimes12896exec/13overlap808;31semanticfamiliesandallglobaluncertaintypreserved.',artifacts=[O/e[6],O/e[7],O/e[8],N/'consolidate_state_engram_v62.py'])]
 persist_refinement(version=62,previous=61,title='C2 state/pool and Engram operator fusion, work, typed memory and reference constraints',
  summary=dict(filename='state-engram-ceiling-summary-v62.json',previous_turn_classification='PROGRESS:v61projection_operator_models_and_observed_regime_coverage',
   reviewed_regimes=276,reviewed_executions=13704,newly_reconciled_regimes=263,newly_reconciled_executions=12896,overlap_projection_regimes=13,overlap_projection_executions=808,
   C2_graph_chains=240,C2_root_chains=192,Engram_graph_bodies=160,Engram_root_bodies=128,operator_decisions=25,M08_M14_semantic_families=31,
   observed_regimes=2082,scoped_reviewed_regimes=448,scoped_reviewed_executions=89293,geometry_only_regimes=5,geometry_only_executions=264,pending_regime_reconciliation=1629,pending_execution_reconciliation=197393,
   native_test=False,implementation=False,new_performance_result=False,all_actual_physical_numeric_ceilings_proved=False,
   remote_job_status='3exactstageoperatorfiledocker-cpreadscomplete;allremainingworklocalimmutableprofile/math/pinnedsource. No model/Torch/NPU/newcapture/health/process/kill/implementation/backgroundanalysis.'),
  changes=['276regimes13704exactnativeexecutionswithC2andEngramoperatorwork/IO/sourceconditionalcontracts;263newregimes12896execafter13projectionoverlap.',
   '25operatorchoicesquantifyC2stateE/B/U/normfork andEngramINT64/UVA/parameter/gate/typedpartitionconstraintswithoutphysicalpeakfiction.',
   '31M08/M14semanticfamilies;448observedregimes89293execwithscopedreview,1629stillpendingexplicitreconciliation. WholeGoalandallmodulesopen.'],
  candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,jobs=jobs,report=report,read_first=[e[7]]+e[:6]+[e[8]],publication_rows={},gate_updates=dict(overall_complete=False),
  scripts=[N/x for x in ['read_state_engram_sources_v62.py','analyze_state_engram_v62.py','model_state_engram_v62.py','consolidate_state_engram_v62.py']],executor_prefix='zcode-state-engram',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
