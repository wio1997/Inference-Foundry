"""Persist router/vector operator decisions with three explicit unclosed MemSet regimes."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
 assert not (O/'REVIEW-CHECKPOINT-v63.json').exists()
 p=read(O/'router-vector-regime-models-v63.json');g=read(O/'router-vector-graph-chains-v63.json');cost=read(O/'router-vector-fusion-costs-v63.json');v=read(O/'router-vector-fusion-variants-v63.json');cov=read(O/'operator-analysis-coverage-ledger-v63.json');inv=read(O/'M07-M11-M12-operator-branch-inventory-v63.json');inputs=read(O/'router-vector-operator-inputs-v63.json');mem=read(O/'router-MemSet-profile-insufficiency-v63.json')
 assert p['regime_count']==145 and p['captured_execution_count']==75802 and p['completed_scoped_conditional_regimes']==142
 assert len(g['chains'])==3200 and g['root_bound_chains']==2560
 assert len(v['variants'])==23 and inv['total_families']==38 and len(inputs['files'])==7
 assert cov['scoped_conditional_review_regimes']==590 and cov['remaining_regime_reconciliation_count']==1487
 assert mem['resolved_attribute_value_counts']=={'':960} and mem['linked_CANN_name_counts']=={'launch':960}
 e=['router-vector-regime-models-v63.json','router-vector-graph-chains-v63.json','router-vector-numerical-contract-v63.json','router-vector-fusion-costs-v63.json','router-vector-fusion-variants-v63.json','router-MemSet-profile-insufficiency-v63.json','operator-regime-refinement-v63.json','operator-analysis-coverage-ledger-v63.json','M07-M11-M12-operator-branch-inventory-v63.json','router-vector-operator-inputs-v63.json','router-vector-local-validation-history-v63.json']
 write(O/'PROFILE-CODE-QUESTION-GATE-v63.json',dict(status='ROUTER_NORM_QUANT_ROPE_CACHE_LAYOUT_OPERATOR_WORK_FUSION_AND_EXTRA_COSTS',previous_turn_classification='PROGRESS:v62C2_Engram_operator_models_and_coverage',
  questions=['WhatexactnormType2score/visionbias/weightnormalizationandoptionalhashpathdoesrouterfusionneedpreserve?',
   'DoescolumnTopKfusionaddcandidate/mergeworkthatoutweighsthelogitinterface?',
   'CanRMS/DynamicQuant/tailRoPE/cache/layout/paddinginterfacesfoldwithoutremovingBF16round/allconsumers/requiredzeros?',
   'Canall3N/AMemSetdefinitionsbeassignedtrafficordeadwritefromsavedprofile?'],
  allowed_source='7exactoriginal166operatornumeric/tilingfiles;reuse5SHA-boundnarrowcallerfilesandoldv29/v60-v62operatorcontracts. Noindependentframework/lifecycle/HTTP/SSE/loader/SDKcatalogresearch.',
  no_model_Torch_NPU_newcapture_health_process_kill_or_implementation=True,current_loaded_native_binary_identity_not_assumed=True,overall_complete=False))
 updates={}
 for cid in ['F04','F05','F06','F07','F08','F16','F20','F21']:
  updates[cid]=dict(review_v63=dict(regime_models=str(O/e[0]),graph_chains=str(O/e[1]),numerical_contract=str(O/e[2]),fusion_costs=str(O/e[3]),variants=str(O/e[4]),MemSet_pending=str(O/e[5]),full_coverage=str(O/e[7]),
   relation='145previouslypendingregimes75802nativeexecutionsaudited;142/74842nowexplicitconditionaloperatorreview,3MemSet/960staypending. 23fusion/split/reorganization/rejectionchoicespricecandidate/partial/padworkandprecisionbarriers.',conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
   profile_review_state='EXPLICIT_OPERATOR_WORK_IO_FUSION_WITH_NONADDITIVE_CANDIDATE_AND_PADDING_COSTS',
   conclusion_changing_gap='Actualnewkernel/aliases/generations/slots/masks/cachetraffic/FLOAT/Vector/sort/specialcapacityandnativeprecisionretainconditions;3MemSetdestinations/lengthsunknown. Thisisconditionalanalysisdeliverable,notnativeimplementationorwholemoduleclosure.',
   next_profile_or_contract_action='NextreconcileM09remainingkeyprojection/keyquant/cacheandM18draftsharedcontext,thenM15-17/23-24sample/acceptanceandmodelinput/indexoperatorfamilies. Useexistingcontracts/profilefirst;do notrepeatRMSBF16round,TopKExpLnSqrt,outflagfalse orC2/Engramlocalquestions.')
 mods={
  'M12':dict(finding='All60previouslypendingstrictM12regimes34504execaudited;57/33544conditionalreviewand3MemSet/960pending. NativeTopK E384/k6anddraftE128/k3,FP32normType2Exp+1LnSqrt/bias/unbiasedselectednorm,IDplans/MC2pads andcandidate12B/8Bmergecost. 15semanticfamiliesincludepriorMC2andexplicitpendingMemSet.',evidence=e[:9],candidates=['F04','F20','F21'],gap='MemSettarget/lengthunrecorded;actualmasks/bias/nonfinites/ties/bankgeneration/newkerneltypedcapacity/numerics;otherapplicableMoEcommunication/hash/fallbackbranchesnotclosed.'),
  'M07':dict(finding='55strictregimes29485execplus2M07|M09ambiguousquants1280execconditionallyreviewed;actualfull5120/1280/512RMS,ordinaryFLOATscaleDQ,tail64RoPEbroadcast/RINTandselectedcachecost. QRfullycomposedinterface8MH+8M,roundingcounterexample;11semanticfamiliesretainingpriorprojectioncontracts.',evidence=e[:9],candidates=['F05','F06','F07','F08'],gap='ExactDQ/nativeRMSprecisionpolicy/newfusionallreaders/auxrstd/slot/rowmask/ownershipandphysicalVector/cachecapacity;otherapplicablebranchesnotclosed.'),
  'M11':dict(finding='28remainingregimes10533exec:inverseRoPE64tail+transpose,960targetprojectedbankcopies,312RSZero/Concatpairsand118CPzeros/40prefixcopies. Directpadded/outputbankmodelsretainBF16partial/collectivecounts;12semanticfamilies,priorG1/panelcontractsunchanged.',evidence=e[:9],candidates=['F16'],gap='Alias/capturedbank/allreaders/CPactualpadding/collectiveprotocol andtypedresource/numerics remainconditions;otheroutputpathsnotclosed.')}
 remaining=deepcopy(read(O/'module-coverage-gaps-v62.json')['remaining'])
 remaining.append('v63router/vector:145pendingregimes75802execaudited=60M12/34504+55M07/29485+28M11/10533+2M07|M09DQ/1280. 142/74842explicitconditionalreview,3MemSet/960staypendingafter32immutableDBschemaandrawattr/launchcheck(allattrInfoempty,launchonly). 3200graphchains/2560root,TopKcurrent7operatorfilespinned(actual166,only3/7matchotherlocalrepo). Norm2Exp→+1→Ln→Sqrt,vision/textbiasrank,unbiasedselectedsum+eps/Div/scalingevenrenorm0;outfalseGMthirdoutputnotwrittenbut4EUBqueueallocated. E384tile16k6:144candidates12Bwrite/read3456Bperrow>logits3072B;8B+re-bias2880Bbeforeothercost,notfreefusion. RMSroundquantBF16[1,129/128]referencewitness126,127vs127,127;QRcomposednormquant/common/joint8MH+8M M3=30744B. 23choices,38M07/M11/M12semanticfamilies,oldM07/M11retained/extendedandM12historyarchived. WO_Btargetcopy131.072MBat6400 conditional,RSzeroConcat/CP830→834zero/copy/layoutbudgetsnonadditive. Whole2082ledger590/164135scopedreview,5/264geometry,1487/122551pending;35candidates0wholemodules. NextM09keyquant/cache,M18sharedcontextandremainingMLP/logits/sampler/input-indexoperators,notframeworkaudit. Allphysicalnumericceilings/globalambiguitiesremainopen.')
 action=(
  '继续原Goal最新v63 PROGRESS。严格profile优先、代码只为具体算子融合/拆分/重组和理论上限，不研究独立framework/HTTP/SSE/loader/lifecycle/SDKcatalog。先读operator-analysis-coverage-ledger-v63、router-vector-regime-models-v63、router-vector-graph-chains-v63、router-vector-numerical-contract-v63、router-vector-fusion-costs-v63、router-vector-fusion-variants-v63、router-MemSet-profile-insufficiency-v63及M07-M11-M12-operator-branch-inventory-v63。'
  '本轮145先前pendingregimes75802TASK=60M12/34504+55M07/29485+28M11/10533+2M07|M09DQ/1280；fullfunctionalkey对32db exactTASK counts/PMU缺口保守核对。142/74842有scopedconditionalwork/IO/fusionreview；3MemSet/960不提升，32immutableDB全schema+STRING_IDS.attrInfo为空960、CANN_API连接仅launch960，无memory/operand地址大小，不能把Pad完整输出当MemSet流量或断言删workspaceclear。'
  '3200graphblocks链2560root/640unbound精确routerNormCast→FLOATGEMM→TopK，IDCast/Equal/Fill/Select和Qnorm/DQ/QueryRoPE/输出inverseRoPE/发送Transpose绑定。rootmedianus NormCast3.16/router17.38/TopK4.82,ID1.3/2.06/1.48/2.48,Qnorm5.04,DQ3.06,KVnorm13.08,Qrope4.04,outputinverse4.00,sendTranspose12.52，均曝光不e2egain；v55router17.36是含unbound3200总体，非性能变化。'
  '从原166只读docker cp7精确operator文件：TopK entry/withoutgroup/e_k/generalized/tiling以及partialRoPE .h/.cpp，136558B；只有3/7和其他本地perf repo hash一致，必须用v63currentsource，不用未绑定其他副本。Loadedbinary/actualtiling仍未证明，opattr_N不能映射ordinal。TopK configG1/sourcecaller normType2、outfalse；typedtid2eidDT_UNDEFINED而inputIDs存在，hashFlag需两者因此sourceconsistent非hash动态TopK，不能把EngramINT64hash代入。tid2eidDtypedefaultINT32处理optionalundefined。'
  'TopK without-group当前源码真实FP32Exp→+1→Ln→Sqrt，然后选择vision/textbias只用于排序；Gather未加bias normscore，selectedsum+eps→Div→routedScalingFactor，即使renorm0 normType2仍归一。不能凭fallback/libSoftplus猜native或换stable函数，x100Exp溢出Inf vsstable约10是反例，实际logits未捕获。E384/k6主模型，E128/k3草稿，FLOATrouter5120×E actualHF32/fullFP32峰值未绑定；Exp/Ln/Sqrt/Sort/依赖链也不能套376BF16TF。第三fullE FLOATout只分配描述但outfalse源guard不GM写，不算新增省HBM；UB outOutQueue仍4E bytes/program可条件裁剪容量。SourceSelectTopKExpertIdx有expertCountAlign copy vs kqueue extent问题只保留bounds验证条件，未据此声称当前native故障/GM流量。'
  'Router新比较：fullrowepilogue可省8ME logits接口，但列分块localTopK+merge增加候选workspace。E384tile16(24programs)C=144，12B(biased/unbiased/id)写读3456B/row>原8E3072B=1.125x。8B(unbiased/id)可重新精确加同bias，写读2304B+4C=576Bmergebias，共2880B/row=.9375x且有新add/merge/sync；tile32/64候选更少但并行度/容量变，不能无成本。EquivalentTOTALorder local-topK union包含globalTopK证明已列，nativeSort ties/NaN/FP32functionorder仍需，GEMM2MHE不因k6变小。Ksplit/RMSfeature split新增partial/reduction与sumorder，不以3blocks推physical低效。'
  'BF16RMS归一化round不能省：reference input[1,129/128],FP32normalized[.9961013794,1.0038833618]→BF16[.99609375,1.0]；普通round(127*z/max)未round[126,127]、rounded[127,127]。这是CPU算术反例非native精度test。RMS→DQ保full1280amax和BF16round、nativezero/nonfinite/round/scale。不是Indexer128headFP16scale。QR来源层两quant+两GEMM全组合budget8MH+8M，M3=30744B：NormQuant4MH、重复quant3MH+4M、jointinputMH+4M，接口只计一次；该familyparentF05，不误归F32(SPgather前移投影候选)。'
  'ID f(x)=INT64x,仅-1→0保其他negative vision IDs。Graphcast12M+equal9M+select17M+fill8=38M+8prep，TopK还读8M；fusedraw32读4M，budget42M+8/layer。M3×40基线5840B，CSE一次producer+40consumer1082B/save4758，inline TopK480B/save5360；这是logicalbytes不是HBM/time，主要可能launch/reduction开销。perinputgeneration/rowplan/原IDs读者/layerbias hash/固定bank失效和collective不删，跨迭代sameaddress不等于samevalues。'
  'RoPE7currentkernel证明tail64 stridedGM，不读写整512，FLOATcos/sin按token广播全64heads，partnerGather/2Mul/paritysign/Add后BF16 CAST_RINT，inverse用negateSin flag/否则python -sin capability分支；不凭oplabel发明Neg GM kernel。QueryMheads64 payloadreadwrite4M*64*64+cos/sin8M64；融合producer仍保BF16beforeRope。KVnorm512先完整reduce，compressed另有IndexKey preRoPE分叉；Scatter只valid updates/INT32indices和cache写，wholecache不是流量。'
  'MC2观察800→816 hiddenBF16/logitsFLOAT/maskINT8/IDsINT64 Pad和BOOLcasts，producer可直接写Pbank有效prefix+suffixzeros，不能把全部M816 gate640calls都当M800 pad320calls；M800norm/router→P816归属不bytime硬配。skip显式falsepad TopK须safe默认IDs/weights和所有dispatch readers协议，Pcollectivecounts仍保。MemSet3regime不判dead。'
  '输出28regime含960targetWO_B projected_output→bank copies(M6400/6528/6672)，directbank写可条件省4TH，6400=131072000B；保BF16localpartial在RS前round，所有alias/capturedbank/ready读者。312RSsuffixzero+Concat M10→P16/M20→P24，directPproducer省4PHConcat写读但2(P-M)Hzeros仍required、Pcollective不变。118CPfullzeros和40M830prefixcopy，source830→834可避免被覆盖prefix清零54394880B；jointemit/pad/inverseRoPE/sendlayout按接口整体预算，不与v60/v61重复领收益。dummy/localempty仍可能要globaloutput。'
  '23条件/排除方案已交付，不需先实现/nativebench才算局部分析；38semanticfamilies=11M07(保留10并新增inputRMS)+12M11(保留9+bank/RS/CP)+15M12含明确MemSetpending/既有MC2。历史archived，M08/M14与M09/M10、M01/M21/M22/M25 branches保护不变。35模型+7legacyframework候选0wholemodule，整体不能完成。全2082/83types/286950ledger=590/164135scopedreview+5/264M18geometry+1487/122551pending，pending需对齐已有family证据不等于历史全没做；不是Goal进度百分比。3136归属歧义、849missingPMU、16unbound保留，实际所有physicalnumericceilings不证明。'
  '下一步优先M09剩余keyprojection/keyquant/indexcache、M18draftsharedcontextconsumer专属、M15-17/23-24 logits/sampling/acceptance与模型输入/索引算子；用已有v55-v63合同做regime明确匹配/语义工作量/跨层跨卡跨迭代budget。M21还有76943exec pending，应按模型输入/索引/采样算子依赖处理而不是独立框架catalog。M23有11528、M18有9672、M09有6040 exec pending，先查profile角色。无Torch/model/NPU/newcapture/health/process/kill/实现；socket ssh166-v56-xinyufeng.sock本轮有效，授权已给无需再问，优先本地缓存。Goalactive，上一轮v62 PROGRESS，本轮v63也PROGRESS。')
 report='''本轮核对145个观察组合、75,802次执行，补齐router、RMS、动态量化、RoPE、缓存写入和输出布局的算子模型。142组、74,842次执行已有条件融合与工作/逻辑IO证据；3个MemSet组合、960次执行继续保留待定。没有运行模型、NPU或新性能测试。

| root绑定图内阶段 | 中位设备耗时 |
|---|---:|
| FFN NormCast / FLOAT Router / TopK | 3.160 / 17.380 / 4.820μs |
| Q RMS / DynamicQuant / KV RMS | 5.040 / 3.060 / 13.080μs |
| Query RoPE / 输出逆RoPE / TP发送Transpose | 4.040 / 4.000 / 12.520μs |

这些统计来自2,560条root绑定链，另640条阶段未绑定链保留。每个算子的时间、字节和依赖分别计账，没有将时间相加宣称端到端收益。

已从原166只读取得7个对应算子文件。当前TopK的normType=2实现按FP32 Exp→加1→Ln→Sqrt计算，然后用vision/text bias后的分数排序。输出权重则读取未加bias的分数，对选中项求和、加eps、除法和缩放；renorm=0没有取消此路径的选中权重归一化。替换成另一种稳定softplus实现可能改变溢出、相等分数和输出权重，必须单独约定数值行为。

TopK内部已经融合了非线性、bias、排序和权重计算。它在片上完成完整专家排序；out_flag=False的第三个完整分数输出在审查路径中不写GM，不能再算一次“省去完整输出”的收益。该输出队列仍分配4E字节片上空间，可以条件研究容量裁剪，但这不是HBM流量收益。

Router GEMM融合TopK还需要比较新增的候选合并成本。E=384、k=6时，原logits写读接口为每行3,072B：

| 专家列tile宽度 | 分块数 | 候选数 | 12B候选写读 | 8B候选写读＋重新读取bias |
|---|---:|---:|---:|---:|
| 16 | 24 | 144 | 3,456B | 2,880B |
| 32 | 12 | 72 | 1,728B | 1,440B |
| 64 | 6 | 36 | 864B | 720B |

12B候选保存排序分数、未加bias的权重分数和expert ID；8B候选保存后两者，合并时重新执行相同FP32 bias加法。表内尚有计算、合并、同步和并行度取舍。细分为24块时，12B方案的候选流量比原logits接口多12.5%。局部TopK再合并在相同总序下有集合包含证明，但还要匹配原排序的相等值、NaN/Inf和ID顺序。GEMM计算量仍为2MHE，不会因为只选6个专家就自动缩至6列。

RMS→量化可以省存储，不能省数值舍入。算术反例中，BF16输入[1,129/128]的FP32归一化约为[0.99610138,1.00388336]；舍入为BF16后是[0.99609375,1]。按round(127z/max)量化，前者为[126,127]，后者为[127,127]。这是规则反例，不是设备精度测试。QR源层若同时组合RMS→量化、重复量化复用和两个GEMM的输入复用，逻辑接口预算合计8MH+8M，M=3、H=1280时为30,744B，不能重复累计同一接口。

RoPE当前实现只读写64维尾部，cos/sin在各head间广播，保留FP32乘法、符号和加法顺序，以及BF16 CAST_RINT。模型因此分别计算完整RMS归约、尾部旋转和选定缓存行写入，没有把512维或整个cache分配体积算成RoPE、Scatter流量。

输出侧补齐960次目标投影结果到输出bank的拷贝。WO_B直接写目标bank，在所有读者、alias和生命周期允许时可省4TH逻辑字节；T=6400时为131.072MB。312组ReduceScatter前的补零/Concat可研究由生产者直接写padded bank，但补零语义和collective计数必须保留。CP的830→834行补齐、逆RoPE和发送布局应作为一组接口组合核算，避免重复计算同一拷贝收益。

3个MemSet组合未提升为“已分析完成”：32份原始DB没有目标地址、长度或操作参数；960项attrInfo均为空，对应CANN记录只有launch。算子名称不足以判断它清除的是输出数据、工作区还是协议状态，报告未猜测流量或宣称可删除。

共形成23个条件方案、拆分取舍或排除结论。统一观察台账目前为590组、164,135次执行有显式条件分析，5组、264次仅几何建模，1,487组、122,551次仍待逐项对齐已有证据。FLOAT router、Vector、排序与特殊函数的实际容量尚未绑定，不能套BF16 Cube峰值。观察样本覆盖、条件模型和报告成稿均不代表全部模型分支或整体目标完成。
'''
 jobs=[
  dict(kind='profile',status='error',description='Retainresolvedlocalschemaauditpathderivationfailure:firstanalysisdirectorychoseninsteadoffinalnestedanalysiscomponent. No target/serverfailure;corrected32DBimmutableauditpassed.',artifacts=[O/e[10]]),
  dict(kind='profile',status='pass',description='145regimes75802exactfunctionalnativeTASKs/PMUcountsconserved;3200graphchains2560root. 3MemSet960N/Arecordsrawattributesemptyandlaunchonlyafter32DBschemaaudit.',artifacts=[O/e[0],O/e[1],O/e[5],N/'analyze_router_vector_v63.py']),
  dict(kind='review',status='pass',description='7exact166TopKandRoPEsourcefilespinned,only3matchotherlocalperfcopy;5oldcallershashbound. Norm2primitive/bias/selectednorm/outfalseandtail-onlyRoPE/RINTcontractsnotinferredfrompubliclatestdocs.',artifacts=[O/e[2],O/e[9],N/'router-vector-source166-v63.json',N/'read_router_vector_contract_v63.py']+[Path(r['local_path']) for r in inputs['files']]),
  dict(kind='design-check',status='pass',description='23operatorchoices,10TopKcandidatecostmodels,6QRjointbudgets,ID/pad/outputmodelsandBF16/nonlinear-biaswitnesses. Candidate/partial/rebias/zeros/sharedresourcecostsnotomitted;MemSetunpriced.',artifacts=[O/e[3],O/e[4],N/'model_router_vector_v63.py']),
  dict(kind='review',status='pass',description='2082regimes286950execconserved:590review164135,5geometry264,1487pending122551including3MemSet960. 38semanticfamiliesM07/M11/M12withpriorbranchesarchivedorretained;globalambiguitiesunchanged.',artifacts=[O/e[6],O/e[7],O/e[8],N/'consolidate_router_vector_v63.py'])]
 persist_refinement(version=63,previous=62,title='Router and vector operator work, candidate merge costs, precision and padding constraints',
  summary=dict(filename='router-vector-ceiling-summary-v63.json',previous_turn_classification='PROGRESS:v62C2_Engram_operator_models_and_coverage',audited_regimes=145,audited_executions=75802,newly_reconciled_regimes=142,newly_reconciled_executions=74842,MemSet_explicit_pending_regimes=3,MemSet_explicit_pending_executions=960,graph_chains=3200,root_graph_chains=2560,operator_decisions=23,M07_M11_M12_semantic_families=38,
   observed_regimes=2082,scoped_reviewed_regimes=590,scoped_reviewed_executions=164135,geometry_only_regimes=5,geometry_only_executions=264,pending_regime_reconciliation=1487,pending_execution_reconciliation=122551,native_test=False,implementation=False,new_performance_result=False,all_actual_physical_numeric_ceilings_proved=False,
   remote_job_status='7exactoperatorfiledocker-cpreadscomplete;remaininglocalimmutableprofile/raw32DBschema/attribute/math/source. No model/Torch/NPU/newcapture/health/process/kill/implementation/backgroundanalysis.'),
  changes=['145regimes75802exactnativeexecutionsaudited;142/74842newconditionaloperatorreviews,3MemSet/960explicitlyremainpending.',
   '23operatorchoicespriceTopKcandidate/rebias/partials,QRrounding/nonadditivebudgets,tailRoPE/cacheanddirectpadding/outputbankinterfaces.',
   '590scopedreviewregimes164135execand38M07/M11/M12semanticfamilies;1487regimes122551execstillpending,allwholemodulesopen.'],
  candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,jobs=jobs,report=report,read_first=[e[7]]+e[:6]+[e[8]],publication_rows={},gate_updates=dict(overall_complete=False),
  scripts=[N/x for x in ['read_router_vector_contract_v63.py','analyze_router_vector_v63.py','model_router_vector_v63.py','consolidate_router_vector_v63.py']],executor_prefix='zcode-router-vector',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
