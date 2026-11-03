"""Persist operator-focused embedding/init/ID analysis with six explicit pending regimes."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not (O/'REVIEW-CHECKPOINT-v67.json').exists()
    p=read(O/'embedding-initialization-regime-models-v67.json');g=read(O/'embedding-initialization-native-chains-v67.json')
    costs=read(O/'embedding-initialization-fusion-costs-v67.json');v=read(O/'embedding-initialization-fusion-variants-v67.json')
    contract=read(O/'embedding-initialization-operator-contract-v67.json')
    cov=read(O/'operator-analysis-coverage-ledger-v67.json');inv=read(O/'M04-M05-M06-M19-M10-operator-branch-inventory-v67.json')
    assert p['regime_count']==236 and p['captured_execution_count']==7616
    assert (p['scoped_reviewed_regimes'],p['scoped_reviewed_executions'])==(230,7224)
    assert (p['explicit_pending_regimes'],p['explicit_pending_executions'])==(6,392)
    assert [len(g[k])for k in ['embedding_groups','target_HC_initializations','draft_initializations','sampled_ID_groups','draft_ID_groups','position_groups']]==[272,104,104,72,72,88]
    assert len(v['variants'])==22 and inv['total_current_family_count']==61 and inv['new_or_reconciled_family_count']==52
    assert (cov['scoped_conditional_review_regimes'],cov['scoped_conditional_review_executions'])==(1245,208655)
    assert (cov['remaining_regime_reconciliation_count'],cov['remaining_execution_reconciliation_count'])==(837,78295)
    assert (cov['explicit_profile_insufficient_regimes'],cov['explicit_profile_insufficient_executions'])==(7,1272)
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v66.json')['modules']}
    assert len(old)==26
    assert all(old[mid].get('branch_schema')is None for mid in ['M04','M05','M06','M19'])
    assert old['M10']['branch_schema']=='SEMANTIC_BRANCHES_V1' and len(old['M10']['branches'])==9
    for a in read(O/'checkpoint-manifest-v66.json')['artifacts']:
        q=Path(a['path']);b=q.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],q
    changing=set(inv['modules']);assert changing=={'M04','M05','M06','M19','M10'}
    protected={mid:m for mid,m in old.items()if mid not in changing};assert len(protected)==len(old)-len(changing)==21
    write(O/'initialization-protected-module-state-v67.json',dict(modules=protected,M10_retained_branches=old['M10']['branches'],
        scope='Other21modulematrixrecordsremainidentical;M10old9semanticrowsretained+2new. FourlegacyinventoriesarchivedinfullincludingM06v64historycorrection.',overall_complete=False))
    e=['embedding-initialization-regime-models-v67.json','embedding-initialization-native-chains-v67.json',
        'embedding-initialization-fusion-costs-v67.json','embedding-initialization-operator-contract-v67.json',
        'embedding-initialization-fusion-variants-v67.json','initialization-profile-insufficiency-v67.json',
        'operator-analysis-coverage-ledger-v67.json','operator-regime-refinement-v67.json',
        'M04-M05-M06-M19-M10-operator-branch-inventory-v67.json','initialization-analysis-validation-history-v67.json',
        'initialization-protected-module-state-v67.json']
    write(O/'PROFILE-CODE-QUESTION-GATE-v67.json',dict(
        status='EMBEDDING_HC_INITIALIZATION_AND_SELECTED_ID_POSITION_OPERATOR_FUSION_MODELS',
        previous_turn_classification='PROGRESS:v66communication_native_count_models_and_repartition_costs',
        questions=['Whichvocabmask/cast/lookup/zeroandownercommunicationinterfacesareprivate?',
            'DoesSPcopy-mapcommutationallowdraftHC4repeataftershardingwithoutchangingmask/padding?',
            'Caninitialonehot/HCrepeatshareloadsorbevirtualwhileallresidual/HF32/numericalreaderspreserved?',
            'WhatselectedK/index-stride/cast/collision/state-generationandfullprefixconditionspermitID/positionfusion?',
            'WhichMemSetoperandsorBOOLfieldreadersremainunknownandmuststaypending?'],
        allowed_source='Sevenprevious166SHA-boundlocaloperatorclauses;workeronlynamedIDgather-scatter/positions/sanitize/embeddingbank/pad-tailquestions. No framework/API/HTTP/SSE/lifecycleaudit.',
        new_remote_source_read=False,local_pinned_operator_clause_review=True,
        no_model_Torch_NPU_capture_health_process_kill_or_implementation=True,
        explicit_new_pending=dict(MemSet_regimes=4,MemSet_exec=312,BOOL_regimes=2,BOOL_exec=80),overall_complete=False))
    scopes={
        'F01':'FirstFP32onehotpremix4TASK28n+8→16nfullwrite/readonlybank/virtualconstantinterfaces,allinputmutation/capture/configlifetimeandfull4H/HF32/BF16Precontracts.',
        'F02':'272vocabmask/gather/zero14TASKgroups,3same-inputwidens/maskedlookup/row0cache/owner-gatherimbalance/nativeSUM-zero;24persistentembeddingcopiesand24SPclones,firstTile-Pre/virtualresidualconditions.',
        'F03':'DummyattentionmetadataNonevirtualzero→HcPostandunusedinputcontentguard;fullHCresidual/newcoefficients/nonfinite/round/collectiveprotocolpreserved,notmetadata-presentno-querybranch.',
        'F28':'104draftglobalHC4repeat-before-SP15TASKinitializations;N10/20A16/24copy-mapcommutation/delayedTile/directownerinit,maskNonepositiveversussuppliedTrueprefix,globalKV/outputreadersandunpricedMemSetsretained.',
        'F31':'72sample4stage+72draft6stageIDgather-scatter,88position4stages;selectedKnot8192storage,cross-storepriority/alias/negativebounds/INT32narrow,fullprefixsanitize-clampandmetadata-tailextents. 4MemSet312+2BOOL80pending.'}
    updates={cid:dict(review_v67=dict(regime_models=str(O/e[0]),native_chains=str(O/e[1]),work_value_models=str(O/e[2]),
        operator_contract=str(O/e[3]),variants=str(O/e[4]),pending_evidence=str(O/e[5]),candidate_specific_scope=s,
        conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False),
        profile_review_state='EMBED_INIT_ID_SELECTED_WORK_IO_READERS_NUMERICS_ALIAS_AND_PENDING_SCOPES_REVIEWED',
        conclusion_changing_gap='ActualownerU/uniqueD/indices/duplicates/alias/allreaders/oldstate-weight-configgeneration/nativeHF32-HCCLpolicy/physicalVector-index-cachecapacityremainconditions. FourMemSetandtwoBOOLfieldregimesgenuinelypending,notclosedbyparent/sourceguess.',
        next_profile_or_contract_action='Next828M21input/metadata/indexoperatorregimes76943exec,groupbyconcreteoperator/datafield/consumerandreuseexistingv37-v54contracts. Codeonlywhenprofilecannotsettlespecificfusion/work/IO/dependency;7MemSet1272and2BOOL80pendingseparately.')for cid,s in scopes.items()}
    info={
        'M04':('59regimes1576TASKsinputID/positions/persistentbanksreviewed,72sample+72draftscatterchainsand88positionchains;selectedK/stride6/logicalB×5/error/collision/fullprefixandtailreaderspriced.13semanticfamiliesarchive20chronologyrecords.',['F31','F02']),
        'M05':('113strictregimes3832TASKsreviewed:27214TASKvocabgroupsand24SPclones. 2M05|M21BOOL24regimes80remainambiguous/pending. Maskedlookup/owner-gathercache/zero-SUM/directbank/added-vocabplansmodeled;10semanticfamiliesarchive9chronologies.',['F02','F28','F31']),
        'M06':('23targetHCinitregimes520TASKsreviewed:104Tile+onehotinitializations. FP32constantbank/materialized-virtualHC/HF32-norm-reassociationguards;14semanticfamiliesretainpriorv58/v59andv64HCcontracts/archive14chronologies.',['F01','F02','F03','F28']),
        'M19':('38draftregimes1560TASKselected;34regimes1248TASKreviewed,4MemSet312pending. 10415TASKinitializations/N10/20copy-mapSP-before-repeat/directowner/maskTrueprefix/premix/capture/globalKVreaders.13semanticfamiliesarchive12chronologies.',['F28','F02','F03','F31']),
        'M10':('1remainingregime48dummyZerosLike20×5120reviewed:metadataNonezero-returnisnotmetadata-presentnoQglobalKV/outputbranch. Existing9semanticfamiliesretained+2dummyconditionalfamilies;allotherattentioncontractsunchanged.',['F03','F28'])}
    mods={mid:dict(finding=f,evidence=e[:9],candidates=cids,
        gap='Allactualrow/alias/oldgeneration/reader/nativeprecision/resourceconditionsandunobservedapplicablemodelpathsremainopen;M05BOOLandM19MemSetsremainexplicitpending.',
        profile_scope='Operator-only230scopedreviewof236selectedregimes,6pendingnotpromoted. ExistingHC/attention/decoder/MMguardsretained,notframeworkclosure.')for mid,(f,cids)in info.items()}
    # Prior detailed findings stay in immutable parent ledgers; current gaps must not retain stale next-actions/counts.
    remaining=[
        'Fulloriginalgoalremains:fixedmodel/version/hardware/loadallmodules,conditionsandcrossoperator/layer/card/iterationfusion/split/reorganization. 26moduleinventories/35modelcandidates/7legacyframeworkcandidatesexist,0wholemodulesclosed. Tables/reports/manifestsareindexes,notscopecompletionproof.',
        'Observedoperatorfrontier2082regimes/286950ComputeInfoexec/83types:1245regimes208655execreviewed,837regimes78295pending=828M21/76943+7MemSet/1272+2M05|M21BOOL/80. SavedHCCL14720callsareseparatelymodeledandnotaddedtoComputeInfodenominator.',
        'M21remaininginput/metadata/indexoperatorsneedfield-specificnecessarywork/selectedIO/alias/ready/owner/generationmodelsandfusion/split/reusechoices. Reusev37-v54answeredCP-slot/rejection/state/sourcecontracts;doreadonlyprofile-firstgrouping,notbroadframework/caller/API/lifecycleaudit.',
        'SevenMemSetregimes1272lackoperandaddress/length/workspace-vs-payloadscope;2BOOLprefixregimes80lackexactfield/allconsumeridentity. Sourceparent/adjacentPadextent/shapenotproof. SharedAttentionState.resetno-ophypothesisexcluded;SPmaskNonecompatiblebutidentitynotproved. Onlyfutureauthorizedexistingarg/IR/runtimeevidencecanclosekeygaps;noneassumed.',
        'Modelbranchapplicabilitystillrequirescompletefixedcaseevidence:prefill/decode/draft/accepted-rejected,SP/CP/DP/EP,compression0/4/32,index-source/consumer,Engram/MM/vision/prompt/None-empty/metadata-capacityguards. Unobserveddoesnotmeaninapplicable;conditionalanalysiscanbecompletewithoutnativeimplementationbutmissingbranchanalysiscannot.',
        'All35candidatechoicesstillneedfulloriginal/transformedsubgraphandallreader/numerical/ownership/synchronization/capacity/generationconditionsandsame-denominatorcost/interactionsreconciledacrosswholemodel. Existingv55-v67localconditionalmodelsarevaliddeliverables;missingimplementation/benchmarkaloneisnotunfinishedanalysis,whilegenuineunpriceddependencies/consumersremainopen.',
        'Actualtypedphysicalcapacitybindingremains:published910Cperdie24Cube48Vector/64GB,376BF16TF/752INT8TOPS/1.6TB/sreferenceisnotactualclock-scaledcapacity. Cached1800MHzdoesnotlicenseinventedscaling. FLOAT/HF32/Vector/INT64mod/sort/special/UVA/L2/HBM/communicationinjection-link-cut-latencyratesandactualtrafficunknown;logicalinterfacesandPMUbusyarenotphysicalMFU/bandwidth.',
        'NativeMC2EP16receiverwork/route-count/internalphase/cache/topologyandcut/sender-receiverpayloadremainseparatefromrankSize8HCCL. v56alreadyfusesdispatchquant/GMM1/Swiglu/GMM2/return;analyzenewvariantsagainstthatbaseline,nofreeoverlaporall-to-allrecordstand-inforMC2.',
        'Memory/stateacrossiterationandlayerstillrequiresrequest/acceptedprefix/CP-SProw/position/slot/group/weight-bank/configgeneration/allfutureobserverandaliasproof. ExistingKV/pool/indexbanks/Engramhistory/metadatareuseisbaseline;sameaddress/shapeisnotvalueidentity. CurrentHC/BF16/HF32/reduction/RNG/TopK-tieprecisionboundariesmustnotbecancelled.',
        'Unchangedglobaluncertainties849missingPMU/16phase-unboundgraphs/3136ambiguousmoduleassignmentsmustbeaccountedwithoutextrapolatingunboundcapturesorrewritingfunctionalattribution. SUMofTASK/COMMintervalsnotrequestcriticalpath;localcoldvalue/referencefloororpayloadratioisnotmeasurede2egain.'
    ]
    remaining.append('v67selected236regimes7616TASK=230review7224+6retainedpending392. 272embedding14TASKgroups/104target5TASK/104draft15TASK,72sample4stage/72draft6stage/88position4stagechains. DraftN20A24lateTilelogicalbudget2222080B/card;ownerinitializerstrongerbutnotadditive,allprovidedmaskvaluesretained. Embeddinginvalidrow0repeatsatmostoneextrauniquerow/cache,ownerfixedAGbalanced.5ARvsconcentrated4ARandnativeSUMsignedzero/NaNgates. IDscatter8192descriptorisstorage,selectedKonly,firstsamplethendraftpriority/sourceB×5viewofBmax×6/negativewrap-vs-errorandalloldstatealiasconditions. 22variantrecords(1explicitunresolved),7old166pinnedlocalclausefiles,52new/reconciledfamilies+9M10retained=61currentfamilies. Globalcoverage1245/208655review,837/78295pending:828M21/76943+7MemSet/1272+2ambigBOOL/80. 35model+7legacyframeworkcandidates,0wholemodules;other21matrixrecordsprotected. NextM21operator/datafield/consumerfamilies,notframeworkresearch.')
    action=(
        '继续原Goal最新v67 PROGRESS。上一轮v66也是PROGRESS。用户要求代码只围绕算子是否可融合及理论上限，充分profile，缺具体消费者/精度/依赖/写入证据再定点代码，不研究框架。'
        '先读operator-analysis-coverage-ledger-v67、embedding-initialization-regime-models-v67、embedding-initialization-native-chains-v67、embedding-initialization-fusion-costs-v67、embedding-initialization-operator-contract-v67、embedding-initialization-fusion-variants-v67、initialization-profile-insufficiency-v67及M04-M05-M06-M19-M10-operator-branch-inventory-v67。'
        '本轮选236/7616，230/7224scopedreview，4M19N/AMemSet312+2ambigM05|M21BOOL80保留pending。全2082/286950现在1245/208655review、837/78295pending、0geometry。余828M21/76943+7MemSet/1272+2BOOL/80，不是Goal完成百分比。'
        '272embedding14TASKgroups，mask源immutableinput3INT32→64+2BOOL→64，原9masktype是no-added-vocab；Gather只选N5120不是16160×5120全表。声明onepassvalue预算8NH+150N→2H(N+U)+4N，U未知。无效ID读row0再masked+0，重复N-U引用最多一个额外unique row10240B，不能当N-U倍HBM收益。Ownergather新协议balancedfixedAG.5AR、集中oneownerfixedpadding4AR，raggedaggregatehalf但owner注入straggler/metadata/nativeSUM负零NaN精度另计。'
        '104target初始化5TASK=localTile+FP32premix4TASK，target本已先SP后repeat。104draft初始化15TASK=globalTile+3Pads+3Casts+2ZerosLike+3MemSets+Fill/Broadcast/ViewCopy，N10/20→A16/24→local2/3。推迟draftrepeat到2DSP后是精确copy-map重排，N20A24可省2222080B/card普通logicalvalue，不是HBM或实测时间。直接ownerinitializer读U行并输出localHC4/ID0-pad/maskTrue-pad/FP32onehot更强，但预算不可相加。104maskZerosLike leaf149证明None→falseprefix；providedmask路径Trueprefix保留。无query卡仍要全局KV/输出通信。'
        'Premixzero+scalar1+broadcast+columncopy28n+8→16n fullwrite，readonlyonehotbank/virtualconstant是其他选择；不可当后续所有pre_mix恒定。Tile+Pre共享读8nH预算保fullHCresidual，虚拟repeat至firstPost两读参考22nH需新Pre/Post/readers/Engram/alias协议；原full4Hnorm/HF32coefficients/FP32postorder/BF16round保留。折叠四通道weight浮点重排反例0vs1只是IEEEFP32参考，不是原生HF32测试；onehot0*Inf仍NaN，不能盲删零项。'
        'M04为输入ID/position算子而非norm。72sample4stage48K→24K，72draft6stage28BD+48K→28K；BD=K10/20时预算48K。输入IDsScatter8192只是bank容量，实际K写；逻辑B×5来自Bmax×6 stride6 offset1，合法负source索引wrapBD、dest负scatter错误/INT32narrow/duplicate别名/先sample后draft覆盖顺序都保。合并parallel写需要disjoint或priority/barrier，不能只按shape。'
        '88positions四阶段IndexCheck+gather32+cast64+add64:60N→28N普通value参考，sourcecapacity4/overflow/error/oldcomputed和所有Rope/position/tailreader保留。placeholder-1→0与clamp_min0只有same-prefix/no-intermediate-readers才合并22N→8N，embedding内私自sanitize不更新persistent IDbank会影响Engram/router其他消费者；只清新scatter K也漏旧prefix。querytail-1、seqtail0、pospad0域不同别CSE。24inputs_embeds大copy可newoutABI/24SPclone可newaliasview，但MM/prompt/sentinel/capturebank/lastreader/prefix-tail条件完整。'
        '48dummyZerosLike是Dsa.metadataNone→zero-return，可conditional虚拟zero/HcPost或删unusednorm/inputAG内容，仍保HCcoeff/residual/FFN/allgroup协议。不同于metadata-presentno-query必做KV/AlltoAll/WA-WB/RS；不是C8请求加速。'
        '22variant记录含最后1项明确待定，230reg补齐条件分析；4N/AMemSet312不能按ConstantPadNd名字/附近Padextent填bytes或删控制clear，另外80graphprefixBOOL24虽知24B store但field/consumers仍未定，歧义未改。精确AST已排除shared_attention_state.reset(仅returnNone)清零假说；sp_padding_mask(None)兼容但非field证明，别重查同个reset。当前总7MemSet1272+2BOOL80待定。7old166SHA-boundlocalfiles复用，seq source73bc..由v30/v62绑定，无remote/NPU/model/Torch/health/process/kill/实现。'
        'M04/M05/M06/M19四legacy矩阵归约13/10/14/13semantic，M10旧9保留+2，总61(52new/reconciled+9retained)，其余21完整modulematrix保护，旧M06v64correction在archive保留。35模型候选+7legacy框架候选，0wholemodules关闭，849missingPMU/16phaseunbound/3136ambiguity保留。'
        '下一步M21 828regimes76943exec：先profile按operator+datafield+caller+shape分组并复用v37-v54已答metadata/CP-slot/rejection/source-contract，不从巨大worker/framework全扫。为每组建必要work/selectedIO/同步和跨算子-跨层-跨卡-跨迭代重组/复用条件；只有具体问题缺证据时定点补调用代码。MemSet和BOOL真缺证据保pending，不因完成报告或本轮范围完就结束原Goal。')
    report='''本轮围绕embedding、HC初始化、草稿切分和输入ID/位置算子，核对236类、7,616次执行。其中230类、7,224次补齐了条件分析；4类MemSet的312次执行及2类布尔清零的80次执行继续待定。全程使用本地保存profile和7份已绑定原166版本的源码，没有连接服务器或执行模型。

原生任务已按源程序顺序对齐：272组embedding、104组目标HC初始化、104组草稿初始化、72组采样ID更新、72组草稿ID更新和88组位置更新。embedding每组仍呈现14个设备任务，源码注释中的“compile会融合”不能替代profile证据。

草稿当前先将完整embedding从N×H复制为N×4×H，再补齐并按卡切分。把精确的复制映射改成先切分2D embedding、再复制局部行，不改变数值运算。20行补齐到24行、每卡3行的案例，普通逻辑读写模型可减少2,222,080B/卡；目标模型本来就先切分再复制，不能在那里重复计算收益。更强的局部初始化kernel可直接读取本卡有效行并输出HC状态、ID、padding mask和FP32 onehot，但这是另一种粒度，预算不能相加。提供的mask若含True前缀，必须保留；本批104条草稿new_zeros路径则明确进入mask为None的分支。

| 算子方案 | 声明的逻辑值读写模型 | 关键约束 |
|---|---|---|
| embedding掩码、索引、lookup和清零组合 | 8NH+150N → 2H(N+U)+4N | U为本卡词表所属有效行；INT64索引、正零、错误及其他词表计划保留 |
| FP32初始onehot一次完整写入 | 28n+8 → 16n | 所有输入只读、bank别名和capture生命周期；后续pre_mix并非常量 |
| 采样ID gather到选定位置 | 48K → 24K | 源索引负值回绕与目标scatter负值错误不同；保持旧值、冲突和顺序 |
| 草稿strided ID gather、转型与写入 | 28BD+48K → 28K | 逻辑B×5来自物理Bmax×6，步长6、偏移1，不能按整个底层容量取索引 |
| 位置gather、转型与加法 | 60N → 28N | INT32旧计数、INT64偏移/输出、源容量和错误、代际及所有位置读者 |

N、n、K和BD在各行使用各自的完整输入行、局部行、选中位置数和草稿逻辑元素数，H=5120。上述预算是明确的常规读写估值，不代表实际HBM字节、物理容量或端到端收益。

embedding有一个容易高估的点：不属于本卡词表的ID先被映射到weight row0，再清零。新kernel可以跳过这些lookup，但所有无效行反复引用的是同一个row0，最多额外涉及一行10,240B的唯一权重，可能已在缓存，不能把每次引用都当成HBM读取。若改成收集各owner的有效embedding而不用全零平面的AllReduce，词表归属均衡时，固定AllGather的参考payload可为原来的1/2；所有token集中到一个owner时，按最大owner行数补齐的固定AllGather则可达到原来的4倍。动态协议、owner注入瓶颈以及原生SUM对负零/NaN的处理都要计入。

输入ID的ScatterElementsV2显示8192个输出容量元素，语义上只写选中的K个位置。当前先写采样token，再写草稿token；合并成并行写入前，需要证明目标不冲突，或实现原来的优先级与同步。未更新位置的旧值也必须保留。placeholder=-1的替换和随后clamp_min0可以在相同前缀、没有中间读者的条件下合并；只在embedding内部清理、或只清理新scatter位置，会改变其他输入ID读者的行为。

初始HC状态还可以研究与第一次HcPre共享输入读取，或把四路重复表示保留到第一次HcPost再物化。后一方案必须改造完整的Pre/Post残差接口，并确认所有读者及中间Engram/别名写入。原4H维归一化、HF32系数投影和BF16边界不能直接代数约简。参考浮点反例中，先折叠四通道权重与原顺序累加可分别得到1与0；onehot的零项遇到Inf也会产生NaN。这些是拒绝无条件改写的证据，不是当前原生kernel数值测试。

48次dummy attention清零来自metadata为None的提前返回。它允许研究虚拟零输入或裁剪未使用的attention输入内容，但必须保留HC系数、残差、FFN及所有通信参与关系。这与metadata存在、仅本卡没有query的分支不同；后者仍有全局KV和输出交换，也不能把dummy路径优化当作C8请求收益。

共形成22条方案记录，其中一条明确保留缺证据状态。当前全观察台账为1,245类、208,655次执行已有显式条件分析，837类、78,295次仍待核对：828类M21输入/metadata/index算子、7类MemSet和2类布尔字段。实际物理上限、未观察但适用的模型分支及整体Goal均未关闭。
'''
    jobs=[
        dict(kind='profile',status='pass',description='236functionalregimes7616uniqueTASKs/PMUcounts;272embedding14TASK/104target5TASK/104draft15TASK/72sample4stage/72draft6stage/88position4stagechainsconserved.',artifacts=[O/e[0],O/e[1],N/'analyze_embedding_initialization_v67.py']),
        dict(kind='review',status='pass',description='7old166SHA-boundlocaloperatorclausesforvocabmask/SPcopy/HCinit/IDpositionandpersistentembeddingbank/dummyNone;workerreadsonlynamedoperatorquestions,noframeworkaudit/remote.',artifacts=[O/e[3]]+[Path(r['path'])for r in contract['source_files']]),
        dict(kind='design-check',status='pass',description='22variantrecordswithtypedembedding/row0cache/owner-skew/draftcopycommutation/premix/selectedID-positioncosts;FP32fold/zeroSUM/stridednegative-indexwitnessesPASS. Nativephysicalceilings/implementationnotclaimed.',artifacts=[O/e[2],O/e[4],N/'model_embedding_initialization_v67.py']),
        dict(kind='profile',status='pass',description='Explicitlyretain4MemSet312withoutargs/size/controlscopeand2BOOLfield80withoutconsumers. ASTexcludesshared_state.resetno-ophypothesis;compatibleSPmaskshapenotidentity. TotalMemSet7/1272.',artifacts=[O/e[5],O/e[9],O/'BOOL-prefix-candidate-review-v67.json',N/'review_prefix_BOOL_candidates_v67.py']),
        dict(kind='review',status='pass',description='1245review208655+837pending78295conserve2082/286950. 61semanticfamilies5modules(52new/reconciled+9M10retained),21otherrecordsprotected;alluncertainty/0wholemoduleclosurekept.',artifacts=[O/e[6],O/e[7],O/e[8],O/e[10],N/'consolidate_embedding_initialization_v67.py'])]
    persist_refinement(version=67,previous=66,title='Embedding, HC initialization and selected input ID/position operator fusion bounds',
        summary=dict(filename='embedding-initialization-ceiling-summary-v67.json',previous_turn_classification='PROGRESS:v66communication_native_count_models_and_repartition_costs',
            selected_regimes=236,selected_executions=7616,reviewed_regimes=230,reviewed_executions=7224,
            retained_pending_regimes=6,retained_pending_executions=392,operator_variant_records=22,variant_records_with_explicit_unresolved_decision=1,
            embedding_groups=272,target_HC_initializations=104,draft_initializations=104,sampled_ID_groups=72,draft_ID_groups=72,position_groups=88,
            semantic_families=61,new_or_reconciled_families=52,retained_M10_families=9,protected_other_modules=21,
            observed_regimes=2082,scoped_reviewed_regimes=1245,scoped_reviewed_executions=208655,geometry_only_regimes=0,
            pending_regime_reconciliation=837,pending_execution_reconciliation=78295,
            MemSet_explicit_pending_regimes=7,MemSet_explicit_pending_executions=1272,unresolved_BOOL_field_regimes=2,unresolved_BOOL_field_executions=80,
            new_remote_source_read=False,local_pinned_operator_clause_review=True,implementation=False,native_test=False,new_performance_result=False,
            all_actual_physical_numeric_ceilings_proved=False,
            remote_job_status='No new remote job/connection. Local saved-profile/pinnedoperator/mathonly;no model/Torch/NPU/capture/health/process/kill/implementation/backgroundanalysis.'),
        changes=['230regimes7224TASKscopedconditionalreviewof236/7616selected;4MemSet312and2BOOL80gapsretained. 272embedding/104targetinit/104draftinit/72sample/72draft-ID/88positionchainsbound.',
            '22variantrecordspriceSP-before-HC-repeat,onehot/read-onlyandvirtualHC,inputlookup/row0cache/owner-skew,selectedID-position/alias/order/typedprecisionconstraints;noinventedphysicaltraffic/speedup.',
            '1245regimes208655reviewed,837pending78295=828M21/76943+7MemSet/1272+2BOOL/80. 61semanticfamilies5moduleswith21othersprotected;globaluncertaintiesand0wholemodulesunchanged.'],
        candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=inv['modules'],remaining=remaining,action=action,
        jobs=jobs,report=report,read_first=[e[6],e[0],e[1],e[2],e[3],e[4],e[5],e[8]],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/x for x in ['analyze_embedding_initialization_v67.py','model_embedding_initialization_v67.py','consolidate_embedding_initialization_v67.py','review_prefix_BOOL_candidates_v67.py','verify_embedding_initialization_v67.py']],
        executor_prefix='zcode-embedding-initialization',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
