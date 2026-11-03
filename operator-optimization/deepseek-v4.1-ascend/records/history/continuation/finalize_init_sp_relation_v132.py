"""Seal scoped first-draft initialization / SP projection composition once."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v132.json').exists()
    pre=read(O/'init-sp-preflight-v132.json');assert pre['status']=='PASS' and pre['checks']>10000
    names=['init-sp-inputs-v132.json','init-sp-native-frontiers-v132.json','init-sp-source-contract-v132.json',
        'init-sp-copy-and-mask-models-v132.json','init-sp-QA-owner-routing-models-v132.json','init-sp-joint-work-content-v132.json',
        'init-sp-first-layer-token-reuse-v132.json','init-sp-selected-graph-and-guards-v132.json']
    detail=dict(evidence=names,first_draft_init_frontiers=104,positive_QA_and_global_KV=69,positive_local_query_empty_global_KV=19,positive_metadata_None_zero=16,
        copy_mask_models=3816,QA_owner_route_models=864,new_scoped_pair_relations=1,new_pair_clauses=1,new_groups=0,new_main_candidates=0,
        actual_CP_SP_ownership_cache_hits_or_native_parity_bound=False,new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    claim=(
        'ComposeSTANDARDembeddingSUMAR→matchingorderedRSwithlocal/virtualHCinitializationandfirstY_BF16/full5120norm_BF16/ownerKV512carrieronlyifallfullE/HC/rawreaders,rank/alias/round/eventcontractsmatch. '
        'SPownerandCPqueryownerarenotprovenequal;retainneededraw5120rowsorrouteownerQA1280/newjoint1792plane,withfullQA1280norm/laterquery/headrolespreserved. '
        'UniformjointQAcomputesA paddedrowsversusNdisjoint-covervalidrows;cannotdoublecountglobalTile/pad/virtualHC/interfacecredits. '
        'FirstHCcoeff/pre/post/comb/residualstaylive;currentcoeffdonotnumericallygateexternalpremixYbutearlypublicationneedsnewsplit/handoff. '
        'Positive19localqueryempty/globalKVarenot16metadataNonezero;onlyuniformdeadinputdomaincanprunenorm/SPinput. '
        'Firstgenerationtokenpre-RoPEKV/rawQAreuseisconditionalonfullparameter/embedding/math/nativeM/branch/captureepochs,cachefillandallIDreaders;stillneedHCcoeff/residual. '
        'Allcountsarelogicalcontents/declareddensework,notwire/HBM/allocatedVRAM/latency/physicalMFUorwholefusionproof.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v131.json'));assert not any({r['a'],r['b']}=={'F28','F32'}for r in rel['clauses'])
    rel['clauses'].append(dict(id='REL132-01',a='F28',b='F32',relation='FIRST_DRAFT_INIT_OWNER_QA_KV_CARRIER_AND_TOKEN_REUSE',operator_claim=claim,
        applicable_domain='SelectedfirstdraftHCgenerationandSWA-onlypaths;STANDARDembedding,CPqueryowner,localqueryemptyandmetadataNonedomainshaveexplicitseparategates.',
        evidence=names,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,init_SP_composition_v132=deepcopy(detail)))
    g=next(g for g in rel['higher_order_groups']if g['id']=='HG85-VERIFY-CONTEXT-DRAFT');assert {'F28','F32'}<=set(g['members']);g['init_SP_composition_v132']=dict(**detail,contract=claim,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v131.json',new_pair_clauses=1,new_higher_order_groups=0,clause_count=139,selected_pair_count=139,init_SP_composition_v132=detail,overall_complete=False)
    write(O/'candidate-relation-clauses-v132.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v131.json'));r=next(r for r in cov['pairs']if {r['a'],r['b']}=={'F28','F32'})
    r.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=['REL132-01'],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None,init_SP_composition_v132=deepcopy(detail))
    cov.update(parent='candidate-relation-coverage-v131.json',parent_relation_index='candidate-relation-clauses-v132.json',scoped_conditional_pairs=139,remaining_pair_relations=456,
        init_SP_composition_v132=detail,scope='139scopedconditionalpairs/26groups/35modelcandidates;456centralpairsremain. FirstdraftF28-F32owner/init/carrier/tokenreuseisscopedconditional;allruntime/allvariantsopen.',overall_complete=False);write(O/'candidate-relation-coverage-v132.json',cov)
    nav=deepcopy(read(O/'candidate-pair-navigation-v131.json'));r=next(r for r in nav['pairs']if {r['a'],r['b']}=={'F28','F32'})
    r.update(conditional_clause_ids=['REL132-01'],semantic_pair_reconciliation_done_in_stated_scope=True,whole_pair_all_variants_closed=False,init_SP_composition_v132=deepcopy(detail),evidence=names)
    nav.update(parent='candidate-pair-navigation-v131.json',parent_footprints='candidate-operator-footprints-v132.json',parent_relation_index='candidate-relation-clauses-v132.json',scoped_conditional_pairs=139,remaining_central_pair_reconciliation=456,init_SP_composition_v132=detail);write(O/'candidate-pair-navigation-v132.json',nav)
    fp=deepcopy(read(O/'candidate-operator-footprints-v131.json'))
    for r in fp['candidates']:
        if r['id']in ['F28','F32']:r['init_SP_composition_v132']=dict(evidence=names,no_new_native_operator_atoms=True,local_graph='init-sp-selected-graph-and-guards-v132.json',all_runtime_or_native_parameters_bound=False)
    fp['parent']='candidate-operator-footprints-v131.json';write(O/'candidate-operator-footprints-v132.json',fp)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v131.json'));op['init_SP_composition_v132']=detail;write(O/'operator-analysis-coverage-ledger-v132.json',op)
    obligations=deepcopy(read(O/'coverage-current-obligations-v131.json'))
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G10','G11','G13']:r['init_SP_composition_v132']=deepcopy(detail)
        if r['id']=='G07':r['remaining']=(
            '139conditionalpairs/26groups,456centralpairsandallvariantsremainopen. SelectoneembeddingSUM/local-or-virtualHC/firstnorm/QAvsKVowner/cachegraph;do notaddoverlappingTile/pad/Y/interfacecredits. '
            'SP/CPmapsandallreadersunbound;newQA1280orjoint1792deliveryneedmatchingweights/nativeM/rounds. UniformA/NpaddingaddsQAwork. '
            'Firsttokenreusehascoldfill/storage/epoch/IDavailabilitycosts;HCcoeff/residual/laterstateandpostselectRoPE/cachepublicationremain. Symbolicmodelsnotphysicalcapacityorspeedups.')
        if r['id']=='G10':r['remaining']='Central139 conditional relations/26 groups leave456 pairs to reconcile;blank rows are not independence. Allhigher-order-subsets/runtimelegalityopen. Do notenumeratearbitraryemptypairsorrelabeloldlocalanswersasnewwork.'
        if r['id']=='G13':r['remaining']=(
            'v132F28-F32firstdraftcomposition:104init/firstHC/RMS/SP/embeddingARfrontiers,69positiveQA+globalKV,19positivepreprocess354localqueryempty/globalKV,16positiveNonezeros. '
            'Exactsource-orderkeysarenotSP/CProwmap/parameter/alias/bankidentity. QAlocalM2/3vsKVglobalM10/20cannotbedirectlyconcatenatedwithoutowner/ABIchange. '
            '3816bitcopy/maskand864ownerroutemodelsretainprovidedTrueprefix,paddingzero/ID0distinctionandrowtags;existingv67local/virtualHC/RSanswersarenotnew. '
            'Newfirst-tokenpre-RoPEKV/rawQAmemoizationisconditional;fullKV126.25MiBandKV+QA441.875MiBvaluecontentsexcludeindices/queues/fill/epochsanddonotprovehitsorallocatedVRAM. '
            'Nextprofile-firstpendingF01-F32:externalpremixYandcoefficient/pre/post/comb fork,earliernorm/projectedcarrierpublicationandmandatoryresidual/readers;reusev58/v59HCsourceandcurrentprofile,notframeworksurvey. '
            'All9/1352native/globalbranch/graph/physicalcapacity/MFU/lowestgatesremain. No newNPU/model/capture/replayauthorization;Padv120consumed,nopendingapproval.')
    obligations.update(parent='coverage-current-obligations-v131.json',new_native_or_whole_closures=0,overall_complete=False);write(O/'coverage-current-obligations-v132.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v131.json'));audit.update(previous_goal_turn='PROGRESS:v131F27-F32checkpointpacket/representation;postcheckPASS11067',input_checkpoint=131,followup_checkpoint=132,init_SP_composition_v132=detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=139,central_pairs_not_reconciled=456,higher_order_groups=26)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v132.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v132.json',dict(status='POSITIVE_FIRST_INIT_QA_CACHEONLY_DUMMY_PROFILE_THEN_TARGETED_OWNER_AND_ROWFUNCTION_CONTRACTS',questions=read(O/'init-sp-inputs-v132.json')['questions'],source_scope='OnlypinnedDSparkinit/firstHC/norm/CPquery/KV/NoneguardandSPcopysegments;reuseHCprecisioncontract,noframeworksurvey/newSDK/remote/deviceexecution.',no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'init-sp-validation-history-v132.json',dict(events=[
        'Predecessorv131revalidatedPASS11067/manifest136;previousgoalturncompletedF27-F32conditionalrelation,notwait/statusonly.',
        'ExactTileTASKjoinsv67draftinitwithv130firstSPfrontier;104embeddingARshaveexactCID/sourceframeDSpark249. Source-orderfrontersarenotactualtensoredgesorclient-stepidentity.',
        '104firstHC/RMS/SPwindows:69QAandglobalKV,19positivepreprocess354cache-onlyKV,16positiveNonezero. No absence-onlybranchinference;allrawTASK/COMPUTE/API/CID/callchainrowschecked.',
        '3816rawBF16bitcopy/paddingmaskmodelsand864CP/SPqueryownerpermutationschecked. Firstnorm/QAlocalityandjoint1792carrierincludeQAextrapadding,allrawconsumersandnumericgates.',
        'Firstdraftpre-RoPEKV/rawQAtokenrowreusevariantrecordsfullparameter/math/nativeM/epoch/key/domain/ID/capturecontracts;finiteexamplesdenytoken-onlyepochkeys,token0forpaddingandfinalpositionreuse. Actualhitsandnativeparityunknown.',
        'Oldlocal/virtualHC/embeddingRSanswersreusedfromv35/v67;newcompositiondoesnotduplicatecreditsorremoveHCcoeff/pre/post/comb/residual. Literalonehot*Infwitnessretainsproducts/nonfinitebehavior.',
        'CorrectedstaleG10prose135/460to139/456;priorauthoritativev131tableswere138/457. Thisbookkeepingfixisnotanoptimizationfinding.',
        'Preflight'+str(pre['checks'])+'PASS. REL132-01addsF28-F32;HG85alreadyincludesboth.139pairs/26groups/456pending;358/26semantic854/1346/48partialgraphallnative/wholeuncertaintyunchanged.'],overall_complete=False))
    gap=('NeedactualSP/CPquery-ownerincidence,embeddingSUMgroup/order/plan/quantandallfullE/HC/rawreaders,parameter/gamma/eps/nativeM/round/event/capture/bankepochs. '
        'PreservefirstHCcoeff/pre/post/comb/residual,providedmasks,position/RoPE/cache/headrestoreandnoQvsdummybranches. PriceQAextraA/Nwork,allnewpack/routing,cachefill/lookup/metadata/epochs/storageandIDreaders. '
        'No actualtokenreuse/nativeparity,wire/HBM/allocatedVRAM/latency/MFU/lowesttimeorwholepair/allbranchclosureestablished.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v132conditionalF28-F32:139pairs/26groups/456pending;104firstinitfrontiers69QA19cacheonly16dummy,3816copy864routingmodels.358/26semantic854/1346/48partialgraph9/1352native849PMU16phase3136ambiguity0wholeunchanged.')
    action=(
        'ContinueoriginalGoal;v132PROGRESSisF28-F32firstdraftinit/owner/carrier/tokenreusecomposition,notnativeoptimization/wholeclosure. Readinit-sp-summary/native-frontiers/source/copy/routing/work/tokenreuse/graph/currentobligations-v132;verify_init_sp_relation_v132mustPASS. '
        '104exactTile/init/firstHC/RMS/SPandembeddingARfrontierswithDSpark249CIDsource;69positiveQABF16M2/3x5120→1280+globalKV,19positivepreprocess354localqueryempty/globalKV,16positiveNonezeros. NoQnotdummy;globalKV/outputcollectivesstay. '
        'Existingv67SPbeforerepeat/directlocal/virtualHC/embeddingAR→RSareprioranswers. NewjointgraphretainsHCcoeff/pre/post/comb/residualandY_BF16/full5120norm_BF16. Literal0*InfmeansonehotYisnotunconditionalrawE;foldedweights/full4Hnormremainseparatenumerics. '
        'SPownerandCPqueryowneractualmapsunbound. Missingraw5120orownerQA1280delivery/newjoint1792carrierneededunlessprovenaligned/allrawreadersclosed;fullQA1280norm/laterquery/headrolesremain.3816copy/mask864routecasesnotnativeproof. '
        'TP8N10/A16:rawSP20480B→KV2048B orKV+QA7168B;globalKVdense80→16rows,disjoint-coverQA10→16rows. N20/A24:30720→3072/10752B;KV160→24,QA20→24. Referencecontent/worknotlatencyoruniversalbounds;AR→RSinputpaddingincreaseswhileoutputcontentdrops. '
        'Optionalfirstgenerationtokenpre-RoPEKV/rawQAcachehascompleteembedding/parameter/math/nativeM/domain/epoch/capturekey,fillandIDreadergates. FullKV132382720B126.25MiB;KV+QA463339520B441.875MiBvaluecontentexcludesmetadata/queues. Actualhits/benefitunknown;HCcoeff/residualstillneedE. Globalreplicatedcache+globalIDsversusowner-demandcachehave differentcommunication/lookupgraphs;donotcachepositionedKVbytokenoruseEtoken0forSPpad. '
        '139conditionalpairs26groups456pending;HG85alreadymembers. G10stale135/460prosefixed.358/26semantic854/1346/48partialgraph2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU/lowestunchanged. '
        'NexttargetpendingF01-F32externalpremixY/coeffforkandearliernorm/projectedcarrierpublication;usev58/v59HCsourcepluscurrentprofile,retainallcoeff/residual/specialvalue/hand-off/capture/readergatesandavoidwhole-opbarriers. '
        'Alllocaljobscomplete;no remote/Torch/CANN/NPU/model/capture/replay/implementation. Padv120consumed,nopendingapproval. Do noteditsealedv132/rerunfinalizer. Versionedpostcheck/CURRENTmustmatchRECOVERY;TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐 F28–F32：第一层 draft embedding/HC 初始化如何与 SP 前移投影、Q_a 行归属及跨迭代复用组合。全部工作基于已有 profile、固定源码和本地模型检查。

按 Tile 原始 TASK 键把104次初始化接到第一层 HcPre、RMSNorm、SP gather；每组前面的 embedding AllReduce 也通过精确 CID 和 dspark249 源码栈确认。分支有69次 Q_a 与全局KV同时出现、19次正面 preprocess354 栈确认的本地查询为空/全局KV仍更新、16次正面 metadata=None 清零。19次空查询不能按 dummy 路径删掉 KV 或输出/head交换。顺序窗口与源码证明范围不等于真实 tensor、指针、请求或 cache 世代绑定。

既有 v35/v67 已提出 embedding AllReduce→匹配 SUM ReduceScatter、先 SP 后 repeat、局部或虚拟 HC 初始化。本轮把它们放入一个图，避免将全局 Tile/Pad、局部初始化和虚拟 residual 的完整收益重复相加。SUM 的成员、rank顺序、舍入、量化及全部 embedding 读者须匹配；ReduceScatter 的 padded 输入内容可能更大，不能单凭输出变小宣布 wire 流量或延迟更低。功能 custom-op 的 clone、别名、capture 和旧读者合同仍须保留。

第一层 HC coefficient/pre/post/comb 以及 first HcPost residual 仍活跃。外部 onehot premix 的 Y 与当前 coefficient 生成是两个数值依赖分支，提前交付 Y 需要新的 split/handoff。它也不是无条件返回 embedding：1×Inf+0×Inf+0×Inf+0×Inf 得到 NaN，直接 Inf 则不同。原 BF16 Y、完整5120维 RMS 的 BF16输出、full4H coefficient 计算和后续舍入不能因融合而消失。

SP owner 的 padded 行与 CP 查询 owner 的行未被实际操作数绑定。若它们不一致，仅把 KV512前移仍不够：须交付缺失的归一化 raw5120行，或由 SP owner 计算并交付 row-tagged Q_a1280；也可在新ABI中统一发送512+1280=1792维平面。Q_a1280 norm、后续查询投影、RoPE与head角色仍保留，全部 raw消费者要闭合。3816组原始 BF16 位模式/掩码用例确认 copy重排保留 signed zero、Inf、NaN payload与 supplied True前缀；864组 owner置换用例核对查询行标签与顺序。这些不是 native数值或真实 CP/SP map 证明。

参考 TP8 均匀 owner 图中，N10需 A16行：原 SP sender内容20480字节，KV-only为2048，KV+Q_a为7168。全卡 KV dense参考行数80→16，但若查询集合原来不重叠且覆盖10行，Q_a参考行数10→16。N20/A24对应30720→3072或10752字节，KV160→24、Q_a20→24。前移通信带来计算与padding取舍；这些是接口内容和指定 dense算法工作，不能当 HBM必要字节、网络wire、关键路径收益或物理上限。

跨迭代还可登记第一层 token行复用：在完整 embedding/reduction/量化、HC初态、norm与WKV、math/nativeM、参数/adapter及capture世代稳定时，预计算或按需缓存 pre-RoPE normalized KV512；可另存 raw Q_a1280。position/RoPE、slot和cache发布仍在 lookup 后执行。一个固定key版本下，完整词表129280行的 KV值内容为132382720字节（126.25MiB），KV+Q_a为463339520字节（441.875MiB），还不含索引、队列、fill与epoch维护。若需同时保留不同nativeM/math/参数版本，值内容还要相应增加，不能假定一张表对全部M等价。原始 embedding、HC coefficient及residual仍有需求，不能因缓存KV就删除。SP padding 的零行也不能拿 token0 的 embedding代替。

owner按需缓存仍可能需要 KV gather；全复制表加全部所需 global IDs 则可研究在每个cache接收者直接lookup，代价变为各卡lookup、全局ID寿命、表容量与维护。当前没有实际重复token/命中率/参数稳定性证据，也没有 native cachefill等价性，因此它只是有明确条件和成本的候选。不能将后续层已改变的HC状态或已加位置的KV按同一token缓存。本轮没有运行设备或修改模型。

新增 REL132-01，条件关系138→139、待核对457→456；HG85 已含两候选，联合组仍26。并修正 G10 文本遗留的135/460旧计数，该修正不算优化发现。35主候选、7历史框架记录、358语义行/26模块和854节点/1346边/48提示的部分图不变；9类/1352次原生字段、849个PMU缺口、16个phase未绑定图、3136项归属歧义继续保留。整模块关闭0，整体目标仍未完成。
'''
    inputs=[Path(r['path'])for r in read(O/'init-sp-inputs-v132.json')['inputs']]
    refs=read(O/'init-sp-native-frontiers-v132.json')['databases'];inputs += [Path(r['path'])for r in refs]+[Path(r['raw_database']['path'])for r in refs]
    updates={cid:dict(review_v132=dict(evidence=names,pair=['F28','F32'],relation_clause='REL132-01',positive_QA_cacheonly_dummy_separated=True,
        first_draft_init_frontiers=104,actual_SP_CP_owner_or_native_numeric_or_joint_fusion_proved=False,first_token_pre_RoPE_reuse_conditional=True,no_new_native_or_physical_gain=True))for cid in ['F28','F32']}
    persist_refinement(version=132,previous=131,title='First draft initialization, SP/CP query owners and pre-RoPE token reuse',
        summary=dict(filename='init-sp-summary-v132.json',previous_turn_classification='PROGRESS:v131F27-F32checkpointpacket/representation;PASS11067',**detail,
            refined_candidates=2,refined_modules=2,branch_records=358,semantic_modules=26,relation_clauses=139,higher_order_groups=26,pending_central_pair_reconciliation=456,
            partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='Alllocalprofile/source/math/verifierworkcomplete;no remote/device/Torch/CANN/model/replay/serviceswork.'),
        changes=['Bind104firstdraftinit/HC/RMS/SP/embeddingARfrontiers;separate69QA+KV,19positivecache-onlyKV,16positiveNonezeros.',
            'Reconcilelocal/virtualHC/embeddingSUM/SPprojectedKVwithCPqueryowner/QArouting,padding/fullnorm/coeff/residual andfirst-tokenpre-RoPEcachecosts.',
            'AddREL132-01F28-F32:139pairs/26groups/456pending;correctG10stalecountproseandretainallglobal/native/wholegates.'],
        candidate_updates=updates,module_updates={
            'M19':dict(candidates=[],finding='FirstHCinit/embeddingSUM/virtualHCandsplitY/norm/carrierneedonegraph.69QA+globalKV/19cache-only/16dummyretained;first-tokenreusekeepsHCcoeff/residualandpostlookupRoPE/cachesemantics.',evidence=names,gap=gap),
            'M23':dict(candidates=[],finding='SP/CPownerincidenceunbound;raw5120/QA1280orjoint1792deliveryandA/NQAworkmustbepricedwithembeddingAR→RS/localinit. Cacheowner-demandandglobalreplicatedlookupgraphsarenotadditivesavings.',evidence=names,gap=gap)},
        remaining=remaining,action=action,jobs=[dict(kind='profile',status='pass',description='104savedexact-keyfirstinit/HC/RMS/SP/embeddingARfrontiersand69QA/19positivecache-only/16dummybranches;raw32DB/APIs/CIDs/sourcepinsverified.',artifacts=[O/'init-sp-native-frontiers-v132.json']+inputs),
            dict(kind='design-check',status='pass',description='3816copy-mask/864owner-routecases,logicalwork/tablecontentsandcache/numericwitnesses;'+str(pre['checks'])+'preflightPASS. Scopedrelationonly,no implementation.',artifacts=[O/'init-sp-preflight-v132.json',O/'init-sp-validation-history-v132.json']+[O/n for n in names]+[O/'coverage-current-obligations-v132.json',O/'whole-scope-requirement-audit-v132.json'])],
        report=report,read_first=names+['coverage-current-obligations-v132.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_init_sp_profile_v132.py',N/'init_sp_models_v132.py',N/'analyze_init_sp_relation_v132.py',N/'verify_init_sp_relation_v132.py',N/'persist_profile_refinement_v90.py'],executor_prefix='init-sp',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
