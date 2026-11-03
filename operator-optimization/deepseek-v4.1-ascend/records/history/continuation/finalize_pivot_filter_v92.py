"""Persist monolithic-pivot operator contracts;no original kernel implementation change."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v92.json').exists()
    assert read(O/'pivot-filter-analysis-preflight-v92.json')['status']=='PASS'
    names=['pivot-filter-source-contract-v92.json','pivot-filter-branch-inventory-v92.json','pivot-filter-work-and-resource-models-v92.json',
      'pivot-filter-workspace-prefix-and-effects-v92.json','pivot-filter-math-validation-v92.json','operator-graph-G7-v92.json',
      'operator-graph-node-coverage-v92.json','operator-graph-edge-coverage-v92.json','operator-graph-candidate-bindings-v92.json','operator-graph-coverage-audit-v92.json']
    variants=[
      dict(id='V92SPLIT',state='ORDERED_FIXED_TILE_SUMMARY_SPLIT_CONDITIONAL',
        change='Parallelizeoriginal8192-elementtile computations,thenmergeeachoriginaltile count/mass/min/fuzzycounttupleinoriginalorder.',
        gates='Identicallocalvalues/reductionresults andperiteration pivots;massFP32addition/duplicateupdates retainorder. Tree-reducingfuzzy(min,count) orchangingtilewidth isnotsafe bydefault. Newstage/launch/global-readinesscosts remain.'),
      dict(id='V92EXP',state='SAME_EXPRESSION_STANDALONE_CACHE_CONDITIONAL',
        change='StorestandaloneE=exp(originalL-max_sample) duringnormalizerpass,readitinsecondpassinstead ofrecomputingExp;serialinplaceprefixcompactioncandidate.',
        gates='SameFP32Expresult/normalizer/sourceepoch,exclusivebank,loadentiretilebeforecompactwrites. Adds4BVstore;fallbackmustrebuildfullP. PostKmaskedthirdExp isnotfourthfullExp.'),
      dict(id='V92SAMPLE',state='SAME_CALL_SAMPLE_STAT_REUSE_CONDITIONAL',
        change='Reuseidenticalfirstsamplevalues/stats acrosssame-callK→standaloneP branchandavoidrequestedtile0rereadwherepossible.',
        gates='SameunchangedL/blocksize/operationresults;alreadycomputedmax/min do notremovethePExp-normalizerpass. CompilerCSEandliveUBpressureunattested. Notcrossiterationactivationreuse.'),
      dict(id='V92VIRTUAL',state='FINAL_MASK_DESCRIPTOR_AND_PREFIX_CONDITIONAL',
        change='Exposeactiveflag/pivot/duplicatelogit/quota;consumerreconstructssamefilteredvaluesfromoriginalL,insteadofwritingmaskedL. Prefixmetadataoption supportsparallelconsumerchunks.',
        gates='Exactsourcefuzzygroupandvocabprefixorder,includinggroupvaluesbelowpivot;inactiveguardreturnsoriginalL. Retainlivefilteredlogit/mutationobserversorchangeABI,sourceepoch,lastread andnativeconsumerprecision. ExtraGwork/prefix/launchcostspriced.'),
      dict(id='V92UB',state='BOUNDED_OUTLIER_BANK_ONCHIP_CONDITIONAL',
        change='Keepboundedcompactoutlierprefixonchip acrosspivotiterations ratherthanmaterializingeachrequestedBUFFERinterface.',
        gates='4O+alllivetilescratch fitsactualUB/registerschedule;dynamicOcap hasoriginalfallback. WholeVrow517120B>196608BUB;49152floatvaluesisnotausablecapacityonceworkspaceiscounted. NativeLowering/cachetrafficunproved.'),
      dict(id='V92BANK',state='EXCLUSIVE_SCRATCH_BANK_REORGANIZATION_CONDITIONAL',
        change='Usegeneration/streamownershiporisolatedscratchbanks for overlappingcallplans;keepreadonlytablesafterready.',
        gates='Cachekeydoesnotcontainstream/generation;newbanksneednotcopyoldcontentsifthecurrentproduced-prefixreadcontractholds. Reuseaddressisnotcontentidentity;actualconcurrencyunobserved.'),
      dict(id='V92NOP',state='PARAMETER_NOOP_CALL_ELISION_CONDITIONAL',
        change='Skipanentireknown-noop invocationwhenallkernelrowsareinactiveandmetadata/layoutprotocolisalreadyvalid.',
        gates='Preservewrappercontiguouscopy/returnedalias/stride,parametercasts/asserts,capturebanksandanyrequiredinitialization. ExistingkernelnoprowhaszeroL-sweeps,butwrapper/launchcostisnotzero.'),
      dict(id='V92NUMERIC',state='NEW_NUMERIC_CONTRACT_REQUIRED_OR_UNPROVED',
        change='Stableglobal/online-max normalizer,exactselection/fullsort ornewpivotwarmstartmaychangealgorithmicwork/stability.',
        gates='Source18iteration/tolerance/finalguardand1e-9fuzzygroups defineoutputs;newrange/scan/ordercanchangevalues. Heuristicsample shift hasnofinite-expguarantee. No silentdefault/backendsubstitution ornewaccuracybudgetassumed.')]
    local=[
      dict(a='V92EXP',b='V92SPLIT',state='INPLACE_READ_BEFORE_WRITE_CONFLICT_CONDITIONAL',rule='Serialforwardcompactionissafe;parallelprogramsneedallinputreadsreadybeforeoverwrites,separateEbankorotherprovedschedule.'),
      dict(a='V92VIRTUAL',b='V92SPLIT',state='PREFIX_AND_NATIVE_REDUCTION_DEPENDENCY',rule='Originaltilegroup-prefixmetadata orserialgroupstate preservesmask;consumerparallelstatsmustalso retainrequiredFP32results.'),
      dict(a='V92BANK',b='V92SPLIT',state='OWNERSHIP_AND_PER_ITERATION_HANDOFF',rule='Tiles/master sharecurrentbankepoch;latercallcannotoverwriteitbeforeallpartials/pivotreadersfinish.'),
      dict(a='V92SAMPLE',b='V92EXP',state='SHARED_SOURCE_EPOCH_NOT_ADDITIVE_FREE_WORK',rule='Reuseonlyidenticalsame-callL/stats;cacheExpadditionalstore andsamplevectorliveness mustbepricedinonecombinedgraph.'),
      dict(a='V92NUMERIC',b='V92SPLIT',state='SOURCE_NUMERIC_CONTRACT_FENCE',rule='Noordinarymonoid/tilewidthchangeassumption forfuzzycounts. Orderedtuplemergeorexactties-separateddomainrequiresitsownproof.')]
    write(O/'pivot-filter-fusion-variants-v92.json',dict(candidate='F35',variants=variants,
      already_fused=['Fullstats/outliergather','TwoKpivotsoneinputscan','Pivotmass/min/duplicatecountoneinputscan','Finalmaskinsideexistingkernel'],
      already_fused_credit_counted_new=False,new_maincandidate=False,implementation=False,native_test=False,overall_complete=False))
    write(O/'pivot-filter-local-variant-relations-v92.json',dict(rows=local,local_relations=5,central_candidate_pair_or_group_counts_increased=False,all_compositions_closed=False,overall_complete=False))
    write(O/'pivot-filter-validation-history-v92.json',dict(corrected_local_tooling_issues=[dict(kind='VerifierSyntaxError',line=71,detail='Missingwhitespaceinelse0literal;correctedbeforepreflight/sharedstatetransition.',native_work=False)],
      math_preflight_status='PASS',prior_fusion_not_counted_new=True,actual_job_running=False,new_Torch_NPU_or_server_work=False,overall_complete=False))
    rel=deepcopy(read(O/'candidate-relation-clauses-v91.json'))
    for r in rel['clauses']:
        if r['id']in ['REL73-22','REL73-26']:
            r['pivot_filter_refinement_v92']=dict(evidence=names,
              scope='Sourcefilterselection/approximate-tie/localworkspace/inputepochcontracts;nohead-private/crosscard/runtimeclosure.',
              exactsource_output_cannot_be_assumed_from_standard_TopK=True,new_pair_or_all_variants_closed=False,critical_path_gain_us=None)
    rel.update(parent_relation_index='candidate-relation-clauses-v91.json',overall_complete=False);write(O/'candidate-relation-clauses-v92.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v91.json'));cov.update(parent='candidate-relation-coverage-v91.json',parent_relation_index='candidate-relation-clauses-v92.json',
      pivot_filter_refinement_v92=dict(new_pairs=0,new_groups=0,local_variant_relations=5,evidence=names),overall_complete=False);write(O/'candidate-relation-coverage-v92.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v91.json'));index={r['id']:r for r in read(O/'candidate-record-index-v91.json')['model_candidates']}
    for r in fps['candidates']:
        q=R/'candidates'/(r['id']+'.json');actual=hashlib.sha256(q.read_bytes()).hexdigest();assert actual==index[r['id']]['master_sha256'];r['captured_master_sha256']=actual
        if r['id']=='F35':
            r['operator_atom_union']+=['sampler.ordered_fuzzy_min_count','sampler.pivot_search_epoch','sampler.virtual_duplicate_quota_mask','sampler.workspace_prefix_ownership']
            r['G7_monolithic_binding_v92']=dict(graph='operator-graph-G7-v92.json',template='UPSTREAM_MONOLITHIC_PIVOT_FILTER',node_rows=16,edge_rows=28,all_F35_primitives_or_variants_mapped=False)
    fps.update(parent='candidate-operator-footprints-v91.json',overall_complete=False);write(O/'candidate-operator-footprints-v92.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v91.json'));byid={r['id']:r for r in fps['candidates']}
    for r in nav['pairs']:r['shared_operator_atom_union']=sorted(set(byid[r['a']]['operator_atom_union'])&set(byid[r['b']]['operator_atom_union']))
    nav.update(parent='candidate-pair-navigation-v91.json',pivot_filter_refinement_v92=dict(new_pairs=0,new_groups=0),overall_complete=False);write(O/'candidate-pair-navigation-v92.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v91.json'));op['pivot_filter_refinement_v92']=dict(evidence=names,
      named_kernel_functions=4,minimal_routing_source_files=1,source_if_sites=51,source_if_arms=102,while_loops=5,for_loops=15,
      source_guard_navigation_not_all_model_coverage=True,work_models=68,local_variant_relations=5,
      ordered_cases=4096,separated_cases=1944,virtual_mask_cases=15360,serial_compaction_cases=1024,
      new_profile_regimes_or_executions=0,actual_native_backend_compile_or_flags_attested=False)
    write(O/'operator-analysis-coverage-ledger-v92.json',op)
    br=deepcopy(read(O/'M17-probability-branch-ledger-v91.json'));br['rows'].append(dict(branch='upstream_monolithic_Triton_pivot_filter',state='NAMED_BODY_AND_ORDERED_STATE_CONTRACT_REVIEWED;NATIVE_COMPILATION_UNPROVED',
      evidence=names[:5],subfamilies=8,CUDA_split_body_not_implicitly_analyzed=True));br.update(parent='M17-probability-branch-ledger-v91.json',branch_universe_exhaustive=False,whole_module_closed=False,overall_complete=False)
    write(O/'M17-probability-branch-ledger-v92.json',br)
    gap=('DeclaredNPUmonolithicroute/sourcefunctiondoesnotproveinstalledTritoncompilation,actualbatchInvariant/random/k/p/finitevalueflags orprogram→AIVmapping. '
      'Fuzzy1e-9tileupdates,18iteration/finalguard,nativeExp/sum/comparator andoriginalsourceepochmustremainorobtainnewnumericcontract. '
      'VirtualfilteredLrequiresallmutation/returnedscore/readersclosed,newbank/prefix/lifetimesandconsumerprecision. Rowparallel/Expbanksneedlayout,barriers,workspaceownershipandcostproof. '
      'Localvalue/IEEE/sourceworkmodelsdonotproveHBMtraffic,MFU,wallclockceiling orspeedup;remainingprocessors/observersandglobalprofilegapsremain.')
    matrix=read(O/'module-phase-branch-matrix-v91.json');m17=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M17'))
    for r in m17:
        if r['id']in ['M17-OP-09','M17-OP-12']:
            r['evidence']+=names;r['pivot_kernel_review_v92']=dict(monolithic_four_named_functions_reviewed=True,ordered_fuzzy_state_not_associative=True,
              existing_pivot_scan_fusions_not_new_credit=True,CUDA_split_body_or_native_compilation_closed=False);r['gap']=gap
    m17.append(dict(id='M17-OP-13',branch='Upstream monolithic pivot kernel:ordered1e-9min/countstatistics,18iteration/cardinality/finalguard,masked-vs-fullExp,implicitmask/quota andscratchprefixownership.',
      state='PINNED_KERNEL_OPERATOR_BRANCH_AND_CONDITIONAL_WORK_DOMAINS_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
      scope='Four namedfunctions,declaredA3NPUroute andsource/math models;notinstalledkernel/GPU-split/allprocessororallreaderclosure.'))
    mids=['M16','M17','M21','M22','M24'];mods={mid:dict(
      finding='Namedmonolithicpivotbody exposesalreadyfusedscans,orderedfuzzystatistics andconditionalsplit/cache/virtual-mask/workspace variants. Sourcealgorithmcostandnumericcontract differfromdefault/fullSort references;noactualNPUattributionchange.',
      evidence=names+['pivot-filter-fusion-variants-v92.json','pivot-filter-local-variant-relations-v92.json'],candidates=['F35'],gap=gap,
      profile_scope='Operatorpivot/reduction/compaction/mask andinput/workspaceepochcontracts;onlyminimalplatformroutingclauses.')for mid in mids}
    update=dict(review_v92=dict(evidence=names+['pivot-filter-fusion-variants-v92.json','pivot-filter-local-variant-relations-v92.json'],
      four_named_monolithic_functions=True,existing_fusions_not_counted_new=True,ordered_fuzzy_state_not_monoid=True,
      source_iteration_fallback_not_exact_standard_TopK_guarantee=True,Exp_input_masks_distinguished=True,
      local_variant_relations=5,G7_child_source_rows=[16,28],all_F35_primitives_mapped=False,implementation=False,native_test=False),
      conclusion_changing_gap=gap,next_profile_or_contract_action='Do notrepeatmonolithicfuzzy/tile/fallback/Exp-expression discoveries. SourceCUDA-onlysplithelpersareoutsideNPUroute;actualA3runnability/flagsunproved. Continueapplicableunclosedprocessor/reader orhigher-impactotheroperatorbranchcontracts,notframeworksurvey;keeporderedsemantics andoneworkspaceepoch perlivecall.')
    write(O/'PROFILE-CODE-QUESTION-GATE-v92.json',dict(status='NAMED_PIVOT_OPERATOR_BODY_AND_FUSION_SPLIT_QUESTIONS',
      questions=['Whichscansarealreadyfused? Pinned.', 'Canfuzzymin/count beparallel-tree-reduced? Notunconditionally;orderedtilepartials suffice.',
        'CanExpandmaskedLinterfacesbereorganized? Onlysameexpression/sourceepoch,quotaprefix andreader/numericdomains.',
        'Doesglobalworkspacecachemeanactivationreuse/concurrentbankavailability? No.'],
      cached_kernel_source_files=1,minimal_routing_source_files=1,named_functions_pinned=4,source_if_sites=51,source_if_arms=102,
      new_target_source_files=0,new_public_document_read=False,new_remote_source_read=False,no_framework_wide_research=True,
      no_Torch_model_NPU_capture_or_server_work=True,overall_complete=False))
    write(O/'pivot-filter-question-status-v92.json',dict(newly_resolved=[
      'Fourmonolithicfunctions,allfive18-capsearchloopsanddifferentrequestedO/Vrangesnowexplicit.',
      'Fuzzymin/count isnoncommutative/nonassociative;changingtileboundariesalterscounts. Originalorderedtilesummaries andseparated-exacttie domainsverified.',
      'Duplicatequotafinalmaskcanbeexpressedwithactiveflagandoriginal-vocabnear-group prefix;privateABIandconsumerprecisionremainconditional.',
      'StandaloneExp repetitionissameexpression;postKthird/fourthmasked/unmaskedExpisnot.',
      'Workspaceproducedprefixpermitsnewuninitializedisolatedbanks;cachedaddressdoesnotgrantconcurrentownership/contentidentity.'],
      already_fused_not_new=['Stats/outlierfirstpass','TwoKpivotsperload','Pivotmass/min/countperload','Finalmaskexistingkernel'],
      unresolved=[gap,'Allrelevantgenericprocessor/logprob/readerprimitivebranches stillnotclosed.'],whole_modules_closed=0,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v91.json')['remaining'])
    remaining.append('v92:1kernel+minimalNPUroutefile,4namedfunctions51if102arms5while15for;NPUdeclared8192/4096tiles8warp-param notAIVcount. Existingstats/twopivots/mass-min-countscansalreadyfused. 1e-9fuzzymincountnoncommutative/nonassoc/tile-sensitive;4096ordered1944separatedcases,15360virtualquota1024serialcompactioncases.18-capKscalarreferencek2/zeros+one1keeps1,constantguardkeepsall—notnativeexactKguarantee. Standaloneheuristicshiftmayoverflow;third/fourthpostKExpmaskdiff forbidsnaiveCSE.68conditionalrequested-workmodelsnotHBM/physicaltime,48AIVvsBprogrammappingseparate.8localvariants5relationsnocentralcountincrease. Scratchprefix/newbank/virtualmask sourceepochgates explicit. G7child16/28currentpartial724/1187;M17OP13=>354records24semantic.115/24/480 and2073/285598review9/1352pending849/16/3136 unchanged0wholeclosed.')
    report='''本轮只围绕上游monolithic Triton过滤kernel的算子问题，核对四个函数及最小NPU路由条款；没有导入Torch、编译kernel或访问服务器。当前profile没有提供此随机过滤路径的执行/参数正证，源码合同与实际A3可运行性保持分开。

先扣除已经实现的融合：首次全词表扫描已合并max/min、有限项计数和outlier压缩；两个top-k pivot共用一次候选读取；pivot的质量和最小值/重复数也共用一次扫描；最终mask已在同一个kernel内。不能再将这些记为新收益。一个kernel仍可能包含多轮依赖扫描：五处while均有18轮或容差退出，搜索范围按分支为outlier前缀O或完整V。

重要的新限制是1e-9近似同值统计。helper对“新最小值”使用严格比较，对“同值计数”使用abs差<1e-9，且遇到更小值时会重置计数。设epsilon=1e-9，按0再epsilon/2合并得到计数2，反序得到1；0、epsilon/2、epsilon/5按不同括号分组得到3与2。这不是可任意树形归约的min/count幺半群。相同数据跨不同tile边界也会改变结果，因此不能把8192改为4096或拆行后无序合并当作数值等价。

可以保留原8192逻辑tile，让各program先生成同样的tile局部count/mass/min/near-count，再由小合并阶段按原顺序更新。整数计数、FP32质量累加和每轮pivot依赖均需保持其合同。V=129280有16个原tile；两pivot top-k局部摘要每行每轮384B，单pivot top-p为192B，另计写回和读取。它可能增加行内并行度，但每轮新增handoff/launch和原生局部归约结果仍未定，不能按program数量推定提速。

这段kernel也不能简单视作标准精确TopK。源码计数的标量模型中，一行前面为0、最后为1，k=2时，18轮回退后正pivot仅保留1项；常量行的pivot达到max则最终guard不做mask、保留全部。该例子使用同样的分支、计数及退出公式，不是已安装CANN kernel的实测结果。它足以说明不能跳过迭代回退、重复项和最终guard，直接套用精确排序合同。v91的Sort32服务界也不适用于这个没有full-sort的pivot算法。

Exp复用须按操作数判断。standalone top-p的前两次全行扫描确实计算相同exp(originalL−max_sample)，可研究第一遍存E、第二遍读E，代价是增加4BV逻辑存储、减少BV次Exp。顺序扫描时，整个tile先读完再向前压缩，可原地覆盖；并行tile可能覆盖其他program尚未读取的E，需要屏障或独立bank。fallback仍须重建完整P。post-k分支第三遍的Exp来自选择后的masked logits，第四遍却来自原buffer logits；它们不是可无条件CSE的同一表达式。

standalone归一化shift取首块mean+10std，并非全行max。首8192项为0而后部有1000时，该shift不能保证FP32 Exp有限。在线稳定max或全局max方案可能改善数值范围，但会改变现有舍入和18轮搜索轨迹，必须作为新的数值合同分析，不能偷偷替换。复用同一调用的sample/max/min也不能消掉仍需计算Exp分母的扫描，且当前编译器是否已消除部分重复load尚未取证。

另一种重组是虚拟最终mask：输出active标志、pivot、duplicate logit和quota，让后续consumer从同一原始L生成完全相同的过滤值。近似重复组须按原词表序号计数，包括未通过严格pivot的同组值；inactive guard必须返回原L，不能把原-inf改为其他mask值。该谓词通过15360组模型对照。若filtered logits或原地修改仍有读者，就必须保留原输出或调整ABI；新的softmax/argmax消费者也须另证原生数值。

可选prefix bank每行17个uint32条目，帮助后续分块定位重复项配额。descriptor-only方案可少一次最终全行读写接口，但要重算部分group/prefix；保留一次prefix生产扫描的方案仅省masked输出存储的一部分。描述符、prefix、多次consumer读取、额外比较、生命周期和启动均另计，不能把两方案额度直接叠加。

缓存workspace按(device,dtype,V)索引，没有stream或调用generation。相同地址证明分配复用，不能证明激活内容可重用或允许并发覆盖。源码六种BUFFER分支只读取本调用已写的O/O2/V前缀；因此隔离新bank不需要复制旧内容，但必须保留前缀、行与last-read协议。B=2/4、num_sm≥B的声明分配中，单bank为1034240/2068480B；并发新bank和readonly表的ready状态均需计价。

成本共68个声明场景，分别记录O/V范围、1或18轮、standalone是否重建P、源请求load/store和Exp/Div工作；不把它们当作实际HBM流量。每行完整FLOAT workspace为517120B，大于单AIV 196608B UB，仅排除整行驻留方案；小O仍可研究有上限的片上bank，需为其他活跃向量留空间。源码每行一个逻辑program不等于已证明每program一个AIV，48-AIV峰值与B-program条件界分别保留，也没有观测频率缩放。

已验证4096组原tile有序摘要、1944组充分分隔/精确同值域、15360组虚拟mask与1024组顺序原地压缩。它们是源语义/数学模型检查，没有原生性能结论。八项条件变体及五条局部组合关系已记录，中央候选pair/group数量没有因内部变体而增加。

G7新增16行monolithic子图节点、28行边，当前部分图724/1187行与版本绑定；这不是唯一算子或全分支覆盖率。累计354条分支记录、24个语义模块；115条条件关系、24组联合合同，中央仍有480对待汇入或复核。2073个regime/285598次执行已作条件分析，9个regime/1352次仍待补；849条PMU缺失、16张阶段未绑定图、3136条模块归属歧义不变。整体任务继续，没有整模块关闭。
'''
    action=('继续原Goal，v92PROGRESS，v91亦PROGRESS。读pivot-filter-summary/PROFILE-FINDINGS-v92、source-contract/branch-inventory/work-and-resource/workspace-prefix/math-validation/fusion-variants/local-variant-relations/question-status及G7v92/node/edge/bindings/audit。'
      '已定点读4monolithicfunctions+NPUPlatform最小clauses：51if102arms5while15for只是localnavigation。NPU声明OOT/devicenpu/isCudaAlikeFalse，wrapperCUDA split不作A3执行，actualinstalledTriton/runtimemode未证。8192/4096tile8warp-param，warp不是AIV。'
      '现有fullstats/outlier、twoKpivotsscan、mass/min/countscan、finalmask已fused不能重记。关键新fuzzyhelperstrictnewmin+absdiff<1e-9same，非commutative/associative；原tile尺寸/顺序会影响计数。4096orderedtuple1944separatedcases，orderedfixedoriginaltilepartials才可保状态；native局部FP32结果/质量sum/新launch未证。'
      '18cap/guard不保证standardexactK：scalar源模型V129280zeros+last1,k2 keeps1，constant keepsall；不是实际NPU记录。Standalonemax_sample首8192mean+10std非globalmax，后部outlier可令Exp非finite。改stablemax/exactselection/warmstart是新numericcontract，不自动合法。'
      'Standalone前两pass同Exp可cache：多4BVstore少BVExp，serialforwardcompact安全；parallel需独立bank/barrier，fallback仍重建P。PostK第三maskedExp和第四fullbufferExp不同不能CSE。Same-call sample/firstTile复用要原inputepoch；编译器是否已CSE未知。'
      'Virtualfinalmask用activeflag+sourcepivot/dup/quota和原vocabnear-group prefix，15360cases证predicate；inactive保持originalL。所有filteredL/mutation/readers+nativeconsumer精度要证。Prefix17u32/row替代方案与descriptor-only去final8V是不同预算，不相加。1024serialcompactioncasesPASS。'
      'Workspacekeydevice/dtype/V不含stream/generation，allocationreuse非activationreuse或并发许可；6prefixdomain说明新隔离bank可uninitialized不必copyoldcontents，保ownership/readiness/lastread。68conditional源工作模型含O/V和1/18迭代不是HBM/physicaltime，Sort32界不适用pivot算法。WholeFLOATrow517120>UB196608仅排wholeplan，boundedO+scratch片上仍可。逻辑programsB2/4与AIV映射未证。'
      '8internalvariants5localrelations不增centralpairs，G7child16/28 currentpartial724/1187；M17OP13=>354records24semantic；35model7legacy0wholeclosed。115/24/480，2073/285598review9/1352pending849PMU16phase3136ambiguity不变。'
      '下一步离开已核实monolithic事实，优先仍适用的directprocessor/logprob/readereffects或其他高影响operator未覆盖分支；不要读CUDA-onlysplit全体来替代A3问题，不重复旧核函数fusions/新本文反例当发现，不盲增pairs，不研究无关framework。仅offline savedprofile/frozenpinpoint/math/必要primary，无server/model/Torch/NPU/capture/service/upload。GoalACTIVE/TaskACTIVE/LoopEVALUATING；verify_pivot_filter_v92须PASS，不重跑finalizer不提前complete。')
    sc=read(O/names[0]);oldgraphs=[Path(p)for k,p in read(O/names[-1])['graph_catalog'].items()if k!='G7']
    jobs=[dict(kind='review',status='pass',description='Pin4monolithicoperatorfunctions/minimalNPUroute;51if102arms5searchloops,existingfusions andorderedfuzzy/fallback/Exp-input/workspace contracts. G7child16/28,currentpartial724/1187.',
      artifacts=[O/n for n in names[:2]+names[3:4]+names[5:]]+[Path(sc['path']),Path(sc['NPU_route']['path'])]+oldgraphs+[R/'fusion-scan.json']),
      dict(kind='design-check',status='pass',description='4096ordered1944separated15360virtualmask1024serialcompactionmodels;68sourceworkcases,8variants5localrelations. NoTorch/NPU/nativeprecision/speedup proof.',
        artifacts=[O/names[2],O/names[4],O/'pivot-filter-analysis-preflight-v92.json',O/'pivot-filter-fusion-variants-v92.json',O/'pivot-filter-local-variant-relations-v92.json',O/'pivot-filter-validation-history-v92.json',O/'typed-capacity-primary-contract-v89.json'])]
    persist_refinement(version=92,previous=91,title='Ordered pivot statistics,implicit filtered values and scratch ownership',
      summary=dict(filename='pivot-filter-summary-v92.json',previous_turn_classification='PROGRESS:v91defaultfilterrace',
        cached_kernel_source_files=1,minimal_routing_source_files=1,new_target_source_files=0,new_public_primary_documents=0,named_functions_pinned=4,
        local_source_if_sites=51,local_source_if_arms=102,while_loops=5,for_loops=15,ordered_tile_cases=4096,separated_exacttie_cases=1944,virtual_mask_cases=15360,serial_compaction_cases=1024,
        source_work_models=68,conditional_variants=8,local_variant_relations=5,new_G7_node_rows=16,new_G7_edge_rows=28,finite_graph_node_rows=724,finite_graph_edge_rows=1187,
        all_F35_primitives_or_readers_mapped=False,refined_candidates=1,refined_modules=5,new_inventory_families=1,branch_records=354,semantic_modules=24,
        new_pairs=0,new_groups=0,relation_clauses=115,higher_order_groups=24,pending_central_pair_reconciliation=480,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_regimes_or_executions_promoted=0,
        actual_A3_Triton_compilation_flags_mapping_or_speedup_proved=False,implementation=False,native_test=False,
        remote_job_status='Alllocal source/math/graphchecks finished;no server/model/Torch/NPU/capture/servicework.'),
      changes=['Pinmonolithicpivotoperator;alreadyfusedscansnotnewcredit,orderedfuzzycount/tile/fallback contracts explicit.',
        'Verifyorderedsummary/virtualmask/serialcompaction models anddecompose68conditionalworkcases;8variants5localrelations.',
        'ExpandG7child16/28,currentpartial724/1187;354branchrecords24semanticwithallnative/globalgapsretained.'],
      candidate_updates={'F35':update},module_updates=mods,semantic_branches={'M17':m17},remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['pivot-filter-fusion-variants-v92.json','pivot-filter-local-variant-relations-v92.json','pivot-filter-question-status-v92.json'],
      publication_rows={'F35':'| F35 | 目标采样行选择、概率过滤、race及输出重组 | 条件 | 默认原P阈值与上游pivot合同分开。Pivot已有多项scan融合，1e-9近似计数非普通并行归约；可条件保原tile有序拆分、复用同表达式Exp、虚拟最终mask及隔离scratchbank。原生精度、读者/生命周期与收益仍待证。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/'analyze_pivot_filter_v92.py',N/'verify_pivot_filter_v92.py',N/'persist_profile_refinement_v90.py'],
      executor_prefix='pivot-filter',main_script=Path(__file__),source_read=False,
      additional_shared_writes={R/'fusion-scan.json':read(O/'fusion-scan-proposed-v92.json')})


if __name__=='__main__':main()
