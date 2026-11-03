"""Persist operator-body fallback coverage, backend/precision counterexamples and G8 finite graphs."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v101.json').exists()
    assert read(O/'recovery-fallback-analysis-preflight-v101.json')['status']=='PASS'
    names=['recovery-fallback-inputs-v101.json','recovery-fallback-source-and-applicability-v101.json','recovery-backend-nonidentity-v101.json',
      'recovery-owner-plan-and-state-reuse-v101.json','recovery-prefix-scan-and-demand-v101.json','recovery-fallback-work-and-interface-models-v101.json',
      'operator-graph-G8-v101.json','operator-graph-node-coverage-v101.json','operator-graph-edge-coverage-v101.json',
      'operator-graph-candidate-bindings-v101.json','operator-graph-coverage-audit-v101.json']
    gap=('NamedPyTorchrecovery/randomverificationbodiesareconditional-sourceanalyzed;currentprofilehas72greedychainsandnoexactrandom/recoveryTASKnames,notcurrentfallbackexecutionproof. '
      'Runtimeflags/backend/rows/cu/validIDs/dtypes/globalvocab/aliases/allobserversandRNG/bankepochsremainunbound. '
      'Preservebackend-specificrecursivecapvsratio-capcumprod,FLOAT64vsFLOAT32uniformcomparison,optionalpi-weightedrecoveryandbadQ/NaN/tail/firsttiecontracts. '
      'RealpairscanisnotIEEEreassociation-equivalent;sourcework/privateplanevaluebudgetsarenotnativeissuedwork,H B Mtraffic,livepeakoractualspeedup. '
      'Noall-moduleMFU,wholephysicalnumericceilingorwhole-goalclosure.')
    evidence=names[:6]+[names[6]]
    rel=deepcopy(read(O/'candidate-relation-clauses-v100.json'))
    row=next(r for r in rel['clauses']if r['id']=='REL73-26');row['recovery_fallback_refinement_v101']=dict(evidence=evidence,
      contract='Draftcu→token/requestownerplanmaybesharedwithmatchingmetadataepochandnewselected-recoveryconsumer;notattentionseqlensidentity. PyTorchordinary/block/optionalweightedrecoveryremainbackend-specific. Thisextendsanexistingpair,nounconditionalproducerbindingornewpair.',all_runtime_gates_proved=False)
    group=next(g for g in rel['higher_order_groups']if g['id']=='HG73-TAIL');group['recovery_fallback_refinement_v101']=dict(evidence=evidence,
      contract='Keepworking/rawPreaders,alloriginalRNGdraws,exactfallbackdecision/entropy/quotient/castandoutputwrite-setsemantics. Mergeprivatevalueplanesorprunedemandonlyforonechosenrowmap/backendgraph.',whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v100.json',new_pair_clauses=0,new_higher_order_groups=0,overall_complete=False);write(O/'candidate-relation-clauses-v101.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v100.json'));cov.update(parent='candidate-relation-coverage-v100.json',parent_relation_index='candidate-relation-clauses-v101.json',
      recovery_fallback_refinement_v101=dict(evidence=evidence,new_pairs=0,new_groups=0,new_native_promotions=0),overall_complete=False);write(O/'candidate-relation-coverage-v101.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v100.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v100.json')['model_candidates']}
    for r in fps['candidates']:
        p=R/'candidates'/(r['id']+'.json');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256'];r['captured_master_sha256']=h;r['captured_master_version']=100
        if r['id']in ['F31','F35']:r['G8_recovery_fallback_v101']=dict(templates=5,evidence=names,cu_source_reader_not_same_native_producer_proof=True,new_observed_operator_atoms=0)
    fps.update(parent='candidate-operator-footprints-v100.json',overall_complete=False);write(O/'candidate-operator-footprints-v101.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v100.json'));nr=next(r for r in nav['pairs']if{r['a'],r['b']}=={'F31','F35'})
    nr['G8_recovery_owner_plan_refinement_v101']=dict(evidence=evidence,existing_relation='REL73-26',no_native_cu_producer_or_alias_binding=True)
    nav.update(parent='candidate-pair-navigation-v100.json',parent_footprints='candidate-operator-footprints-v101.json',parent_relation_index='candidate-relation-clauses-v101.json',overall_complete=False);write(O/'candidate-pair-navigation-v101.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v100.json'));op['recovery_fallback_refinement_v101']=dict(conditional_source_templates=5,new_native_regimes=0,new_native_executions=0,new_native_attribution_or_promotions=0,
      source_profile_scope='72greedychains;exactrandom/recoverynames0;noabsence-to-impossibilityinference.',source_graph_rows=[70,88],operator_graph_counts=[813,1297])
    write(O/'operator-analysis-coverage-ledger-v101.json',op)
    matrix=read(O/'module-phase-branch-matrix-v100.json');branches=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M17'))
    for r in branches:
        if r['id']in ['M17-OP-04','M17-OP-07','M17-OP-09','M17-OP-12']:
            r['recovery_fallback_refinement_v101']=dict(evidence=evidence,local_named_source_bodies_reviewed=True,all_runtime_flags_or_branches_closed=False)
    additions=[
      ('M17-OP-14','PyTorchordinaryrecovery:cuownerplan,NGRAMclone/excludeorpositiveP-D,Qgather/twoexception-safeplanes,guardedDiv/Argmax;right-biasedorinterval-ownerplansandtilefusion.',
        'Mapidentitypreserveszero-draftduplicatesandTtail→req0underorderedcu. SixFLOATprivateplaneinterfacescanbefusedonlywithoriginalP/Q/Dobservers,badQ/NaN/tie/dtype/errorandnewtailmaskcontracts.'),
      ('M17-OP-15','PyTorchordinary/blockrandomverification:precisedraft/prob/uniform/entropyreadsets,firstfailureversuslastlegal,prefix/bonus/old-outputwrites;backend-specificcap/cumprodanduniformdtypes.',
        'ExactcounterexamplesseparateTritonrecursivecapfromPyTorchratio-capcumprodandF32/F64drawcomparisons. RealmonoidscanexistsbutIEEEunderflowcounterblocksdrop-inreassociation. Currentruntimebackend/flagsunknown.'),
      ('M17-OP-16','OptionalPyTorchblockwiserecovery:prepositionp_i-weightedresidual,token/request/positionmap,max(n).itemandprefixloop;staticSboundalternativeandnewselected-onlyconsumer.',
        'Rootcalldoesnotpassuse_block_verify;ordinaryrecoveryremainsthedefaultdespitesubsequentblockverification. Weightedresidualdiffersfromstandard;staticS>=max(n),validT/cu/pos/error/RNGandreadercontractsrequired.')]
    for cid,label,finding in additions:
        assert not any(r['id']==cid for r in branches);branches.append(dict(id=cid,branch=label,finding=finding,state='CONDITIONAL_SOURCE_BODY_WORK_IO_AND_ALGEBRA_REVIEWED;RUNTIME_BINDING_OPEN',
          evidence=evidence,gap=gap,scope='Namedfallbackoperatorbodies/branchesonly;notnewobservedTASKsorfullF35/modelclosure.'))
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v100.json'));audit.update(previous_goal_turn='PROGRESS:v100headwiseattention/outputstateandcostcontracts;262postchecks',input_checkpoint=100,followup_checkpoint=101)
    ga=read(O/'operator-graph-coverage-audit-v101.json')
    for r in audit['requirements']:
        if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES':r['branch_records']=362
        if r['id']=='CURRENT_FINITE_GRAPH_LEDGER':r.update(evidence=['operator-graph-coverage-audit-v101.json'],rows=[813,1297]);r['gap']='FiveG8fallbacksource-operatorgraphsadded;allconnectedsets2..4remainstructural. Actualkernel/backend/numerical/readerandunmappedprimitivebranchdomainsstillopen.'
        if r['id']=='THEORY_AND_MEASURED_COMPARISON':r['evidence']+=names
    audit['recovery_fallback_refinement_v101']=dict(named_fallback_bodies_reviewed=True,backend_equivalence_not_assumed=True,
      conditional_source_graph_rows=[70,88],runtime_new_executions=0,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v101.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v101.json',dict(status='SAVED_GREEDY_ONLY_PROFILE_THEN_PINPOINT_UNEXPANDED_RANDOM_RECOVERY_OPERATOR_BODIES',
      questions=['Whichnewoperatorwork/ioisinsidepreviouslyopaquefallbackbodies?',
        'ArebackendblockandRNGcomparisonformulasactuallyequivalent?',
        'CanT×Browownership/privateV-wideplanesbereplacedwhilepreservingcu/ID/Q/precision/errorcontracts?',
        'Whichoptionalweightedrecovery/staticS/prefixscanbranchesareseparatealgorithms?'],
      cached_source_files=3,new_target_source_files=0,new_public_documents=0,no_framework_wide_research=True,no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    write(O/'recovery-fallback-validation-history-v101.json',dict(refinements=[
      'v100PROGRESSrevalidated262checks;profileexactrandom/recoverynamematches0doesnotprovecase-wideimpossibilityoractivatefallback. Noexistingoperatorregimepromotion.',
      'Ascendkernelisnotupstreamrecoveryreciprocal-multiply/FP64path. Onlyupstreamgenerate_uniform_probsisimportedhere;itsFLOAT64isboundwiththehistoricalsourceSHA.',
      'Blockcomparisoncounteruses3validpositions,maxspec>=3,finitepositiveP/Dandsynthetic/entropyoff;PyTorchlast=-1versusTritonlast=2inexactandroundedoperandmodels.',
      'Ownersearchmustberight-biasedandkeeptailfallback0;intervalfillrequiresnonnegativeorderedcu. 5456finitecasessupportidentity,includingzero-lengthandcapacitytails,notallbad-metadataerrorparity.',
      'Realpairscanassociative125cases;binary32underflowchangesfinalpi0vs2^-100andacceptance. Currentorderedrecurrenceisnotallalgorithmspanproof.',
      'SixprivateFLOATplanesareadeclaredmaterializedsourcebaseline,notmeasuredHBMorsummedliveallocations. Fusion/lazycreditsareoneunion;alloriginalRNG/fullP/rawreadersremain.',
      'Analysis/verifierlocalPythonconditionalsyntaxspacingwasfixedbeforeexecution;nocheckpointorprotectedtargetwasmodifiedinthosefailedparseattempts.',
      'G8adds70nodes88edgesinfiveconditionaltemplates;connectedsets2..4checkedindependently,notsemantic/nativefullcoverage. NewF31/F35structuralhintalreadyhasREL73-26,no newcentralpaircount.'],
      all_local_jobs_finished=True,overall_complete=False))
    updates={cid:dict(review_v101=dict(evidence=names,named_fallback_operator_templates=5,
      new_graph_source_rows=[70,88],local_backend_nonidentity_and_IEEE_counterexamples=True,map_identity_cases=5456,real_monoid_cases=125,
      private_interface_models=80,new_relation_ids=[],refined_existing_relation='REL73-26',implementation=False,native_test=False,new_runtime_executions=0),conclusion_changing_gap=gap)for cid in ['F31','F35']}
    mods={'M17':dict(candidates=['F31','F35'],finding='PreviouslyopaquePyTorchrecovery/ordinary-blockverificationbodiesarenowfiveoperatorgraphs. Map/grid/privateFLOAT/guardedargmaxandoutputreadsetspriced;backendclip/precision/weightedrecoverydistinguished. 3newsemanticbranchrows,nativefallbackexecutionstillunbound.',evidence=names,gap=gap),
      'M21':dict(candidates=['F31','F35'],finding='Draftcurow-planconsumeridentifiedinrecovery;right-biasedownersearch/disjointfillorreuseconditionalonsamecu/row/stride/dtype/epochandTtailfallback0. Noattentionseqlens/cudraftproduceridentityinferred.',evidence=[names[1],names[3],names[6]],gap=gap)}
    remaining=list(read(O/'module-coverage-gaps-v100.json')['remaining']);remaining.append('v101: NamedPyTorchfallbackrecovery/randomverificationbodygaprefined,notruntimeactivation. Exactbackendrecursivecapvsratio-cappedcumprodandF64/F32uniformcounterexamples;optionalweightedrecoverynotrootdefault. 5456cuowner-mapcases/125realmonoidcases/IEEEunderflowfalsifier/80privateplaneinterfacecases. FiveG8sourcegraphs70nodes88edges;partialcatalog813/1297/46hints;362branchrecords24semantic. 131pairs25groups464pendingunchanged,2073/285598conditional9/1352pending849PMU16phase3136uncertainty0wholeclosed/noallmodulelowestMFU. Actualvalidcu/IDs/V/backend/flags/allP-RNG-readers/epochsandnativearithmetictype/capacity/alias/capturegatesremain;do notborrowupstreamkernelformulas,removeRNGdrawsorconvertsourceerrorsasoptimization.')
    report='''本轮补的是此前只有接口合同、尚未展开算子步骤的随机拒绝采样PyTorch回退分支。先复用已有profile证据：72条greedy链可见；原始保存的精确random/recovery名称查询为0。本轮因此是条件源码分析，没有把回退分支写成当前负载已执行，也没有新采集、设备测试或目标代码修改。

新增G8的5张局部算子图，覆盖恢复采样的token→request映射、普通恢复值计算、普通随机验证、block随机验证，以及可选blockwise恢复。共70个源图节点、88条边；这些是source边界行数，不是新增原生算子。源码只取对应算子函数、调用guard及导入的随机数生成器，没有展开框架研究。

最重要的结论是后端不能按同名算子直接互换。Triton block逐步计算pi=min(pi·p/q,1)，PyTorch回退先把每个p/q截到1，再做cumprod。取3个位置的p/q=[1/2,2,1]，uniform=[9/10,4/5,4/5]，Triton后续位置可重新通过，最终接受到第2个位置并加bonus；PyTorch前缀质量一直为1/2，没有合法位置，会读恢复行0。这是有限正数上的公式差别，不是浮点小误差，也没有断言当前C8实际触发。

普通验证还有比较精度边界：导入的uniform生成器返回FLOAT64；Triton调用显式转FLOAT32，PyTorch回退直接接收原值。p/q=1/2、uniform64=1/2+2^-26时，原值比较拒绝，转FLOAT32后uniform变为1/2，比较接受。融合需要保留所选后端的路径。

可选blockwise恢复也不是普通max(P−D,0)。它使用max(p_before_position·P−D,0)；一个同Q反例的两个恢复ID分别为0和1。读取的root调用没有传use_block_verify，默认仍False，因此不能仅因后续启用block验证，就把这个可选恢复公式套进去。

有三类具体优化机会。第一，当前恢复映射用T×B区间比较、BOOL→INT32、argmax/any/where找request。对于非负、有序cu，可以改为右偏upper_bound；零长度请求使相同前缀出现重复值，左偏搜索会错。也可以按互不重叠的request区间写入owner/position，或由融合kernel的request/position网格直接寻址。容量尾部t≥cu[-1]原本映射request0，通用helper不能无条件跳过。5456组有限用例覆盖了零长度和容量尾部；无序cu给出了反例，必须声明或验证适用域。

第二，普通PyTorch恢复有private clone/残差、Q gather、两份Q safe中间值、除法结果和异常guard结果等可融合接口。按6个FLOAT32临时平面、各写读一次的独立物化参考，预算为48·T·V字节；例如B=2、S=5、V=129280时是62054400B。这个数不是实测HBM、分配峰值或必需流量，编译器也可能已消掉部分平面。新tile内核可按原顺序算残差/排除draft列、保留Q异常guard后直接归约argmax，避免物化这些私有平面，但不能删除原P的接受判断、raw/logprob/entropy等其他读者。

新padding还必须单独处理：PyTorch只有实际V列。若全部有效列因Q=0/Inf被置为−1e10，而新kernel把额外lane填0，padding会错误获胜；应按对应有限argmax合同排除无效lane。NaN、dtype、first-tie、原除法舍入和NGRAM/概率残差分支仍需保持。不能借用另一个upstream kernel的reciprocal-multiply或tail行为来代替Ascend当前实现。

第三，已有v33的decision→selected recovery合同可用于这些回退body；它不是本轮新发现。需要真正改消费者的读取集合：当前PyTorch验证先gather所有recovered值，之后才按first/last决策选用。保留原uniform/Q默认及seeded draw顺序的基础上，可以只算最后需要的恢复行，并与临时平面融合按一个基线计费；不能叠加两份覆盖同一临时量的收益。普通验证的私有entropy可以按有效非greedy访问前缀按需计算，block验证则一般仍需全部相关位置。减少词表工作也可能增加决策阶段和损失并行度，未声称实测更快。

block递推还存在一个理论重组方向：精确实数下f_r(x)=min(rx,1)可用(a,b)表示min(ax,b)，函数复合(a2a1,min(a2b1,b2))满足结合律，可做并行prefix scan。但它不是当前FLOAT32递推的直接替换。有限正数P=[2^-100,2^-100,1]、D=[1,1,2^-100]时，原顺序在第二步下溢为0，后续仍0；重新结合可以保留2^-100并改变最后接受判断。125组实数复合用例与该IEEE反例分别保存。源码S步循环也不能因此宣称是所有合法算法的硬件下界；精确组合舍入函数是另一种有成本的表示域。

可选blockwise helper的max(n).item()还可以登记静态S≥max(n)的新接口候选，代价是更多无效prefix槽位；必须保持有效T/cu/position、零长度和错误行为。它没有被计作当前root正在支付的同步开销。

理论部分保留SYMBOLIC_CEILING：在无缓存winner或额外结构的独立有限score比较模型中，K个完整词表行最坏需要KV次score观察、K(V−1)次比较决定；这不是融合后强制GM/HBM读取次数或每列Div次数。实际LOG/EXP/DIV/归约与存储层容量仍需同口径上界，不能借BF16矩阵峰值代算。

台账新增3条M17语义记录，并细化既有F31/F35关系，没有新增中央pair或原生执行。当前为362条分支记录、24个语义模块；局部源图813节点行/1297边行，46条结构提示；中央131条条件关系、25组联合合同，464对待汇入或复核。2073个regime/285598次执行条件分析，9个regime/1352次待关键字段，849 PMU/16阶段图/3136归属歧义未变。整体融合覆盖、最低模块MFU和物理理论上限仍未完成。
'''
    action=('继续原Goal,v101PROGRESS;v100PASS262checks。用户operatorfusion/split/reorg/theoreticalceilings,profilefirstthenpointsource,noframeworksurvey。'
      '先读recovery-fallback-summary-v101/PROFILE-FINDINGS/source/applicability/backendnonidentity/ownerplan/prefixscan/workinterfaces/G8graphaudit/wholeaudit。'
      'Saved72greedychains/exactrandomrecoverynames0;newworkconditionalfallbacksourceNOTnewruntimeexecution。'
      'AscendTritoncaprecurrencepi=min(pi*p/q,1)≠PyTorchcumprod(clamp(eachp/q,max1));S3ratio[.5,2,1],u[.9,.8,.8]giveslast2vs-1. UpstreamonlyuniformgeneratorimportedF64,AscendTritoncastsF32vsPyTorchbareF64counteru=.5+2^-26. Optionalblockwiserecoveryusespi_before*P-D,androotdoesnotpassuse_block_verify,donotactivatefromblockflagalone。'
      '5456owner-mapcasesvalidnonnegativeorderedcu:RIGHTupper_bound+tailowner0orO(T+B)disjointintervalfill. Nonmonotonecu/negativeIDs/publicTtail/zero/error/stride/dtype/generationguardsremain. NewperreqgridcanavoidmapbutTtailnotfreeprune。'
      '80logicalstandalone6privateFLOATplanevsfused/selectedRgraphs:48TVbudgetnotHBM/livepeak,alloriginalP/RNGreaders/drawsremain. Newpaddedlanes-infvsactualPyTorchbadQ-1e10,NaN/tie/Divprecisionneeded. v33lazyrecovery/RNGanswerreusednotnew;consumeroriginallyeagervectorRecoveredgatherneedsnewreadsetorvalidunusedslots。'
      'Realcappedmultiplicativefunctionmonoid125casesallowsparallelprefixdomain,butIEEEunderflowP[2^-100,2^-100,1]/D[1,1,2^-100]seq0vsreassoc2^-100 changeslast-accept. NotdropinFP32orallalgoSspanLB. EntropyordinaryactiveprefixvsblockallpositionandstaticSoptionalhelpertradeoffscopepreserved。'
      'G8fivefallbacktemplates70nodes88edges=>partialcatalog813/1297/46hints;3newM17OP14/15/16=>362records24semantic. ExistingREL73-26/HG73TAILrefined,no newpair/group:131/25/464;35model+7legacy,2073/285598conditional9/1352pending849PMU16phase3136ambiguousunchanged0wholeclosed/noallmodulelowestMFU。'
      'Nextprioritizeunexpandedoperatorconditionaldomainsorcriticalnativeevidenceandcurrentloadapplicability,not464arbitraryemptycentralpairs. Do notrepeatresolvedfallbackbody/dtype/capformula/ownersearch;sourceprofilebranchreadiness/modelphysicalratesremainunknown. M00/M03onlyoperatorselection/readinessscope,noframeworkAPIcatalog。'
      'Alllocaljobsfinished,no server/model/Torch/NPU/capture/service/implementation. GoalACTIVE,TaskACTIVE,LoopEVALUATING;verify_recovery_fallback_v101mustPASS,neverreruncompletedfinalizer。')
    fusion=deepcopy(read(R/'fusion-scan.json'));fusion['per_node_edge_coverage_ledger']=dict(summary=str(O/names[-1]),nodes=str(O/names[7]),edges=str(O/names[8]),graph_catalog=ga['graph_catalog'],structural_only=True,all_native_or_semantic_variants_complete=False)
    fusion['operator_coverage_v101_scope']='FiveconditionalPyTorchrecovery/verificationoperator-bodytemplatesandbackendprecisionguards;notobservedfallbackexecutionorallF35branches.'
    inputs=[Path(x['path'])for x in read(O/names[0])['inputs']];sources=[Path(x['path'])for x in read(O/names[1])['parts']]
    jobs=[dict(kind='profile',status='pass',description='Reuse72greedy/exactrandom-recoverynameabsenceevidence;pin3sourcefilesandfiveconditionaloperatorfallbacktemplates,noruntimeexecutionpromotion.',
      artifacts=[O/n for n in names]+inputs+sources),dict(kind='design-check',status='pass',description='5456ownerplan/125realmonoid/IEEEandbackendcounterexamples,80private-interfacegraphs;G8connected2..4setscheckedindependently,3semanticrowsnotwholeclosure.',
        artifacts=[O/'recovery-fallback-analysis-preflight-v101.json',O/'recovery-fallback-validation-history-v101.json',O/'whole-scope-requirement-audit-v101.json'])]
    persist_refinement(version=101,previous=100,title='Random recovery fallback operator graphs and backend-specific fusion domains',
      summary=dict(filename='recovery-fallback-summary-v101.json',previous_turn_classification='PROGRESS:v100attentionheadstateandoutputreorganizationcontracts;262postchecks',
        conditional_fallback_templates=5,observed_greedy_chains=72,new_runtime_executions=0,map_identity_cases=5456,real_prefix_monoid_cases=125,interface_models=80,
        new_source_graph_node_rows=70,new_source_graph_edge_rows=88,partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
        new_target_source_files=0,new_public_documents=0,refined_candidates=2,refined_modules=2,new_semantic_families=3,branch_records=362,semantic_modules=24,
        new_pairs=0,new_groups=0,refined_relation='REL73-26',relation_clauses=131,higher_order_groups=25,pending_central_pair_reconciliation=464,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
        actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_test=False,remote_job_status='Alllocalexistingprofile/pinpointcachedoperatorsource/mathworkfinished;no server/model/Torch/NPU/capture/serviceorimplementation.'),
      changes=['Expand5previouslyopaqueconditionalPyTorchrecovery/verificationoperatorbodiesintoG8sourcegraphswithoutinventingruntimeexecution.',
        'Separatebackendcap/cumprod,uniformdtype,optionalweightedresidual;realprefixscanhasIEEEfalsifierandvalidcuownerplanhasrightbias/tailguards.',
        'Priceonechosenprivate-plane/recovery-demandgraphwithunchangedRNG/Preaders;logicalinterfacebudgetsnotHBM/timeoruniversalcomputeceilings.',
        'Add3semanticM17rows=>362/24;partialgraphs813/1297/46hints;131/25/464relationsandnativecriticalgapsremainopen.'],
      candidate_updates=updates,module_updates=mods,semantic_branches={'M17':branches},remaining=remaining,action=action,jobs=jobs,report=report,read_first=names+['whole-scope-requirement-audit-v101.json'],
      publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_recovery_fallback_v101.py',N/'verify_recovery_fallback_v101.py',N/'reconcile_operator_graph_coverage_v88.py',N/'persist_profile_refinement_v90.py'],
      executor_prefix='recovery-fallback',main_script=Path(__file__),source_read=False,additional_shared_writes={R/'fusion-scan.json':fusion})

if __name__=='__main__':main()
