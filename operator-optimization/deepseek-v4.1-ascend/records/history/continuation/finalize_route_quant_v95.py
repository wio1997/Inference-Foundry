"""Persist MC2 input-quant reuse, source partition work and scoped ceiling evidence."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v95.json').exists()
    assert read(O/'route-quant-analysis-preflight-v95.json')['status']=='PASS'
    names=['route-quant-operator-source-contract-v95.json','route-quant-tiling-and-work-models-v95.json','route-quant-native-frontiers-v95.json',
      'route-quant-reuse-interface-models-v95.json','route-quant-variants-v95.json']
    members=['F21','F22']
    gap=('Same-quantizer/default-DynamicQuantV2 versus routing FP16-TRUNC/INT8-ROUND,zero/nonfinite/scale policy and all input epochs/readers remain unbound. '
      'New prequantized MC2 ABI must preserve sorted row maps/masks/padding/counters,peer records,bank aliases and generation/readiness;producer-before-MC2 changes the dependency graph. '
      'Source inner40 budget and outer platform24/48 are different fields;source-conditioned key/row-loop arithmetic is not loaded-binary/instruction timing attestation. '
      'Direct padded target816 rows cannot be assumed identical to800-row shared input. Sender quant loops are not receiver GEMM assignments. '
      'GM interface models are not HBM or wire bytes;weights/expert receiver counts/physical capacities/critical path remain separate. No measured gain or whole-module closure.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v94.json'));c=next(r for r in rel['clauses']if r['id']=='REL73-08')
    c['routing_quant_refinement_v95']=dict(evidence=names,
      contract='Current no-smooth routing already reuses quantized rows within core intervals;cross-boundary repetition depends on topK3/6 and inner40 tiling. Share canonical q only under exact quantizer/new-ABI/epoch contract;otherwise preserve two quantizers and price their outputs. Full-row aligned partition is an alternative with new parallelism/publication costs.',
      all_variants_or_native_precision_closed=False)
    for g in rel['higher_order_groups']:
        if g['id']in ['HG73-MOE','HG80-A3-LOCAL-PATH','HG84-HC-FFN-FORK']:
            g['routing_quant_refinement_v95']=dict(evidence=names,contract=c['routing_quant_refinement_v95']['contract'],whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v94.json',new_pair_clauses=0,new_higher_order_groups=0,overall_complete=False);write(O/'candidate-relation-clauses-v95.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v94.json'));cov.update(parent='candidate-relation-coverage-v94.json',parent_relation_index='candidate-relation-clauses-v95.json',
      routing_quant_refinement_v95=dict(existing_clause='REL73-08',new_pairs=0,new_groups=0,evidence=names),overall_complete=False);write(O/'candidate-relation-coverage-v95.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v94.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v94.json')['model_candidates']}
    for r in fps['candidates']:
        p=R/'candidates'/(r['id']+'.json');sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==ix[r['id']]['master_sha256'];r['captured_master_sha256']=sha;r['captured_master_version']=94
        if r['id']in members:r['operator_atom_union']+=['moe.input_quant.token_row_replication'];r['routing_quant_refinement_v95']=dict(evidence=names,shared_work_atom_not_new_operator=True)
    fps.update(parent='candidate-operator-footprints-v94.json',overall_complete=False);write(O/'candidate-operator-footprints-v95.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v94.json'));fm={r['id']:r for r in fps['candidates']}
    for pair in nav['pairs']:pair['shared_operator_atom_union']=sorted(set(fm[pair['a']]['operator_atom_union'])&set(fm[pair['b']]['operator_atom_union']))
    nav.update(parent='candidate-pair-navigation-v94.json',parent_footprints='candidate-operator-footprints-v95.json',overall_complete=False);write(O/'candidate-pair-navigation-v95.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v94.json'));shape_keys={r['saved_input_shapes']for r in read(O/names[1])['cases']}
    regimes=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']if r['signature']['inputShapes_text']in shape_keys]
    assert sum(r['executions']for r in regimes)==4472
    ids={r['regime_id']for r in regimes}
    for r in op['regimes']:
        if r['regime_id']in ids:assert r['state']=='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED';r['scoped_conditional_review_evidence']+=names
    op['routing_quant_refinement_v95']=dict(existing_regimes_refined=len(ids),existing_MC2_executions=4472,existing_graph_frontiers=3200,reused_TASKs=6400,
      source_loop_evaluations_conditional=862040,row_once_all_input_reference=799560,new_native_regime_or_execution_promotions=0)
    write(O/'operator-analysis-coverage-ledger-v95.json',op)
    matrix=read(O/'module-phase-branch-matrix-v94.json');branches=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M12'))
    assert not any(r['id']=='M12-OP-17'for r in branches)
    branches.append(dict(id='M12-OP-17',branch='Shared/routed input quantization reuse versus token-aligned route replication:topK3/6,inner40 partitions,source full/gather kernels,new prequantized ABI and finite precision gates.',
      finding='Five current MC2 descriptor cases/4472 calls and3200 graph shared-quant frontiers bind the scope. Source loop counts3K6=18,3K3=9,2K3=6,816K6=836,834K6=834. Outer24/48 platform launch is separate frominner40. Common/dual producer androw-aligned variants priced conditionally;no native equivalence or speedup.',
      state='SCOPED_OPERATOR_REUSE_PARTITION_AND_REPRESENTATION_CEILING_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
      scope='Frozen source program and typed captured cohorts,all-input-row reference. Not loaded internal tiling or same-quantizer proof;all padding/state/reader cases remain explicit.'))
    mods={mid:dict(finding='Routed/shared quant producer reuse now has source partition work counts,quantizer/new-ABI guards and one nominal GM graph budget. Core count effects and unchanged routing/receiver work remain explicit.',
      evidence=names,candidates=[cid for cid in members if cid in next(m['candidate_links']for m in matrix['modules']if m['id']==mid)],gap=gap,
      profile_scope='Operator reuse/fusion and necessary work/resource domain only;pinpoint custom tiling/kernel dependencies.')for mid in ['M12','M13','M22','M23']}
    updates={cid:dict(review_v95=dict(evidence=names,graph_frontiers=3200,reused_TASKs=6400,MC2_existing_executions=4472,source_case_count=5,
      source_internal_AIV_budget=40,recorded_outer_launch_fields=[24,48],these_fields_are_distinct=True,
      conditional_same_quantizer_M3_GM_input_difference=92088,conditional_dual_quantizer_difference=76716,
      new_pairs=0,new_groups=0,all_reader_or_loaded_precision_or_gain_proved=False,implementation=False,native_test=False),conclusion_changing_gap=gap)for cid in members}
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v94.json'));audit.update(previous_goal_turn='PROGRESS:v94norm-quantprecision/GM/resourcecontracts persisted;143postchecks',input_checkpoint=94,followup_checkpoint=95)
    for r in audit['requirements']:
        if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES':r['branch_records']=357
        if r['id']=='THEORY_AND_MEASURED_COMPARISON':r['evidence']+=names
    audit['routing_quant_refinement_v95']=dict(source_selection_not_framework_research=True,source_loop_work_not_necessary_bound=True,
      physical_numeric_ceiling_not_claimed=True,no_whole_requirement_closed=True)
    write(O/'whole-scope-requirement-audit-v95.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v95.json',dict(status='PROFILE_ROUTED_SHARED_INPUT_QUESTION_THEN_PINPOINT_CUSTOM_OPERATOR_TILING',
      questions=['Does no-smooth routing already quantize once per token or per assignment?',
        'How do token-splitting core boundaries change work for captured topK/rows?',
        'Can shared INT8q+scale replace routing BF16 input without new ABI or quantizer precision proof?',
        'Does outer launch24/48 identify the internal quantization partition budget?'],
      new_target_source_files=0,cached_pinpoint_source_files=7,no_framework_wide_research=True,
      no_public_web_read=True,no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    write(O/'route-quant-validation-history-v95.json',dict(corrected_assumptions=[
      dict(topic='Outer launch versus inner tiling',failed_check='Expected recorded MC2 block fields20/20 from the constantBLOCK_NUM20.',
        observed='3200 profiles all report blockNum24/mixBlockNum48.',resolution='Host lines246-252 derive outer SetBlockDim from platform APIs;line273 independentlysetsinnerquantAIV40. The fields have different meanings;no source/binary mismatch inferred. Corrected check binds both domains.'),
      dict(topic='Observed routing width',resolution='Actual descriptor topK6target/topK3draft used,not generic8;all five source cases retain roles/row counts.'),
      dict(topic='Sourcework versus execution',resolution='Derived row loops862040 weighted over saved calls are conditional source estimates,not measured native instructions or physical lower work.')],
      native_or_whole_binary_proof=False,all_local_jobs_finished=True,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v94.json')['remaining']);remaining.append('v95: ExistingF21/F22commonquantcontract refined fromsource/profile,notnew candidate.5MC2shapes4472calls topK6target3draft;source innerAIV40 fromBLOCK_NUM20 independentouterplatform24/48 profiles. Full21000/gather11000 no-smooth reusewithininterval buttokenboundariesduplicate:3K6→18,3K3→9,2K3→6,816K6→836,834K6→834. Weightedsource862040vs799560row-oncereference;notnativeinstructionornecessaryworkbound. A48variants816→816but834→858;row-alignedalternativechangesparallelism/protocol.3200graphs6400TASKs reused. Commonq/dualquantrequireexactquantizer/newMC2ABI/epochs/masks/readiness;M3samebaseGM338520→246432commonor261804dual (differences92088/76716),unchangedH+32routecopies/receiverwork;notHBM/wire/gain. Fixedfreshperassignmentuncompressedq+scale representation has A*(H+4) symbolicpublicationwritecut,capacityunknown,notallalgorithms. REL73-08/3groupsrefined122/25/473 unchanged;357branches24semantic;9/1352pending849PMU16phase3136ambiguityunchanged0wholeclosed/lowestMFUunestablished.')
    report='''本轮核对 routed expert 与 shared expert 的输入量化共用条件，限定在算子输入、内部路由量化、分区和表示成本。

Profile中MC2共4472次执行、5种输入组合：target为TopK=6、draft为TopK=3。3200组图内MC2与shared DynamicQuant已按原生TASK连接，复用6400条旧TASK，其中2560组有阶段归属、640组仍阶段未知。当前shared量化在同一记录stream的MC2之后；把它变为共同生产者，需要改变提交和就绪关系，不能直接把该kernel耗时当作收益。

冻结host给内部路由量化传入40个AIV分区预算、UB=196352B、quantMode=1、scaleDim0=0。外层launch由平台CalcTschBlockDim单独设置；profile的blockNum=24、mixBlockNum=48不是内部量化预算。最初把两者直接对应的检查已纠正，未据此判断版本冲突。父算子源码与原166已保存副本的8个文件哈希一致；这仍不是加载ELF和内部tiling参数证明。

full-load和gather的无smooth全列分支，在每个core区间中对同一个token量化一次，然后复制到该token的多条路由。区间边界若切开token，就会重复量化。按冻结源码推导，3行/TopK6为18次行量化，3行/TopK3为9次，2行/TopK3为6次；816行/TopK6为836次，834行/TopK6为834次。五种组合按调用次数加权，源程序参考862040次，对比全部输入行各一次的799560次。这些是条件源码循环量，不是已实测的内部指令数，也不能乘除成耗时或加速倍数。

分区数和工作量并非单调：把同样的assignment分区预算改为48时，816行可到816次，而834行会到858次。按完整token的路由组对齐可以不重复行量化，但M=3、K=6时只有3个token分区，原方案则有18个assignment分区。总工作量下降与并行度变化必须同时评估；映射、最后写者、计数器、跨核flag及通信准备仍是合法性条件。

当前routing量化含BF16→FLOAT、absmax/127、FLOAT除法、FP16 CAST_TRUNC、INT8 CAST_ROUND。shared DynamicQuantV2是否具有同一舍入、scale、零与非有限值规则，尚未证明。当前MC2也接收BF16输入，不能把INT8指针直接塞入原ABI。条件方案包括：证明量化器一致后复用共同q/scale；或保留两个量化器，只共享被证明相同的输入/amax工作并支付双输出成本；或调整token分区。每种方案都保留参数与行generation、padding/mask、路由映射、peer记录和全部读者。

M=3、H=5120、K=6、18条有效assignment的声明GM值模型中，共同基线为338520B。复用已存在的shared q/scale后为246432B，差92088B；若两个量化器都保留并增加routing q/scale输出，则为261804B，差76716B。H+32路由记录的92736B写入仍在，receiver的dispatch/GMM/SwiGLU/combine、各自权重和网络协议也未删去。这不是HBM流量、线缆流量或真实收益。直接调用的816行可能包含800行之外的padding，不能按同形或相邻关系认定与shared输入全等。

在保留新鲜、互异、未压缩的逐assignment GM q+FLOATscale记录这一表示域中，必要写值至少为A·(H+4)，本例92232B；当前H+32记录多出的padding不是通用必要量。时间界保留T≥A·(H+4)/B_upper_GM_publication，容量上限未知，不能套用HBM带宽。若改为索引/共享记录等新表示，必须重建这个界。sender量化行次数与receiver专家GEMM assignment数保持分开。

已有REL73-08及三组联合合同得到细化，未增加独立候选或虚增关系覆盖。累计357条分支记录、24个语义模块；中央122条条件关系/25组联合合同，473对仍待汇入或复核。2073个regime/285598次执行的条件分析计数不变，9个regime/1352次执行仍缺关键证据；849条PMU缺失、16张阶段未知图、3136条模块归属歧义保留。全模块最低MFU和整体优化分析尚未完成。
'''
    action=('继续原Goal。v95PROGRESS，v94PROGRESS已143checks。先读route-quant-summary-v95/PROFILE-FINDINGS/source/tiling-work/native/interfaces/variants及wholeaudit。'
      '用户只要operatorfusion/split/reorg/theory，profile优先，pinpoint缺口，不做framework研究。'
      '本轮4472MC2calls5shapes targettopK6/draft3，3200same-streamMC2→sharedDQfrontier6400oldTASKs root2560unbound640。SourcehostinnerquantAIV40=2*BLOCK_NUM20，而outerlaunchplatformCalcTschBlockDim/recorded24,48另字段，最初错误check已修复不推versionconflict。8parentfileSHAsmatchoriginal166stage，notloadedELF。'
      'NoSmooth quant full/gather coreinterval rowcache已经baseline；key21000small3/2，11000M816/834，H5120onecol。sourceconditionalqrows3K6=18,3K3=9,2K3=6,816K6=836,834K6=834；weighted862040vs799560row-once，不是instruction/measured/necessarywork。A48sensitivity816816/834858，tokenalignedalternativegainsworknotfreeparallelism，newflags/maps/bankprotocolgates。'
      'Commonrouting/sharedqneedsquantizeridentity(路由BF16→F32amax/127/Div→FP16TRUNC→INT8ROUND)，newprequantMC2ABI，allmasks/padding,row/gamma/generation/aliases/readers/producerbeforeMC2。Differentqcanconditionaldualfuseinput/amaxbutkeepoutputrounds。M3K6baseGM338520/common246432差92088/dual261804差76716，同基线替代不相加；H+32recordwrites92736/receiverGMM/weights/network保留，notHBM/wire/gain。FixedfreshperassignmentGMrecorddomainwritecutA(H+4)=92232symbolicBunknown；newrepresentationnewcut。SenderqRnotreceiverassignmentA。'
      'REL73-08及HG73MOE/HG80/HG84refined0newpairs/groups，122/25/473;357records24semantic，2073/285598review9/1352pending849/16/3136不变0wholeclosed/noallmodulelowestMFU。'
      '下一步优先尚未定案且影响全范围结论的operator分支/组合，复用已有答案不再复读QR/HCfork/commoninputquant。后续可核对routingfull-load各core重复sort/映射与TopK交接的真实baseline，只作算子合同与理论，不扩框架。保持offline/frozen/profile/math/必要primary,no server/model/Torch/NPU/capture/implementation/upload。GoalACTIVE/TaskACTIVE/LoopEVALUATING，verify_route_quant_v95须PASS，不重跑成稿finalizer。')
    sources=[Path(p['path'])for p in read(O/names[0])['parts']]
    jobs=[dict(kind='profile',status='pass',description='Bind3200existingMC2/sharedDQfrontiers6400TASKs and5descriptors4472calls;keepcurrenttopK3/6,phase/sourceidentity/loadedkeygates.',artifacts=[O/names[0],O/names[2]]+sources),
      dict(kind='design-check',status='pass',description='49152partitionmodels1872copy-commutationmodels;source40vsouterlaunch24/48,canonical/dualGMgraphsandsymbolicrepresentationwritecut,noactualnativegain.',
        artifacts=[O/names[1],O/names[3],O/names[4],O/'route-quant-analysis-preflight-v95.json',O/'route-quant-validation-history-v95.json',O/'whole-scope-requirement-audit-v95.json',R/'fusion-scan.json'])]
    persist_refinement(version=95,previous=94,title='Routed/shared quantization reuse with token partition and representation costs',
      summary=dict(filename='route-quant-summary-v95.json',previous_turn_classification='PROGRESS:v94normquantprecision/sharedinterfacespersistedandverified',
        native_frontiers=3200,reused_native_TASKs=6400,saved_MC2_calls=4472,source_descriptor_cases=5,source_conditional_quant_row_evaluations=862040,canonical_all_input_row_reference=799560,
        partition_models=49152,row_function_copy_models=1872,new_target_source_files=0,new_public_documents=0,refined_candidates=2,refined_modules=4,
        new_pairs=0,new_groups=0,relation_clauses=122,higher_order_groups=25,pending_central_pair_reconciliation=473,branch_records=357,semantic_modules=24,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
        native_precision_or_actual_hardware_ceiling_or_speedup_proved=False,implementation=False,native_test=False,
        remote_job_status='Alllocalprofile/source/mathematicalmodelworkfinished;no server/model/Torch/NPU/capture/serviceorimplementation.'),
      changes=['Resolveinnerquantpartition40versusplatformouterlaunch24/48;actualsavedtopK3/6shapeinputs/sourcepathconditionsexplicit.',
        'Sourceintervalquantreuseandtoken-boundaryduplicateworkderivedfor5capturedshapes;alternativepartitioncostsnotunconditionalgain.',
        'Common/dualquantnewABIandprecisiongates,oneGMbaselinewithunchangedreceiverwork,andscoped symbolicwritecut.',
        'ExistingREL73-08andthreejointgroupsrefinedwithoutcountinflation;357records24semantic,allcriticalgapsandGoalACTIVEretained.'],
      candidate_updates=updates,module_updates=mods,semantic_branches={'M12':branches},remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['whole-scope-requirement-audit-v95.json'],publication_rows={},gate_updates=dict(overall_complete=False),
      scripts=[N/'analyze_route_quant_v95.py',N/'verify_route_quant_v95.py',N/'persist_profile_refinement_v90.py'],executor_prefix='route-quant',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
