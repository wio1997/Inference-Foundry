"""Persist new primary instruction capacity cases,without promoting physical/runtime closure."""
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v89.json').exists()
    assert read(O/'typed-capacity-analysis-preflight-v89.json')['status']=='PASS'
    names=['typed-capacity-primary-receipts-v89.json','Vector-theoretical-primary-v89.html','Cube-theoretical-primary-v89.html',
      'HF32-mode-primary-v89.html','Vector-primary-tables-v89.json','Cube-primary-tables-v89.json','Cube-formula-decoded-v89.json',
      'typed-capacity-primary-contract-v89.json','A3-matrix-typed-capacity-cases-v89.json','mixed-projection-capacity-repricing-v89.json',
      'router-Vector-instruction-cuts-v89.json','HC-precision-mode-capacity-cases-v89.json']
    matrices=read(O/names[8]);source_matrix=read(O/'module-phase-branch-matrix-v88.json')
    groups=defaultdict(list)
    for r in matrices['regimes']:
        for m in r['module_candidates']:groups[m].append(r)
    gap=('Official current A3/2201 instruction capacities now have typed cases. Installed CANN/header/binary identity,actual FLOAT/HF32 mode/rounding and issue schedule,physical maximum frequency,HBM/cache/collective service and transformed numeric/ownership/readiness still require evidence. '
      'Separate instruction peaks do not prove a joint issue simplex;dense-Mmad or fullSort lowering bounds do not cover all equivalent algorithms. No actual module MFU,wallclock upper or fusion speedup is proved.')
    module_ids=sorted(set(groups)|{'M06'})
    theory_rows=[]
    for m in source_matrix['modules']:
        rows=groups[m['id']]
        theory_rows.append(dict(module=m['id'],captured_independent_matrix_regimes=len(rows),
          captured_matrix_execution_associations=sum(r['executions']for r in rows),
          FLOAT_matrix_regimes=sum(r['input_dtype']=='FLOAT'for r in rows),
          extra_source_instruction_domains=(['HcPre AB/HF32 and A2FLOAT mode cases','HcPost separate FP32Mul/Add']if m['id']=='M06'else
            ['normType2 FP32 nonlinears,selectedDiv/Gather and conditional fullSort32']if m['id']=='M12'else[]),
          state='SCOPED_TYPED_INSTRUCTION_CASES_ADDED'if m['id']in module_ids else'NO_NEW_INSTRUCTION_CASE_THIS_VERSION',
          all_operator_algorithms_branches_or_physical_capacities_closed=False,whole_module_closed=False))
    write(O/'module-theory-capacity-coverage-v89.json',dict(modules=theory_rows,rows=26,refined_modules=len(module_ids),
      unique_matrix_regimes=118,unique_matrix_executions=35334,module_associations_must_not_be_summed_as_unique_work=True,
      new_documentary_gap_closed='Previously absent official typed instruction-rate cases for FLOAT/HF32 and key A3Vector instructions,within the specified current-document architecture and lowering domains.',
      outstanding=gap,whole_model_lowest_MFU_module_established=False,whole_modules_closed=0,overall_complete=False))
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v88.json'))
    op['typed_instruction_capacity_review_v89']=dict(evidence=names,matrix_regimes=118,matrix_executions=35334,
      FLOAT_regimes=21,FLOAT_executions=5304,prior_BF16_INT8_regimes=97,router_Vector_regimes=7,router_Vector_executions=4472,
      new_regime_promotions=0,new_execution_promotions=0,native_attribution_or_global_uncertainty_changed=False,
      public_documentary_capacity_gap_refined=True,installed_mode_or_actual_hardware_ceiling_complete=False)
    write(O/'operator-analysis-coverage-ledger-v89.json',op)
    rel=deepcopy(read(O/'candidate-relation-clauses-v88.json'))
    refine_ids=['REL73-01','REL73-03','REL73-05','REL88-01','REL88-02','REL88-03']
    for c in rel['clauses']:
        if c['id']in refine_ids:
            c['typed_capacity_refinement_v89']=dict(evidence=names[7:],
              scope='Instruction/domain capacity repricing only;existing dataflow and numerical relation clauses remain conditional.',
              native_mode_or_joint_mixed_issue_proved=False,physical_gain_us=None)
    assert set(refine_ids)<={r['id']for r in rel['clauses']}
    rel.update(parent_relation_index='candidate-relation-clauses-v88.json',overall_complete=False)
    write(O/'candidate-relation-clauses-v89.json',rel)
    for base in ['candidate-relation-coverage','candidate-pair-navigation','candidate-operator-footprints']:
        d=deepcopy(read(O/(base+'-v88.json')))
        d.update(parent=base+'-v88.json',typed_capacity_refinement_v89=dict(evidence=names[7:],new_pair_clauses=0,new_groups=0,no_runtime_closure=True),overall_complete=False)
        if base=='candidate-relation-coverage':d['parent_relation_index']='candidate-relation-clauses-v89.json'
        write(O/(base+'-v89.json'),d)
    selected={
      'M06':{'M06-OP-06','M06-OP-10','M06-OP-12'},
      'M07':{'M07-OP-13'},'M08':{'M08-OP-02','M08-OP-12','M08-OP-14'},
      'M12':{'M12-OP-02','M12-OP-03','M12-OP-05'},'M14':{'M14-OP-13','M14-OP-15'}}
    branches={m['id']:deepcopy(m['branches'])for m in source_matrix['modules']if m['id']in selected}
    for mid,rows in branches.items():
        assert selected[mid]<={r['id']for r in rows}
        for row in rows:
            if row['id']not in selected[mid]:continue
            row['evidence']+=names[7:];row['typed_instruction_review_v89']=dict(
              public_document_instruction_rate_cases_known=True,installed_mode_or_native_mixed_issue_attested=False,
              actual_clock_memory_capacity_or_speedup_proved=False,scope='Declared typed instruction and resource-cut cases,not all algorithms or complete module coverage.')
            row['gap']=gap
            if row['id']=='M12-OP-02':row['branch']='RouterFP32dense5120→384target/128draft,M2/3/800/816/834domains;official2201FLOAT/HF32 instruction-rate cases now known,active mode/actual capacity remain unbound.'
    members=['F02','F03','F04','F08','F17','F20','F23','F25']
    updates={cid:dict(review_v89=dict(evidence=names,
      typed_instruction_capacity_cases_documented=True,all_legal_algorithms_or_native_mode_closed=False,
      existing_HF32_source_fact_not_counted_new=True,new_pair_clauses=0,new_groups=0,implementation=False,native_test=False),
      conclusion_changing_gap=gap)for cid in members}
    updates['F17']['next_profile_or_contract_action']='N24/M3 fixedB/L1/uncompressed domain retains8022cycle weight cut in bothFLOAT mode cases;new representation/precision,alternate datapath and all readers remain conditions. Do not infer actual memory bottleneck or speedup.'
    updates['F04']['next_profile_or_contract_action']='Use typed FLOAT matrix and per-instruction Vector capacity cases;retain fullE/ties/unbiased-score/selected-normalization and BF16norm boundaries. FullSort32 first-pass case isnot a native lowering or universal TopK proof.'
    updates['F02']['next_profile_or_contract_action']='Preserve AB HF32 mode/rounding and disabled state after AB;sourceA2 first-entry state remains unbound. Price sourceGram vs semantic sumsq separately;no new HF32 optimization claimed.'
    mods={mid:dict(finding='Official2201 typed instruction cases reprice saved matrix geometry and,where applicable,HC mode/Vector nonlinear work. Existing fusion/dataflow clauses remain conditional;no new native execution or module ranking.',
      evidence=names+['module-theory-capacity-coverage-v89.json'],candidates=[cid for cid in members if cid in next(m['candidate_links']for m in source_matrix['modules']if m['id']==mid)],gap=gap,
      profile_scope='Operator fusion precision/typed data paths and mathematical throughput/resource cuts only.')for mid in module_ids}
    write(O/'PROFILE-CODE-QUESTION-GATE-v89.json',dict(status='PROFILE_LED_OPERATOR_CAPACITY_AND_FUSION_PRECISION_QUESTIONS',
      questions=['What are official A3/2201 FLOAT/HF32 and keyVector instruction capacities?',
        'Do improved compute rates alone lower the mixed SWA/C2 declared max-cut bound?',
        'What Mmad precision-mode and typed CC→L1 gates must a fused graph preserve?'],
      saved_matrix_population=[118,35334],new_target_source_files_obtained=0,pinpoint_cached_source_files_reread=1,
      public_primary_documents_read=3,new_public_document_read=True,new_remote_source_read=False,
      no_framework_wide_research=True,no_model_Torch_NPU_capture_or_server_work=True,overall_complete=False))
    write(O/'typed-capacity-question-status-v89.json',dict(
      answered_within_declared_domain=['Official2201FLOAT full1024MAC/cycle vsHF322048 vsBF164096 perAIC',
        'KeyA3VectorFLOATExp/Ln/Sqrt32elements/cycle and Add/Mul64,with distinct reduction/sort cases',
        'All118captured independent dense matrix geometries have typed capacity cases',
        'FixedB/L1oneinitialgrant mixedN24andM3 models retain8022cycle weight cut in bothFLOATcases'],
      old_facts_reused=['ABenablesHF32anddisablesafterMmad previouslyreviewedv58','ExistingC2castsharing andnumericgatesv88'],
      unanswered=['Installed actualmode/binary and initialHF32state','Joint mixedissue/full highlevel reduction/sort lowering',
        'Physical maxfrequency/cache/HBM/communication/criticalpath','Allmodelbranches/allreaders/allcandidatecompositions'],
      source_build_scope='Publicpages advertise different master commits and uncommitted modifications. Received HTML hashes bind documentary evidence;no exact installed CANN9.2 correspondence is claimed.',
      whole_modules_closed=0,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v88.json')['remaining'])
    remaining.append('v89: public2201Mmad and A3Vector rate cases now documented,not universally unknown:FLOATfull16x16x4,Hf3216x16x8,BF1616x16x16;do notimport3510k1.118ordinarydense matrixregimes35334calls incl21FLOAT5304,97oldBF16INT8.7routerVector4472calls useindividualtypedcuts,fullSort32onlyloweringcase.6mixedmodels fixedB/L1N24andM3 retain8022cycle weightcut;usefulFLOATworkratio8vsindependentpaddingref2 notspeedup.5HCmodelsrepriceoldABHF32/A2FLOATsourcefacts;entrymode/newenum mapping/loadedbinaryunbound. Publiccurrentdirtybuildsnapshotsnotinstalled9.2;actualfreq/HBM/mixedissue/allalgorithms/readers open. No pair/group/branch/regime promotions:115/24/480,351/24semantic,2073/285598review9/1352pending849/16/3136 unchanged0wholeclosed.')
    report='''代码查阅继续只服务于算子融合、精度合同和理论上限。本轮以已有profile中的shape/dtype为入口，补充官方A3/2201指令容量，重算相关候选；没有开展框架遍历或服务器实验。

官方Cube表1给出2201硬件并行度：BF16为16×16×16，常规FLOAT为16×16×4，HF32为16×16×8。按一次MAC计两次运算，分别为每AIC每周期8192、2048、4096次运算。3510表中的FLOAT k=1不适用于这里。计算模型采用既有每die24AIC参考，时间仍以周期表示；未把观测1800MHz当作物理最高频率。[官方Cube理论性能](https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html)

现有118类MatMulV2/MatMulV3/QuantBatchMatmulV3二维dense几何、35334次执行，现在都有对应类型的容量模型；其中21类FLOAT、5304次执行补上了此前缺失的模式分支，其余97类沿用v80输入证据。这个集合不包括封装在attention、MC2、HcPre内部的全部矩阵指令，也不覆盖所有等价算法。每条记录保留原模块候选、角色、阶段和dtype，未改动原生归属。

对SWA BF16与两路C2 FLOAT投影，N=24时，两路FLOAT有效dense工作在常规模式下的理想容量下界为5120周期，HF32为2560；SWA单独为640周期。先局部投影的每卡3行方案，对应FLOAT有效工作为640或320周期。但在v88声明的固定B、L1→L0、原类型不压缩、仅一次初始L0驻留额度的模型中，两种行数仍需同一组权重，权重服务下界为8022周期。各资源界取max，不能把不同峰值倒数直接相加。提高矩阵吞吐本身没有降低这一特定下界；它不证明实际瓶颈或实际收益。GM直达L0、改变表示或算法等方案须另建模型。

小M还需注意分块。只按官方独立矩阵m/n对齐16的公式估价，N24与M3的两路FLOAT服务参考相差2倍；有效dense算术量相差8倍。这两个比例都不是实际加速比，真实切分、并行度、启动和通信成本仍未定。BF16重编码所需的权重精确可表示、原归约顺序和C2 FLOAT状态输出条件保持v88结论。

路由算子的7类、4472次既有执行增加了Vector指令模型。官方A3表中，FLOAT Exp/Ln/Sqrt各为32元素/周期，Add/Mul各为64；逐项计算必要工作界，再取max。不能把高层RMS/softmax归约统一套用64元素/周期，ReduceRepeat另有速率，而高层ReduceSum没有独立数值速率。Sort32/MrgSort的2元素/周期只用于明确的指令实现分支；完整TopK不必采用同一算法，实际Sort二进制也尚未取证。[官方Vector理论性能](https://asc.gitcode.com/api/appendix/vector_instruction_theoretical_perf_summary.html)

融合还须保持Mmad模式状态。HcPre的AB显式开启HF32、设置转换模式并在计算后关闭，这在v58已核实，本轮只补容量估价。A2函数自身不设置模式，首个入口状态仍缺原生证据；重排后不能让本应常规FLOAT的计算沿用HF32。官方接口说明HF32会改变输入舍入，结果仍是FLOAT，且模式持续至显式更改。旧SetHF32TransMode的数值1没有擅自映射到新API枚举。[官方HF32接口](https://asc.gitcode.com/api/SIMD-API/c_api/cube_compute/asc_enable_hf32.html)

2201搬出表列出的L0C FLOAT→L1 BF16/half通路会改变类型。该表不能支撑C2 FLOAT状态或HC系数无损改走这条路径；表中未列FLOAT直通也不等于所有其他API一概不支持。所需通路与精度要共同证明。

三份官方页面已保存URL、HTML和SHA；页脚标记为不同master构建且含未提交修改，因此证据以收到的快照为准，不冒充已安装CANN9.2或执行二进制。实际模式、物理最高频率、缓存/HBM/通信容量、混合指令联合吞吐及完整关键路径仍未闭合。

本轮没有增加候选数量或把条件分析提升为实测证明。累计351条分支、24个语义模块，115条条件关系、24组联合合同；中央仍有480对待汇入或复核。2073个regime/285598次执行已作条件分析，9个regime/1352次执行仍待补，849条PMU缺失、16张阶段未绑定图及3136条模块归属歧义不变。没有整模块关闭，整体任务继续。
'''
    action=('继续原Goal，v89PROGRESS，v88亦PROGRESS。读typed-capacity-summary/PROFILE-FINDINGS-v89、primary-contract/receipts/Cube-formula/matrix-cases/mixed-repricing/router-Vector-cuts/HC-precision-mode-cases/module-theory-capacity-coverage/question-status。'
      '新官方primary证据补A3/2201Mmadtypedrates：BF168192ops/AICcycle,FLOATfull2048,HF324096,INT816384；不要import3510FLOATk1或950directUB，publicdirtymasterHTMLsha不代表installed9.2。参考24AIC48AIV，f变量不观测1800当最高频。'
      '118ordinary2Dmatrixregimes35334calls incl21FLOAT5304、97oldBF16INT830030有typedcases；不是attention/MC2/HC内部全矩阵或所有算法。物理模式/路径/混合issue/freq/HBM/comm/rankMFU仍open。6mixedNmodels含newownerlocalM3，N24两FLOATuseful5120full/2560HF32、M3640/320，SWA640/80；fixedB/L1/uncompressed/onewarmgrant原8022weightcut不变，max不加不同peak倒数。官方独立paddingM24→32/M3→16使service参考ratio2不是useful8或speedup。'
      '7routerVectorregimes4472calls ExpLnSqrt32 vsAddMul64elements/AIVcycle、Gather5、Sort32/MrgSort2为明确instructioncases；ReduceRepeat64/7，高层ReduceSumsoft/noindependentrate，不能统一generic64。FullSortfirstpassonlyconditionalnative/universalTopKnotproved。'
      'HcPreABenableHF32/transmode1/disableafterMmad是v58旧事实非新优化，本轮5HCmodels重价。A2FLOATbody无modesetter，firstentryunbound；新fusion必须保每次mode/rounding/resets，oldliteral1不映射新enum。2201CC→L1listedFLOAT→BF16/half非无损FLOATbridge；absence不普遍unsupported。'
      '无新pair/group/branch/nativepromotions：115/24/480，351records24semantic，2073/285598review9/1352pending849PMU16phase3136ambiguity0wholeclosed。已知文档typedcapacitygap有条件补齐，勿再泛称完全无FP32/HF32/Vector资料；actualcapacity/wholealgorithm仍缺。'
      '后续优先仍会改变融合结论的operatorprimitive/allbranch/allreader映射或可由保存profile/精确定点source/primary资料补的真实证据；不盲增pairs，不复读旧结论报新进展，不研究无关框架。不做server/model/Torch/NPU/capture/servicework，不上传。GoalACTIVE/TaskACTIVE/LoopEVALUATING；verify_typed_instruction_capacity_v89须PASS，不重跑finalizer不提前complete。')
    jobs=[dict(kind='review',status='pass',description='Bind3currentofficialprimarysnapshotsand2201/A3tableguards,decodeMmadformula;notinstalledCANNbinary/modeattestation.',artifacts=[O/n for n in names[:8]]),
      dict(kind='design-check',status='pass',description='Reprice118savedmatrixregimes35334calls,6mixedrowcases,7routerVectorregimes4472calls,5HCsourcecases;maxcuts/precision/pathconditions,notphysicaltimeorspeedup.',
        artifacts=[O/n for n in names[8:]]+[O/'typed-capacity-analysis-preflight-v89.json',O/'module-theory-capacity-coverage-v89.json',R/'fusion-scan.json'])]
    persist_refinement(version=89,previous=88,title='Typed A3 instruction capacities and operator fusion precision modes',
      summary=dict(filename='typed-capacity-summary-v89.json',previous_turn_classification='PROGRESS:v88mixedprojectionandgraphreconciliation',
        new_public_primary_documents=3,new_target_source_files=0,cached_pinpoint_source_files_reread=1,
        matrix_regimes=118,matrix_executions=35334,FLOAT_regimes=21,FLOAT_executions=5304,prior_BF16_INT8_regimes=97,
        router_Vector_regimes=7,router_Vector_executions=4472,mixed_row_models=6,HC_models=5,refined_modules=len(module_ids),refined_candidates=len(members),
        new_pairs=0,new_groups=0,relation_clauses=115,higher_order_groups=24,pending_central_pair_reconciliation=480,
        branch_records=351,semantic_modules=24,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
        pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_regimes_or_executions_promoted=0,
        actual_native_mode_frequency_or_physical_wallclock_ceiling_proved=False,implementation=False,native_test=False,
        remote_job_status='All local saved-profile/source/math checks finished;no server/model/Torch/NPU/capture/service work.'),
      changes=['Document official2201 FLOAT/HF32/BF16/INT8 and A3Vector typed rate cases,with current-publicbuild/installedbinary distinction.',
        'Extend capacity cases to118saved independent matrixregimes35334executions,including21FLOAT5304.',
        'MixedsmallM fixedB/L1weightcut remains8022cycles;routertypedVector cuts and HCprecision-mode gates reprice existing fusion candidates.',
        'No new pair/group/branch/regime promotion or whole-module closure;global evidence gaps retained.'],
      candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names[7:]+['module-theory-capacity-coverage-v89.json','typed-capacity-question-status-v89.json'],
      publication_rows={
        'F04':'| F04 | RMSNorm/类型转换与路由矩阵、非线性、TopK融合 | 条件 | 已有双输出norm不重复计收益。2201FLOAT/HF32与A3Vector指令速率已建分支模型；保BF16边界、全E选择、无偏权重及舍入合同。实际模式、联合吞吐和物理收益待证。 |',
        'F17':'| F17 | C2两路FLOAT投影及与SWA BF16联合投影/分区 | 条件 | 原cast已共享；全BF16重编码须精确性证明。N24/M3有效算术不同，但声明固定B/L1/原类型/一次初始驻留下的权重服务界仍为8022周期。输入、通信与分块收益另核，不能用HF32容量推定实际模式或加速。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/'analyze_typed_instruction_capacity_v89.py',N/'verify_typed_instruction_capacity_v89.py'],
      executor_prefix='typed-capacity',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
