"""Persist typed Indexer projection costs and a source-derived upstream QA/Weight panel edge."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v106.json').exists()
    assert read(O/'index-projection-analysis-preflight-v106.json')['status']=='PASS'
    names=['index-projection-inputs-v106.json','index-projection-source-contract-v106.json','index-projection-native-frontiers-v106.json',
        'index-projection-work-and-resource-models-v106.json','index-projection-rounding-and-readiness-v106.json',
        'index-weight-partition-and-precision-v106.json','index-projection-composition-v106.json']
    packet=read(O/names[-1]);ids=['F05','F09','F10']
    gap=('Five observed projection M domains and640 saved same-graph frontiers are typed/work/stream-order reviewed. '
        'Actual row/hidden/QR/weight/config/alias epochs,native scale/cast/quant/reduction policy,all observers and changed tile/capture/event protocols remain conditions. '
        'Weight/RoPE overlap is already baseline andWeight/IndexQuant share one auxiliary stream;early start/late join are not measured overlap gains. '
        'FixedB cycle cuts require the stated24AIC/must-load distinct uncompressed planes/one warm grant/path and no computed-output reuse;not all algorithms or native wallclock. '
        'Split-K and upstreamQA32panel need independent RMS/output/precision/publication costs. No MFU,physical numeric ceiling,speedup or whole closure.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v105.json'));rel['clauses']+=deepcopy(packet['clauses'])
    group=next(g for g in rel['higher_order_groups']if g['id']==packet['extend_group']);assert 'F10'not in group['members'];group['members'].append('F10')
    group['index_projection_review_v106']=dict(evidence=names,
        contract='Optional IndexWeight32 panel shares local query_hidden with QA1280 upstream producer. Keep local/query-global Jmap and separateQA1280 RMS,Weight32 scaling andSWA/C1 norms;choose one explicit ragged graph,not all group variants simultaneous. One work/weight/bridge/input credit union.',whole_group_closed=False)
    old=next(c for c in rel['clauses']if c['id']=='REL83-02')
    old['index_projection_review_v106']=dict(evidence=names,
        contract='IndexQ remains INT8 QR1280→4096 andWeight remains BF16 hidden5120→32. Model their separate typed cuts andexactFP16-before-product boundary. Early Weight start/late first-reader join must preserveexistingauxstream serialization anderrors/readers;not direct common-inputIndexQ/WeightGEMM.',all_runtime_gates_proved=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v105.json',clause_count=132,selected_pair_count=132,new_pair_clauses=1,new_higher_order_groups=0,overall_complete=False)
    write(O/'candidate-relation-clauses-v106.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v105.json'));clause=packet['clauses'][0]
    row=next(r for r in cov['pairs']if{r['a'],r['b']}=={'F05','F10'});assert row['clauses']==[]
    row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[clause['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(parent='candidate-relation-coverage-v105.json',parent_relation_index='candidate-relation-clauses-v106.json',scoped_conditional_pairs=132,remaining_pair_relations=463,higher_order_groups=25,
        scope='132scopedconditionalpairs/25groups/35modelcandidates;463centralpairreconciliations andallvariantgates remain. Newpairuses upstreamQA producerextension,not same IndexQ/Weight input.',
        index_projection_review_v106=dict(new_pairs=1,new_groups=0,evidence=names),overall_complete=False)
    write(O/'candidate-relation-coverage-v106.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v105.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v105.json')['model_candidates']}
    for r in fps['candidates']:
        h=hashlib.sha256((R/'candidates'/(r['id']+'.json')).read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256'];r['captured_master_sha256']=h;r['captured_master_version']=105
        if r['id']in ids:r['index_projection_review_v106']=dict(evidence=names,new_native_atoms=0,
            F05_upstream_QA_producer_variant_included=r['id']=='F05',ordinary_IndexQ_Weight_common_input_not_assumed=True)
    fps.update(parent='candidate-operator-footprints-v105.json',overall_complete=False);write(O/'candidate-operator-footprints-v106.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v105.json'));row=next(r for r in nav['pairs']if{r['a'],r['b']}=={'F05','F10'});assert row['conditional_clause_ids']==[]
    row.update(conditional_clause_ids=[clause['id']],semantic_pair_reconciliation_done_in_stated_scope=True,
        index_projection_review_v106=dict(shared_hidden_input_edge_not_shared_operator_atom=True,evidence=names))
    nav.update(parent='candidate-pair-navigation-v105.json',parent_footprints='candidate-operator-footprints-v106.json',parent_relation_index='candidate-relation-clauses-v106.json',
        scoped_conditional_pairs=132,remaining_central_pair_reconciliation=463,index_projection_review_v106=dict(new_pairs=1,all_atom_sets_unchanged=True,evidence=names),overall_complete=False)
    write(O/'candidate-pair-navigation-v106.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v105.json'))
    op['index_projection_review_v106']=dict(saved_frontiers=640,reused_TASKs=5760,projection_shape_cohorts=10,M_domains=[3,800,816,830,834],
        source_typed_roles_not_new_functional_attribution=True,new_native_regimes=0,new_native_executions=0,new_native_attribution_or_phase_promotions=0,new_conditional_relation='REL106-01')
    write(O/'operator-analysis-coverage-ledger-v106.json',op)
    matrix=read(O/'module-phase-branch-matrix-v105.json');branches=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M09'))
    for r in branches:
        if r['id']in ['M09-OP-04','M09-OP-05']:
            r['evidence']=list(dict.fromkeys(r['evidence']+names))
            r['index_projection_review_v106']=dict(observed_M_domains=[3,800,816,830,834],typed_work_and_fixedB_cuts_reviewed=True,
                original_cast_interface_budget_reused=True,actual_physical_ceilings_or_all_consumers_closed=False,evidence=names)
            r['gap']=gap
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v105.json'))
    audit.update(previous_goal_turn='PROGRESS:v105 current dependency audit and two exact PCP selectors;1089postchecks',input_checkpoint=105,followup_checkpoint=106)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=132,higher_order_groups=25,central_pairs_not_reconciled=463)
        if r['id']in ['THEORY_AND_MEASURED_COMPARISON','ALL_MODULE_CONDITIONAL_BRANCHES']:r.setdefault('evidence',[]).extend(names)
    audit['index_projection_review_v106']=dict(five_observed_shapes_scoped_priced=True,upstream_QA_weight_common_input_relation=True,
        baseline_weight_RoPE_overlap_and_aux_serialization_preserved=True,new_native_promotions=0,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v106.json',audit)
    obligations=deepcopy(read(O/'coverage-current-obligations-v105.json'))
    for r in obligations['obligations']:
        if r['id']=='G10':r['remaining']=r['remaining'].replace('131 conditional','132 conditional').replace('464 pairs','463 pairs')
        if r['id']=='G13':
            r['index_projection_review_v106']=dict(evidence=names,scoped_observed_projection_work_resource_gap_priced=True,
                actual_native_values_and_physical_capacity_still_open=True)
            r['remaining']='M09 five observed query/weight shapes now have typed work/resource,readiness andprecision models;this resolves that bounded cost question,not all native modes. Continue genuine optional filter/processor/readers and captured-bank identity questions after checking prior answers;do not infer unread solely from old wording.'
    obligations.update(parent='coverage-current-obligations-v105.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v106.json',obligations)
    write(O/'PROFILE-CODE-QUESTION-GATE-v106.json',dict(status='SAVED_TYPED_PROJECTION_AND_QLI_FRONTIERS_THEN_ONLY_INDEXER_PROJECTION_AND_STREAM_JOIN_SOURCE',
        questions=['Which input/weight arithmetic belongs to IndexQ versus Weight projection in all observed row domains?',
            'Which weight-start/join edges are control ordering, and what overlap/aux serialization already exists?',
            'Can QA andWeight share hidden input upstream without changing QA RMS or Weight rounding?',
            'What split-K/epilogue costs and conditional shared-storage cuts constrain the selected graph?'],
        cached_source_files=3,new_target_source_files=0,new_public_documents=0,no_framework_wide_research=True,
        no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    write(O/'index-projection-validation-history-v106.json',dict(refinements=[
        'v105 predecessor revalidated PASS1089. Priorv64Cast/Mul/Cast budgets,v82IndexQextension,v83distinct-inputjoin,v99raggedQA/SWA/C1panels reused,not rediscovered.',
        'Exact saved native geometry confirms IndexQ uses QuantBatchMatmulV3 INT8 inputs withBF16/FLOAT scales,packed128×80×16×32 weight plane andBF16 output. Weight is ordinaryBF16MatMul5120→32. Ten shape cohorts coverM3/800/816/830/834;no dtype inferred from config alone.',
        '640 same-graph frontiers/5760 existingTASKs crosschecked against32 saved DB projections;512root/128unbound retained. Every Weight starts after IndexQ and overlapsIndexRoPE;Weight andIndexQuant share auxiliary stream in all640. No false new-overlap credit.',
        'A local new analyzer initially parsed N/A non-matrix shapes as integers and stopped before checkpoint/shared mutation. It now filters operator type first and preserves N/A as unknown;all source-native assertions completed.',
        'Newshared-hidden QA1280+Weight32 panel extendsF05upstreamproducer domain;IndexQ1280 input isstill distinct. KeepQA1280 RMS separate fromWeight32 andoptionalglobalSWA/C1Jmap/norm groups. No new native operator atom or mode binding.',
        'FixedB24AICshared L1→L0B capacity3072B/cycle andoneinitialgrant1572864B giveM3jointIndexQ/Weight1302cycles against separate usefulcompute80/5cycles;no INT8/BF16mixedissue sum or frequency scaling. UpstreamBF16panel5568/7275cycle cases remain fixed-path/weight-output-domain bounds,not actual wallclock/physicalMFU.',
        'FiniteBF16 Weight3×2^-20 andvalidqueryamax520192→FP16scale4096:required firstFP16round givesproduct2^-12;dropping it gives3×2^-14. NativeFTZ/cast modes notattested.448sufficient-domain references preserve exact finiteFP16scaledWeight.',
        '256 ideal DAG schedules illustrate earlyWeightstart/late first-readerjoin;they are not native capacity lower bounds or speedups and preserveauxserial limits.30splitKgraphs retaininput/weightunion andonefinalBF16round,whileFP32products[2^25,-2^25,1] giveordered1 versusregrouped0. Reference2Ntiles isnot actualactive-corecount.',
        'One newREL106-01 F05/F10,existingREL83-02 refined,HG75-Q-KV-JOIN extendedF10.132relations/25groups/463pending;353semanticrows26modules unchanged. Native/PMU/phase/ambiguity/whole/MFU gaps remain;alllocaljobs terminal.'],overall_complete=False))
    updates={cid:dict(review_v106=dict(evidence=names,saved_frontiers=640,reused_TASKs=5760,shape_domains=[3,800,816,830,834],
        projection_cohorts=10,new_relation_ids=['REL106-01'],refined_relation='REL83-02',extended_group='HG75-Q-KV-JOIN',
        safe_rounding_cases=448,ideal_DAG_examples=256,splitK_reference_graphs=30,
        actual_MFU_or_physical_speedup_proved=False,implementation=False,native_test=False,new_runtime_executions=0),conclusion_changing_gap=gap)for cid in ids}
    mods={'M07':dict(candidates=ids,finding='QA upstreamproducer can conditionally emit independent32Weight outputs sharingquery_hidden;IndexQ remains distinctQRINT8 input. Raggedpanel work/one-shared-weight-ingress cut andindependentRMS/release constraints priced.',evidence=names,gap=gap),
        'M09':dict(candidates=['F09','F10'],finding='Allfive observedIndexQ/WeightMdomains now have typedmatrix/weight/interface bounds.640frontiers preserveexistingWeight/RoPE overlap andauxserialization;earlystart/latejoin,FP16round andsplitK alternatives retainnumerical/resourceconditions.',evidence=names,gap=gap)}
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('Current obligation authority:coverage-current-obligations-v106.json;historical55records retained in coverage-gap-history-before-v105.json. No whole closure;critical9native operands and physical allmodule MFU remain open. Do not count current fixed-path/reference/DAG/splitK models as actual speedups.')
    report='''本轮补的是G13中具名的Indexer query/weight投影成本范围。先核对原profile，再定点阅读3份已缓存算子源码。未登录服务器、新采集、设备测试或修改目标代码。

原生记录确认两条投影不同：IndexQ为INT8 QuantBatchMatmulV3，QR宽1280、输出4096、BF16 weight scales/FLOAT row scales、输出BF16；IndexWeight为BF16 MatMul，hidden宽5120、输出32。全部保存的shape cohort为M=3/800/816/830/834，两条投影各5组。640条QLI相关graph局部链、5760个既有TASK中，512条root绑定、128条阶段未绑定，归属不作提升。

一个具体重组方向是weight提前启动、延后其第一次真正消费处的等待。源码当前先提交IndexQ，再记录weights_start供辅助流等待；weight只依赖hidden输入，可以在声明的输入只读/版本/就绪条件下更早启动。当前weight已与IndexRoPE重叠，640条记录全部如此，这不是新增收益。weight与query quantization还在同一辅助流，单独删除主流的早等待不会让它们彼此并行；独立压缩/准备工作能否提前，仍需事件、读者、buffer和竞争成本证明。

另发现一个上游共同输入边：QA1280投影和IndexWeight32投影读取同一个query_hidden表达式，可条件性组成两个独立BF16输出panel。这里共享的是QA的输入，不能把读取norm后QR1280的IndexQ与Weight直接视为共同输入GEMM。QA仍对自己的1280维做RMS；Weight32走自己的BF16→FLOAT缩放→FP16路径，不能合成1312维RMS。与已有SWA/global24行及C1投影拼接时，必须沿用局部3行/全局24行的Jmap及各自norm/输出就绪合同。

M=3时，IndexQ常规参考为31457280 INT8 OPS，Weight为983040 BF16 OPS；native TASK中位数分别为14040ns和10840ns。这些是对应独立算子的记录，不是相加可回收延迟，也不是模块MFU。常规M16完整tile的issued-work参考另存，不能凭小N=32断言当前只用了2个核；split-K也可能改变原生累加数值。

理论模型现在覆盖这5个观察到的M域。在24AIC、不同未压缩权重都必须经固定B的L1→L0B路径、全局仅一次初始L0B驻留额度、没有已算结果复用或换操作数路径的声明域中，两条权重共5570560B，额度1572864B、能力上界3072B/cycle，给出1302个整数服务cycle参考cut。M=3各自有用算术cut为80 INT8 cycles与5 BF16 cycles；不同类型算术没有擅自相加为混合发射下界。实际最高频率及路径未绑定，因此没有换算成原生时间上限或提速。

把Weight32加入v99的QA/SWA和QA/SWA/C1 ragged panel后，常规BF16工作分别为166133760和291962880 OPS，固定B权重cut为5568和7275 cycles。全部扩大到global24行会分别产生456/169和584/297倍的矩形参考工作。M3额外共享输入的逻辑读值预算为30720B；它与先前panel信用必须对同一独立物化基线计一次，不是实测HBM节省。

融合中不能无条件删掉Weight的中间FP16舍入。一个有限参考中，BF16投影值为3·2^-20，缩放1/64；有效BF16 query amax=520192使query scale为FP16的4096。按原顺序先转FP16，最终product为2^-12；省掉这次FP16舍入则为3·2^-14。融合可以在寄存器保留该舍入并省去物化，删除舍入本身要另证值域和原生FTZ/cast模式。另保存448组充分域参考，未将它们当作实际输入值证明。

30个split-K参考图为窄32列Weight增加独立partial和FLOAT merge，权重/input并集不随partition重复计算，部分和不提前转BF16。即便如此，FLOAT32乘积[2^25,−2^25,1]按不同结合顺序可得到1或0，最终BF16也不同。因此新并行图仍需原生累加/容差合同。256个提前启动/晚合流DAG例子只表示理想服务排程，不能倒数成物理容量上界或宣称测得收益。

本轮新增F05/F10条件关系REL106-01，细化既有F09/F10关系REL83-02并把F10加入已有联合组；没有新增模型候选、原生执行或branch记录。中央现132条条件关系、25组联合合同、463对待汇入或复核。353条语义记录/26个暂定模块及9个regime/1352次执行的关键字段、849 PMU/16阶段图/3136归属歧义、全模块最低MFU和整体物理理论上限仍未完成。
'''
    action=('继续原Goal,v106PROGRESS；v105PASS1089，本轮preflight545。先读index-projection-summary-v106/report/source/native-frontiers/work-resource/rounding-readiness/splitK/composition及current-obligations-v106。'
        'G13 M09projection具体5Mshape(3,800,816,830,834)已typed work/resourcepriced；原v64postcast budgets/v82extension/v83distinctIndexQWeightinputs/v99raggedpanels非新发现。640frontiers5760existingTASK512root128unbound，IndexQ INT8 QR1280→4096、WeightBF16H5120→32；均source/shapegrade无归属提升。'
        'Weight现有全部640和RoPE重叠且和IndexQuant同aux流，不能重复算overlap；新earlystart/latefirstreaderjoin需inputepochs/errors/capture/events/竞争证明。新QA/Weight共享query_hidden上游panel是F05producerextension，不是IndexQ/Weight同input。QA1280norm和Weight32缩放分开，globalKV24Jmap保留。'
        'M3Qops31457280/Wops983040，fixedB24AIC/shared128Bpercorecycle/oneL0Bwarmgrant cut1302cycles，compute80/5max非混合simplex；无actualfmax/time/MFU。v99upstreampanel加W32得到5568/7275cycles，逻辑inputcredit30720B需oneunion，不是HBM。'
        'Weight3*2^-20→/64→FP16 thenqscale4096 gives2^-12，dropF16round gives3*2^-14；448safecases非actualvalues。30splitKrefs FLOAT32 [2^25,-2^25,1] assoc1vs0，256idealDAG非speedup/capacityLB。'
        'NewREL106-01F05F10/existingREL83-02refined/HG75joinedF10:132/25/463；353/26rows,partial813/1297/46graphs,2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/no allmoduleMFU。'
        'Next choose conclusion-changing G13 remaining operator body/reader/graph-bank source question or critical evidence from saved assets;do not repeat newprojection cost or create near-duplicate variants. Optional upstream filter/processors and captured bank/parameter relations remain named review surfaces;first check prior answers. G03/G04 minimal evidencev98 remains prepared/unexecuted. '
        'Alllocaljobs terminal,no server/model/Torch/NPU/capture/service/environment/implementation. GoalACTIVE TaskACTIVE Loop004EVALUATING；verify_index_projection_v106 mustPASS；never rerun completed finalizer or edit completed artifacts.')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    jobs=[dict(kind='profile',status='pass',description='Reuse640 nine-stage IndexQ/Weight/QLI frontiers andallten shape/type cohorts from32 saved DB projections;preserve actual overlap/auxserialization andall attribution uncertainty.',artifacts=[O/n for n in names]+inputs),
        dict(kind='design-check',status='pass',description='Five typedprojection andtwo upstream raggedpanel cost/cut models;FP16 intermediate counterexample,448safe-domain references,256idealDAG and30splitK graphs;newsource-derived QA/Weight relation,no native gain claim.',
            artifacts=[O/'index-projection-analysis-preflight-v106.json',O/'index-projection-validation-history-v106.json',O/'whole-scope-requirement-audit-v106.json',O/'coverage-current-obligations-v106.json'])]
    persist_refinement(version=106,previous=105,title='Indexer projection resource ceilings and upstream QA/Weight fusion',
        summary=dict(filename='index-projection-summary-v106.json',previous_turn_classification='PROGRESS:v105 dependency audit and exactPCPselectors;1089postchecks',
            saved_frontiers=640,reused_TASKs=5760,root_bound=512,phase_unbound=128,projection_cohorts=10,observed_M=[3,800,816,830,834],
            projection_resource_cases=5,upstream_panel_cases=2,safe_rounding_cases=448,ideal_DAG_examples=256,splitK_reference_graphs=30,
            new_target_source_files=0,new_public_documents=0,refined_candidates=3,refined_modules=2,new_semantic_families=0,branch_records=353,semantic_modules=26,
            new_pairs=1,new_groups=0,new_relation='REL106-01',extended_group='HG75-Q-KV-JOIN',relation_clauses=132,higher_order_groups=25,pending_central_pair_reconciliation=463,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            implementation=False,native_test=False,remote_job_status='All local saved-profile/operator-source/math work finished;no server/model/Torch/NPU/capture/service/environment or implementation.'),
        changes=['Price five observed typedIndexQ/Weight projection domains and one-shared-fixedB resourcecuts without native clock/MFU inference.',
            'Distinguish existingWeight/RoPE overlap andauxserialization fromnewearly-start/late-join candidates.',
            'Addsource-derived upstreamQA/Weight32common-inputpanel relation withindependentnorm/round/release andone-credit union.',
            'PreserveFP16intermediate andFP32splitK reduction contracts;one newpair132/25/463,allnative/fullscopegapsopen.'],
        candidate_updates=updates,module_updates=mods,semantic_branches={'M09':branches},remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names+['coverage-current-obligations-v106.json','whole-scope-requirement-audit-v106.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_index_projection_v106.py',N/'verify_index_projection_v106.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='index-projection',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
