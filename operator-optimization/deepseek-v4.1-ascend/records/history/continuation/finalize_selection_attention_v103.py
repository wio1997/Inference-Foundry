"""Persist provisional-selection/attention fusion domains and work/representation tradeoffs."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v103.json').exists()
    assert read(O/'selection-attention-analysis-preflight-v103.json')['status']=='PASS'
    names=['selection-attention-inputs-v103.json','selection-attention-source-contract-v103.json',
        'selection-attention-native-frontiers-v103.json','selection-winner-readiness-and-cover-v103.json',
        'selection-attention-retraction-and-rounding-v103.json','selection-attention-speculation-cost-models-v103.json',
        'selection-attention-variants-v103.json']
    gap=('640 saved QLI/prepare/SMLA frontiers give task order only, not provisional winner or native tile-ready events. '
        'Speculative per-key QK/logits and certified original-order tiles require new interfaces and exact native score/precision/tie/mask/alias/cache/row-generation contracts. '
        'Rounded-summary collision excludes only that unconditional replacement; exact arithmetic, augmented state or reread/recompute remain separate allowed domains. '
        '36 declared work/storage graphs are not actual selected lengths, HBM/peak memory, physical numeric ceilings, MFU or critical-path speedup. Whole scope remains open.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v102.json'))
    row=next(r for r in rel['clauses']if r['id']=='REL83-08')
    row['selection_frontier_refinement_v103']=dict(evidence=names,
        contract='Finalized direct handoff is an existing candidate. Provisional TopK eviction is not append-only attention; irreversible updates need certified membership and original complete tile order, or another proved precision/state domain. Per-item-logit heap/local-cover alternatives retain final banks/readers and add QK work/storage.',
        new_pair=False,all_runtime_gates_proved=False)
    group=next(g for g in rel['higher_order_groups']if g['id']=='HG83-INDEX-ATTENTION')
    group['selection_frontier_refinement_v103']=dict(evidence=names,
        contract='Keep distinct index/attention scores, source candidate-block output, sorted final-index publication and original BF16 online order. SWA independence and old interface credits are baseline, not extra new savings. No forced whole-operator barrier when certified tiles or per-item speculation suffice.',whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v102.json',new_pair_clauses=0,new_higher_order_groups=0,overall_complete=False)
    write(O/'candidate-relation-clauses-v103.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v102.json'))
    cov.update(parent='candidate-relation-coverage-v102.json',parent_relation_index='candidate-relation-clauses-v103.json',
        selection_frontier_refinement_v103=dict(evidence=names,new_pairs=0,new_groups=0,new_native_promotions=0),overall_complete=False)
    write(O/'candidate-relation-coverage-v103.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v102.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v102.json')['model_candidates']}
    for r in fps['candidates']:
        h=hashlib.sha256((R/'candidates'/(r['id']+'.json')).read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256']
        r['captured_master_sha256']=h;r['captured_master_version']=102
        if r['id']in ['F12','F13']:r['selection_frontier_refinement_v103']=dict(evidence=names,new_native_atoms=0,new_candidate_graph_interfaces_only=True)
    fps.update(parent='candidate-operator-footprints-v102.json',overall_complete=False);write(O/'candidate-operator-footprints-v103.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v102.json'));pair=next(r for r in nav['pairs']if{r['a'],r['b']}=={'F12','F13'})
    pair['selection_frontier_refinement_v103']=dict(existing_relation='REL83-08',evidence=names,all_native_or_semantic_variants_closed=False)
    nav.update(parent='candidate-pair-navigation-v102.json',parent_footprints='candidate-operator-footprints-v103.json',parent_relation_index='candidate-relation-clauses-v103.json',overall_complete=False)
    write(O/'candidate-pair-navigation-v103.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v102.json'))
    op['selection_frontier_refinement_v103']=dict(reused_frontiers=640,reused_TASKs=1920,root_bound=512,phase_unbound=128,
        new_native_regimes=0,new_native_executions=0,new_native_attribution_or_promotions=0,partial_operator_graph_counts_unchanged=[813,1297])
    write(O/'operator-analysis-coverage-ledger-v103.json',op)
    matrix=read(O/'module-phase-branch-matrix-v102.json');branches={mid:deepcopy(next(m['branches']for m in matrix['modules']if m['id']==mid))for mid in ['M09','M10']}
    additions={
        'M09':('M09-OP-11','Provisional TopK membership versus final/certified original-order tiles; local complete-score TopK cover or versioned resident winner-logit heap, including candidate-mask/filler/tie and future-bank contracts.',
            'A finished score tile does not certify its tentative winners against unseen scores. Compatible local TopK union contains global TopK; per-item attention QK speculation is a distinct work/storage graph, not existing QLI fusion or a free TP factor.'),
        'M10':('M10-OP-14','Attention over changing provisional selection: per-item QK/logit retention then final chronological softmax/PV versus rounded partial-PV eviction/replay.',
            'Exact-real fixed-anchor subtraction exists, but a rounded (m,l,a) collision loses survivor content even with the removed item known. Reordered BF16 probability quantization can differ. Certified tiles or augmented/recomputed state can permit early work without a whole-operator barrier.')}
    for mid,(cid,label,finding)in additions.items():
        assert not any(r['id']==cid for r in branches[mid]);branches[mid].append(dict(id=cid,branch=label,finding=finding,
            state='CONDITIONAL_SELECTION_FUSION_WORK_STATE_AND_READINESS_REVIEWED;RUNTIME_NUMERICS_AND_FIELDS_OPEN',
            evidence=names,gap=gap,scope='Operator fusion/split/reorganization alternative domain; not a new observed runtime branch or whole-module closure.'))
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v102.json'))
    audit.update(previous_goal_turn='PROGRESS:v102 capture-bound sampling request intent and cost routing;249 postchecks',input_checkpoint=102,followup_checkpoint=103)
    for r in audit['requirements']:
        if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES':r['branch_records']=364
        if r['id']in ['CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON']:r.setdefault('evidence',[]).extend(names)
    audit['selection_frontier_refinement_v103']=dict(reused_frontiers=640,distinct_score_contract_reused=True,
        tentative_winner_eviction_and_certified_tile_domains_reviewed=True,new_native_promotions=0,
        source_graph_rows_unchanged=[813,1297],branch_records=364,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v103.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v103.json',dict(status='SAVED_QLI_PREPARE_SMLA_FRONTIERS_THEN_FOUR_PINPOINT_OPERATOR_FILES',
        questions=['Does finishing a QLI score tile certify its provisional winners?',
            'Can rounded attention summary state remove an evicted winner exactly?',
            'Can per-key QK speculation preserve the original final softmax/PV order, and what additional work/storage does it require?',
            'Which certified tile interfaces permit early progress without forcing a whole-kernel barrier?'],
        cached_operator_source_files=4,new_target_source_files=0,new_public_documents=0,
        no_framework_wide_research=True,no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    write(O/'selection-attention-validation-history-v103.json',dict(refinements=[
        'v102 predecessor revalidated PASS249 before refinement; no completed artifact/finalizer changed or rerun.',
        'Selected 640 existing v100 same-graph QLI/prepare/SMLA frontiers (512 root/128 unbound),1920 existing TASKs; native sequential spans are observations, not critical-path recoverable time.',
        'Existing final handoff, candidate publication, independent SWA overlap and distinct QLI/attention scores retained as prior v60/v83/v100 answers; no new credit for already fused QLI/SMLA internals.',
        '15088 distinct-score partition/rank cases and1215 tied-total-order reference cases verify local TopK cover. Reference tie order is not native SortAll attestation. Unseen completion witness excludes certification from tile completion alone in the stated independent-score model.',
        '2187 positive-mass exact-real fixed-anchor retraction cases supplement algebra; FLOAT32 round-nearest collision shows identical rounded summary and removed item can leave survivor outputs0 or1. Additional original KV/score reads or augmented state remain allowed.',
        'BF16 round-nearest reference away from BF16 ties: old relative exp.5019 and rebase factor.75 give rounded-old*.75=.375 versus fresh rounded.376953125. Reference final outputs.271484375 and.2734375 differ. Not a CANN Exp/CAST_ROUND/SoftmaxFlashV2 native test.',
        '36 chosen graph work/storage examples include N34816,K512,P8,A4096:compressed attention arithmetic9/2 baseline,or19/5 including128SWA;chosen FLOAT logit union1048576B versus resident heap131072B per query. Not actual runtime lengths/HBM/peak/time gain or lower bound on all algorithms.',
        'Six variants refine existing REL83-08/HG83-INDEX-ATTENTION; two conditional opportunity rows added, no new pair/group/regime/graph row. No all-module MFU or numeric whole physical ceiling. All local jobs terminal.'],overall_complete=False))
    updates={cid:dict(review_v103=dict(evidence=names,reused_frontiers=640,reused_TASKs=1920,
        partition_cases=15088,reference_tie_cases=1215,real_retraction_cases=2187,work_storage_models=36,
        rounded_summary_and_BF16_rebase_reference_counterexamples=True,new_relation_ids=[],refined_relation='REL83-08',
        implementation=False,native_test=False,new_runtime_executions=0),conclusion_changing_gap=gap)for cid in ['F12','F13']}
    mods={mid:dict(candidates=['F12','F13'],finding=finding,evidence=names,gap=gap)for mid,(_,_,finding)in additions.items()}
    remaining=list(read(O/'module-coverage-gaps-v102.json')['remaining']);remaining.append(
        'v103: QLI→prepare→SMLA speculative-selection domain refined with640/1920 existing frontiers/TASKs,15088rank/1215tie reference cases and2187real retractions. Final/certified original-order tile handoff, resident per-key logits and local-cover speculation remain conditional. Rounded summary does not generally support exact eviction; rebase can change BF16 probabilities. Actual selected IDs/native ties/row-cache generation/new ABI/readiness/precision/capacities unknown. 36 work/storage graphs are not speedup/HBM/peak evidence. +2opportunity rows=>364/24;partial813/1297/46graphs,131/25/464relations,2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/no allmoduleMFU unchanged.')
    report='''本轮围绕Indexer TopK→索引修整→SparseFlashMla的融合边界，复用640条保存的局部链、1920个TASK（512条root绑定、128条阶段未绑定）。所有链上均记录了QLI、prepare、SMLA同流先后执行；这只是当前执行顺序，没有给出内部tile就绪时间、临时winner、选中ID或可回收关键路径延迟。源码只核对4份已缓存的算子文件，没有展开框架研究，也没有新服务器、设备、采集或实现工作。

已有v60/v83方案已覆盖最终TopK→prepare→attention直接交接、候选bank共享和独立SWA提前。本轮不重复计收益，补的是“能否在TopK尚未结束时计算压缩attention”。在独立完整score且没有未扫描部分上界的模型中，一个tile结束不能证明其临时winner最终入选：已见score=[4,3]、K=1时，后续score为−1或5，对应最终ID分别为0或2。因此不能把临时入选项直接当作不可撤回的attention贡献。

这不构成全算子屏障。若有可信未扫描score上界、成员资格证书和完整的原始顺序tile，可以提前处理已确定的最终tile；也可先做允许丢弃的逐项计算。在只需要选中集合且N≤K的特殊域中，选择依赖也可不同，但原输出顺序、候选块bank及其他读者仍需保留。没有把当前SortAll或整体QLI完成自动当成所有实现的必需等待。

精确实数下，固定anchor的softmax分母l和加权和a可以减去被淘汰项的贡献；2187组有限正质量用例核对了这个公式。有限精度汇总却不一定可撤回。设两项质量为1和2^-25，被删除项value=1，另一项value分别为0或1。FLOAT32最近舍入后，两种历史的(m,l,a)都可成为(0,1,1)，删除项信息也相同；删除后的正确单项输出却分别为0和1。只凭该舍入汇总无法区分，必须保留额外逐项信息，或重读/重算。这是声明精度模型的反例，没有声称当前负载实际出现，也没有否定其他状态表示。

临时分块还可能改变BF16概率边界。一个最近舍入参考中，旧相对指数0.5019先转BF16为0.5，再乘新anchor因子0.75得0.375；按最终tile的anchor先相乘再转BF16则是0.376953125。保留相同FLOAT32分母的标量PV参考，最终BF16输出分别为0.271484375和0.2734375。该用例不是CANN Exp、CAST_ROUND或SoftmaxFlashV2实测；它说明实数重标定等式不足以证明重分块后保持原量化结果。源实现的概率转KV_T、在线max/sum、PV缩放/累加及最终输出边界仍需逐项匹配。

一个条件候选是提前计算临时TopK新进入项的attention QK，并按键、query及cache版本保留H个logits；被淘汰项安全释放，最终按原有的可见性、时间顺序、tile划分、softmax、BF16概率和PV流程读取。选定的resident表示同时最多保留K项，但整个扫描可能先后计算N项，不能把它写成只有K项工作。QLI和attention使用不同score，不能用前者代替后者。

另一候选是给每个逻辑key分区计算local TopK，再为其并集提前算attention QK。相同完整score总比较器下，global TopK一定包含在local TopK并集中；否则一个global winner在本分区已有K个更优项，矛盾。15088组不同score与1215组含tie的参考核对了该覆盖关系，参考tie顺序没有被当成原生SortAll证明。候选外填充位置、membership算术、mask/offset和候选bank合同仍需保留。

代价必须重新计。按每query、H=64、D=512、N=34816、K=512、8个逻辑分区的示例，并集有4096项。提前算其QK、最终只算512项PV，压缩attention矩阵工作是原参考的4.5倍；若另有128项SWA且保持其工作不变，总矩阵工作为3.8倍。并集FLOAT logits值大小为1048576B，选定resident heap为131072B；这些不是HBM流量、分配峰值或普适容量下界，也不是8张卡自动带来的加速。提前开始可能隐藏依赖，同时增加竞争和工作，不能据此宣称更快。

理论部分维持SYMBOLIC_CEILING。在1≤K<N、独立任意score、无额外证书的比较模型中，精确选中集合最坏需要N次score观察和至少N−K次比较；这不证明N次原生INT8 GEMM。选定图的必要矩阵、Vector、分层存储和通信工作需使用匹配的能力上界；独立资源允许重叠，同资源负载不得重复计。当前仍没有实际选择轨迹、原生tile等价和物理能力证据来给数值提速或全模块MFU。

本轮细化既有F12/F13关系REL83-08及联合组，新增2条条件优化机会记录，未新增原生分支、regime、pair或图节点。当前364条记录、24个语义模块，131条条件关系、25组联合合同、464对待汇入或复核。局部图813节点行/1297边行、9个regime/1352次执行关键字段缺口、849 PMU/16阶段图/3136归属歧义及整体未完成状态均保持明确。
'''
    action=('继续原Goal,v103PROGRESS；v102PASS249，本轮source/native/math preflightPASS123。先读selection-attention-summary-v103/report/source-contract/winner-readiness/retraction-rounding/speculation-cost/wholeaudit。'
        '已复用640QLI/prepare/SMLA frontiers1920TASKs(512root128unbound)。不要重复v60/v83最终handoff/SWAoverlap/score-distinct/v100state答案；新域是provisionalTopK替换与certified final original-order tile/per-item QK logits。'
        '15088rank partition+1215totalorder ties；native SortAll ties未绑定。Real fixed-anchor subtraction2187cases；FLOAT32 summarycollision同(m,l,a)与removeditem但survivor0/1，不能无条件撤回。BF16 rebase参考.5019*.75 round boundary≠先round再scale；不是CANN测试。'
        'Residentheap liveK/everK..N与localcover A=sum min(K,n_partition)是不同表示；N34816K512P8A4096压缩矩阵9/2、含SWA128总19/5，FLOAT logits union1MiB/heap128KiB每query，不是时间/HBM/peak/所有算法下界。保留最终原序softmax/BF16/PV/sink/metadata/candidatebanks/readers/generation；certifiedtiles允许早开，无强制全算子barrier。'
        '+2条件机会记录=>364/24；关系仍131/25/464，partialgraphs813/1297/46，2073/285598条件9/1352pending849PMU16phase3136歧义0whole/no allmoduleMFU。'
        'Next audit current coverage for genuinely missing operator-selection/readiness semantics, especially remaining chronological M00/M03 only where they choose operator implementations or constrain input/collective readiness. Do not broaden to framework API catalogue or keep enumerating similar speculative variants. Critical native operands/capacities remain evidence gaps, not solved by more code; minimal planv98 remains prepared/unexecuted. '
        'All local jobs finished, no server/model/Torch/NPU/capture/service/environment/implementation. GoalACTIVE TaskACTIVE Loop004EVALUATING；verify_selection_attention_v103 mustPASS；never rerun completed finalizer or edit completed artifacts.')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    jobs=[dict(kind='profile',status='pass',description='Reuse640 existing QLI/prepare/SMLA frontiers and four cached operator sources, preserve all actual field/tile-readiness/precision uncertainty; no new native work.',artifacts=[O/n for n in names]+inputs),
        dict(kind='design-check',status='pass',description='15088rank/1215tie reference cases,2187real retractions,rounded state and BF16 rebase witnesses,36 chosen work/storage graphs; refine existing F12/F13 edge without physical speedup claims.',
            artifacts=[O/'selection-attention-analysis-preflight-v103.json',O/'selection-attention-validation-history-v103.json',O/'whole-scope-requirement-audit-v103.json'])]
    persist_refinement(version=103,previous=102,title='Provisional TopK selection and sparse-attention fusion frontier',
        summary=dict(filename='selection-attention-summary-v103.json',previous_turn_classification='PROGRESS:v102 capture-bound sampling intent and fusion cost routing;249 postchecks',
            reused_frontiers=640,reused_TASKs=1920,rank_partition_cases=15088,totalorder_tie_cases=1215,real_retraction_cases=2187,
            work_storage_models=36,new_target_source_files=0,new_public_documents=0,refined_candidates=2,refined_modules=2,
            new_semantic_families=2,branch_records=364,semantic_modules=24,partial_graph_node_rows=813,partial_graph_edge_rows=1297,
            structural_pair_hints=46,new_pairs=0,new_groups=0,refined_relation='REL83-08',relation_clauses=131,higher_order_groups=25,pending_central_pair_reconciliation=464,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_test=False,
            remote_job_status='All local saved-profile/operator-source/math work finished; no server/model/Torch/NPU/capture/service/environment or implementation.'),
        changes=['Refine provisional-winner eviction and certified-final-tile readiness on640 saved QLI/prepare/SMLA frontiers.',
            'Separate exact-real retraction from rounded-summary collision and BF16 rebase reference counterexamples.',
            'Price per-item-logit heap/local-TopK-cover speculative graphs; no free reduction of work, storage or current critical path.',
            'Add2 conditional opportunity rows=>364/24, refine existing REL83-08/group with all native/physical/full-scope gates open.'],
        candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names+['whole-scope-requirement-audit-v103.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_selection_attention_v103.py',N/'verify_selection_attention_v103.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='selection-attention',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
