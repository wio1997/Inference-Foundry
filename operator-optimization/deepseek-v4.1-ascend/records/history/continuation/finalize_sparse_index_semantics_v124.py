"""Persist sparse-index occurrence and request-window operator coverage."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v124.json').exists()
    pre=read(O/'sparse-index-preflight-v124.json');assert pre['status']=='PASS'
    names=['sparse-index-inputs-v124.json','sparse-index-source-contract-v124.json',
        'sparse-index-multiset-and-rounding-v124.json','sparse-index-request-reuse-and-cost-v124.json',
        'sparse-index-profile-scope-v124.json','operator-graph-G9-v124.json',
        'operator-graph-node-coverage-v124.json','operator-graph-edge-coverage-v124.json',
        'operator-graph-candidate-bindings-v124.json','operator-graph-coverage-audit-v124.json']
    ga=read(O/names[-1])
    detail=dict(evidence=names,reused_profile_frontiers=640,reused_unique_TASKs=1920,
        profile_root_bound_frontiers=512,profile_phase_unbound_frontiers=128,
        multiset_reference_cases=7500,request_interval_reference_cases=1680,metadata_layout_models=9,
        new_source_graph_nodes=41,new_source_graph_edges=49,partial_graph_node_rows=854,
        partial_graph_edge_rows=1346,structural_pair_hints=48,new_semantic_rows=2,
        no_actual_duplicate_rate_or_new_native_traffic=True,new_pairs=0,new_groups=0,
        new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    contract=(
        'Prepare preserves a multiset of visible IDs and counts occurrences,not distinct keys. '
        'Within-query repeated valid keys cannot be dropped without retaining multiplicity/sink and native rounding/order. '
        'Local QLI tile IDs are distinct;global uniqueness needs disjoint partitions/record-preserving merge/range/offset conditions. '
        'Selected noncausal DSpark logical windows repeat one per-request row across query rows;descriptor/KV-load reuse does not establish query-output reuse. '
        'Keep bank/page/request/weight/mask generations,source physical-pair gather order and all stable publication/readers.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v123.json'))
    for cid in ['REL83-08','REL73-27','REL73-28']:
        row=next(r for r in rel['clauses']if r['id']==cid)
        row['evidence']=list(dict.fromkeys(row['evidence']+names))
        row['sparse_index_semantics_v124']=dict(**detail,contract=contract)
    group=next(g for g in rel['higher_order_groups']if g['id']=='HG83-INDEX-ATTENTION')
    group['sparse_index_semantics_v124']=dict(**detail,contract=contract)
    rel.update(parent_relation_index='candidate-relation-clauses-v123.json',new_pair_clauses=0,
        new_higher_order_groups=0,sparse_index_semantics_v124=detail,overall_complete=False)
    write(O/'candidate-relation-clauses-v124.json',rel)
    selected_pairs=[{'F12','F13'},{'F12','F31'},{'F13','F31'}]
    cov=deepcopy(read(O/'candidate-relation-coverage-v123.json'))
    for p in cov['pairs']:
        if {p['a'],p['b']}in selected_pairs:p['sparse_index_semantics_v124']=deepcopy(detail)
    cov.update(parent='candidate-relation-coverage-v123.json',parent_relation_index='candidate-relation-clauses-v124.json',
        sparse_index_semantics_v124=detail,
        scope='135 scoped conditional pairs/26 groups/35 model candidates;460 central pair reconciliations remain. Existing F12/F13/F31 clauses gain multiset and request-window source contracts;no all-variant closure.',overall_complete=False)
    write(O/'candidate-relation-coverage-v124.json',cov)
    nav=deepcopy(read(O/'candidate-pair-navigation-v123.json'))
    for p in nav['pairs']:
        if {p['a'],p['b']}in selected_pairs:
            p['sparse_index_semantics_v124']=deepcopy(detail)
            p['evidence']=list(dict.fromkeys(p.get('evidence',[])+names))
    nav.update(parent='candidate-pair-navigation-v123.json',parent_footprints='candidate-operator-footprints-v124.json',
        parent_relation_index='candidate-relation-clauses-v124.json',sparse_index_semantics_v124=detail)
    write(O/'candidate-pair-navigation-v124.json',nav)
    footprints=deepcopy(read(O/'candidate-operator-footprints-v123.json'))
    for r in footprints['candidates']:
        if r['id']in ['F12','F13','F31']:
            r['sparse_index_semantics_v124']=dict(evidence=names,source_graph_supplement='G9',
                no_new_native_operator_atoms=True,all_variants_or_native_parameters_bound=False)
    footprints.update(parent='candidate-operator-footprints-v123.json')
    write(O/'candidate-operator-footprints-v124.json',footprints)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v123.json'))
    op['sparse_index_semantics_v124']=detail
    write(O/'operator-analysis-coverage-ledger-v124.json',op)

    obligations=deepcopy(read(O/'coverage-current-obligations-v123.json'))
    for r in obligations['obligations']:
        if r['id']=='G00':r['remaining']='26 provisional module inventories,358 semantic rows and35 model candidates are indexes;all applicable operator branches and graph variants are not proven exhausted.'
        if r['id']=='G09':r['remaining']=(
            'Partial source catalog854 nodes/1346 edges/48 structural hints now includes G9 prepare-multiset,noncausal logical-window broadcast and selected PA gather branches. '
            'These are source rows,not unique native operators. Complete QLI/LD partition identity,all lowering branches/actual IDs/owner and bank maps/parameters remain open;enumerated sets2..4 do not prove global semantics.')
        if r['id']in ['G00','G02','G05','G06','G07','G08','G09','G10','G11','G13']:
            r['sparse_index_semantics_v124']=deepcopy(detail)
        if r['id']=='G13':r['remaining']=(
            'v124 resolves prepare-as-multiset and selected DSpark same-request logical-window repeat semantics. '
            'No actual sparse ID/duplicate rate/global QLI disjoint-partition/loaded offset or per-query native gather count is captured. '
            'Need actual valid row/position/range,source scheduler/merge and page/request/bank/mask epochs;do not infer payload duplicates from -1 padding or discard multiplicity/sink/BF16 rounding. '
            'Descriptor or KV reuse needs new consumer ABI/all reader/capture lifetime and baseline tiling/cache evidence;equal index rows do not prove equal query output. '
            'v123 MoE placement/packet tradeoff andv122 fanout are prior answers,not unread code. '
            'Original9/1352,physical capacities/comparable MFU/lowest and full graph/branch coverage remain. Pad v120 grant consumed,no equivalent retry/pending device approval.')
    obligations.update(parent='coverage-current-obligations-v123.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v124.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v123.json'))
    audit.update(previous_goal_turn='PROGRESS:v123 saved hyperedge placement/packet composition,215 mappings/435 comparisons;postcheck PASS467287',
        input_checkpoint=123,followup_checkpoint=124,sparse_index_semantics_v124=detail)
    for r in audit['requirements']:
        if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES':r.update(branch_records=358)
        if r['id']=='CURRENT_FINITE_GRAPH_LEDGER':
            r.update(evidence=['operator-graph-coverage-audit-v124.json'],rows=[854,1346],
                gap='G9 adds41 source nodes/49 edges and selected helper producer/consumer mapping. All connected sets2..4 remain structural;complete native graph/parameters/branch and numeric/reader closure is unproved.')
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v124.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v124.json',dict(
        status='REUSE_640_ORDERED_PROFILE_FRONTIERS_THEN_ONLY_INDEX_PRODUCER_PREPARE_AND_CONSUMER_SEMANTICS',
        questions=read(O/names[0])['operator_questions'],source_segments='Selected existing frozen prepare/QLI/indexer/schema/DSpark/PA-gather segments only;no repository-wide framework audit.',
        new_source_downloads_or_remote_device_Torch_model_work=False,
        missing_payloads_not_inferred_from_shapes=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'sparse-index-validation-history-v124.json',dict(events=[
        'v123 revalidated PASS467287/manifest64. Prior turn progressed by recovering hyperedge/owner evidence and joint packet-placement counts,not by testing another native probe.',
        'All46 prior structural pair hints already had central conditional clauses. Select a genuine named M09/M10 gap instead of arbitrary460 empty central pairs:valid sparse duplicate semantics versus invalid padding and cross-query reuse.',
        '640 existing QLI/prepare/SMLA frontiers/1920TASKs preserve512root128phase-unbound counts;no ID payload or actual duplicate rate inferred. DSpark descriptor domain is source-qualified,not attested merely by a QLI chain.',
        'Source prepare filters/sorts every slot and sums valid occurrences;it never deduplicates. Normal nonoverflowing visibility<=INT32_MAX domain required. Reversible normal FLOAT bit keys preserve multiset and avoid NaNs/zeros in encoded domain.',
        'Local source QLI lane IDs are distinct,but global disjoint S2/LD partitions/record-preserving merge are explicit gates. Pinned V41 caller omits output_idx_offset/schema None;generic nonzero branch adds to -1 and can create positive repeated IDs,so no blanket unique-output claim.',
        'SMLA selected PA path consumes occurrences. Negative indices already skip valid KV copies;paired copies can order by physical offsets,so preserve emitted-row/numerical order and page epochs rather than assuming chronological logical-ID order alone.',
        'Sink counterexample:duplicate value1/score0 twice plus sink0 yields2/3,not dedup1/2. Declared BF16 exp/PV boundary witness 257/512 rounds0.5:three originals numerator1.5 versus aggregate BF16 1.5078125;final examples0.59765625/0.6015625 differ. Not native exp/log execution.',
        'Selected noncausal DSpark logical branch repeats one contiguous request interval across its query rows. New descriptor references preserve per-query outputs and require changed consumer ABI/full reader/capture lifetime;native KV reuse may already be baseline.',
        '7500 multiset references,1680 independent interval/broadcast references,10006 key boundary/sample values and9 metadata-layout models checked. Source-style reference arithmetic is not physical traffic/time/MFU.',
        'G9 three helper/source templates add41nodes49edges to partial catalog854/1346. Two new structural hints F12/F31 and F13/F31 already have REL73-28/REL73-27;48hints do not increase135 central pairs or prove joint legality.',
        'Independent connected-set2..4 counts match G9;two new semantic families M09-OP-12/M10-OP-15 produce358rows/26modules. Original9/1352 and849PMU16phase3136ambiguity remain;whole module closures0. No server/device/Torch/model/implementation.'],preflight=pre,overall_complete=False))
    gap=(
        'Need actual sparse ID/valid-row/position/length values,source QLI partition/merge/offset and loaded numerical identity. '
        'Global uniqueness is a conditional producer certificate,not inferred from TopK shape;physical pointer equality is not key/epoch identity. '
        'Preserve valid multiplicity,sink once,BF16 exponent/PV and accumulation order,actual physical-pair gather order and all original masks/bank/page/request epochs. '
        'Same-request index descriptors or KV-load reuse require consumer ABI/coherent publication/capture/all-reader retirement and baseline native tiling/cache evidence. '
        'No native duplicate collapse,new physical traffic/time/MFU/lowest or full source/native branch coverage is established.')
    branches={}
    matrix=read(O/'module-phase-branch-matrix-v123.json')
    newrows={
        'M09':dict(id='M09-OP-12',branch='Visible sparse-index multiset and conditional producer uniqueness;normal INT32/FLOAT-bit sort domain and generic output-offset exception.',
            finding='Prepare preserves valid duplicates and counts occurrences. Source local IDs are distinct per tile;global set certificate additionally requires disjoint partitions/merge/offset/range. Pinned caller offset None is not historical loaded ABI proof.'),
        'M10':dict(id='M10-OP-15',branch='Within-query repeated-key multiplicity versus same-request repeated index-row/interval and KV reuse;per-query outputs remain distinct.',
            finding='Dropping duplicate valid keys changes sink/softmax mass;real multiplicity folding does not preserve native BF16 rounding automatically. DSpark same-request logical window repeat supports a new descriptor/read-reuse question,not output reuse or measured gain.')}
    for mid,row in newrows.items():
        rows=deepcopy(next(m['branches']for m in matrix['modules']if m['id']==mid))
        assert row['id']not in {r['id']for r in rows}
        rows.append(dict(**row,state='CONDITIONAL_SOURCE_OPERATOR_BRANCH_REVIEWED;NATIVE_AND_WHOLE_GATES_OPEN',
            evidence=names,gap=gap,scope='Selected fixed-source normal integer/noncausal logical/PA branches;actual ID payload/rates/native bit identity unbound.'))
        branches[mid]=rows
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v124 adds2 sparse-index occurrence/request-reuse semantic families=>358/26 andG9 source41/49=>854/1346/48hints. Two added hints map existing central clauses;135/26/460 unchanged. Valid multisets preserve multiplicity/sink/rounding;DSpark equal index rows do not imply equal outputs. No new native regimes or whole closures.')
    report='''本轮转向注意力侧，核对稀疏索引重复是否能作为融合/重组收益。复用已有640条QLI→prepare→SMLA顺序前沿，共1920个TASK，保留512条root绑定、128条phase未绑定；这些profile没有索引值、重复率、每query gather次数或新方案ready事件。因此本轮只对影响判断的已有冻结算子代码局部取证，不把shape当成重复数据证据，没有服务器、Torch、NPU或模型运行。

prepare_indexer_indices并不去重。它按slot过滤selected>=0且selected<visible，替换为INT32_MAX sentinel，经可逆FLOAT bit-key排序后恢复ID，最后sentinel变-1；长度是valid slot的数量。例如selected=[1,1,-1,5]、position=3、ratio=1时输出[1,1,-1,-1]，长度2而不是distinct数1。结论限定position+1不溢出、0<=visible<=INT32_MAX的正常整数域；不能跨越位宽/sentinel边界移植。7500个multiset参考用例保持每个有效ID的出现次数，另复核了10006个边界/样例key值的有序可逆编码，未执行Triton或native sort。

QLI源码的单个S2 tile使用ArithProgression加cuBaseS2Idx，局部有效lane ID确实不同。但全局有效TopK唯一性还需要每query的S2/LD分区不重叠、sort/merge保留record以及offset/range一致，不能单凭TopK输出shape证明。固定V41 ordinary caller没有传output_idx_offset，注册schema默认None；另一个通用非零offset分支会对包括-1在内的输出slot统一加offset，可能把无效填充变成正ID。该分支没有被断言在当前case生效，也不是实测bug；它说明“所有调用都唯一/填充总为负”不能脱离调用域推广。

SMLA所读的是索引出现项。源码先按sparseBlockSize展开每个slot，再检查真实position，PA分支经request block table取地址。对正blockSize，-1只展开负位置，已有代码会跳过对应有效KV拷贝；不能把重复的-1 padding当成重复有效key载荷，重新领取整份gather/QK/PV收益。CopyInKv还会根据物理offset选择成对拷贝的最小起点/绝对stride，重叠或范围不适合时回退两次SingleKv；新gather/reuse必须保留实际发出的行序与数值契约，不能只凭逻辑索引已按时间排序就假设物理累加顺序相同。

若一个query/head确有重复有效项，直接删除会改变softmax权重总量。例如同一score0/value1出现两次，另有sink score0且sink value0，原输出为2/3，删为一项后为1/2。精确实数域可以按组保留multiplicity c：denominator=exp(sink-m)+sum(c*exp(score-m))，numerator=sum(c*exp(score-m)*value)，sink仍只出现一次；但前提是组内完整有效score、value、mask/bias/position和bank epoch都一致，不能只比较裸指针。

实数式也不是无条件native bitwise替换。声明的source-style BF16 exponential/PV参考中，exp值257/512舍入为0.5，三份先舍入再参与PV得到1.5；先合并exp再BF16舍入得到1.5078125。带一份exp1的背景/sink时，参考最终BF16输出分别为0.59765625和0.6015625。这个有限反例没有测native exp/log或MMA，只是否定“加log(c)/乘c总能无损替换”这一通用推断。保留原occurrence顺序、复用一次KV读取后重复原算术，是另一候选域；它也须证明读者、舍入与并发写入契约，不能自动把所有QK/PV工作删掉。若真实producer给出唯一有效ID证书，则去重是恒等操作，没有这份重复算术收益；当前归档没有证明其全局native证书或实际重复率。

DSpark非因果、use_logical_indices=True的另一个来源给出了跨query的结构性重复：每request先构造start=max(seq_len-query_len-window,0)、visible=seq_len-start的连续逻辑窗口，再按该request query_len重复同一索引行和长度。1680个含零query请求、不同prefix/window的独立参考核对了恢复顺序和长度。它支持一份request描述符供多个query读取的条件候选，也给v60既有KV-union敏感IO公式提供了新的source限定实例；不是再加一份独立IO收益。native可能已经通过tiling或cache复用了KV，profile缺少相应证据，未认领新的HBM或耗时减少。

同一个索引窗口不表示query相同。相同keys=[0,1]、values=[0,1]、无sink时，q=0与q=log(3)的实数输出分别1/2和3/4。每query的QK/softmax/PV、位置变换、mask与输出仍有需求。新descriptor接口必须支持request owner、cache/page/head/layout/compression/mask generation，并保证index/length同步发布、capture固定地址、旧reader及queued writer退役；现有persistent bank是基线，不能直接拿8B替换当前native期望的整行buffer。

9个明确布局参考分开计入descriptor和INT32 query-owner。例B=1、T=3、index width256，当前活跃indices+lengths内容3084B；一份request index row+length+owner为1040B，区间(start,length)+owner为20B。差额2044B/3064B仅是选定新ABI的逻辑内容参考，不是实际分配、HBM必要字节或可达提速。原_max_tokens持久容量、其它读者、对齐/metadata和新地址生成开销仍需计入。理论状态保持SYMBOLIC_CEILING，数值时间下界/吞吐上界/实测上界比为未知，没有从descriptor字节推导MFU。

把三个以前未展开到primitive的selected source分支加入G9：prepare multiset、DSpark logical-window broadcast和SMLA PA occurrence gather，共41个source节点、49条边。部分全图台账从813/1297更新为854/1346；source行不等于唯一native算子，连通2..4节点集合只是结构枚举，未证明整张图或所有条件分支闭合。两个新结构提示F12/F31与F13/F31已有中央REL73-28/REL73-27，结构提示46→48，中央条件关系仍135、联合组26、待核对460。现有REL83-08与上述两条关系和HG83补入了语义证据。

新增M09-OP-12、M10-OP-15两个语义families，台账356→358行、仍26模块；没有因局部参考用例或新增source图而关闭整模块。原9类/1352次原生字段、849缺PMU、16 phase未绑定图及3136含糊归属均不变。实际ID/mask/page epochs、全局QLI分区与native算术、完整读者/生命周期、物理容量上界及同口径最低利用率仍缺关键证据，整体目标继续。v120 Pad授权已完成消耗，本轮没有新的设备运行。
'''
    action=(
        'Continue original Goal;v124 PROGRESS on sparse index multiset/request-window source semantics andG9 graph mapping. '
        'Read sparse-index summary/source-contract/multiset-rounding/request-reuse/profile-scope/G9audit/current-obligations-v124;verify_sparse_index_semantics_v124 must PASS. '
        'Predecessorv123 PASS467287/manifest64. Reuse640QLI/prepare/SMLA frontiers1920TASKs512root128unbound,no ID payload/rates. '
        'Prepare preserves visible valid-ID multiplicity and counts slots,not distinctIDs;normal no-overflow0<=visible<=INT32_MAX. LocalQLItile unique,globalS2/LDpartition/merge/offsetcertificate notproved. Caller omitsoutput_idx_offset/schemaNone;generic nonzerooffset adds to-1 so retainbranch,not currentbugclaim. '
        'Dropping duplicates changes sink mass2/3vs1/2;exactrealcountfold differs afterBF16 exp/PV witness257/512=>0.5,3copies1.5 vs aggregate1.5078125,final0.59765625vs0.6015625. Notnativeexecution. '
        'SMLA invalid-1 already skips valid KVcopy;preservephysical-pairgatherorder,not onlychronologicalIDs. Equalphysicalpointer isnotlogicalkey/epoch. '
        'SelectedDSpark noncausal logicalbranch repeats one contiguousrequestwindow forqrows. Descriptor/KVreadreuse isconditional;queriesoutputsremain distinct;nativeKVreusealreadypossiblebaseline. '
        '9rawlayoutmodels B1T3I256 current3084B/request-row1040B/interval+owner20B,not allocator/HBM/gain. NeednewconsumerABI,allreaders/capture/bank/page/head/mask/owner generation and retirement. '
        '7500 multiset refs1680interval refs10006key samples;G9three selectedsourcegraphs41nodes49edges=>partial854/1346/48hints.2newhintsmap existingF12F31/F13F31clauses;no centralpairs added135/26/460. '
        'M09OP12/M10OP15 newsemanticfamilies=>358/26;35model7legacy0wholeclosed,2073/285598 conditional9/1352 pending849PMU16phase3136ambiguity unchanged. Symbolicphysicalceiling/noallMFU. '
        'All local jobs complete,no server/Torch/NPU/model/implementation;Padv120grant consumed,no pending/equivalentprobe. Do not edit sealedv124/rerunfinalizer. '
        'Global fusion-scan per-node/edge pointer nowG9v124. Continue genuine missing operator branches/relations or matching-call evidence,not arbitraryblankpair enumeration. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    fusion=deepcopy(read(R/'fusion-scan.json'))
    assert fusion['per_node_edge_coverage_ledger']['summary']==str(O/'operator-graph-coverage-audit-v101.json')
    write(O/'fusion-scan-before-v124.json',fusion)
    fusion['per_node_edge_coverage_ledger']=dict(summary=str(O/'operator-graph-coverage-audit-v124.json'),
        nodes=str(O/'operator-graph-node-coverage-v124.json'),edges=str(O/'operator-graph-edge-coverage-v124.json'),
        graph_catalog=ga['graph_catalog'],structural_only=True,all_native_or_semantic_variants_complete=False)
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=124,previous=123,title='Sparse-index occurrence semantics and same-request descriptor reuse',
        summary=dict(filename='sparse-index-summary-v124.json',
            previous_turn_classification='PROGRESS:v123 saved hyperedge placement/packet composition;PASS467287',
            **detail,refined_candidates=3,refined_modules=3,branch_records=358,semantic_modules=26,
            relation_clauses=135,higher_order_groups=26,pending_central_pair_reconciliation=460,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            theory_state='SYMBOLIC_CEILING',actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            Pad_authorization='GRANTED_AND_CONSUMED',implementation=False,native_test=False,
            remote_job_status='All local profile/source/multiset/descriptor/graph checks finished. No remote/Torch/NPU/model/replay/implementation;completed Pad grant not reused.'),
        changes=[
            'Resolve visible-index multiset/length semantics and local-versus-global uniqueness/offset domains;retain occurrence,sink,BF16 rounding and actual PA gather order.',
            'Derive selected noncausal DSpark per-request logical-window repeat;separate descriptor/KV-load reuse from query-output reuse and price9 explicit metadata layouts without native/HBM gain claims.',
            'Add G9 selected-source41node49edge primitive graphs and2 existing-clause hints;extend M09/M10 with2 semantic families,keep whole/numerical/native coverage open.'],
        candidate_updates={c:dict(review_v124=deepcopy(detail),conclusion_changing_gap=gap)for c in ['F12','F13','F31']},
        module_updates={
            'M09':dict(candidates=['F12','F31'],finding='Prepare preserves valid multiset and occurrence length. Local source IDs are unique;global uniqueness/merge/offset/range needs explicit certificate. New source graph exposes filter/bit-sort/publication primitive boundaries.',evidence=names,gap=gap),
            'M10':dict(candidates=['F13','F31'],finding='Repeated valid keys retain softmax multiplicity/sink/rounding and physical gather order. Equal per-request index windows permit descriptor/KV-read reuse questions,not reuse of query-dependent attention output.',evidence=names,gap=gap),
            'M21':dict(candidates=['F31'],finding='Selected DSpark logical-window builder repeats one request interval/length across query rows;new compact descriptor ABI is conditional on ownership/mask/capture/all-reader generation and extra metadata/address costs.',evidence=names,gap=gap)},
        semantic_branches=branches,remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Reuse640orderedQLI/prepare/SMLAfrontiers1920TASKs,read only selected frozen index-producer/schema/DSpark/PA-gather source;no new native payload or devicework.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='7500multiset/1680interval/10006key references,finite multiplicity/BF16counterexamples,9metadata layouts and independent G9connected2..4 sets;2newsemanticfamilies,allwholecoverageopen.',artifacts=[O/'sparse-index-preflight-v124.json',O/'sparse-index-validation-history-v124.json',O/'coverage-current-obligations-v124.json',O/'whole-scope-requirement-audit-v124.json'])],
        report=report,read_first=names+['coverage-current-obligations-v124.json'],publication_rows={},
        gate_updates=dict(overall_complete=False),scripts=[N/'analyze_sparse_index_semantics_v124.py',N/'verify_sparse_index_semantics_v124.py',N/'reconcile_operator_graph_coverage_v88.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='sparse-index',main_script=Path(__file__),source_read=True,
        additional_shared_writes={R/'fusion-scan.json':fusion})


if __name__=='__main__':main()
