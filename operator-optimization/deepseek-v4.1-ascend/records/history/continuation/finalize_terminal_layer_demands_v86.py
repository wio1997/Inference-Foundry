"""Persist terminal-layer demand/state cut, keeping all unresolved global scope."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v86.json').exists()
    names=['terminal-layer-prior-evidence-v86.json','terminal-layer-operator-source-v86.json',
      'terminal-layer-native-frontier-v86.json','terminal-layer-demand-cut-v86.json',
      'terminal-layer-owner-and-work-models-v86.json','terminal-layer-weight-and-dependency-proof-v86.json',
      'terminal-layer-composition-clauses-v86.json']
    packet=read(O/names[-1]);assert len(packet['clauses'])==10 and len(packet['higher_order_groups'])==2
    relations=deepcopy(read(O/'candidate-relation-clauses-v85.json'))
    relations['clauses']+=deepcopy(packet['clauses']);relations['higher_order_groups']+=deepcopy(packet['higher_order_groups'])
    for c in relations['clauses']:
        if c['id']in packet['refine_existing_relations']:
            c['terminal_demand_refinement_v86']=dict(evidence=names,
              prior_selector_and_row_union_rules_not_new=True,all_readers_closed=False,
              conditional_cut='Final-output U can affect Q/output/FFN under all-reader/numeric/state guards; full required KV/aux/preceding-layer input remains.',
              no_double_credit='Head already has K rows; localized/private/selected graphs share one baseline; no extra K-row head savings.')
    relations.update(clause_count=105,selected_pair_count=105,no_new_maincandidate=True,overall_complete=False)
    write(O/'candidate-relation-clauses-v86.json',relations)
    cov=deepcopy(read(O/'candidate-relation-coverage-v85.json'))
    for c in packet['clauses']:
        pair=next(p for p in cov['pairs']if {p['a'],p['b']}=={c['a'],c['b']});assert not pair['clauses']
        pair.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(scoped_conditional_pairs=105,remaining_pair_relations=490,higher_order_groups=21,
      parent_relation_index='candidate-relation-clauses-v85.json',parent_coverage='candidate-relation-coverage-v85.json',
      terminal_demand_refinement_v86=dict(new_pairs=10,new_groups=2,evidence=names,all_readers_or_state_writers_closed=False))
    write(O/'candidate-relation-coverage-v86.json',cov)
    footprint=deepcopy(read(O/'candidate-operator-footprints-v85.json'))
    registry={r['id']:r for r in read(O/'candidate-record-index-v85.json')['model_candidates']}
    extra_modules=['M07','M10','M11','M12','M13']
    for row in footprint['candidates']:
        p=R/'candidates'/(row['id']+'.json');assert hashlib.sha256(p.read_bytes()).hexdigest()==registry[row['id']]['master_sha256']
        row['captured_master_sha256']=registry[row['id']]['master_sha256']
        if row['id']=='F26':
            row['operator_atom_union']+=['terminal.query_output_demand','terminal.FFN_output_demand','state.KV_writer_demand_cut']
            row['modules']=sorted(set(row['modules']+extra_modules))
            row['source_scope_v86']='Conditional terminal-layer output demand; full KV/state branch preserved. Module links are analytical navigation,not altered native assignment.'
    footprint.update(parent='candidate-operator-footprints-v85.json',overall_complete=False)
    write(O/'candidate-operator-footprints-v86.json',footprint)
    nav=deepcopy(read(O/'candidate-pair-navigation-v85.json'));byid={r['id']:r for r in footprint['candidates']}
    for pair in nav['pairs']:
        pair['conditional_clause_ids']=[c['id']for c in relations['clauses']if {pair['a'],pair['b']}=={c['a'],c['b']}]
        pair['semantic_pair_reconciliation_done_in_stated_scope']=bool(pair['conditional_clause_ids'])
        pair['common_module_associations']=sorted(set(byid[pair['a']]['modules'])&set(byid[pair['b']]['modules']))
        pair['shared_operator_atom_union']=sorted(set(byid[pair['a']]['operator_atom_union'])&set(byid[pair['b']]['operator_atom_union']))
    nav.update(scoped_conditional_pairs=105,remaining_central_pair_reconciliation=490,parent='candidate-pair-navigation-v85.json',overall_complete=False)
    write(O/'candidate-pair-navigation-v86.json',nav)
    ops=deepcopy(read(O/'operator-analysis-coverage-ledger-v85.json'))
    ops['terminal_layer_demand_refinement_v86']=dict(evidence=names,reused_ComputeInfo_executions=2800,last_layer_graphs=[80,64,16],
      new_regimes_or_executions_promoted=0,all_readers_closed=False,new_target_or_device_work=False)
    write(O/'operator-analysis-coverage-ledger-v86.json',ops)
    matrix=read(O/'module-phase-branch-matrix-v85.json')
    selected={'M04':{'M04-OP-09'},'M06':{'M06-OP-11','M06-OP-12','M06-OP-13'},
      'M07':{'M07-OP-04','M07-OP-05','M07-OP-08','M07-OP-09'},'M10':{'M10-OP-05','M10-OP-07'},
      'M11':{'M11-OP-06','M11-OP-07','M11-OP-14'},'M12':{'M12-OP-01','M12-OP-08','M12-OP-13','M12-OP-14'},
      'M13':{'M13-OP-12','M13-OP-13','M13-OP-15'},'M15':{'M15-OP-05','M15-OP-07','M15-OP-09'},
      'M16':{'M16-OP-01','M16-OP-07'},'M21':{'M21-S05','M21-S07','M21-S15'},
      'M22':{'M22-S12','M22-S13'},'M23':{'M23-OP-05','M23-OP-06','M23-OP-15'},'M24':{'M24-OP-06','M24-OP-10'}}
    branches={m['id']:deepcopy(m['branches'])for m in matrix['modules']if m['id']in selected}
    gap=('Complete raw/normalized/output readers and actual I/J/U/owner values remain unbound; prior v31 readiness/reader rules and v65max(U_p) are not new proofs. '
      'Native changed-shape precision,SP/CP/TP/EP mapping,causal positions,all KV/aux/EPLB state effects,capture lifetimes and actual capacities/critical path remain open.')
    for mid,rows in branches.items():
        assert selected[mid]<={r['id']for r in rows}
        for row in rows:
            if row['id']in selected[mid]:
                row['evidence']+=names
                row['terminal_demand_review_v86']=dict(finding='Conditional last-layer output demand stops at full required KV/aux/state branch. Owner quota is max(u_p); all headK rows and cold weight planes remain separately priced.',gap=gap)
    newrows={
      'M10':dict(id='M10-OP-12',branch='Final target-layer selective-query graph: preserve full-required KV/cache and original causal coordinates while conditionally restricting Q/output/attention-Post/FFN to complete final-reader union.',
        finding='80last-layer39graphs have35typed ComputeInfo tasks each. Selected compact query ordinals are not positions; full last-layer KV input demand blocks output-only arithmetic pruning of preceding layers.'),
      'M12':dict(id='M12-OP-16',branch='Final-layer FFN row-demand specialization across HcPre/RmsNormCast/router/shared/routed/merge/Post,conditional on all output and routing-state observers.',
        finding='Conventional router/shared/6-expert source work scales with selected origin rows,not necessarily receiver physical MC2 blocks or cold weight planes. Preserve EPLB/heat/load/ID/parameter epochs and all native rounding.')}
    for mid,row in newrows.items():
        assert row['id']not in {r['id']for r in branches[mid]}
        row.update(state='CONDITIONAL_TERMINAL_OUTPUT_DEMAND_AND_STATE_WRITER_CUT_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
          scope='Pinned final target layer and stated row-reader/numerical/ownership domain; no implementation or all-reader closure.')
        branches[mid].append(row)
    write(O/'terminal-layer-branch-inventory-v86.json',dict(modules=branches,new_inventory_families=['M10-OP-12','M12-OP-16'],
      prior_row_selector_or_sparse_publish_rules_counted_new=False,branch_records=347,semantic_modules=24,refined_modules=13,untouched_other_modules=13,
      native_attribution_changed=False,whole_modules_closed=0,overall_complete=False))
    members=sorted({c[k]for c in packet['clauses']for k in ['a','b']})
    updates={cid:dict(review_v86=dict(evidence=names,conditional_terminal_demand_analysis=True,all_readers_closed=False,
      relation_ids=[c['id']for c in packet['clauses']if cid in[c['a'],c['b']]],implementation=False,native_test=False),
      conclusion_changing_gap=gap)for cid in members}
    updates['F26'].update(profile_review_state='CONDITIONAL_LAST_LAYER_QUERY_FFN_DEMAND_AND_FULL_KV_STATE_CUT_REVIEWED',
      next_profile_or_contract_action='Do not rediscover v31 head-selector readiness or v65 owner quota. Actual prompt/other observer union and row owners still unknown. Next extend remaining operator combinations and capacity cuts using saved profile/primary evidence; full KV/aux/EPLB state demand stays explicit.')
    mods={mid:dict(finding='Conditional final-output demand graph crosses final Q/output/FFN but preserves full required KV/aux/preceding-layer state inputs. Concrete owner-quota and typed work/weight models refine existing row-selection rules.',
      evidence=names,candidates=sorted(set([cid for cid in members if any(cid in m['candidate_links']for m in matrix['modules']if m['id']==mid)]+(['F26']if mid in extra_modules else[]))),
      gap=gap,profile_scope='Operator row dependence,fusion/split/reorganization and theoretical resource constraints; no framework research.')for mid in selected}
    write(O/'PROFILE-CODE-QUESTION-GATE-v86.json',dict(status='PROFILE_LED_TERMINAL_OPERATOR_DEMAND_CUT',
      questions=['Where does final-output row demand stop at future KV/state writers?',
        'Which Q/output/FFN operators can conditionally share the same selected row generation?',
        'How do actual owner quotas,cold weights and EP receiver work limit the theoretical opportunity?'],
      prior_v31_head_selector_and_v65_quota_not_new=True,local_pinpoint_source_files=3,new_target_source_files_obtained=0,
      new_remote_source_read=False,new_public_document_read=False,no_model_Torch_NPU_server_or_capture_work=True,overall_complete=False))
    write(O/'terminal-layer-question-status-v86.json',dict(prior_answers_reused=['Head I ready before forward (v31)','Prompt/state normalized readers (v31)','8max(U_p) sparse publication (v65)'],
      newly_reviewed_conditional_scope=['Terminal query/output/FFN demand propagation with full KV/state cut','Concrete owner maximum envelopes and typed work/weight resource domains'],
      still_open=['Actual all-reader I/J/U/owner plans and prompt guard','Native changed-shape precision/ABI','All state side effects/aliases/lifetime','Physical capacities/critical-path benefit'],
      all_readers_closed=False,whole_modules_closed=0,overall_complete=False))
    write(O/'terminal-layer-validation-history-v86.json',dict(errors=[],
      schema_label_refinement='Before persistence renamed identical_typed_operator_schema_count to operator-type multiplicity schema count. Types/counts agree; per-case shapes/dtypes retained without claiming cross-case equality.',
      prior_evidence_novelty_checked=True,all_local_analysis_jobs_complete=True,new_target_execution=False,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v85.json')['remaining'])
    remaining.append('v86: v31 headI/readers and v65 maxUp were prior,not new. Exact80lastlayer39graphs/2800ComputeInfo/35nodes each64root16unbound support conditional backward output-demand cut acrossQ/output/FFN; preserve full requiredKV/aux/previous-layer input and EPLB state. ActualI/J/U/owners/prompt/alias/native numerics unbound.5root-shapeU=K illustrative cases88roots; N24U12 equal-count16..24slots,not12; K=N alone not U=N. Coldplanes and EP16 receiver work not proportional.10newpairs2groups=>105/21/490pending.347records24semantic,35model7legacy,0wholeclosed;2073/285598review9/1352pending and849/16/3136 unchanged.')
    report='''本轮把最终输出的选行需求向前追到最后一层的算子，并标出了必须保留的状态分支。

保存的80组最后层39计算图，每组有35个相关ComputeInfo执行，共2800项既有执行。其中64组阶段已绑定，16组仍未知。Q、KV、attention输出投影、两次HC、FFN及缓存写入按原执行身份保留；这不是新采集或新PMU证据。

条件图允许将完整最终读者并集U向前传到最终Post、FFN、attention输出及Q分支。它不能直接删除KV/cache写入：后续迭代仍会读这些值。因此，最后层输入仍可能需要全部有效行，不能再按U/N删除前一层计算。所有辅助checkpoint、原始query位置、因果窗口、KV槽、EPLB负载观察及原生舍入都保留约束。当前尚未证明新图的全部适用条件。

head索引在forward前生成、prompt额外读者和全读者并集在v31已有结论，等长通信8max(U_p)规则在v65已有。本轮只增加具体的所有者极值预算：TP8、24行、假设U为12个不同有效行时，等长通信需16～24个行槽；最差分布下没有载荷下降。实际I、prompt集合J、U和各卡分布仍未知，不能把这一例子当成已测收益。

理论工作量按BF16、INT8、FLOAT及其他资源分别计数。Q按查询所有者u_p、输出投影按各权重所有者的Uglobal行计；6U只是该源组的专家分配参考，不能替代EP16接收方实际padding。只要还有正数行，常规GEMM仍可能读取整片权重。现有head本就只计算K行，本轮不能再给head记一次N→U收益。

新增10条条件关系、2组联合合同，累计105条关系、21组合同，中央仍有490对待汇入或复核。分支台账新增最后层查询/状态分界和最终FFN行需求两项，共347条记录、24个语义模块。未关闭整模块，整体目标继续。
'''
    action=('继续原Goal，v86为PROGRESS，v85亦PROGRESS。先读terminal-layer-prior-evidence/summary/PROFILE-FINDINGS-v86、operator-source/native-frontier/demand-cut/owner-and-work-models/weight-and-dependency-proof/composition-clauses。'
      '禁止再把v31 headI在forward前就绪、prompt/state额外读者/rowunion和v65 8maxUp规则算新发现。当前真实新增是80lastlayer39/2800ComputeInfo逐身份35节点图与条件最终输出需求传播/完整KV状态切口、具体5种根shape88roots的owner极值和typedwork/weight联合成本。'
      'U=distinct union(headI,promptJ,allotherreaders)，I含order/duplicates/pad，K=N不等于U=N；实际I/J/U/owner/prompt仍unknown。q=maxUp满足ceil(U/P)..min(U,N/P)。N24U12仅假设，需16..24slots不保证通信降。'
      'Q/attention输出/最终FFN可以条件selected；attentionHCPre/inputNorm及KV/cache仍有完整state需求，前层fullrows不可按最终U删除。保3aux、originalposition/causal/sink/slot/acceptedrollback和EPLB load/heat/counters/weights所有观察者。Prune输出与删state不同；fusion可省接口但必要值仍需算。'
      'QsourcegroupU与WO每TP权重ownerUglobal不同，6U不能当EP16receiverwork；coldweightplanes不随行数线性下降。HcPreHF32/attentionkeys/quant metadata与物理capacity仍单列。Head已有K不重复计减行收益；local/private/selected三图信用不能独立相加。'
      '10pairs2groups=>105pairs21groups490中央pending；347records24semantic/35model7legacy，F26跨M07/M10/M11/M12/M13关联只是分析导航，native attribution/regimes不改。2073/285598review、9/1352pending、849/16/3136、0wholeclosure保持。'
      '下一步继续剩余跨算子/跨层/跨卡/迭代条件组合与必要工作/硬件容量切口，优先真正未分析模块/分支或会改结论的证据；不能为成稿complete，也不以重复旧源码答案作进展。仅已有profile/必要定点代码/公开原始资料，no remote/model/Torch/NPU/capture/service/processwork。GoalACTIVE/TaskACTIVE/LoopEVALUATING，verify_terminal_layer_demands_v86应PASS，不回放finalizer。')
    preflight=O/'terminal-layer-analysis-preflight-v86.json';assert read(preflight)['status']=='PASS'
    sourcepaths=[Path(r['path'])for r in read(O/names[1])['source_files']]
    jobs=[dict(kind='review',status='pass',description='Reuse3pinned operator/caller files and priorv31/v65 answers without novelty promotion; output-demand/state-writer cut remains conditional.',artifacts=[O/names[0],O/names[1],O/names[3]]+sourcepaths),
      dict(kind='profile',status='pass',description='80lastlayer39 graph identities/2800ComputeInfo/35operator tasks each64root16unbound;per-case shapes/dtypes retained,not equal-shape or critical-path proof.',artifacts=[O/names[2]]),
      dict(kind='design-check',status='pass',description='5recorded-root-shape conditional owner/work models,6extremal examples,cold-weight/state dependency cut and10pair/2group clauses; no actual speedup or new native execution.',artifacts=[O/n for n in names[4:]]+[preflight])]
    persist_refinement(version=86,previous=85,title='Conditional terminal-layer query/FFN demand and full KV-state cut',
      summary=dict(filename='terminal-layer-summary-v86.json',previous_turn_classification='PROGRESS:v85conditionaltailandcommonbudget',
        prior_head_selector_answer_version=31,prior_owner_quota_rule_version=65,local_pinpoint_source_files=3,new_target_source_files=0,
        last_layer_graphs=[80,64,16],reused_ComputeInfo_executions=2800,operator_tasks_per_graph=35,root_shape_models=5,modeled_head_roots=88,owner_extremal_examples=6,
        all_readers_closed=False,all_state_writers_closed=False,new_pairs=10,relation_clauses=105,new_groups=2,higher_order_groups=21,pending_central_pair_reconciliation=490,
        operator_footprints=35,pair_navigation_rows=595,refined_candidates=len(members),refined_modules=13,new_inventory_families=2,branch_records=347,semantic_modules=24,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
        new_regimes_or_executions_promoted=0,implementation=False,native_test=False,new_remote_source_read=False,actual_physical_ceiling_or_speedup_proved=False,
        remote_job_status='All local saved-profile/source/math analysis finished; no remote/model/Torch/NPU/capture/service/process work.'),
      changes=['Reused v31/v65 reader/selector/quota facts without recounting;3pinpoint local source files only.',
        'Bound80last-layer39graphs/2800ComputeInfo and separated conditional Q/output/FFN pruning from full required KV/aux/preceding-layer state demand.',
        'Derived concrete owner-quota endpoint models and typed work/weight costs,including EP16 receiver and cold-plane limits.',
        '10new pairs2groups=>105/21/490pending;347records24semantic,global gaps and GoalACTIVE retained.'],
      candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['terminal-layer-question-status-v86.json','candidate-pair-navigation-v86.json'],
      publication_rows={'F26':'| F26 | 最后层Post、collapse、Norm、选行及条件Q/FFN重组 | 条件 | 完整读者并集可条件前传到最后Q/输出/FFN；KV/cache/aux及EPLB状态另保留。TP8的U12可能仍需16～24行槽；head已有K行、权重和MC2接收方成本不可再按U/N缩放。实际读者/索引/数值/容量仍open。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/n for n in ['analyze_terminal_layer_demands_v86.py','model_terminal_layer_demands_v86.py','reconcile_terminal_layer_relations_v86.py','verify_terminal_layer_demands_v86.py']],
      executor_prefix='terminal-layer',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
