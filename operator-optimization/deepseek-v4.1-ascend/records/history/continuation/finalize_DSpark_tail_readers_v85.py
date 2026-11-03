"""Persist conditional operator tail costs; direct-reader answer is prior v11 evidence."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v85.json').exists()
    names=['DSpark-tail-source-index-v85.json','DSpark-tail-reader-closure-v85.json','DSpark-tail-native-boundaries-v85.json',
      'DSpark-tail-reorganization-variants-v85.json','DSpark-tail-row-local-proof-v85.json','DSpark-tail-work-communication-models-v85.json',
      'DSpark-tail-rounding-witnesses-v85.json','DSpark-tail-transfer-credit-v85.json','DSpark-tail-composition-clauses-v85.json',
      'tail-shared-memory-budget-v85.json','tail-shared-memory-relations-v85.json',
      'DSpark-tail-prior-evidence-reconciliation-v85.json','DSpark-tail-ceiling-domains-v85.json']
    assert read(O/names[11])['prior_answer_first_established_version']==11
    assert not read(O/names[1])['all_snapshot_observers_closed']
    packets=[read(O/names[8]),read(O/names[10])]
    additions=[c for p in packets for c in p['clauses']];groups=[g for p in packets for g in p['higher_order_groups']]
    assert len(additions)==11 and len(groups)==3
    relations=deepcopy(read(O/'candidate-relation-clauses-v84.json'))
    relations['clauses']+=deepcopy(additions);relations['higher_order_groups']+=deepcopy(groups)
    for c in relations['clauses']:
        if c['id']in['REL73-21','REL74-08','REL74-07']:
            c['DSpark_reader_refinement_v85']=dict(prior_v11_direct_drafter_content_answer_revalidated=True,
              all_snapshot_observers_closed=False,remaining_guards=read(O/names[1])['remaining_observer_guards'],
              protected_values='Normalized target rows,3aux,contextKV,draft raw confidence,sampling raw/probability/RNG values remain separate.',evidence=names)
    relations.update(clause_count=95,selected_pair_count=95,no_new_maincandidate=True,overall_complete=False)
    write(O/'candidate-relation-clauses-v85.json',relations)
    coverage=deepcopy(read(O/'candidate-relation-coverage-v84.json'))
    for c in additions:
        pair=next(p for p in coverage['pairs']if {p['a'],p['b']}=={c['a'],c['b']});assert not pair['clauses']
        pair.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    coverage.update(scoped_conditional_pairs=95,remaining_pair_relations=500,higher_order_groups=19,
      parent_relation_index='candidate-relation-clauses-v84.json',parent_coverage='candidate-relation-coverage-v84.json',
      DSpark_tail_refinement_v85=dict(new_pairs=11,new_groups=3,prior_direct_content_answer_version=11,all_snapshot_observers_closed=False,evidence=names))
    write(O/'candidate-relation-coverage-v85.json',coverage)
    footprints=deepcopy(read(O/'candidate-operator-footprints-v84.json'))
    registry={r['id']:r for r in read(O/'candidate-record-index-v84.json')['model_candidates']}
    for row in footprints['candidates']:
        path=R/'candidates'/(row['id']+'.json');assert hashlib.sha256(path.read_bytes()).hexdigest()==registry[row['id']]['master_sha256']
        row['captured_master_sha256']=registry[row['id']]['master_sha256']
        if row['id']=='F03':row['operator_atom_union']+=['mhc.final_ffn_post']
        if row['id']=='F26':
            row['operator_atom_union']+=['state.target.raw_HC_snapshot','mhc.final_ffn_post']
            row['source_scope_v85']='Prior v11 direct DSpark drafter-content answer revalidated; registered/external/native observers and alias/lifetime guards remain open.'
        if row['id']in['F25','F26']and'M02'not in row['modules']:row['modules']=sorted(row['modules']+['M02'])
    footprints.update(parent='candidate-operator-footprints-v84.json',source='Exact v84 master hashes before mutation. Final-Post/common-budget associations are conditional navigation, not native functional attribution.',overall_complete=False)
    write(O/'candidate-operator-footprints-v85.json',footprints)
    nav=deepcopy(read(O/'candidate-pair-navigation-v84.json'));byid={r['id']:r for r in footprints['candidates']}
    for p in nav['pairs']:
        p['conditional_clause_ids']=[c['id']for c in relations['clauses']if {p['a'],p['b']}=={c['a'],c['b']}]
        p['semantic_pair_reconciliation_done_in_stated_scope']=bool(p['conditional_clause_ids'])
        p['shared_operator_atom_union']=sorted(set(byid[p['a']]['operator_atom_union'])&set(byid[p['b']]['operator_atom_union']))
        p['common_module_associations']=sorted(set(byid[p['a']]['modules'])&set(byid[p['b']]['modules']))
    nav.update(scoped_conditional_pairs=95,remaining_central_pair_reconciliation=500,parent='candidate-pair-navigation-v84.json',overall_complete=False)
    write(O/'candidate-pair-navigation-v85.json',nav)
    ops=deepcopy(read(O/'operator-analysis-coverage-ledger-v84.json'))
    ops['DSpark_tail_refinement_v85']=dict(existing_target_tails=104,existing_rawHC_AG=104,existing_premix_AG=104,
      graph_frontiers=[80,64,16],promoted_regimes=0,promoted_executions=0,prior_direct_content_answer_version=11,all_snapshot_observers_closed=False,evidence=names)
    write(O/'operator-analysis-coverage-ledger-v85.json',ops)
    matrix=read(O/'module-phase-branch-matrix-v84.json')
    selected={'M02':{'M02-OP-12'},'M06':{'M06-OP-10','M06-OP-11','M06-OP-12'},'M13':{'M13-OP-12','M13-OP-13'},'M14':{'M14-OP-20'},
      'M15':{f'M15-OP-{i:02d}'for i in range(1,9)},'M18':{'M18-OP-01','M18-OP-02','M18-OP-03','M18-OP-16'},
      'M19':{'M19-OP-10'},'M22':{'M22-S12','M22-S13','M22-S14'},'M23':{'M23-OP-12','M23-OP-14','M23-OP-16'},'M24':{'M24-OP-09','M24-OP-12'}}
    branches={m['id']:deepcopy(m['branches'])for m in matrix['modules']if m['id']in selected}
    gap=('Direct Ascend v1 DSpark drafter-content question answered v11 and revalidated. Registered/external/native snapshot observers, aliases/events/lifetime are NOT closed. '
      'Normalized-row readers/head I readiness, changed-shape numerics, TP gamma equality for local Norm, new ABI and actual capacities/usable memory/critical path remain explicit gates.')
    replacements={
      'M15-OP-03':'104 baseline copies. Prior v11 direct Ascend-v1 DSpark argument does not consume raw HC;3aux are live separate inputs. No-copy is conditional on additional observers/aliases/lifetime and retains buffer control/two gathers.',
      'M15-OP-04':'Prior v11 bufferNone alternative: local collapse→collapsed BF16 AG→global RMS. Remaining observer/alias/lifetime and changed-shape numeric gates stay open. Moving Norm before AG adds equal gamma/eps/numeric policy across TP.',
      'M15-OP-08':'Raw HC/premix pack2→1 remains an alternative where raw is needed. Conditional no-buffer/local-collapse changes payload. Byte or launch reduction does not prove latency dominance.',
      'M06-OP-11':'Interior Post feeds future Pre/Post/aux/Engram. Final FFN Post output may be private only after remaining raw/output-observer and lifetime guards close; input residual/coefficients remain. No extra final HcPre/Sinkhorn.',
      'M23-OP-12':'Packing raw HC/premix keeps payload with a new ABI. Existing local-collapse/one-BF16-AG path is conditional on observer/rank/row/trim/numeric/capture contracts. Direct-content fact preexisting v11.',
      'M24-OP-09':'Direct Ascend-v1 DSpark raw snapshot argument settled v11. Remaining raw observers/aliases/lifetime are open. Target normalized rows,3aux checkpoints,draft raw confidence and memo/state epochs are distinct live generations.'}
    for mid,rows in branches.items():
        for row in rows:
            if row['id']in selected[mid]:
                row['evidence']+=names
                row['DSpark_tail_review_v85']='Prior-v11 direct content answer revalidated, not global observer closure; conditional final-Post/typed cost/common-budget domains refined.'
            if row['id']in replacements:row.update(branch=replacements[row['id']],gap=gap,state='PRIOR_DIRECT_READER_ANSWER_REVALIDATED_AND_CONDITIONAL_OPERATOR_REPARTITION_REVIEWED;WHOLE_MODULE_OPEN')
    branches['M15'].append(dict(id='M15-OP-09',branch='Conditional final FFN Post→collapse→optional local RMS; private final output only after remaining observers/aliases/lifetime are discharged.',
      state='CONDITIONAL_FINAL_TILE_FUSION_AND_TYPED_RESOURCE_BOUNDARY_ANALYZED;WHOLE_MODULE_OPEN',evidence=names,
      finding='Preserve merge/Post/collapse/norm BF16 barriers and Post input residual/coefficients. No extra HcPre/Sinkhorn. Localized92/96MH interface union avoids counting old global and new local intermediates as independent gains.',
      gap=gap,scope='Pinned operator value boundaries and declared row-local/numeric/capacity domains; not framework research.'))
    write(O/'DSpark-tail-branch-inventory-v85.json',dict(modules=branches,new_family='M15-OP-09',clarified_prior_conditional_families=sorted(replacements),
      branch_records=345,semantic_modules=24,refined_modules=10,untouched_other_module_matrices=16,whole_modules_closed=0,overall_complete=False))
    members=sorted({c[k]for c in additions for k in['a','b']})
    updates={cid:dict(review_v85=dict(prior_direct_drafter_content_answer_version=11,all_snapshot_observers_closed=False,evidence=names,
      relation_ids=[c['id']for c in additions if cid in[c['a'],c['b']]],conditional_operator_analysis_deliverable=True,implementation=False,native_test=False),conclusion_changing_gap=gap)for cid in members}
    updates['F26'].update(profile_review_state='PRIOR_V11_DIRECT_CONTENT_ANSWER_REVALIDATED_CONDITIONAL_FINAL_POST_AND_COMMON_BUDGET_REVIEWED',
      raw_snapshot_observer_status='DIRECT_DRAFTER_CONTENT_SETTLED_V11;REGISTERED_EXTERNAL_NATIVE_OBSERVERS_AND_ALIAS_LIFETIME_GUARDS_OPEN',
      next_profile_or_contract_action='Resolve normalized-target readers and head logits_indices availability before forward using named operator value contracts. Preserve remaining snapshot observer/alias/lifetime guards. Do not recount v11 as new proof.')
    updates['F03']['next_profile_or_contract_action']='Keep interior live residual constraints. Final Post→pure collapse may remove its output plane only under remaining observer/alias/lifetime guards; preserve BF16 barriers, Post inputs and tile ownership. No extra final HcPre.'
    mods={mid:dict(finding='Prior v11 direct-content answer revalidated.104tails/208AGs and80last-merge/Post frontiers support conditional work/communication/private-transfer and common-memory-budget models. All-observer closure is unproved.',
      evidence=names,candidates=sorted(set([cid for cid in members if any(cid in m['candidate_links']for m in matrix['modules']if m['id']==mid)]+(['F25','F26']if mid=='M02'else[]))),
      gap=gap,profile_scope='Operator fusion/repartition and theoretical work/storage/communication; no framework catalog or device execution.')for mid in selected}
    write(O/'PROFILE-CODE-QUESTION-GATE-v85.json',dict(status='PROFILE_DRIVEN_NAMED_OPERATOR_QUESTIONS_AND_PRIOR_EVIDENCE_RECONCILED',
      questions=['Direct raw-HC drafter-content? Prior v11 answer revalidated; all observers not closed.',
        'Which conditional final Post/collapse/Norm boundaries can stay private?',
        'Which quantities are necessary boundary costs versus comparison references?',
        'How do raw credit,WKV memo and C2 state compete under one budget?'],
      named_symbol_search='8pinned files; frozen4-reference-file inventory supplements prior installed90matches/26files.',
      new_target_source_files_obtained=0,new_remote_source_read=False,new_public_document_read=False,no_model_Torch_NPU_capture_or_server_work=True,overall_complete=False))
    write(O/'DSpark-tail-question-status-v85.json',dict(
      prior_bounded_question_revalidated=dict(id='ASCEND-V1-DSPARK-DIRECT-DRAFTER-CONTENT',first_answer_version=11,state='PRIOR_SOURCE_ANSWER_REVALIDATED',evidence=names[:3]+[names[11]]),
      still_open=['Additional registered/external/native snapshot observers and alias/event/lifetime','All normalized-target readers/head I readiness','Native numerical/ABI changes','Physical capacities,usable memory and critical-path benefit'],
      reviewed_conditional_questions=['C/G row-local identity and G1/G2/G3 typed costs','One common memory-budget remainder','Fixed-boundary necessary payload and symbolic capacity bounds'],whole_modules_closed=0,overall_complete=False))
    write(O/'DSpark-tail-validation-history-v85.json',dict(errors=[dict(kind='STRICT_REFERENCE_INVENTORY_GUARD',error_type='AssertionError',
      cause='Guard expected config definition file to contain raw-buffer reference; actual frozen symbol set is4files.',
      correction='Exact equality guard corrected, not weakened; abort before outputs/target mutation.',archived_script=str(N/'analyze_DSpark_tail_readers_v85.before-reference-set-fix.txt'),resolved=True),
      dict(kind='LOCAL_PATCH_SHAPE',cause='apply_patch rejected delete+add of same path in one patch before mutation.',correction='Separate delete/add rebuilt unrun finalizer; no target work.',resolved=True)],
      analytical_correction=dict(kind='PRIOR_EVIDENCE_SCOPE_AND_NOVELTY',resolved=True,
        reason='v11 already settled direct content using stronger installed90/26 inventory. Staged drafts overstated novelty/all-observer scope.',evidence=names[11],superseded_staged_drafts='DSpark-tail-pre-reconciliation-v85.json'),
      all_analysis_jobs_complete=True,new_device_work=False,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v84.json')['remaining'])
    remaining.append('v85: Prior v11 direct Ascend-v1 DSpark content answer revalidated, installed90/26 inventory retained; no all-observer/alias/lifetime closure. Conditional final-Post frontiers80graphs64root16unbound,104tails/208AGs incl24DIRECT rawwithoutCompute.320MiB is requested capacity.5refined graph domains/4shape budgets/2BF16witnesses; fixed-boundary Q=2M(h+1)H+4Mh+2H and symbolic capacities, not numeric speedup. One common memory remainder with WKV/C2.95pairs19groups500pending;345records24semantic;35model7legacy/0wholeclosed.2073/285598review,9/1352pending,849/16/3136 unchanged. Next: normalized readers/head selector readiness.')
    action=('继续原Goal。v85条件算子分析PROGRESS。先读DSpark-tail-summary/PROFILE-FINDINGS-v85、prior-evidence-reconciliation、native-boundaries、reorganization-variants、work-communication-models、transfer-credit、ceiling-domains、tail-shared-memory-budget-v85。'
      'v11已有Ascend-v1 DSpark直接草稿输入不读rawHC结论与安装源90matches/26files；本轮8文件/冻结4引用文件仅复核，不计新发现或全观察者闭合。注册/外部/native观察者和alias/event/lifetime仍open，旧条件guards未推翻。'
      '104tailsN24x80/6400x8/6528x8/6672x8，104raw+104premixAG，24DIRECTraw无Compute不能漏；80lastmerge/Post图64root16unbound。MTP角色标签非method证据；N6672不可强拼context6668。'
      '条件G0S只删copy保2AG；G1已有None路径localC→AGcollapsed→globalNorm；G2Norm前移须TPgamma/eps/numeric相同；G3最后Post→C/Norm仅剩余观察者与生命周期闭合时private，输入residual/coefficients仍在。无额外HcPre/Sinkhorn；interior残差仍live。保merge/Post/C/Norm四处BF16。local92/96MH与optionalDownAdd4MH不能重复计global/local/copy信用。'
      'Qboundary=2M(h+1)H+4Mh+2H仅固定GM边界域，rstd4M非普遍必要；容量upper未绑定不能算数值时间。P8ringpayload比4.0015625非提速。320MiB仅请求，实际DeltaB未知；统一DeltaFree=Srelease+floor((r+DeltaB-Dextra)/C)，C540928，假设joint raw/private32/memo256为597..598非独立596..599。'
      '95pairs19groups500中央pending，345records24semantic/35model7legacy；F25/F26 M02仅预算导航非native归属。2073/285598review、9/1352pending、849PMU缺失16phase3136歧义、0wholeclosure保持。'
      '下一具名问题：normalized-target所有数值读者与head logits_indices在forward前就绪否；采样/拒绝后context I是另一selector。优先已有profile/合同，仅为融合/理论上限补必要源码，不研究框架、不重复旧证据。GoalACTIVE/TaskACTIVE/LoopEVALUATING；无server/model/Torch/NPU/capture/process/service工作；verify_DSpark_tail_readers_v85应PASS，不重跑finalizer，不提前complete。')
    report='''本轮分析最终 HC Post、collapse、Norm 与尾部通信的条件重组及理论边界。

v11已经确认：对应 Ascend v1 DSpark 直接草稿输入不使用 getter 返回的完整 HC 快照，并已搜索安装目录中的90条引用、26个文件。本轮历史文件哈希一致，只作复核，不算新发现。额外注册、外部及原生观察者、别名与生命周期仍未闭合。

原 profile 的104组目标尾部、104次raw HC AllGather和104次premix AllGather均保留。80组图与最后一层merge、FFN Post精确对应，其中64组有阶段归属，16组仍未知；24次DIRECT raw通信没有ComputeInfo，也纳入预算。

方案分为只省快照复制、已有bufferNone局部collapse后通信、条件前移Norm、最终Post融合，以及merge/Post/collapse/Norm组合。前两种在v11已有定义。本轮细化最终Post的私有输出条件和联合成本。其输入残差及系数继续保留，末尾没有另一轮HcPre或Sinkhorn。每种方案仍受剩余观察者、行映射、精度、ABI和生命周期约束。

BF16舍入不能随融合取消。标量例子中略去Post舍入可令collapse从5变为4；略去collapse舍入也改变Norm。完整目标归一化输出、三个辅助均值、草稿raw置信度、采样概率、RNG及接受状态继续保留。

源式中间张量读写和常规FP32操作数是比较参考；92/96MH是条件私有接口的搬运并集。固定局部collapse/Norm的GM输入输出边界下，必要值载荷为2M(h+1)H+4Mh+2H；rstd不是普遍必要输出。容量上界已知时才能形成数值时间下界。目前混合资源容量未绑定。TP8的ring发送载荷参考比4.0015625不等于速度倍数。

320MiB是快照请求容量，实际可用显存增量未知。该增量、WKV缓存和C2私有状态共用一个预算余数。明确假设的例子中，共同计算得597～598个global ID的变化；独立范围相加会错误得到596～599。这不是实际容量测量。

累计95条条件关系、19组联合合同；中央仍有500对待汇入或复核。新增最终Post边界家族后共345条分支、24个语义模块。保留2073个regime/285598次执行的条件审阅、9个regime/1352次执行待补，以及849项PMU缺失、16张图阶段未知、3136项归属歧义。没有整模块或整体完成。
'''
    preflight=O/'DSpark-tail-analysis-preflight-v85.json';assert read(preflight)['status']=='PASS'
    src=read(O/names[0]);sourcepaths=[Path(x['path'])for x in src['source_files']]+[Path(src['saved_runtime_script']['path'])]
    jobs=[dict(kind='review',status='pass',description='Revalidate prior-v11 direct-content answer and source hashes; retain installed90/26 inventory and all remaining observer/alias/lifetime guards. No novelty promotion.',artifacts=[O/names[0],O/names[1],O/names[11],O/'F26-consumer-contract-v11.json',N/'installed-fullhc-consumers166-v11.json']+sourcepaths),
      dict(kind='profile',status='pass',description='104saved tails/208AGs and80last-merge/Post graphs64root16unbound,24DIRECT rawwithoutCompute included; no native-address or critical-path gain claim.',artifacts=[O/names[2]]),
      dict(kind='design-check',status='pass',description='5refined graph domains,4typed work/communication/boundary models,2round witnesses,transfer union and6common-budget cases; numeric capacities/recovery/gain unbound.',artifacts=[O/x for x in names[3:]]+[preflight]),
      dict(kind='review',status='error',description='Resolved strict inventory guard and patch-shape errors; draft novelty/scope reconciled against v11 before persistence. No target mutation.',artifacts=[O/'DSpark-tail-validation-history-v85.json',O/'DSpark-tail-pre-reconciliation-v85.json',N/'analyze_DSpark_tail_readers_v85.before-reference-set-fix.txt'])]
    persist_refinement(version=85,previous=84,title='Conditional final HC Post/tail resource domains and common budget; prior v11 reader proof revalidated',
      summary=dict(filename='DSpark-tail-summary-v85.json',previous_turn_classification='PROGRESS:v84WKVhoist/memoandHCfrontier',
        reused_operator_reader_files=8,current_partial_frozen_reference_files=4,prior_installed_reference_inventory=[90,26],
        direct_drafter_content_answer_first_version=11,all_snapshot_observers_closed=False,new_full_dead_buffer_proof=False,
        source_snapshot_requested_bytes=335544320,actual_memory_recovery_proved=False,target_tails=104,tail_AG_ops=208,lastmerge_post_tail_graphs=[80,64,16],
        refined_conditional_variants=5,shape_models=4,numerical_counterexamples=2,shared_memory_budget_models=6,fixed_boundary_payload_models=4,
        new_pair_clauses=11,relation_clauses=95,new_higher_order_groups=3,higher_order_groups=19,pending_central_pair_reconciliation=500,
        operator_footprints=35,pair_navigation_rows=595,refined_candidates=len(members),refined_modules=10,new_semantic_families=1,
        branch_records=345,semantic_modules=24,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
        new_target_source_files=0,new_remote_source_read=False,implementation=False,native_test=False,actual_native_numerics_or_latency_gain_proved=False,actual_lowest_MFU_module_established=False,
        remote_job_status='All local operator/profile/math work finished; no server/model/Torch/NPU/capture/service/process work.'),
      changes=['Prior-v11 direct drafter-content answer revalidated; stronger90/26 inventory and remaining all-observer/alias/lifetime guards retained.',
        '80last-Post graph frontiers/104tails/208AGs support conditional private-interface,typed work/communication and fixed-boundary payload models.',
        'Raw capacity credit,WKV memo and C2 state share one budget remainder; no measured release or speedup.',
        '11new pair clauses3groups=>95/19/500pending;345branchrecords24semantic. Global profile gaps/GoalACTIVE preserved.'],
      candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['DSpark-tail-question-status-v85.json','candidate-pair-navigation-v85.json'],
      publication_rows={
        'F26':'| F26 | 最后层Post、collapse、norm、有效行与通信重组 | 条件 | v11直接DSpark草稿输入不读raw快照，本轮复核；注册/外部/native读者及alias/lifetime仍open。条件局部归约、最终Post私有融合与共享显存预算已建模；320MiB请求容量和ring字节比非实测提速。 |',
        'F03':'| F03 | attention 或 FFN mHC post → 后续pre / 最终collapse重组 | 条件 | 内层保留后续残差、aux及Engram读者。最终FFN Post仅在剩余快照/输出观察者和生命周期条件闭合时可私有融合collapse；保留BF16舍入及Post输入残差/系数。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/n for n in ['analyze_DSpark_tail_readers_v85.py','model_DSpark_tail_reorganization_v85.py',
        'reconcile_DSpark_tail_relations_v85.py','model_tail_shared_memory_budget_v85.py','reconcile_v11_tail_evidence_v85.py','verify_DSpark_tail_readers_v85.py']],
      executor_prefix='DSpark-tail',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
