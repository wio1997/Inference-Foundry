"""Persist conditional KV publication alternatives and baseline-aware theory."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v87.json').exists()
    names=['KV-publication-prior-evidence-v87.json','KV-publication-source-contract-v87.json','KV-publication-native-scope-v87.json',
      'KV-publication-prefix-and-dependency-proof-v87.json','KV-publication-variants-v87.json',
      'KV-publication-value-work-models-v87.json','KV-publication-memory-page-models-v87.json',
      'KV-publication-state-plane-conditions-v87.json','KV-publication-shared-reader-topology-v87.json','KV-publication-composition-clauses-v87.json']
    packet=read(O/names[-1]);assert len(packet['clauses'])==7 and len(packet['higher_order_groups'])==2
    rel=deepcopy(read(O/'candidate-relation-clauses-v86.json'));rel['clauses']+=deepcopy(packet['clauses']);rel['higher_order_groups']+=deepcopy(packet['higher_order_groups'])
    for c in rel['clauses']:
        if c['id']in packet['refine_existing_relations']:
            c['KV_publication_refinement_v87']=dict(evidence=names,current_read_before_acceptance=True,all_readers_closed=False,
              domain='Explicit overlay or PA-COW alternatives preserve current reads;acceptance-delayed commit is not a current Scatter deletion.',
              baseline='Retained-bank overlay may save bytes vs source Scatter yet adds2Ab vs direct producer-to-cache;memory credit shared once.')
    rel.update(clause_count=112,selected_pair_count=112,no_new_maincandidate=True,overall_complete=False);write(O/'candidate-relation-clauses-v87.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v86.json'))
    for c in packet['clauses']:
        pair=next(p for p in cov['pairs']if {p['a'],p['b']}=={c['a'],c['b']});assert not pair['clauses']
        pair.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(scoped_conditional_pairs=112,remaining_pair_relations=483,higher_order_groups=23,parent_relation_index='candidate-relation-clauses-v86.json',
      parent_coverage='candidate-relation-coverage-v86.json',KV_publication_refinement_v87=dict(new_pairs=7,new_groups=2,evidence=names,all_readers_closed=False))
    write(O/'candidate-relation-coverage-v87.json',cov)
    footprints=deepcopy(read(O/'candidate-operator-footprints-v86.json'));registry={r['id']:r for r in read(O/'candidate-record-index-v86.json')['model_candidates']}
    extra=['M02','M15','M22','M24']
    for row in footprints['candidates']:
        p=R/'candidates'/(row['id']+'.json');assert hashlib.sha256(p.read_bytes()).hexdigest()==registry[row['id']]['master_sha256']
        row['captured_master_sha256']=registry[row['id']]['master_sha256']
        if row['id']=='F08':
            row['operator_atom_union']+=['KV.current_read_overlay','KV.accepted_prefix_publication','KV.PA_COW_page_adoption']
            row['modules']=sorted(set(row['modules']+extra));row['scope_v87']='Conditional KV publication alternatives;module links are analytical navigation,not changed native attribution.'
    footprints.update(parent='candidate-operator-footprints-v86.json',overall_complete=False);write(O/'candidate-operator-footprints-v87.json',footprints)
    nav=deepcopy(read(O/'candidate-pair-navigation-v86.json'));byid={r['id']:r for r in footprints['candidates']}
    for p in nav['pairs']:
        p['conditional_clause_ids']=[c['id']for c in rel['clauses']if {p['a'],p['b']}=={c['a'],c['b']}]
        p['semantic_pair_reconciliation_done_in_stated_scope']=bool(p['conditional_clause_ids'])
        p['common_module_associations']=sorted(set(byid[p['a']]['modules'])&set(byid[p['b']]['modules']))
        p['shared_operator_atom_union']=sorted(set(byid[p['a']]['operator_atom_union'])&set(byid[p['b']]['operator_atom_union']))
    nav.update(scoped_conditional_pairs=112,remaining_central_pair_reconciliation=483,parent='candidate-pair-navigation-v86.json',overall_complete=False)
    write(O/'candidate-pair-navigation-v87.json',nav)
    ops=deepcopy(read(O/'operator-analysis-coverage-ledger-v86.json'))
    ops['KV_publication_review_v87']=dict(evidence=names,strict_SWA_projection_executions_reused=3120,
      existing_layer20_SWA_C1_ambiguities_retained=160,terminal_frontiers=[80,64,16],new_regimes_or_executions_promoted=0,all_readers_closed=False)
    write(O/'operator-analysis-coverage-ledger-v87.json',ops)
    matrix=read(O/'module-phase-branch-matrix-v86.json')
    selected={'M02':{'M02-OP-03','M02-OP-10','M02-OP-12'},'M07':{'M07-OP-05'},
      'M08':{'M08-OP-07','M08-OP-08','M08-OP-10','M08-OP-11','M08-OP-12'},
      'M09':{'M09-OP-06','M09-OP-08','M09-OP-10'},'M10':{'M10-OP-05','M10-OP-12'},
      'M14':{'M14-OP-20'},'M15':{'M15-OP-03','M15-OP-09'},
      'M21':{'M21-S04','M21-S09','M21-S10','M21-S15'},'M22':{'M22-S12','M22-S13'},
      'M23':{'M23-OP-05','M23-OP-16'},'M24':{'M24-OP-05','M24-OP-06','M24-OP-08','M24-OP-12'}}
    branches={m['id']:deepcopy(m['branches'])for m in matrix['modules']if m['id']in selected}
    gap=('Actual valid slots/duplicates/required A and all current/future/external observers remain unbound;count alone is not accepted input identity. '
      'Native overlay/COW/recompute precision,map/tuple coherence,old-reader+queued-writer lifetimes and actual capacities/incremental memory/critical path remain conditions.')
    for mid,rows in branches.items():
        assert selected[mid]<={r['id']for r in rows}
        for row in rows:
            if row['id']in selected[mid]:row['evidence']+=names;row['KV_publication_review_v87']=dict(finding='Current verify reads precede acceptance;retained/copied bank,PA-COW,on-chip and recompute graphs have separate value/memory/epoch costs. Compare against direct cache fusion.',gap=gap)
    newrows={
      'M10':dict(id='M10-OP-13',branch='Conditional current-KV read-through versus acceptance-delayed publication;existing-cache-only attention forbids a simple delayed Scatter.',
        finding='Source current reader uses SWAcache;layer39 also uses compressedKV/indexK source20. External-bank overlay changes reader semantics;COW inside existing PA pool can keep cache argument but adds map/alias/lifetime gates. No original-overlay interpretation of cmp_residual_kv.'),
      'M24':dict(id='M24-OP-13',branch='Cross-iteration KV accepted-input publication with exact prefix/seed identities,coherent state planes and last-reader/queued-writer retirement.',
        finding='For baseline-prefix domain a=q-r,r=d+1-c ifd>0 else0. No-draft/prefill need not equal c;input seed+draft differs from accepted+replacement output IDs. Retained-bank credit vs source2(R-A)b becomes extra2Ab vs direct-cache fusion;no raw320MiB double spend.')}
    for mid,row in newrows.items():
        assert row['id']not in {r['id']for r in branches[mid]}
        row.update(state='CONDITIONAL_KV_PUBLICATION_ALTERNATIVES_AND_RESOURCE_DOMAINS_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
          scope='Pinned unpacked BF16 SWA/current compressed reader contracts and explicit accepted-prefix/GM/page domains;no implementation or all-reader closure.')
        branches[mid].append(row)
    write(O/'KV-publication-branch-inventory-v87.json',dict(modules=branches,new_inventory_families=['M10-OP-13','M24-OP-13'],
      branch_records=349,semantic_modules=24,refined_modules=11,untouched_other_modules=15,native_attribution_changed=False,whole_modules_closed=0,overall_complete=False))
    members=sorted({x for g in packet['higher_order_groups']for x in g['members']})
    updates={cid:dict(review_v87=dict(evidence=names,conditional_KV_publication_analysis=True,all_readers_closed=False,
      relation_ids=[c['id']for c in packet['clauses']if cid in[c['a'],c['b']]],implementation=False,native_test=False),conclusion_changing_gap=gap)for cid in members}
    updates['F08'].update(profile_review_state='CURRENT_KV_READ_AND_DELAYED_PUBLICATION_ALTERNATIVES_WITH_BASELINE_COSTS_REVIEWED',
      next_profile_or_contract_action='Do not delete/delay current Scatter with unchanged current reader. Overlay,COW,on-chip and recompute are separate domains;compare with conditional direct-cache producer fusion. Next continue other unreviewed operator combinations or capacity evidence;actual A/aliases/new numerics remain open.')
    mods={mid:dict(finding='Current verify needs candidate KV before acceptance. Conditional delayed publication alternatives are priced against source Scatter and direct-cache fusion;prefix/seed semantics,COW tuple aliases,current source20 consumers and one memory budget retained.',
      evidence=names,candidates=sorted(set([cid for cid in members if any(cid in m['candidate_links']for m in matrix['modules']if m['id']==mid)]+(['F08']if mid in extra else[]))),gap=gap,
      profile_scope='Operator cache write/read/publication dependencies and theoretical resource domains;no broader framework or allocator audit.')for mid in selected}
    write(O/'PROFILE-CODE-QUESTION-GATE-v87.json',dict(status='PROFILE_LED_CURRENT_KV_READ_VS_CROSS_ITERATION_PUBLICATION',
      questions=['Can existing cache write simply wait for acceptance? No,unchanged current reader distinguishes it.',
        'Which overlay/COW/on-chip/recompute alternatives preserve current reads and what do they cost versus direct cache fusion?',
        'Which input-prefix identities,shared source20 readers and multi-plane lifetime guards remain?'],
      pinpoint_code_files=4,reused_model_configs=1,new_target_source_files_obtained=0,new_remote_source_read=False,new_public_document_read=False,
      no_model_Torch_NPU_remote_or_capture_work=True,overall_complete=False))
    write(O/'KV-publication-question-status-v87.json',dict(newly_answered_conditional_scope=['Unchanged-reader delayed-write counterexample/dependency cycle',
      'Bank-overlay versus direct-cache value comparison','Existing-PA COW avoids mandatory separate reader argument but not current R-row stores',
      'Retained input-prefix/seed shift and source20 reader implications for commit'],
      still_open=['Actual required future-row A and every observer','Native new-path equivalence/map coherence','Real incremental tier memory and physical capacity/benefit'],
      all_readers_closed=False,whole_modules_closed=0,overall_complete=False))
    write(O/'KV-publication-validation-history-v87.json',dict(errors=[],
      observation_note='An initial read-only exploratory query had no retained stdout/handle in the displayed result. A bounded local process-comm check showed no newly running Python job;the authoritative derivation was then independently persisted by analyze_KV_publication_v87 with its observed session completed. No target job was restarted.',
      prior_C2_fork_topology_and_prefix_rules_reused_without_novelty_promotion=True,all_local_analysis_jobs_complete=True,new_target_execution=False,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v86.json')['remaining'])
    remaining.append('v87: currentSWAread requires candidateKVbeforeacceptance;simpledelayedScatter excluded.6conditional domains(retain/copyGMbank,existingPA-COW,onchip,recompute plusexcludedplain),7value models/4bank/5pageexamples. Vs source3Rb retained(R+2A)b saves2(R-A)b,butvsdirectRb extra2Ab;no physicalspeedup. Prefixa=q-r conditional,r=d+1-c ifd>0else0;no-draft/prefillA!=c andseed+draft!=sampleoutput. Layer39compressedKV/indexK source20,queryselection36;tuple/page/code-scale/ringepochsremain.3120strictSWA+160existinglayer20ambiguities/80terminal64root16unbound preserved.40*8192*512*BF16=320MiB bankreference mayconsume assumedrawcredit,actualDunknown.7newpairs2groups=>112/23/483pending.349records24semantic35model7legacy0wholeclosed;2073/285598review9/1352pending849/16/3136 unchanged.')
    report='''本轮检查跨迭代KV发布能否延后，以及它与现有融合方案的成本关系。

固定源码中，当前attention读取的是预处理已经写入的SWA缓存。保留原读入口而仅把Scatter移到接受结果之后，会让当前校验读取旧值，并形成“校验→接受→写缓存→校验”的依赖环。因此这一直接改写不成立。

条件替代图包括保留现有KV数组并增加当前读通路、复制到私有数组、在现有PA缓存池内使用COW页、片上保留，以及接受后精确重算。PA池内私有页可沿用现有attention缓存参数，不能笼统声称所有方案都需要新增参数；它仍需写入当前校验用到的全部R行，并证明页表、所有别名平面及生命周期一致。

按相同BF16表示、每行b字节、R个有效不同写入行、A个未来必需保留行的边界参考：原生产者数组加Scatter为3Rb；保留数组后提交为(R+2A)b；另复制私有数组为(3R+2A)b；条件直接写缓存为Rb。保留数组方案相对原Scatter可少2(R-A)b，但相对直接写缓存仍多2Ab。没有物理带宽或延迟优劣结论。

保留前缀不能直接套采样输出计数或ID。只在对应基线前缀域内，a=q-r，r在d>0时为d+1-c，否则为0。无草稿及混合prefill可能要求更多行；输入的seed+draft与采样输出的accepted+replacement存在一位偏移。实际未来读者并集仍待绑定。

第39层还读取第20层生成的压缩KV和index key/scale，最近index查询/选择来源为第36层。当前后续层必须在最终接受结果前看到这些值。3120项严格SWA投影和160项既有第20层SWA/C1归属歧义均保留；没有把歧义改成零工作。

若40层各保留8192×512的BF16数组，独立数组容量参考即320MiB，等于原raw HC快照的请求容量。这不是实测增量或可释放显存，不能把同一份预算再分给WKV缓存、C2状态和私有KV。页COW还受V41多平面tuple分配约束。

累计112条条件关系、23组联合合同，中央仍有483对待汇入或复核。分支台账349条、24个语义模块。整模块与整体均未标记完成。
'''
    action=('继续原Goal，v87PROGRESS，v86亦PROGRESS。先读KV-publication-summary/PROFILE-FINDINGS-v87、prior-evidence/source-contract/native-scope/prefix-and-dependency-proof/variants/value-work-models/memory-page-models/state-plane-conditions/shared-reader-topology/composition-clauses。'
      'v24C2pool/updatefork、v25tupleallocation、v38rejectionprefix、v83SWA非私有临时值、v86fullKVcut均旧证据，不能算新发现。新比较：当前cache-onlyattention禁止单纯delayScatter；保留GMoverlay/额外copyoverlay/现有PA池COW/onchip/accept后重算不同域。cmp_residual_kv不是originaloverlay；PA-COW可保旧reader参数但要求tuple所有平面/页表/epochs一致，仍写所有currentR。'
      'GM边界source3Rb/retain(R+2A)b/copy(3R+2A)b/directRb。对source省2(R-A)b，对direct反而多2Ab；不能只数少写rejected行。当前attentionkeyreads/metadata/压缩/index/copy/capacity另价，非runtime支配证明。'
      'a=q-r仅对应基线retainedprefix域；r=d+1-c ifd>0else0，nofraft/prefillA不直接c；inputseed+draft不是sampleaccepted+replacement。实际R/A/slots/duplicates/futureallreaders未知，保持exactcount/errorpredicate。'
      '源码config层39 ratio1，compressedKV与indexKscale来自20，最近query/indexselection来自36。当前source20数据被20..39层在accept前读，不能因39局部图无compressor就删除早先publisher。3120严格SWAgraphprojection+160层20歧义保留；80terminalfrontiers64root16unbound，nativealias/newloadedcode未证。'
      '40*8192*512BF16=320MiB仅unaliasedbankreference，绝对存活不等实际incremental。rawsnapshot请求320MiB不等可回收，统一WKV/C2/overlay/COW/metadata预算；PA128KiB不是独立alloc成本。C2acceptedhistory和lastqueuedwriter不能由A直接替代。'
      '7newpairs2groups=>112pairs23groups483中央pending；349records24semantic35model7legacy，11模块refined15untouched，F08新模块链接仅分析导航不改native归属。2073/285598review9/1352pending、849PMU缺失16phase3136歧义、0wholeclosure保留。'
      '下一步继续真正未审阅的算子组合/分支与硬件容量证据，不再次重复当前cache读依赖/旧前缀/拓扑规则当进展。仍只savedprofile/必要定点代码/公开primary资料；no remote/model/Torch/NPU/capture/service/processwork。GoalACTIVE/TaskACTIVE/LoopEVALUATING，verify_KV_publication_v87应PASS，不回放finalizer、不因成稿complete。')
    preflight=O/'KV-publication-analysis-preflight-v87.json';assert read(preflight)['status']=='PASS'
    sourcepaths=[Path(r['path'])for r in read(O/names[1])['source_files']]
    jobs=[dict(kind='review',status='pass',description='Pin4code files+1modelconfig for cache current-read,PA interface and source20/36 value namespaces;reuse old prefix/tuple/C2 contracts,no all-reader closure.',artifacts=[O/names[0],O/names[1],O/names[3],O/names[8]]+sourcepaths),
      dict(kind='profile',status='pass',description='3120strict SWA projections/160retained layer20 SWA-C1 ambiguities and80terminal scatter/attention frontiers64root16unbound from saved native identities;no new device or pointer proof.',artifacts=[O/names[2]]),
      dict(kind='design-check',status='pass',description='6publication domains,7GM/work models,4bank/5page models,prefix/identity/dependency witnesses and7pair/2group clauses compared with direct-cache fusion;actual gain/capacity unbound.',artifacts=[O/n for n in names[4:]]+[preflight])]
    persist_refinement(version=87,previous=86,title='Current KV reads versus conditional acceptance-delayed publication and direct-cache fusion',
      summary=dict(filename='KV-publication-summary-v87.json',previous_turn_classification='PROGRESS:v86terminaldemandandfullKVcut',
        pinpoint_code_files=4,reused_model_configs=1,new_target_source_files=0,strict_graph_SWA_projection_executions=3120,
        existing_layer20_SWA_C1_ambiguities_retained=160,terminal_frontiers=[80,64,16],publication_graph_variants=6,value_work_models=7,row_bank_models=4,page_examples=5,
        all_readers_closed=False,native_pointer_or_rewrite_equivalence_proved=False,new_pairs=7,relation_clauses=112,new_groups=2,higher_order_groups=23,pending_central_pair_reconciliation=483,
        operator_footprints=35,pair_navigation_rows=595,refined_candidates=len(members),refined_modules=11,new_inventory_families=2,branch_records=349,semantic_modules=24,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
        new_regimes_or_executions_promoted=0,implementation=False,native_test=False,new_remote_source_read=False,actual_physical_ceiling_or_speedup_proved=False,
        remote_job_status='All local operator/profile/math analysis completed;no remote/model/Torch/NPU/capture/service/process work.'),
      changes=['Exclude unchanged-reader lateScatter using current-read dependency/witness;separate retained/copiedbank,existingPA-COW,onchip and recomputation domains.',
        'Compare3Rb/(R+2A)b/(3R+2A)b/Rb and one memory budget;no rejected-write-only speedup inference.',
        'Bind input-prefix/seed shift and source20/36 reader implications;3120strict+160ambiguous saved projection identities preserved.',
        '7new pairs2groups=>112/23/483pending;349records24semantic,global gaps and GoalACTIVE retained.'],
      candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['KV-publication-question-status-v87.json','candidate-pair-navigation-v87.json'],
      publication_rows={'F08':'| F08 | KV投影、norm、RoPE、缓存写入及条件延后发布 | 条件 | 当前attention先需候选KV，不能仅延后Scatter。保留数组overlay、PA池COW、片上保留与重算是不同图；对原Scatter少搬运不代表优于直接写缓存。实际读者/前缀/tuple别名/精度/容量仍open。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/n for n in ['analyze_KV_publication_v87.py','model_KV_publication_v87.py','refine_KV_shared_readers_v87.py','reconcile_KV_publication_relations_v87.py','verify_KV_publication_v87.py']],
      executor_prefix='KV-publication',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
