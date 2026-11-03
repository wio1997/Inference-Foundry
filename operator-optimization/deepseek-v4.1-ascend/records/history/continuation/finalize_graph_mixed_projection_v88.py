"""Persist version-bound coverage and new conditional mixed-projection resource models."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v5 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v88.json').exists()
    names=['operator-graph-coverage-audit-v88.json','operator-graph-G1-v88.json','operator-graph-G5-v88.json',
      'operator-graph-node-coverage-v88.json','operator-graph-edge-coverage-v88.json','operator-graph-candidate-bindings-v88.json',
      'mixed-SWA-C2-source-contract-v88.json','mixed-SWA-C2-native-frontiers-v88.json','mixed-SWA-C2-cost-models-v88.json',
      'mixed-SWA-C2-project-before-gather-v88.json','mixed-SWA-C2-precision-witnesses-v88.json','mixed-SWA-C2-variants-v88.json',
      'mixed-SWA-C2-composition-clauses-v88.json']
    packet=read(O/names[-1]);assert len(packet['clauses'])==3 and len(packet['higher_order_groups'])==1
    rel=deepcopy(read(O/'candidate-relation-clauses-v87.json'));rel['clauses']+=deepcopy(packet['clauses']);rel['higher_order_groups']+=deepcopy(packet['higher_order_groups'])
    for c in rel['clauses']:
        if c['id']in packet['refine_existing_relations']:
            c['mixed_projection_refinement_v88']=dict(evidence=names[6:],
              typed_joint_exchange='SWA1024B+twoC2FLOAT4096B per projected row vs shared hidden10240B in N24/M3/P8 domain;one baseline,not independent savings.',
              representability_or_native_precision_proved=False,physical_speedup_proved=False)
    rel.update(clause_count=115,selected_pair_count=115,no_new_maincandidate=True,overall_complete=False);write(O/'candidate-relation-clauses-v88.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v87.json'))
    for c in packet['clauses']:
        p=next(p for p in cov['pairs']if{p['a'],p['b']}=={c['a'],c['b']});assert not p['clauses']
        p.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(scoped_conditional_pairs=115,remaining_pair_relations=480,higher_order_groups=24,
      parent_relation_index='candidate-relation-clauses-v87.json',parent_coverage='candidate-relation-coverage-v87.json',
      mixed_projection_refinement_v88=dict(new_pairs=3,new_groups=1,evidence=names,finite_graph_enumeration_is_not_semantic_coverage=True))
    write(O/'candidate-relation-coverage-v88.json',cov)
    footprints=deepcopy(read(O/'candidate-operator-footprints-v87.json'));reg={r['id']:r for r in read(O/'candidate-record-index-v87.json')['model_candidates']}
    for row in footprints['candidates']:
        p=R/'candidates'/(row['id']+'.json');assert hashlib.sha256(p.read_bytes()).hexdigest()==reg[row['id']]['master_sha256'];row['captured_master_sha256']=reg[row['id']]['master_sha256']
        if row['id']=='F08':row['modules']=sorted(set(row['modules']+['M08']));row['operator_atom_union']+=['mixed.SWA_C2_projection_input']
        if row['id']=='F17':row['modules']=sorted(set(row['modules']+['M07']));row['operator_atom_union']+=['mixed.SWA_C2_projection_input']
        if row['id']in['F30','F31']:
            row['partial_source_graph_binding_v88']=dict(graph='operator-graph-G5-v88.json',template='DSPARK_BOUNDARIES',source_answer_version=36,
              scope='Only raw confidence/dynamic-policy branch;not full primitive or native all-variant graph.')
    footprints.update(parent='candidate-operator-footprints-v87.json',overall_complete=False);write(O/'candidate-operator-footprints-v88.json',footprints)
    nav=deepcopy(read(O/'candidate-pair-navigation-v87.json'));byid={r['id']:r for r in footprints['candidates']}
    for p in nav['pairs']:
        p['conditional_clause_ids']=[c['id']for c in rel['clauses']if{p['a'],p['b']}=={c['a'],c['b']}]
        p['semantic_pair_reconciliation_done_in_stated_scope']=bool(p['conditional_clause_ids'])
        p['common_module_associations']=sorted(set(byid[p['a']]['modules'])&set(byid[p['b']]['modules']))
        p['shared_operator_atom_union']=sorted(set(byid[p['a']]['operator_atom_union'])&set(byid[p['b']]['operator_atom_union']))
    nav.update(scoped_conditional_pairs=115,remaining_central_pair_reconciliation=480,parent='candidate-pair-navigation-v87.json',overall_complete=False);write(O/'candidate-pair-navigation-v88.json',nav)
    ops=deepcopy(read(O/'operator-analysis-coverage-ledger-v87.json'))
    ops['mixed_projection_and_graph_review_v88']=dict(evidence=names,mixed_frontiers=[240,192,48],reused_GEMMs=720,reused_casts=240,
      new_regimes_or_executions_promoted=0,current_finite_source_graph_rows=[632,1050],graph_rows_not_native_or_all_branch_coverage=True)
    write(O/'operator-analysis-coverage-ledger-v88.json',ops)
    matrix=read(O/'module-phase-branch-matrix-v87.json');selected={
      'M07':{'M07-OP-04','M07-OP-08','M07-OP-09'},'M08':{'M08-OP-02','M08-OP-06','M08-OP-11','M08-OP-12'},
      'M09':{'M09-OP-06','M09-OP-08'},'M10':{'M10-OP-02'},'M16':{'M16-OP-07'},'M19':{'M19-OP-10'},
      'M20':{'M20-OP-03','M20-OP-04','M20-OP-09','M20-OP-10'},'M21':{'M21-S10','M21-S15'},
      'M22':{'M22-S11','M22-S12'},'M23':{'M23-OP-05','M23-OP-18'},'M24':{'M24-OP-06'}}
    branches={m['id']:deepcopy(m['branches'])for m in matrix['modules']if m['id']in selected}
    gap=('Actual same value/storage/row ownership and C2weight BF16 representability/native FLOAT-HF32 reduction policy remain unbound. '
      'New encoding/segmented output/collective/capture/reader epochs and physical mixed capacities remain conditions. '
      'Current graph structural rows and v36 prior confidence answer do not close all primitive/native/runtime branches.')
    for mid,rows in branches.items():
        assert selected[mid]<={r['id']for r in rows}
        for row in rows:
            if row['id']in selected[mid]:row['evidence']+=names;row['graph_mixed_review_v88']=dict(finding='Current graph revision coverage reconciled;raw confidence source answer preexistingv36. Conditional SWA BF16+C2FLOAT joint input/projection/repartition costs retain distinct types,rows and GM/weight bridges.',gap=gap)
    newrows={
      'M07':dict(id='M07-OP-13',branch='C2-source SWA BF16 and twoFLOAT projections:joint typed input/weight/output graph or owner-local projection before shared gather.',
        finding='240graph groups bind SWA BF16+twoFLOAT GEMMs and onecast. N24global versusQlocal3;typed output communication107520B vs215040B ringreference. Source cold weight set unchanged,so local-row arithmetic reduction doesnot lower declared L1weight-ingress floor.'),
      'M08':dict(id='M08-OP-14',branch='Mixed SWA/C2 input representation,conditional exact C2 BF16weight encoding and segmented output,distinct pool/norm/state readers.',
        finding='Originalinputreference16NH,F17FLOATpairjoin12NH;2NH route needs exact reencoding/accumulation proof. KeepSWA BF16andC2FLOATbridges. Genericweightcounter256.5vs256 and exactBF16operand0vs1 show precision gates;wholeM24dualA/L0Cplan exceeds capacity,tiles remain candidates.')}
    for mid,row in newrows.items():
        assert row['id']not in {r['id']for r in branches[mid]}
        row.update(state='CONDITIONAL_MIXED_PROJECTION_AND_TYPED_RESOURCE_DOMAINS_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
          scope='Pinned C2source layers2/8/14 and declared representation/owner/path domains;no implementation or actual speedup proof.');branches[mid].append(row)
    write(O/'graph-mixed-branch-inventory-v88.json',dict(modules=branches,new_inventory_families=['M07-OP-13','M08-OP-14'],branch_records=351,
      semantic_modules=24,refined_modules=11,untouched_other_modules=15,native_attribution_changed=False,whole_modules_closed=0,overall_complete=False))
    members=sorted(set(packet['higher_order_groups'][0]['members']+['F28','F30','F31','F35']))
    updates={cid:dict(review_v88=dict(evidence=names,conditional_mixed_projection_analysis=(cid in packet['higher_order_groups'][0]['members']),
      graph_coverage_reconciliation_is_not_new_semantic_closure=True,all_readers_closed=False,
      relation_ids=[c['id']for c in packet['clauses']if cid in[c['a'],c['b']]],implementation=False,native_test=False),conclusion_changing_gap=gap)for cid in members}
    updates['F17'].update(profile_review_state='SWA_BF16_C2_FLOAT_JOINT_PROJECTION_REPRESENTATION_COMMUNICATION_AND_INGRESS_REVIEWED',
      next_profile_or_contract_action='Reencoding needs actual weight representability and original accumulator policy;no cast-sharing novelty. Continue unresolved finite-graph mappings and high-impact source/resource gaps without assuming new UB→L1 route or8x runtime benefit.')
    updates['F30']['source_graph_authority_v88']='operator-graph-G5-v88.json#DSPARK_BOUNDARIES;v36raw confidence/policy answer now synchronized,not newly discovered.'
    mods={mid:dict(finding='Version-bound partial graph coverage and confidence source branch reconciled;conditional mixed BF16SWA/FLOATC2 graph has explicit row/type/communication/weight/output costs. Enumeration doesnot prove semantic completion.',
      evidence=names,candidates=sorted(set([cid for cid in members if any(cid in m['candidate_links']for m in matrix['modules']if m['id']==mid)]+(['F17']if mid=='M07'else[])+(['F08']if mid=='M08'else[]))),gap=gap,
      profile_scope='Operator input/output/fusion/precision/resource and finite source-graph coverage;no new framework research.')for mid in selected}
    write(O/'PROFILE-CODE-QUESTION-GATE-v88.json',dict(status='WHOLE_SCOPE_AUDIT_THEN_PROFILE_LED_MIXED_OPERATOR_QUESTION',
      questions=['Does old coverage hash match current partial source graphs? No;current revision now linked.',
        'Was raw confidence source input unknown? Already answeredv36;now graph synchronized.',
        'Can BF16SWA and FLOATC2 project jointly without losing precision or hiding ingress/comm cost? Conditional domains and counterexamples documented.'],
      pinpoint_source_files=5,new_target_source_files_obtained=0,new_remote_source_read=False,new_public_document_read=False,
      no_model_Torch_NPU_capture_or_server_work=True,overall_complete=False))
    audit=read(O/names[0]);audit.update(requirement_audit_input_checkpoint=87,followup_refinement_checkpoint=88,
      followup_branch_records=351,followup_semantic_modules=24,requirement_statuses_unchanged_by_structural_enumeration=True)
    write(O/names[0],audit)
    write(O/'graph-mixed-question-status-v88.json',dict(reconciled=['Stale finite graph/coverage hashes','v36 confidence raw/ordered-input/globalpolicy dataflow'],
      newly_reviewed=['Mixed BF16/FLOAT input/output precision and A3capacity domains','Typed owner-local projection exchange and shared weight-ingress floor'],
      open=['All applicable primitive/branch/observer mappings','C2weight value identity and native reduction policy','Real typed mixed capacity/clock/bandwidth','Nine pending regime/native-field questions'],
      all_readers_closed=False,whole_modules_closed=0,overall_complete=False))
    write(O/'graph-mixed-validation-history-v88.json',dict(errors=[],prior_confidence_answer_not_counted_new=True,
      finite_connected_enumeration_is_not_global_exhaustion=True,all_local_analysis_jobs_complete=True,new_target_execution=False,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v87.json')['remaining'])
    remaining.append('v88: whole-scopeauditfinds old623node/1029edgeledger hashesstale. CurrentpartialG1(338/690),G5(294/360) afterv36confidence-sync gives632/1050rows,notuniqueopsorsemanticclosure. 35sourcebindingrows/38structuralpairhints;metadata/SP/C1/samplingprimitivegraphsnotallmapped. New240mixedBF16SWA+2FLOATC2+castfrontiers192root48unbound bind720GEMM240cast. Oldcastalreadyshared. Typedinput16NH→12NH viaexistingF17;2NHrouteconditionalweightencoding/numerics. Ownerlocaltypedexchange107520vs215040B ring,not2xlatency;QA3vsglobal24. Originalweights26,214,400B/source unchangedbylocalM,declaredfixedB L1cut8022cycles/source or25088for3withsinglewarmgrant;GMdirectL0outside,noactualwallclockupper. Genericweight/reductionwitnesses and wholeA737280>L1524288/C147456>L0C131072 plans.3newpairs1group=>115/24/480pending.351records24semantic35model7legacy0wholeclosed;2073/285598review9/1352pending849/16/3136 unchanged.')
    report='''本轮先核对整体覆盖，再分析由实际输入分叉定位的混合精度投影。

旧结构覆盖台账为623行节点、1029行边，绑定较早G1/G5哈希。本轮对齐当前辅助状态修订，并把v36已核实的confidence原始隐藏状态、seed/前序预测输入及全局survival预算关系同步到图。当前部分图台账为632行节点、1050行边；这是可追溯的结构记录，不是唯一算子数量、融合可行性或全覆盖百分比。原图仍明确不完整，旧confidence结论未计为新发现。

新分析定位到240组C2源层图：同一源输入进入SWA的BF16投影，以及C2的两路FLOAT投影和一个共享输入cast。其中192组有阶段归属、48组来自16张阶段未知图。实际输入指针、FLOAT/HF32模式和权重值仍未取证。

按指定边界，原输入平面参考为16NH；已有F17两路FLOAT合并后为12NH，不能再重复记一次cast复用。进一步使用全部BF16操作数，需要C2权重精确可表示和原归约顺序等价。标量例子中重量重编码使结果256.5变256；即使操作数都可精确表示，改变归约树仍可得到0与1。C2的FLOAT状态结果不能因后续存在BF16归一化而省去。

在TP8、每卡3行的条件重组中，先局部计算完整SWA/C2投影，再交换BF16/FLOAT输出，每卡ring发送载荷参考为107,520B，原隐藏状态为215,040B。三路输出共享同一个旧gather基线；打包、多个通信启动、行映射和新shape数值策略都须计价，不能推导2倍提速。

理论上限也保留了权重服务约束。每源层原类型权重共26,214,400B，改为局部行或减少启动并不减少这组权重。在声明的24-AIC、固定B、L1→L0通路模型中，扣一次允许初始L0驻留后需至少8022个聚合周期；跨三个源层的共同窗口为25088周期。该界有路径、表示和初态条件，不能用观测频率换成实际耗时下界。其他合法通路另建模型。

单AIC同时放全24行BF16/FLOAT输入需737,280B，超过512KiB L1；全三路FP32累加结果需147,456B，超过128KiB L0C。这只排除相应整矩阵方案，行/K/列分块仍可分析。

累计115条条件关系、24组联合合同，中央仍有480对待汇入或复核。当前351条分支、24个语义模块，9个regime/1352次执行待补，整体目标继续。
'''
    action=('继续原Goal，v88PROGRESS，v87亦PROGRESS。先读graph-mixed-summary/PROFILE-FINDINGS-v88和operator-graph-coverage-audit/G1/G5/node-coverage/edge-coverage/candidate-bindings、mixed-SWA-C2-source/native/cost/project-before-gather/precision/variants/composition。'
      '旧Rcoverage623/1029绑定baseG1/G5不是aux修订；v88结构632node/1050edge包含G1338/690、G5294/360，G1/G5语义重叠不能加成唯一机会。枚举2/3/4connectedsets含fork/stateadjacency，非所有jointsemantics或runtimeproof。v36confidence rawlastHidden/T[:,:D]/globalTopK旧结论已同步partialG5，不算新发现；metadata/SP/C1/samplingprimitive和allbranch映射仍open。'
      '240C2source2/8/14图body192root48unbound包含SWA BF16 GEMM+两FLOAT GEMM+一次cast，720matrix240cast全旧profile，未promote任何regime。Sameinput源契约非nativealias。原typed input16NH，F17join12NH，2NH只在C2weights exactBF16+原nativeaccum/BF16SWA/FLOATC2outputs证明域；不能重复castCSE或引入A5UB→L1。'
      'Ownerlocal beforeAG保Qlocal3，发SWA1024+两FLOAT4096=5120B/row，N24P8每rank107520vs215040 ringreference。原输入gather一次不分别记90%和60%收益；多个typedcollectives或packednewABI另价，globalQjoint会多算QA。原weights26,214,400B/source不随N→M减少。'
      '声明24AIC/L1only固定B/uncompressed distinct planes/无prewindowinflight/初始L0warm grant一次，原8022cycles/source、3source25088。别改成actualruntime上界，GM→L0alternative不受此L1界，HF32/FLOAT/Vector/freq/BWactualupper仍unknown。两generic数值反例与整A737280>L1524288、整C147456>L0C131072只排除特定wholematrixplan，不否认tiling。'
      '3newpairs1group=>115pairs24groups480中央pending，351records24semantic35model7legacy；F17M07/F08M08只是分析导航不改native归属。2073/285598review9/1352pending849PMU缺失16phase3136ambiguity0wholeclosure保持。'
      '下一步优先真正未闭合operator图primitive/条件分支映射或会改融合结论的capacity/semantics证据，利用当前graphpairhints而非继续盲增cartesianpairs；不重复旧confidence/graphsource结论当新优化，不研究无关框架。仅savedprofile/必要定点source/primary资料；no remote/model/Torch/NPU/capture/servicework。GoalACTIVE/TaskACTIVE/LoopEVALUATING，verify_graph_mixed_projection_v88应PASS，不重跑finalizer，不提前complete。')
    preflight=O/'graph-mixed-analysis-preflight-v88.json';assert read(preflight)['status']=='PASS'
    scan=R/'fusion-scan.json';assert not any(a['path']==str(scan)for a in read(O/'checkpoint-manifest-v87.json')['artifacts'])
    write(O/'fusion-scan-before-v88.json',read(scan));s=read(scan)
    s['per_node_edge_coverage_ledger']=dict(summary=str(O/names[0]),nodes=str(O/names[3]),edges=str(O/names[4]),
      graph_catalog=dict(G1=str(O/names[1]),G5=str(O/names[2])),structural_only=True,all_native_or_semantic_variants_complete=False)
    s['operator_coverage_v88_scope']='Partial source graph versions and all known scope gaps are explicit;legacy R/coverage retained as history. Node/edge enumeration is not semantic exhaustion.'
    write(scan,s)
    sourcepaths=[Path(r['path'])for r in read(O/names[6])['source_files']]
    jobs=[dict(kind='review',status='pass',description='Reconcile stale partial G1/G5 hashes and finite632node/1050edge structural rows,explicitly synchronize priorv36 confidence branch,no semantic/global completion promotion.',artifacts=[O/n for n in names[:6]]+[scan,O/'fusion-scan-before-v88.json']),
      dict(kind='profile',status='pass',description='240mixed SWABF16/C2FLOAT/cast groups192root48unbound bind720GEMMs240casts using saved identities and5pinpoint source files;no nativepointer/newmode proof.',artifacts=[O/names[6],O/names[7]]+sourcepaths),
      dict(kind='design-check',status='pass',description='5mixed graph variants/5typed row models/4conditional ingress cuts,typed1/2communication reference,2numeric witnesses and3pair/1group clauses;physical time/capacity/weight identity remain unbound.',artifacts=[O/n for n in names[8:]]+[preflight])]
    persist_refinement(version=88,previous=87,title='Version-bound operator graph coverage and mixed SWA BF16/C2 FLOAT projections',
      summary=dict(filename='graph-mixed-summary-v88.json',previous_turn_classification='PROGRESS:v87KVpublicationalternatives',
        graph_catalog=dict(G1=str(O/names[1]),G5=str(O/names[2])),finite_graph_node_rows=632,finite_graph_edge_rows=1050,
        legacy_node_edge_rows=[623,1029],confidence_source_answer_first_version=36,confidence_fact_counted_new=False,structural_pair_hints=38,
        local_pinpoint_source_files=5,new_target_source_files=0,mixed_projection_frontiers=[240,192,48],reused_matrix_executions=720,reused_cast_executions=240,
        mixed_projection_variants=5,row_cost_models=5,conditional_weight_ingress_cuts=4,numerical_witnesses=2,
        all_primitive_graphs_or_readers_closed=False,new_pairs=3,relation_clauses=115,new_groups=1,higher_order_groups=24,pending_central_pair_reconciliation=480,
        operator_footprints=35,pair_navigation_rows=595,refined_candidates=len(members),refined_modules=11,new_inventory_families=2,branch_records=351,semantic_modules=24,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
        new_regimes_or_executions_promoted=0,new_remote_source_read=False,implementation=False,native_test=False,actual_physical_wallclock_ceiling_or_speedup_proved=False,
        remote_job_status='All local finite-graph/profile/source/math work finished;no remote/model/Torch/NPU/capture/service work.'),
      changes=['Version-bind finite graph coverage and publish current632node/1050edge ledger;v36confidence source answer synchronized without novelty promotion.',
        '240existing mixed projection groups support BF16/FLOAT representation/typed output/owner-local communication analysis.',
        'Reprice shared input/communication and unchanged weight service;conditional cycle cuts,precision/capacity witnesses,not actual time/speedup.',
        '3new pairs1group=>115/24/480pending;351branchrecords24semantic,global missing evidence and GoalACTIVE retained.'],
      candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['graph-mixed-question-status-v88.json','candidate-pair-navigation-v88.json'],
      publication_rows={'F17':'| F17 | C2两路FLOAT投影及与SWA BF16投影的联合输入/分块 | 条件 | 一次输入cast已是基线；typed input参考16NH→12NH属已有两路合并。BF16重编码须权重精确与归约等价；owner-local三路交换字节可减半，但冷权重服务、C2FLOAT输出和A3 GM桥仍需计价。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/n for n in ['reconcile_operator_graph_coverage_v88.py','analyze_mixed_SWA_C2_v88.py','model_mixed_SWA_C2_v88.py','reconcile_mixed_projection_relations_v88.py','verify_graph_mixed_projection_v88.py']],
      executor_prefix='graph-mixed',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
