"""Persist receiver capacity/clipping proofs and co-resident workspace/cache/state budgets."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v97.json').exists()
    assert read(O/'receiver-budget-analysis-preflight-v97.json')['status']=='PASS'
    names=['receiver-budget-operator-source-contract-v97.json','receiver-capacity-evidence-status-v97.json','receiver-capacity-and-workspace-models-v97.json',
      'receiver-workspace-shared-live-budget-v97.json','receiver-budget-composition-clauses-v97.json','receiver-budget-variants-v97.json']
    packet=read(O/names[4]);members=['F20','F21','F25','F26','F34']
    gap=('Current166 loaded C/max_output_size and same-call all-rank input bounds/receiver counts are not attested by source defaults or old169hashes. '
      'C_new clipping equivalence is conditional on A<=B and preserved row order/maps/aliases;reference128row guard is not a native write-extent proof. '
      'Full-load generic scratch reservation is distinct from explicit map/count/peer buffers and16MiBSYSTEM;new workspace/capture addresses and all old readers must remain valid. '
      'Requested workspace size is not actual live/reserved allocation,HBM traffic,compute work or speedup. Peak credit and WKV/C2 feasibility require same-device overlapping lifetimes and allocator/pool-rebudget proof. '
      'All numerical/resource/native precision/critical-path/whole-candidate and full goal gaps remain open.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v96.json'));rel['clauses']+=deepcopy(packet['clauses'])
    g=next(g for g in rel['higher_order_groups']if g['id']==packet['extend_group']);assert 'F21'not in g['members'];g['members'].append('F21')
    g['receiver_workspace_refinement_v97']=dict(evidence=names,
      contract='One co-resident peak budget for selected customMC2 workspace capacity/scratch and genuinely incremental pureWKV/privateC2 banks. Max-over-phase and allocator liveness determine usable credit,not a sum of percall requests or local decode deltas.',
      actual_credit_or_whole_group_closed=False)
    c=next(r for r in rel['clauses']if r['id']=='REL73-07');c['receiver_capacity_refinement_v97']=dict(evidence=names,
      contract='Same-call routing/input bounds can reduce a capacity attribute while preserving min(A,C_old);do not increase it or borrow anotherMegaMoe path logic. Count-first exact sizing changes the fused dependency/ABI.',all_runtime_gates_proved=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v96.json',clause_count=124,selected_pair_count=124,new_pair_clauses=2,new_higher_order_groups=0,overall_complete=False);write(O/'candidate-relation-clauses-v97.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v96.json'))
    for c in packet['clauses']:
        row=next(r for r in cov['pairs']if{r['a'],r['b']}=={c['a'],c['b']});assert not row['clauses']
        row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(parent='candidate-relation-coverage-v96.json',parent_relation_index='candidate-relation-clauses-v97.json',scoped_conditional_pairs=124,remaining_pair_relations=471,higher_order_groups=25,
      receiver_budget_refinement_v97=dict(new_pairs=2,new_groups=0,extended_group=packet['extend_group'],evidence=names),overall_complete=False);write(O/'candidate-relation-coverage-v97.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v96.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v96.json')['model_candidates']}
    for r in fps['candidates']:
        p=R/'candidates'/(r['id']+'.json');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256'];r['captured_master_sha256']=h;r['captured_master_version']=96
        if r['id']in members:r['receiver_budget_refinement_v97']=dict(evidence=names,resource_coupling_not_shared_operator_work_identity=True)
    fps.update(parent='candidate-operator-footprints-v96.json',overall_complete=False);write(O/'candidate-operator-footprints-v97.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v96.json'))
    for row in nav['pairs']:
        row['conditional_clause_ids']=[c['id']for c in rel['clauses']if{c['a'],c['b']}=={row['a'],row['b']}]
        row['semantic_pair_reconciliation_done_in_stated_scope']=bool(row['conditional_clause_ids'])
    nav.update(parent='candidate-pair-navigation-v96.json',scoped_conditional_pairs=124,remaining_central_pair_reconciliation=471,overall_complete=False);write(O/'candidate-pair-navigation-v97.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v96.json'));op['receiver_budget_refinement_v97']=dict(cached_current_MC2_TASKs=4472,databases=32,
      current_loaded_capacity_or_all_rank_bounds_known=False,conditional_source_cases=5,workspace_reference_cases=24,new_native_regime_or_execution_promotions=0,new_relations=2)
    write(O/'operator-analysis-coverage-ledger-v97.json',op)
    matrix=read(O/'module-phase-branch-matrix-v96.json');branches=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M12'))
    assert not any(r['id']=='M12-OP-19'for r in branches)
    branches.append(dict(id='M12-OP-19',branch='CustomMC2 receiver capacity/clipping-equivalent reduction,phase-specific workspace max/full-load generic-scratch reservation and shared WKV/C2 live-memory budget.',
      finding='For A<=B,C_new=min(C_old,max(1,B)) preserves original clipping. Host request16MiB+max(coc,init),coc capacity coefficient15368B,row/scale bank aliases counted once. Full-load genericworkspace argument unused but explicit map/peer/count buffers remain. Actual C/all-rank bounds/allocator lifetimes unbound;two new same-device memory coupling contracts recorded.',
      state='SCOPED_CAPACITY_AND_WORKSPACE_REPRESENTATION_CONTRACT_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
      scope='Frozen customMC2 source and stated input/routing/allocation domains;source default65536 and 128row guard are reference cases,not current runtime or physical buffer safety attestations.'))
    mids=['M08','M12','M14','M22','M23','M24']
    mods={mid:dict(finding='CustomMC2 source-requested workspace and clipping capacity now couple explicitly with live WKV/privateC2 banks under one peak budget. Local decode request reduction is not automatically reusable peak memory or fewer HBM transfers.',
      evidence=names,candidates=[cid for cid in members if cid in next(m['candidate_links']for m in matrix['modules']if m['id']==mid)],gap=gap,
      profile_scope='Operator capacity/representation/lifetime constraints and resource coupling,not framework configuration research.')for mid in mids}
    updates={cid:dict(review_v97=dict(evidence=names,current_MC2_TASKs_reused=4472,actual_current_C_attested=False,
      capacity_identity_models=1073280,source_workspace_models=24,clipping_equivalence_only_in_declared_global_bound=True,
      native_write_extent_or_allocator_peak_credit_or_actual_speedup_proved=False,
      new_relation_ids=[c['id']for c in packet['clauses']if cid in[c['a'],c['b']]],extended_group='HG85-SHARED-MEMORY-BUDGET',
      implementation=False,native_test=False),conclusion_changing_gap=gap)for cid in members}
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v96.json'));audit.update(previous_goal_turn='PROGRESS:v96inverse-map/postmask/operator-cost contracts persisted;132postchecks',input_checkpoint=96,followup_checkpoint=97)
    for r in audit['requirements']:
        if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES':r['branch_records']=359
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=124,higher_order_groups=25,central_pairs_not_reconciled=471)
        if r['id']=='THEORY_AND_MEASURED_COMPARISON':r['evidence']+=names
    audit['receiver_budget_refinement_v97']=dict(actual_current_C_or_native_memory_peak_not_proved=True,
      precise_minimum_missing_evidence='CurrentC binding,same-call rankbounds/counts,loaded write extents and allocator/capture bank lifetimes.',whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v97.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v97.json',dict(status='PROFILE_CAPACITY_GAP_THEN_PINPOINT_OPERATOR_ARGUMENT_AND_WORKSPACE_FORMULA',
      questions=['Does current customMC2 apply the adjacent MegaMoe capacity protection?',
        'Which smaller C preserves original clipping for allreceiverA<=B?',
        'Does reducing C lower the whole max(coc,init) allocation and actual peak?',
        'Can that credit fund co-resident WKV/C2 banks across prefill/decode and graph lifetimes?'],
      cached_source_files=8,new_target_source_files=0,new_public_documents=0,no_framework_wide_research=True,
      no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    write(O/'receiver-budget-validation-history-v97.json',dict(scope_refinements=[
      'Old169 config omission/two hash matches are retained evidence,not new166runtime C proof. Cached32DBprojection has no decoded C field;raw DB/provider may contain additional evidence.',
      'Current customdispatch_ffn_combine forwards configured C;adjacent CANNMegaMoe code is a different branch,not a current clipping guard.',
      'Capacity optimization preserves min(A,C_old);largegeometryB>C_old keeps originalC instead of changing clipping semantics.',
      'Initial reference test cardinalities corrected to1296geometryand24workspacecases;arithmetic identities and workspace formulas passed independent checks.',
      'FixedSYSTEM16MiB,explicit inverse/peer buffers and large gathered init path remain. 128row guard is a sizing reference,not a write-bounds proof.',
      'Onebank request,max-over-phase actual live peak,fixed KV pool and newly rebudgeted pool have distinct denominators;no 40layer multiplication or freed-HBM-time claim.'],
      all_local_jobs_finished=True,new_native_work=False,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v96.json')['remaining']);remaining.append('v97: CurrentcustomMC2 C unbound;default65536/old1692hashes not166proof. Cached32DB4472tasks bound,butno decoded C inprojection andno samecallallrankN/k. C_new=min(C_old,max1B) preservesmin(A,C_old)forallA<=B;do notincreaseC. 16rankhomogeneousreference B288/144/96/78336/80064 forM3K6/M3K3/M2K3/M816K6/M834K6,notactualsamecallproof. Workspace16MiB+max(coc(C),init);a=15368B/C,aliasescountedonce. Full-loadgenericworkspaceunused(nonnullguardstill) permitsconditionalreservationelision,notexplicitmap/peer/countbuffers orSYSTEM/largegather. DefaultM3reference1023993344B;guardC512-only33555328B;pairedunusedinit24704512B;notactualfreedmemory/HBM/gain/nativewritesafety. Peakprefillcanerasealldecodecredit;pureWKV/privateC2banksrequirecommonbudget/lifetimes/allocator/epoch. 2newREL97pairsF21F25/F34+HG85budgetmember=>124/25/471,359records24semantic;2073/285598review9/1352pending849/16/3136uncertainty0wholeclosed/noallmodulelowestMFU. Nextcriticalevidence C/run/rank,allrankbound/counts,loadedwriteextent/capturebanklifetime remain.')
    report='''本轮把 MoE 接收容量、工作区申请和跨迭代状态/缓存的共同内存约束连到一起。所有数值均注明来源域，没有修改配置或运行设备。

当前custom DispatchFFNCombine直接把mega_moe_max_tokens传给max_output_size；附近另一条CANNMegaMoe路径的安全容量计算没有用于这次custom调用。冻结源码默认值为65536，但它不是当前166实际参数证明。旧169的配置缺省检查和两条65536候选哈希保留为历史证据；32份当前缓存导出中的4472条MC2 TASK没有直接解码出C。该导出缺字段也不等于原始DB一定没有参数。

设同次调用目标receiver的assignment数为A，可靠上界为B。每个source最多发送n_s·k_s条；若每token选中的physical expert互异，则该receiver最多收到Σn_s·min(k_s,E_receiver)。在A≤B的域中，取C_new=min(C_old,max(1,B))保持min(A,C_new)=min(A,C_old)。因此可缩小容量而不改变原裁剪；增大C会改变可能触发的裁剪行为，没有作为等价优化。

| 假定同次16个source均不超过所列M、使用相同k | receiver几何上界B | 以C_old=65536为参考的C_new |
|---|---:|---:|
| M=3、k=6 | 288 | 288 |
| M=3、k=3 | 144 | 144 |
| M=2、k=3 | 96 | 96 |
| M=816、k=6 | 78336 | 65536 |
| M=834、k=6 | 80064 | 65536 |

这张表不是各卡同一次通信已完成对齐的证明。大行数中，几何上界超过默认参考容量，不能仅凭本地shape或默认值认定裁剪从未发生。现有87组路由观测与后来的hostprof TASK仍是不同采集群，未把它们伪联接成同次接收量。

工作区申请公式已逐项核对：W_req=16MiB+max(O_fixed+15368·C,W_init)。15368=两份FLOAT行scale的8字节+FP16中间bank的10240字节+INT8中间bank的5120字节；C1/C2及input/activation各自已复用的bank只计一次。初始化与CoC工作区取max，不能相加。固定SYSTEM的16MiB也没有删除。

在选定full-load21000的源码中，通用workspace参数只有声明/定义，没有读取或转发；入口仍要求非空。显式inverse-map前缀、peer量化结果、计数等缓冲继续使用，不能因此删除。故可登记“复用有效非空地址、取消未用通用scratch预留”的条件工作区合同；大行数gather分支不适用。

以M=3、k=6、C_old=65536的源码默认参考为例：原请求1023993344B；选择含128行guard的参考C=512后，请求33555328B；联合去掉未用初始化预留后为24704512B。单独去掉该预留时原CoC项仍占max，申请量不变，两项存在条件协同。这个128行guard没有被证明覆盖实际内核所有写范围，较小workspace仍需加载内核写界、bank地址、旧读者和capture合同证据。这些数值是单次API申请参考，不是当前实际分配、已释放显存、HBM流量、kernel耗时或逐层可累加的空间收益。

若要按精确A_r而非静态几何界分桶，A_r要到路由计数和交换之后才可得。当前融合调用在launch前就申请workspace；新的count-first切分、预分配bank选择或capture ABI必须显式付出依赖、交换和就绪成本，不能提前免费使用未来计数。

同一个设备的WKV缓存、C2新状态bank和MC2 workspace还受共同live peak约束。引用既有表示模型，64个WKV结果加完整ID键是3282944B，另加一个owner的三层C2 payload393216B；这是新bank与旧池同时驻留、尚无旧池退还信用的参考，仍未计管理/对齐等开销。一个跨prefill/decode的模型中，decode请求虽然下降，但prefill仍决定峰值，所以可重新投入的峰值信用为0；新增持久bank可以在decode单阶段可行、跨两阶段却超预算。实际KV池已分配的headroom与重新规划前的总预算必须区分，不能重复扣workspace或把free ID计数当成释放显存。

本轮新增F21↔F25、F21↔F34两条同设备live-memory条件合同，并扩展既有共同预算组；它们没有共享算子工作或独立提速含义。中央累计124条条件关系、25组联合合同，471对待汇入或复核；分支台账359条、24个语义模块。原生2073个regime/285598次执行条件分析、9个regime/1352次执行待补关键字段，及849条PMU缺失、16张阶段未知图、3136条模块归属歧义均未改变。目标和全模块最低MFU仍未完成。
'''
    action=('继续原Goal，v97PROGRESS，v96PROGRESS已132checks。先读receiver-budget-summary-v97/PROFILE-FINDINGS/source/evidence-status/capacity-workspace/shared-live/composition/variants及wholeaudit。'
      '用户聚焦operatorfusion/split/reorg/theory，profile优先pinpoint，不扩framework。CurrentcustomMC2只forward mega_moe_max_tokens，邻近CANNMegaMoe自动cap是另一分支。Frozen default65536和old169两hash/CLI遗漏不绑定166实际C，32cachedDB4472TASK没有decodedC，rawDB可能另有证据。'
      'Parametricclipidentity：可靠samecallallrank几何B下A<=B,Cnew=min(Cold,max1B)保持min(A,C)；大B保持Cold，不增cap改语义。Reference16ranks同M/k B288/144/96/78336/80064非samecall测量。'
      'Wreq=SYSTEM16MiB+max(cocFixed+15368C,init16MiB+32Mk+320)；两scale/C1C2alias/inputactalias数一次。FullLoad21000 genericworkspace参数unused但nonnullguard&explicit inverse/peer/count args保留；largegather不elide。M3k6default参考1023993344B，Cguard512-only33555328，pairedinitelide24704512；guard128未证明native写界，非实际allocation/peak/HBM/gain。standalone initelide0收益，联合有max协同，不逐层/跨iter累加。'
      '精确Ar晚于counts/exchange，count-first bankselection要新split/ABI和capture/ready成本，不能旧prelaunchallocator免费用未来计数。共同预算D_WKV64*(51200+96)=3282944,D_C2owner三层393216为genuinelyincremental/nooldpoolcredit参考；decode节省不等peak，prefill可使peakcredit0。现有KVpool预算和rebudget-beforepool不同分母，no doubleworkspace/freeIDs≠physicalbytes。'
      '2newREL97 F21F25/F34，HG85-SHARED-MEMORY-BUDGET新增F21成员，无新group，124pairs25groups471pending；resourcecoupling不伪装sharedoperatoratom。359branches24semantic,2073/285598review9/1352pending849/16/3136unchanged0wholeclosed/noallmodulelowestMFU。'
      '后续优先能改变剩余全范围算子融合/理论结论的证据。C/run/rank/samecallallsourcebounds/nativewriteextent/allocatorcapturelifetime是当前量化收益关键；不能用默认值/旧hash/不同群路由矩阵补齐。已有局部cap/scratch身份已回答，勿继续复读。保留offlineexistingprofile/frozen/math/必要primary，no server/model/Torch/NPU/capture/implementation/upload。GoalACTIVE/TaskACTIVE/LoopEVALUATING，verify_receiver_budget_v97须PASS，不重跑已成finalizer。')
    sources=[Path(p['path'])for p in read(O/names[0])['parts']];hist=[Path(x['path'])for x in read(O/names[1])['historical']]
    jobs=[dict(kind='profile',status='pass',description='Audit32cachedDB4472MC2TASKs andactualcapacityevidencegaps;old169/defaultconfig/notcurrent166value,distinctcustomversusCANNMegaMoebranch.',artifacts=[O/names[0],O/names[1]]+sources+hist),
      dict(kind='design-check',status='pass',description='1073280clipidentity/1296geometry/24workspacecases,full-loadscratch/cocmaxsynergyandsharedlivebudgetcounterexample;2conditionalrelationsnotactualspeedups.',
        artifacts=[O/n for n in names[2:]]+[O/'receiver-budget-analysis-preflight-v97.json',O/'receiver-budget-validation-history-v97.json',O/'whole-scope-requirement-audit-v97.json',
          O/'Engram-WKV-storage-and-work-models-v84.json',O/'ring-storage-capacity-models-v79.json',O/'tail-shared-memory-budget-v85.json',R/'fusion-scan.json'])]
    persist_refinement(version=97,previous=96,title='Clipping-equivalent receiver workspace and shared live-memory budget',
      summary=dict(filename='receiver-budget-summary-v97.json',previous_turn_classification='PROGRESS:v96inversemap/operatorcostcontractspersistedandverified',
        current_cached_MC2_TASKs=4472,current_cached_databases=32,current_C_or_samecallbounds_attested=False,capacity_identity_models=1073280,geometry_models=1296,workspace_models=24,
        new_target_source_files=0,new_public_documents=0,refined_candidates=5,refined_modules=6,new_semantic_families=1,branch_records=359,semantic_modules=24,
        new_pairs=2,new_groups=0,extended_group='HG85-SHARED-MEMORY-BUDGET',relation_clauses=124,higher_order_groups=25,pending_central_pair_reconciliation=471,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
        actual_peak_memory_HBM_or_speedup_or_whole_candidate_proved=False,implementation=False,native_test=False,
        remote_job_status='Alllocalcachedprofile/pinpointsource/mathworkfinished;no server/model/Torch/NPU/capture/serviceorimplementation.'),
      changes=['SeparatecurrentcustomMC2capacityargumentfromotherMegaMoebranchandold169evidence;criticalcurrentC/samecall/write/lifetimegapsremainexplicit.',
        'Proveclipping-equivalentcaptransformunderreliableglobalB;deriveexactsourceworkspace max andconditionalfull-loadunusedgenericreservation.',
        'Jointcap/scratchmaxcostandsame-deviceWKV/C2peakbudgetreconciled;requestedspaceisnotphysicalcreditorHBMtime.',
        '2newconditionalpairsandoneextendedbudgetgroup=>124/25/471;359records24semantic,originalgoalremainsactiveandincomplete.'],
      candidate_updates=updates,module_updates=mods,semantic_branches={'M12':branches},remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=names+['whole-scope-requirement-audit-v97.json'],publication_rows={},gate_updates=dict(overall_complete=False),
      scripts=[N/'analyze_receiver_budget_v97.py',N/'verify_receiver_budget_v97.py',N/'persist_profile_refinement_v90.py'],executor_prefix='receiver-budget',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
