from datetime import datetime, timezone, timedelta
import hashlib, json
from finalize_origin_coverage_v9 import N, O, R, W, T, L, read, write, ctl

def main():
    assert not (O/'REVIEW-CHECKPOINT-v11.json').exists()
    now=datetime.now(timezone.utc)
    summary=read(O/'outer-consumer-summary-v11.json')
    assert summary['outer_compute']['definitions']==111
    assert summary['graph_memory']['exact_trace_matches']==561
    assert len(read(O/'candidate-profile-ledger-v11.json')['candidates'])==32
    assert len(read(O/'module-profile-ledger-v11.json')['modules'])==26
    assert not summary['overall_complete']
    receipt_names=['server-readonly-health166-v11.json','profile-outer-segments-copies166-v11.json','installed-fullhc-consumers166-v11.json',
                   'profile-copy-trace-patterns166-v11.json','prepare-profile-fields166-v11-v2.json','prepare-copy-pin-event-origins166-v11.json','vector-zero-init-origins166-v11.json']
    assert all(read(N/n)['returncode']==0 for n in receipt_names)
    assert read(N/'prepare-profile-fields166-v11.json')['returncode']==1
    zero=json.loads(read(N/'vector-zero-init-origins166-v11.json')['stdout'])
    write(O/'vector-zero-initialization-profile-v11.json',dict(**zero,scope='Exact vector-output ZerosLike census in pinned EP3;15source-linked sp_padding_mask callsites exist, but no type/shape-only join to the unlinked graph prefix is made.',no_new_source_read=True,overall_complete=False))
    # Keep existing candidate IDs aligned to their actual subphase.
    p=O/'candidate-profile-ledger-v11.json';d=read(p)
    next(c for c in d['candidates'] if c['id']=='F25')['review_v11']['finding']='Captured Engram projection/gate/residual compute sequence has38records per configured gap;8/8 comparisons with source-bound eager interior pass. Parameter/WKV ownership, lookup-ready and boundary copies remain separate contracts; no whole Engram closure.'
    write(p,d)
    action=('继续Loop004。v11已取得实质进展，现有profile优先、无法单独定案再定点补代码的工作顺序保持。'
      '每图111块外计算已逐条分七段21/38/38/2/2/2/8，11图1221记录，111definitions位置不变；这只是分级语义overlay，不是张量/别名join。'
      '8/8Engram类型序列匹配：graph38=eager40去首尾TensorMove，复制消失未获证明。561/561图MEMCPY精确匹配原trace，每图51，位置4首前+41块内+6两处Engram间隙；args无bytes/ptr/value，勿再扫536MB原trace。'
      '完整原结果保存在166的ep3-outer-segments-and-memory-tasks-v11.json SHAbce26ccaa1c16eb6d5c979d8d33feaf7f1efd3381a3317cb7211c0b3483bda2f和ep3-copy-trace-and-outer-patterns-v11.json SHA3cf7932c8f3ace424bac006cd70e7c0ff2745949135626518306de45fd0e730a。'
      'F26 exactsymbol完整安装根搜索90处/26文件，4活动读取文件SHA与冻结一致。Ascend v1 DSpark getter返回buffer但只在method==mtp时替换hidden，所以该直接草稿内容消费者已排除；3auxmean/gather(源层37/38/39)仍真实活跃。'
      'F26a仅省未观察快照写入，F26b单独考虑已有bufferNone局部collapse再gather路径；保留其他MTP/GPU/Ascendv2/外部观察者、数值/别名/图输出和归一化hidden消费者。不得再把getter存在本身当活跃读取，也不得全局删除。单次逻辑983040B不是HBM实测/收益。'
      'H02既有profile证据已用尽关键问题：官方rank0/8各12prepare根、14412选中嵌套API；每rank1032H2D/684pin/48sync，全部callchainId空且无PYTORCH_CALLCHAINS表。带栈旧EP3有0prepare根，不能移植时序/字段。'
      '新增prepare-copy-pin-event-origins-v11.json SHA dd01460b703f575ffd5e9a3a73edc1b3e04d32d1a8e7fed140b0c793a7614883，不要重跑已完成原DB查询。'
      '下一轮应落实有界166补采，优先同一model-callID绑定的worker输入准备完整Python/native调用栈、shape/字段及copy/pin/event ready与代际，同时取H01EngineCore onCPU及producer-ready/read和H03实际hash/frontend函数成本；按缺口分可独立小诊断，不必一次拼大采集。'
      '先读当前恢复包与已有隔离诊断/恢复脚本，重新核验当前健康/占用/容器ID/NPU进程归属，已有授权无需重复问；仅操作自己的模型clone/进程，恢复正常服务，不kill新增未归属进程。此次健康检查5e1bd9f...只为当时证明，不能代替新设备运行前核验。'
      '如先处理prefix额外ZerosLike，已有50vectorZero记录/39带栈，来源含15sp_padding_mask；读saved ep3-vector-zero-initialization-origins-v11.json SHAa4e12d01f41d853a7bcb7d9b669a1a53a0756ef350fb23b991cf9e48030f5d3d，不能仅同type/shape强配graph。'
      '初次graph0、其他rank同调用、全模块条件/别名/消费者、资源同域上界/因果/global最低利用率仍未闭合。0整模块关闭，Goal active，不因报告或这项profile分账完成而关闭。')
    jobs=[
      ('review','pass','Fresh read-only166ownership/health checks pass: expected containers/images/script;health/models200;all17NPU PIDs owned by base;no workload/service mutation.',[N/'server-readonly-health166-v11.json',N/'check_readonly_server_health_v11_166.py']),
      ('profile','pass','All111outer compute definitions assigned to7stable locations across11replays;77segments/1221records;2156separate MEMCPY tasks inventoried,561graph tasks.',[N/'profile-outer-segments-copies166-v11.json',N/'query_graph_outer_segments_and_copies_v11_166.py']),
      ('review','pass','Exact two-symbol search over installed source roots:90matches26files;read only active getters/allocation/branch and one GPUcounterexample. Four read source SHA pins equal frozen versions.',[N/'installed-fullhc-consumers166-v11.json',N/'query_fullhc_installed_consumers_v11_166.py',O/'F26-consumer-contract-v11.json',O/'PROFILE-CODE-QUESTION-REVIEW-v11.json']),
      ('profile','pass','561/561graph MEMCPY tasks each exact interval-match one original trace event;all7argskeys lack bytes/ptr.8/8Engram graph38/eager40interior comparisons pass;no inferred copy deletion.',[N/'profile-copy-trace-patterns166-v11.json',N/'query_copy_trace_and_outer_patterns_v11_166.py',O/'graph-copy-accounting-v11.json']),
      ('profile','error','First prepare-capability script referenced a locally-derived continuation JSON absent from server archive and stopped before database work;original script and failing receipt preserved.',[N/'prepare-profile-fields166-v11.json',N/'query_prepare_profile_fields_v11_166.py']),
      ('profile','pass','Corrected capability script embeds only24recorded numerical root indexes;reads same archived official profile folders. Both rank0/8DBs contain153050PYTORCH_API but no PYTORCH_CALLCHAINS table;SHA/stats pinned.',[N/'prepare-profile-fields166-v11-v2.json',N/'query_prepare_profile_fields_v11_166_v2.py']),
      ('profile','pass','Independent exact official24root interval check and copy/pin/event census:eachrank14412nested records;1032H2D/684pin/48sync;0callchains. Older stackedEP3capture has0prepare roots;explicit necessity for fresh bounded same-call evidence.',[N/'prepare-copy-pin-event-origins166-v11.json',N/'query_prepare_origins_v11_166.py',O/'prepare-profile-gap-proof-v11.json']),
      ('profile','pass','Read saved completeEP3origin evidence only:50vector-output ZerosLike tasks,39source-linked;15sp_padding_mask source API records. No forced join by type/shape to graph prefix.',[N/'vector-zero-init-origins166-v11.json',N/'query_zero_init_origins_v11_166.py',O/'vector-zero-initialization-profile-v11.json']),
      ('review','pass','Graded111semantic overlay and scoped F26a/b consumer contracts;32candidate and26module ledgers retain all open branch/alias/ready/resource questions. No target optimization implementation or performance gain claim.',[N/'reconcile_outer_consumers_v11.py',O/'outer-consumer-summary-v11.json',O/'outer-compute-owner-ledger-v11.json',O/'candidate-profile-ledger-v11.json',O/'framework-candidate-ledger-v11.json',O/'module-profile-ledger-v11.json',O/'module-phase-branch-matrix-v11.json',O/'module-coverage-gaps-v11.json',O/'PROFILE-FINDINGS-v11.md'])]
    registered=[]
    for i,(kind,status,description,artifacts) in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90','--kind',kind,
                '--command','Prepared v11 evidence execution; retrospective registration: '+description,
                '--code-ref','Frozen target unchanged;old169archive read on166;exact installed symbol read only;no new inference/NPU/service operation','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
        for p in artifacts:
            assert p.exists(),p
            args.extend(['--artifact',str(p)])
        ctl(*args);registered.append(dict(run_dir=run,status=status,description=description))
    checkpoint=dict(status='PROGRESS',previous_turn_classification='PROGRESS:v10replay/CSV/schedule andsmallM evidence',
      current_verified_changes=['111outer compute definitions in7stable segments across11replays;graded semantic overlay retains one exact zero-init field gap.',
       '561graph MEMCPY exact trace correspondence and51/task replay distribution;8Engram sequence checks pass without copy-elimination claim.',
       'F26 exact Ascend v1 DSpark direct buffer-content consumer excluded;F26a/bseparate conditional variants preserve other implementation/observer and numeric/lifetime contracts.',
       'H02official24prepare roots independently audited:all selected callchainId null,no callchains table;older stackedEP3capture has0prepare roots. Fresh bounded same-call evidence is justified, not source substitution.',
       'Fresh point-in-time166health/ownership pass;only read-only/archive/source actions thispass.'],
      summary=str(O/'outer-consumer-summary-v11.json'),candidate_ledger=str(O/'candidate-profile-ledger-v11.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v11.json'),
      module_ledger=str(O/'module-profile-ledger-v11.json'),branch_matrix=str(O/'module-phase-branch-matrix-v11.json'),report=str(O/'PROFILE-FINDINGS-v11.md'),
      source_gate=str(O/'PROFILE-CODE-QUESTION-REVIEW-v11.json'),coverage_gaps=str(O/'module-coverage-gaps-v11.json'),
      remote_job_status='All v11jobs terminal.1missing local-derived archive-path failure preserved and corrected by embedding only numerical profile root indexes. No new inference/NPU/service mutation.',
      pending_full_scope_domains=summary['remaining'],next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=now.isoformat())
    write(O/'REVIEW-CHECKPOINT-v11.json',checkpoint)
    if not (O/'profile-summary-before-v11.json').exists():
        (O/'profile-summary-before-v11.json').write_bytes((R/'profile-summary.json').read_bytes())
    prof=read(R/'profile-summary.json')
    prof['supplementary_outer_consumer_v11']=dict(summary=checkpoint['summary'],scope='Separate pinned EP3diagnostic andofficial rank0/8prepare populations;no timing/request join.',
      outer_compute_definitions=111,outer_compute_segments=7,graph_MEMCPY_exact_matches=561,F26_direct_Ascend_v1_DSpark_snapshot_content_consumer=False,
      F26_other_observers_globally_excluded=False,H02_official_roots_with_callchains=0,whole_modules_closed=0,overall_complete=False)
    write(R/'profile-summary.json',prof)
    for p in [L/'state.json',T/'index.json']:
        d=read(p);assert d['status'] in ['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=now.isoformat());write(p,d)
    for p in [L/'status.md',T/'status.md']:
        ls=p.read_text().splitlines()
        ls=[('- 下一步：'+action) if x.startswith('- 下一步：') else ('- 更新时间：`'+now.isoformat()+'`') if x.startswith('- 更新时间：') else x for x in ls]
        p.write_text('\n'.join(ls)+'\n')
    for p in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(p);d.update(latest_profile_first_review=checkpoint,overall_complete=False,whole_modules_closed=0,updated_utc=now.isoformat())
        d['next_read_only_action' if p==R/'STATE.json' else 'next_action']=action
        if p==W/'STATE.json':d.update(profile_candidate_ledger=checkpoint['candidate_ledger'],profile_module_ledger=checkpoint['module_ledger'])
        if p in [N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[checkpoint['summary'],str(O/'REVIEW-CHECKPOINT-v11.json'),str(O/'prepare-profile-gap-proof-v11.json'),checkpoint['coverage_gaps']]
        if p==N/'RECOVERY.json':d['original_service']='Fresh v11read-onlyhealth/ownership pass at2026-10-30T20:53:18.946882UTC;expected containers/image/script,health/models200,17NPU PIDs allbase-owned. Point-in-time only;recheck before live work. No mutation thispass.'
        write(p,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T))
    write(O/'task-run-registration-v11.json',dict(registered_utc=now.isoformat(),runs=registered,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    paths=[O/n for n in ['PROFILE-FINDINGS-v11.md','outer-consumer-summary-v11.json','outer-compute-owner-ledger-v11.json','F26-consumer-contract-v11.json','graph-copy-accounting-v11.json',
      'candidate-profile-ledger-v11.json','framework-candidate-ledger-v11.json','module-profile-ledger-v11.json','module-phase-branch-matrix-v11.json','module-coverage-gaps-v11.json',
      'PROFILE-CODE-QUESTION-GATE-v11.json','PROFILE-CODE-QUESTION-REVIEW-v11.json','prepare-profile-gap-proof-v11.json','vector-zero-initialization-profile-v11.json',
      'REVIEW-CHECKPOINT-v11.json','task-run-registration-v11.json','profile-summary-before-v11.json']]
    paths += [N/n for n in receipt_names+['prepare-profile-fields166-v11.json']]
    paths += [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
    write(O/'checkpoint-manifest-v11.json',dict(generated_utc=now.isoformat(),artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v11.json'),registered_runs=len(registered),statuses=[r['status'] for r in registered],validation_output=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
