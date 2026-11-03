"""Seal Linux CPU-only adapter validation; no native operand or fusion promotion."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v127.json').exists()
    pre=read(O/'host-adapter-preflight-v127.json');assert pre['status']=='PASS'and pre['checks']==553
    names=['host-adapter-inputs-v127.json','host-adapter-test-audit-v127.json',
        'host-adapter-collector-readiness-v127.json','host-adapter-validation-history-v127.json']
    detail=dict(evidence=names,CPU_test_platform='Linux aarch64',CPU_fake_runtime_cases=23,
        ASan_UBSan_cases=4,compiler_builds=6,CPU_adapter_forwarding_and_limits_validated=True,
        actual_CANN_interception_tested=False,installed_into_model=False,
        actual_new_CANN_host_packets_observed=0,original_pending_operands_recovered=0,
        new_fusion_findings=0,new_pairs=0,new_groups=0,new_native_promotions=0,
        no_log_is_not_no_work_witnesses=True,last_task_not_outer_task_witness=True,
        new_device_work=False,whole_requirement_closed=False)
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-pair-navigation','candidate-operator-footprints','operator-analysis-coverage-ledger']:
        obj=deepcopy(read(O/(base+'-v126.json')))
        obj['host_adapter_CPU_validation_v127']=deepcopy(detail)
        if base!='operator-analysis-coverage-ledger':obj['parent']='%s-v126.json'%base
        if base=='candidate-relation-clauses':obj.update(parent_relation_index=base+'-v126.json',new_pair_clauses=0,new_higher_order_groups=0,overall_complete=False)
        if base in ['candidate-relation-coverage','candidate-pair-navigation']:obj['parent_relation_index']='candidate-relation-clauses-v127.json'
        if base=='candidate-pair-navigation':obj['parent_footprints']='candidate-operator-footprints-v127.json'
        write(O/(base+'-v127.json'),obj)
    obligations=deepcopy(read(O/'coverage-current-obligations-v126.json'))
    for row in obligations['obligations']:
        if row['id']in ['G01','G03','G05','G09','G11','G13']:row['host_adapter_CPU_validation_v127']=deepcopy(detail)
        if row['id']=='G03':row['remaining']=(
            '1272 original eager MemSet scopes still uniquely traverse HostArgs,but no original packet destination/count/workspace/span has been recovered. '
            'The adapter now passes23 Linux CPU fake-runtime cases and4 sanitizer cases;this is not actual CANN interception or native operand evidence. '
            'Prove actual public-symbol/provider coverage and full same-run process/rank/device/thread/stream/task/CID identity before binding packets. '
            'A direct backend lookup and compiler-local recursion can bypass the observer;last-task getter can return an internal nested task. '
            'Actual parameter layout,allocation/alias generation,workspace/payload role and all readers/writers remain open. '
            'Both captured BOOL24 regimes/80 tasks need separate capture-creation/value/storage/update/readers evidence;the16same-shape eager calls are not these tasks.')
        if row['id']=='G13':row['remaining']=(
            'CPU adapter is implemented and tested in an independent /tmp directory in the existing Linux aarch64 container,not installed into the model. '
            'Do not restart core implementation or claim native readiness from23 passing fake-runtime cases. '
            'Next inspect the narrowly scoped actual symbol/provider call path and assemble full identity/lifetime acceptance evidence before any fresh device-window request. '
            'Real CANN metadata getter/internal-error semantics,actual forwarding,interposition bypass coverage,original host packets andcapturedBOOLstorage remain unvalidated. '
            'Require a complete footer and account for quota/time/name/IO losses;rc0 means submission,not completion. Zero reentrant skips does not prove one task. '
            'All9/1352 remain pending;v120Pad permission is consumed,no additional device/model capture is authorized or pending. '
            'No whole module,physical numeric capacity,MFU or lowest-time gate is closed.')
    obligations.update(parent='coverage-current-obligations-v126.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v127.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v126.json'))
    audit.update(previous_goal_turn='PROGRESS:v126 actual1272/1272 HostArgs API scopes and bounded Darwin CPU snapshot core;postcheck PASS3982',
        input_checkpoint=126,followup_checkpoint=127,host_adapter_CPU_validation_v127=detail)
    for row in audit['requirements']:
        if row['id']in ['MODEL_VERSION_HARDWARE_LOAD','NATIVE_OPERATOR_REGIMES','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            row.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v127.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v127.json',dict(
        status='TARGETED_HOST_PACKET_DIAGNOSTIC_ADAPTER_AND_CPU_FAKE_RUNTIME_ONLY',
        questions=read(O/names[0])['questions'],source_scope='New standalone diagnostic C code and pinned SDK ABI;no model/framework source survey or optimization changes.',
        no_target_model_operator_implementation=True,real_CANN_runtime_loaded=False,
        CPU_remote_container_used=True,remote_service_or_container_restart=False,
        Pad_authorization='GRANTED_AND_CONSUMED',additional_device_authorization=False,new_device_work=False,overall_complete=False))
    gap=(
        'CPU fake-runtime forwarding,prelaunch byte preservation,record/byte/time budgets,thread/TLS guards andloader negative cases now tested. '
        'Actual CANN interception/provider/error-state semantics and complete same-run native identity are unproved;CPU records do not recover original operands. '
        'Backend-handle lookups/internal local calls can bypass observer;last-task metadata can bind a nested task. '
        'Needactual packet layout,destination/count/storage/alias/workspace andreader retirement;captured BOOL80 has a separate route. '
        'No native fusion,physical capacity,MFU or whole-module conclusion closes.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v127 Linux CPU adapter23cases/4sanitizer cases pass;actualCANN interception/originaloperands untested.9/1352pending,358/26semantic,854/1346/48partialgraph,135/26/460relations,849PMU/16phase/3136ambiguity and0whole unchanged.')
    action=(
        'Continue original Goal;v127 PROGRESS is Linux CPU fake-runtime adapter validation,not native operator evidence. '
        'Read host-adapter summary/test-audit/collector-readiness/validation-history/current-obligations-v127;verify_host_adapter_cpu_v127 mustPASS. '
        'v126postPASS3982;1272/1272originaleagerMemSet HostArgs API scopes remain proven. v12723CPUcases/6builds/4ASanUBSanPASS onLinuxaarch64 in existingcontainerPID2032500;no realCANN/Torch/NPU loaded,no model/service restart. '
        'Defaultdisabled observer copiesliveHostArgs beforeforward,neverdeviceptrdereference;fakebackend verifies originalblocks/stream/cfg/packet/PH,rc/Cerrno. Limits4096B/32PH,56records/1MiB/60s(default),hard256/16MiB/3600s;arm,mutex,TLS,PIDscope/footer implemented. '
        '8threads160calls quota7writes7;sanitizedquota56writes56;expiry stopsmetadata;bytebudget retainsfooter;missingdir anddirectbackendlookup canproduce nofile despitework. '
        'Attempta-Werror unusedwrite fixed;attemptb compiler-local recursion bypassedobserver. Bothpreserved. Attemptc explicitRTLD_DEFAULTreentry testsbackend2/seen2/skipped1/onepacket;getterlasttask2isnested. Zero skipcounter doesnotprove one runtime task. '
        'AllremoteCPU subprocesses terminal,SSHmaster explicitlyclosed. No pendingdevicepermission;v120Padgrantconsumed. Adapteruninstalledintomodel,notfullcaptureready. '
        'Next targetonlyactualpublicsymbol/providercallpath+same-runfullidentity acceptance plan;useexistingcachedSDK/providerbytes or narrowreadonlyfetch. Do not expand frameworkresearch. '
        'NeedactualCANNforwarding/getterinternalerrorsemantics/bypasscoverage,actualhostpacket/providerlayout/dst/count/allocation/alias/workspacerole/readers. CapturedBOOL80separatecreation/value/storage/update/readers. No newNPU/model/capture/replay withoutfreshauthorization. '
        'All9/1352pending,2073/285598conditional,358/26semantic,854/1346/48partialgraph,135/26/460relations,849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU/lowest unchanged. '
        'Do noteditsealedv127/rerunfinalizer orremotecompiledartifacts. Versionedpostchecksidecar isauthority;CURRENT-POSTCHECKoverlayvalidonlyformatchingRECOVERY. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮完成了 HostArgs 取证适配器的 Linux CPU 验证。适配器独立放在评估目录中，默认关闭；通过显式开关、输出目录和 run ID 启用。它只在真实 HostArgs API 边界复制主机参数及 placeholder，不读取参数里设备地址指向的内容。模型、框架和算子实现没有修改，适配器没有装入模型进程。

在166对应的原105主机、既有容器内，以纯 CPU 假运行库完成6次编译和23个测试场景，其中4个包含ASan/UBSan。编译器将警告作为错误处理。测试库没有依赖或加载真实CANN/Torch，也没有建立NPU上下文。对安装CANN库只做固定SHA检查及readelf导出检查，不构成真实调用验证。所有测试进程已结束，SSH控制连接已明确关闭，容器PID保持2032500。

CPU后端核对了转发的blocks、stream、cfg、参数和placeholder，验证后端返回值及C errno仍能到达原调用方。后端会修改原参数中的一个指针槽，而记录保留修改前的零槽，因此测试覆盖了调用前快照。它没有证明CANN内部错误状态或实际元数据getter无副作用；真实运行库的这些性质仍需独立证据。

采集默认上限为单包4096B、32个placeholder、56条记录、1MiB和60秒，可配置硬上限为256条、16MiB和3600秒。8线程共调用160次，7条记录配额下写入7条并报告153次配额跳过；sanitizer场景写56条并报告104次跳过。时间到期后仍转发调用并停止元数据查询，1024B预算场景放弃正文但留下footer。未arm、非目标、名称查询失败、超大包、零配额及后端失败也有明确检查。footer表示日志结构结束，不表示原工作负载全部被观察；所有跳过和未知范围必须保留。

保留了两次有价值的失败记录：第一次编译因忽略write返回值被-Werror拒绝，修复后重编；第二次的递归测试虽然返回成功，但编译器生成的库内直接递归绕过了拦截入口，实际没有测到TLS保护。第三次通过RTLD_DEFAULT强制走公共入口，普通及sanitizer递归场景均得到后端调用2次、observer见2次、跳过递归1次、只写外层快照1条。没有把前一次测试成功误算为保护生效。

递归场景还证实：外层调用结束后查询的最后任务ID可以属于内层调用。即使reentrant_skips为0，库内绕过仍可能存在，所以该计数也不能证明一次API恰好对应一个任务。必须依靠同run的完整process/rank/device/thread/stream/task/connection/capture/update关联，rc0也只表示提交成功，不等于执行完成。

另一个负例直接通过backend句柄dlsym并成功执行，却没有observer日志；RTLD_LOCAL延迟加载则在显式已加载库回退路径下采集成功。这些结果限定了preload的覆盖范围，不能用缺少日志推断没有算子。fork子进程只转发、不声称采集；_exit留下正文但没有footer，判定不完整；目录不可写仍转发成功并没有文件。实际CANN/provider是否经过同一入口、是否存在类似绕过，尚未验证。

原profile已证明的1272次eager MemSet均经HostArgs，但本轮没有新增任何真实CANN host packet，更没有恢复旧destination、count、workspace或实际读写范围。真实provider布局、allocation/alias generation、workspace/payload角色和全部读者退役仍是融合合法性前提。80次捕获图BOOL24还需要单独追踪creation/value/storage/update/readers，不能由这个eager适配器代替。

本轮只是取证工具的准备与CPU验证。已消耗的v120 Pad授权没有扩展为新的NPU探针、模型运行或capture/replay授权。原9类/1352次pending、849个PMU缺口、16个phase未绑定图及3136项归属歧义保持原状；358条语义分支/26模块、135条条件关系/26联合组/460待核对pair、854节点/1346边/48提示的部分图也未作完成性提升。整模块关闭仍为0，全模块可比较的物理MFU和最低时间下界仍未建立，整体目标继续ACTIVE。
'''
    inputs=[Path(r['path'])for r in read(O/'host-adapter-inputs-v127.json')['inputs']]
    directory=N/'eager-host-adapter-v127'
    persist_refinement(version=127,previous=126,title='Linux CPU host-argument adapter validation and interception coverage limits',
        summary=dict(filename='host-adapter-summary-v127.json',previous_turn_classification='PROGRESS:v126 original1272/1272HostArgs route and bounded CPU core;PASS3982',
            **detail,refined_candidates=0,refined_modules=2,branch_records=358,semantic_modules=26,
            relation_clauses=135,higher_order_groups=26,pending_central_pair_reconciliation=460,
            partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            diagnostic_adapter_implemented=True,model_operator_implementation=False,native_test=False,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',
            remote_job_status='All CPU fake-runtime tests completed in existing Linux container;SSHmaster closed. RealCANN/Torch/NPU/model/replay not executed;adapter not installed into model.'),
        changes=[
            'Implement default-disabled bounded HostArgs adapter and verify23Linuxaarch64CPUfake-runtime cases/4sanitizer cases without realCANN/NPU.',
            'Prove CPU forwarding/prelaunch copying/quota/time/byte/thread/late-loader behavior;retain initial compile error and recursion-fixture coverage correction.',
            'Record explicit no-log bypass and nested-last-task counterexamples;leave actualCANN provider/fullidentity/layout/lifetime/capturedBOOL and allnative/whole gates open.'],
        candidate_updates={},module_updates={
            'M12':dict(candidates=[],finding='Router Pad helper diagnostic now has a Linux CPU-tested adapter;actual original MemSet packet/layout/storage and complete interposition coverage remain unproved.',evidence=names,gap=gap),
            'M19':dict(candidates=[],finding='Initialization observation tooling passes CPU forwarding/budget/loader tests,but no native write extent or legal helper deletion can be inferred;last-task may be nested.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='design-check',status='error',description='Retain initialCPUadapter build rejection:ignoredwrite return with-Werror;no runtime test/device launched.',artifacts=[N/'host-adapter-CPU166-v127.json',O/'host-adapter-CPU-results-v127.json',directory/'host_args_adapter-attempt-a.c']),
            dict(kind='design-check',status='error',description='Retain CPUrecursion fixture coverage failure:compiler-localcall bypassed observer despite23successful process exits;notTLSguard evidence.',artifacts=[N/'host-adapter-CPU166-v127b.json',O/'host-adapter-CPU-results-v127b.json',directory/'fake_runtime-attempt-b.c']),
            dict(kind='design-check',status='pass',description='Sixbuilds/23CPUfakecases/4sanitizer casesPASS;explicitpublicreentry,rc/Cerrno,budgets/thread/loadernegative cases;realCANNunloaded.',artifacts=[N/'host-adapter-CPU166-v127c.json',O/'host-adapter-CPU-results-v127c.json']+inputs),
            dict(kind='design-check',status='pass',description='553preflight checksPASS;rawlogbytes/SHA reconstructed,sourcepins verified,unresolvednativegates retained andSSHclosed.',artifacts=[O/'host-adapter-preflight-v127.json']+[O/n for n in names]+[O/'coverage-current-obligations-v127.json',O/'whole-scope-requirement-audit-v127.json'])],
        report=report,read_first=names+['coverage-current-obligations-v127.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_host_adapter_cpu_v127.py',N/'verify_host_adapter_cpu_v127.py',N/'run_host_adapter_cpu166_v127.py',N/'persist_profile_refinement_v90.py']+sorted(directory.glob('*.c'))+[directory/'host_packet_snapshot.h'],
        executor_prefix='host-adapter',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
