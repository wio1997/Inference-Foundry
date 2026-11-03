"""Persist host-argument API coverage and CPU diagnostic core,not live recovery."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v126.json').exists()
    pre=read(O/'host-launch-preflight-v126.json');assert pre['status']=='PASS'and pre['checks']==3939
    names=['host-launch-inputs-v126.json','host-launch-original-API-coverage-v126.json',
        'host-launch-installed-ABI-contract-v126.json','host-packet-CPU-core-result-v126.json',
        'host-launch-collector-readiness-v126.json']
    detail=dict(evidence=names,original_eager_MemSet_tasks=1272,same_thread_unique_HostArgs_scopes=1272,
        tested_CPU_core=True,ASan_UBSan='PASS',CPU_test_platform='Darwin arm64',
        max_bytes_per_copy=4096,max_placeholders_per_copy=32,CPU_snapshot_struct_bytes=4368,
        actual_new_host_packets_observed=0,live_CANN_interception_tested=False,
        original_pending_operands_recovered=0,new_fusion_findings=0,new_pairs=0,new_groups=0,
        new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    rel=deepcopy(read(O/'candidate-relation-clauses-v125.json'))
    rel.update(parent_relation_index='candidate-relation-clauses-v125.json',new_pair_clauses=0,
        new_higher_order_groups=0,host_launch_argument_route_v126=detail,overall_complete=False)
    write(O/'candidate-relation-clauses-v126.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v125.json'))
    cov.update(parent='candidate-relation-coverage-v125.json',parent_relation_index='candidate-relation-clauses-v126.json',host_launch_argument_route_v126=detail,overall_complete=False)
    write(O/'candidate-relation-coverage-v126.json',cov)
    nav=deepcopy(read(O/'candidate-pair-navigation-v125.json'))
    nav.update(parent='candidate-pair-navigation-v125.json',parent_footprints='candidate-operator-footprints-v126.json',
        parent_relation_index='candidate-relation-clauses-v126.json',host_launch_argument_route_v126=detail)
    write(O/'candidate-pair-navigation-v126.json',nav)
    footprints=deepcopy(read(O/'candidate-operator-footprints-v125.json'))
    footprints.update(parent='candidate-operator-footprints-v125.json',host_launch_argument_route_v126=detail)
    write(O/'candidate-operator-footprints-v126.json',footprints)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v125.json'));op['host_launch_argument_route_v126']=detail
    write(O/'operator-analysis-coverage-ledger-v126.json',op)
    obligations=deepcopy(read(O/'coverage-current-obligations-v125.json'))
    for r in obligations['obligations']:
        if r['id']=='G03':r['remaining']=(
            'All7 MemSet regimes/1272 eager tasks now have one exact generic launch link and one same-thread nested aclrtLaunchKernelWithHostArgs event each. '
            'Installed header and primary API docs identify a host packet and explicit relocation offsets,but the saved profile still contains no packet contents/destination/count/overlap/workspace role. '
            'A4096B/32placeholder CPU copy core passed local bounds and ASan/UBSan checks;its own-Pad fixture is constructed,not a newly observed host call. '
            'Need real Linux/CANN adapter/provider/layout/identity/window/forwarding validation and actual same-call operands before closure. '
            'Both captured BOOL regimes/80 tasks still require capture-creation/value/storage/update/all-reader identity;the16same-shape eager C2 calls remain different tasks.')
        if r['id']in ['G01','G03','G05','G09','G11','G13']:r['host_launch_argument_route_v126']=deepcopy(detail)
        if r['id']=='G13':r['remaining']=(
            'v126 proves the original1272 eager MemSet launch scopes all traverse a unique HostArgs API;do not restart the generic API-path search or treat GetParams device pointers as host data. '
            'Pure CPU core is compiled/tested on Darwin arm64 only. Next prepare Linux adapter/real-forwarding/loader/provider selection,full process/stream/task/capture/update correlation and bounded total output/window gates;do not call it an installed collector. '
            'Use supplied placeholder offsets,not universal3pointer/128Bpacking/slot8/data24 assumptions. Fixture32Bsource-span remains v120-own-only. '
            'No raw old packets or pending fields have been recovered. Graph BOOL80 remains separate;no new device permission is pending or granted andv120Pad grant is consumed. '
            'All9/1352,actual precision/physical capacities/whole graph/branch/comparable MFU/lowest gates remain open.')
    obligations.update(parent='coverage-current-obligations-v125.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v126.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v125.json'))
    audit.update(previous_goal_turn='PROGRESS:v125 exact eager/captured split and same-shape/modelId false-match evidence;postcheck PASS5589',
        input_checkpoint=125,followup_checkpoint=126,host_launch_argument_route_v126=detail)
    for r in audit['requirements']:
        if r['id']in ['MODEL_VERSION_HARDWARE_LOAD','NATIVE_OPERATOR_REGIMES','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v126.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v126.json',dict(
        status='ORIGINAL_EAGER_LAUNCH_API_SCOPES_THEN_PINNED_CANN_HOST_ARGUMENT_DECLARATIONS',
        questions=read(O/names[0])['questions'],source_scope='Only selected pinned runtime launch/placeholder/task/name declarations and primary official API pages;no framework survey.',
        diagnostic_copy_core_only=True,no_model_operator_implementation=True,
        no_CANN_library_device_or_remote_execution=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'host-launch-validation-history-v126.json',dict(events=[
        'Predecessorv125 revalidated PASS5589/manifest57. Its actual classification1272eagerMemSet/80capturedBOOL directs this targeted launch interface question.',
        'Each of1272 original eager MemSet runtime CIDs identifies one generic launch scope;same-thread bounded nesting contains exactly one aclrtLaunchKernelWithHostArgs event. Checked all32 raw cachedDBs against prior SHA/stat. No nearest timestamp or unrelated BOOL field binding used.',
        'Pinned installed9.2 header declares hostArgs/argsSize and8B{addrOffset,dataOffset}placeholder. Official9.1 HostArgs page and8.5 thread-last-task page archived by direct HTTP200;version differences explicitly supplemental,not installed-loader proof. Browser not used.',
        'Pure CPU C core implemented with disabled/device-source early guards,4096Bpacket/32placeholder limits,valid live-host-span contract,disjoint output,range/unique-relocation validation. It does not resolve symbols,log files,forward launches or call CANN.',
        'Darwinarm64 compilation,normal selftest,ASan/UBSan and ctypes fixture passed. Binary links only ordinary host system libraries. Existing device pointer words are copied as bytes,never dereferenced.',
        'Two initial CPU test-driver attempts failed because rendered_uint64_words in old JSON is a count19,not the word array. Preserved errors;reader corrected to actual saved Kernel Args text. No old packet/checkpoint modified.',
        'Declared fixture uses v120post-relocation152Bbytes with a constructed zero pointer slot and explicit8->24placeholder. Resolve96Bsource record gives old32Bsource span. It is not newly observed host memory or proof allold packets use those offsets.',
        'Native adapter remains absent/unloaded:Linux real-symbol/interposition/provider/layout/forwarding/returncode/fullidentity/capture-update and total-window-budget gates explicitly open. Per-copy bounds do not enforce total records/time/files.',
        'ThreadLastTaskId is current-thread/user-stream scoped,not a global or no-wrap correlation guarantee. Do not copy device-resident LaunchKernel/V2args or GetParamsisHostArgs0 pointers using this HostArgs-only primitive.',
        'Independent API nesting/header/pins/CPU executable/sanitizer/readiness verifier preflight3939checks PASS. No native operand promotions,candidate/pair/graph/semantic counts unchanged. TaskACTIVE LoopEVALUATING GoalACTIVE.'],overall_complete=False))
    gap=(
        'HostArgs API occurrence is proven for original eager MemSet calls,but no old argument packet is stored. '
        'The CPU-only copy core is not a deployed/validated LinuxCANNinterceptor;need real forwarding/loader/provider/layout/span ownership/full identity and bounded total capture window. '
        'Use validated supplied placeholder offsets and preserve unknown bytes;do not infer write extent from argsSize or a single fixture. '
        'Destination/allocator/alias generation,workspace role and all readers/writers remain. '
        'Captured BOOL80 needs its own creation/value/update/readers route. No physical upper/MFU/whole-module or native fusion conclusion is closed.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v126 actual1272/1272eagerHostArgs API path proven;bounded CPU snapshot core+ASan/UBSan passed,live LinuxCANNadapter not installed/tested. Constructed own152Bfixture/32Bsource span is not old packet evidence. All9/1352,358/26,854/1346/48,135/26/460 and0whole closures unchanged.')
    report='''本轮把v125提出的eager取证路线推进到一个已核实、可测试的主机参数入口。复查32份原始缓存数据库，1272次待补MemSet的runtime connection均精确指向一个generic launch；该scope内同globalTid恰好有一个aclrtLaunchKernelWithHostArgs事件，覆盖1272/1272。这里使用已绑定runtimeCID及明确的API嵌套边界，不按最近时间或同名shape猜测字段。原始profile仍没有保存hostArgs内容，所以尚未恢复旧destination/length。

安装版本9.2的固定acl_rt.h声明了该入口的hostArgs、字节argsSize和aclrtPlaceHolderInfo{addrOffset,dataOffset}。官方HostArgs文档说明参数区位于主机，placeholder描述待更新指针槽及inline数据区偏移；LaunchKernel/V2的device参数指针不能套用同一复制规则。公开页面为9.1/8.5补充资料，已直接HTTP归档并核对本地声明；它们不是安装库导出、910C动态绑定或observer实际生效的运行证明。[HostArgs接口说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1961.html)

围绕这个入口已实现并运行纯CPU快照核心，未接入模型：disabled或device-source时在任何payload读取前返回；每次最多4096B参数和32个placeholder；检查独立输出、边界及唯一inline relocation，原参数只读。核心不解析设备地址指向的内容，不分配CANN内存、不写日志、不解析动态符号、不调用或转发CANN。调用者仍须提供真实live HostArgs API中的有效主机span，不能把任意数值或GetParamsisHostArgs0地址标成主机内存来读取。

在本地Darwin arm64上，C编译、边界测试、ASan/UBSan及ctypes测试通过；快照结构4368B，placeholder结构8B。正常CPU动态库只依赖主机系统库，没有加载CANN或Torch/NPU。初始两次测试驱动把旧JSON中的rendered_uint64_words计数19误当数组，已保留失败记录并改为读取实际debug-dump Kernel Args文本，原封存数据未改变。

解析测试使用一个明确构造的fixture：取v120自己Pad捕获的152B post-relocation字节，把指针槽置零，再提供addrOffset8/dataOffset24，复制后只按所给偏移取96B source record，结果仍是既有32B source-span。输入字节未改变，重复/歧义placeholder和越界数据被拒绝。它没有观测新的真实hostArgs，更不能证明原1272次都采用slot8/data24、固定三指针prefix或32B写范围；未知尾部字节仍保留，不称padding或多余工作。

下一层native adapter还未安装或测试。它必须先验证Linux/CANN真实符号和interposition路径、函数/provider与参数布局，确保原指针/cfg/stream/参数和返回值原样转发；再加入按目标kernel选择、完整run/process/rank/device/thread/stream/task/capture/update identity以及总记录/输出/时长预算。现有单次复制边界不等于整次采集预算已生效，也不能证明不会漏掉未匹配provider。

可选ThreadLastTaskId只描述当前线程在用户显式创建stream上的最后任务，不能替代完整关联，失败、重复或歧义须保持未知。未来实际参数观察应与同run的API/task证据结合；只取得host packet也仍需证明目标allocation/alias generation、workspace/payload角色、重叠读写与读者退役，才可判断helper删除或融合合法性。[线程任务ID说明](https://www.hiascend.com/doc_center/source/en/canncommercial/850/API/appdevgapi/aclcppdevg_03_1829.html)

本轮没有新增设备窗口、模型加载、replay、服务重启或NPU探针，Padv120授权保持已消耗。80次捕获图BOOL字段仍须单独追踪capture-creation/value/storage/update及全部读者；eager入口本身不能解决它们。前置3939项核验通过，原9类/1352次pending没有提升；35模型候选、7历史框架记录、358语义行/26模块、135条件关系/26联合组/460待核对pair、部分图854/1346/48和全部原生/PMU/phase/归属不确定性均保留，整模块关闭0。新增的是已由原profile覆盖、由CPU测试支撑的取证能力，整体优化分析与理论上限仍未全部完成。
'''
    action=(
        'Continue original Goal;v126 PROGRESS on actualHostArgs route+testedCPUcopycore,not originaloperand recovery. '
        'Read host-launch summary/original-API-coverage/installed-ABI/CPU-result/collector-readiness/current-obligations-v126;verify_host_launch_route_v126 must PASS. '
        'v125postPASS5589/manifest57.1272/1272originaleagerMemSetlaunchCIDscopes eachcontainone same-threadHostArgsAPI;all32DBSHA/stat checked. '
        'Pinned9.2header hostArgs+byte argsSize,8Bplaceholder{addrOffset,dataOffset};official9.1/8.5docs directHTTP cached,supplement notruntimeattestation. '
        'CPUcore in eager-host-launch-source-v126 compiled/testedDarwinarm64,Cbounds+ASanUBSan+ctypesPASS:4096Bpacket32placeholders,4368Bsnapshot,disabled/deviceearlyreturn,read-onlyinput/disjointoutput/uniqueinlinebounds. NoCANN/Torch/NPUloaded. '
        'Constructed fixture fromownv120152Bpost-relocationbytes+zero slot+explicit8->24 resolves96Brecord/32Bsource-span;notnewhostpacket oruniversalpacking. Oldrendered_uint64_wordsiscount19;twoCPUdrivererrorsretained,correctreadKernelArgs. '
        'Next implement/test actualLinuxCANNadapter againstCPUfakebackend/realforwarding/loader/provider/selection/identity/totalbudget before anydevicewindowrequest. Coredoesnotinterpose/log/forward;do notcallcollectorreadyorinstalled. '
        'UsegenuineHostArgsvalidhostspans only,neverdeviceLaunchV2/GetParamsisHostArgs0. LastTaskID onlycurrentthread/userstream;fullrun/process/rank/device/capture/update/stream/task correlation mandatory,ambiguitiesunknown. '
        'Needactualpacket/provider/dst/count/allocation/alias/workspace/readers;graphBOOL80separatecreationfieldroute. All9/1352pending,358/26,854/1346/48,135/26/460,2073/285598conditional849PMU16phase3136ambiguity0whole/noMFU unchanged. '
        'Alllocaljobscomplete,no server/model/device/replay/operatoroptimization. Padgrantconsumed,no pendingnewpermission. Do noteditsealedv126/rerunfinalizer or rebuildsealedCPUbinaries. Postcheckoverlayusedonlyformatchingcheckpoint. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    source_dir=N/'eager-host-launch-source-v126'
    persist_refinement(version=126,previous=125,title='Host launch parameter evidence route and bounded CPU snapshot core',
        summary=dict(filename='host-launch-summary-v126.json',previous_turn_classification='PROGRESS:v125 exact eager/captured operand provenance;PASS5589',
            **detail,refined_candidates=0,refined_modules=2,branch_records=358,semantic_modules=26,
            relation_clauses=135,higher_order_groups=26,pending_central_pair_reconciliation=460,
            partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            diagnostic_core_implemented=True,model_operator_implementation=False,native_test=False,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',
            remote_job_status='All local raw-profile/API/header/HTTP/CPU-core checks complete. No remote/Torch/CANN/NPU/model/replay/service work;native adapter remains uninstalled/untested.'),
        changes=[
            'Prove all1272 original eager MemSet generic launch scopes contain exactlyone same-thread HostArgsAPI;actualroutecoverage now known without rawpacket contents.',
            'Pin installed host-packet/placeholder declarations and primary docs;compile/runbounded CPUcopy/inline-offsetcore withASanUBSan anddeclaredownpacketfixture,without CANN/device access.',
            'Retain explicitLinuxnativeadapter/forwarding/provider/layout/identity/totalbudget/readerlifetime andcapturedBOOL gates;no originalnative/wholeorperformancepromotion.'],
        candidate_updates={},module_updates={
            'M12':dict(candidates=[],finding='Original eager router Pad helpers all use the verified HostArgs boundary;bounded CPU snapshot prototype can preserve bytes/offsets,but actual native packets/provider/forwarding and write/read scope remain unobserved.',evidence=names,gap=gap),
            'M19':dict(candidates=[],finding='Initialization MemSet launch path has an explicit host-parameter observation candidate;do not replace native spans by packet bytes or extrapolate own152/96/32fixture to old tasks.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='design-check',status='error',description='First local CPU fixture reader used count19asarray;test-driver TypeError,old source packet unchanged/no devicework.',artifacts=[O/'host-launch-CPU-test-history-v126.json']),
            dict(kind='design-check',status='error',description='Second local CPU fixture attempt retained same schema error before final reader correction;core bounds/sanitizer stage did not access CANN/NPU.',artifacts=[O/'host-launch-CPU-test-history-v126.json']),
            dict(kind='profile',status='pass',description='Verify1272originaleagerMemSet scopes eachuniquelynest HostArgsAPI across32pinnedDBs;pinCANNheader andtwoofficialHTTPsourcepages.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='LocalCPU Ccore/selftests/ASanUBSan/ctypesfixture and3939preflightchecksPASS;liveadapter/oldoperands/wholecoverage stillopen.',artifacts=[O/'host-launch-preflight-v126.json',O/'host-launch-validation-history-v126.json',O/'coverage-current-obligations-v126.json',O/'whole-scope-requirement-audit-v126.json'])],
        report=report,read_first=names+['coverage-current-obligations-v126.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_host_launch_route_v126.py',N/'test_host_packet_snapshot_v126.py',N/'reconcile_host_launch_core_v126.py',
            N/'verify_host_launch_route_v126.py',source_dir/'host_packet_snapshot.h',source_dir/'host_packet_snapshot.c',source_dir/'selftest_snapshot.c',N/'persist_profile_refinement_v90.py'],
        executor_prefix='host-launch',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
