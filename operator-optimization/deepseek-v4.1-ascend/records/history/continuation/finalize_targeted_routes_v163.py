"""Seal the exact-source/targeted supplementary route delta once."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v163 import persist_delta
assert not (O/'REVIEW-CHECKPOINT-v163.json').exists()
old=read(O/'REVIEW-CHECKPOINT-v161.json')
action=(
 '继续全部模块/分支和跨算子/层/卡/迭代原目标。上一轮真实69B联合探针与恢复是PROGRESS，本轮精确图源合同和35项新CPU观察器合同也是PROGRESS，不构成整模块或原始native闭合。'
 '优先把171同一完整C8×32768→1024/前缀90%共享builtin采集入口做成可审查的有界执行包：预先捕获必要creation阶段，正式prefill/decode窗口复用内置字段，actual guard/graph scope取证；默认关闭未验证的TorchDispatchMode和native额外hook。'
 '不要把全部native witness、所有reader或全部模块先齐备作为这条独立profile路线的前置条件，也不能把缺字段当独立；完整目标和未采到的native/offset/epoch/读者等分别继续保留。先准备全局产物/时间/占用护栏、相同完整Case、profiler-off参考与恢复，最后再取得确切新窗口授权；旧171探针授权已消耗。'
 '已实现有界offset/extent/version、capture/update/replay、实际runner/backend guard等补充观察接口，35项CPU合同仅验证原调用一次、返回/异常/配额及UNKNOWN边界；没有真实Torch dispatch/CANN/编译兼容性验证，不自动开启。'
 'actual TaskID bridge需经同run/process incarnation/device/context/显式stream/API witness核对，不能用task0哨兵或最近时间；推理_version可能缺失，不能当完整更新代次。'
 '图resource factory候选、weak refs、provider bind及局部PA workspace复用是有条件源合同，不等于实际存活显存、全部读者退役或当前DSv41选中PA路线；先记录use_compress/use_sparse/backendMRO等实际guard，减少无关分支工作。'
 '所有35+7候选、26模块358行的穷尽性、G00–G14、原9/1352、849PMU、16phase、3136归属、copy/control、实际数值/容量和最低MFU缺口继续open。整体Goal保持active、未完成。')
gate={'version':163,'parent':pin(Path(old['source_gate'])),
 'policy':'Profile first;shared conclusion-changing operator facts;no Cartesian pair sweep or unrelated framework study.',
 'answered_in_stated_scope':[
  {'question':'Can the CLI record_shapes flag alone enable the current Ascend field producer?',
   'answer':'No. Exact3413B factory omits forwarding it;temporary2-line route patch prepared,plus with_memory config required. Not installed.',
   'evidence':'profile-field-route-readiness-v162.json'},
  {'question':'Do graph resource/cache/update source records prove fixed independent workspaces or reader retirement?',
   'answer':'No. Factory candidate replacement,weak refs and provider overrides need explicit epochs/readers. PA same-key workspace sharing is conditional and not attested selected for DSv41.',
   'evidence':'capture-workspace-epoch-contract-v163.json'},
  {'question':'Are the newly prepared supplementary interfaces real-runtime proof?',
   'answer':'No.35 CPU contracts check exact-source forwarding and UNKNOWN guards only. Actual Torch dispatch/compiled graph/NPU paths remain untested;dispatch default off.',
   'evidence':'targeted-observer-readiness-v163.json'}],
 'next_targeted_questions':[
  {'id':'Q163-BUILTIN-COHORT','question':'Can one full fixed-case171 cohort collect compatible builtin and startup graph/guard evidence with enforced global budget and rollback?',
   'allowed':'Prepare isolated reviewable controller/entrypoint and CPU static checks;no new NPU/model execution without a new exact window.'},
  {'id':'Q163-MISSING-EXTRA','question':'Which specific cut premises still require native witness,offset/epoch/value/reader evidence after that builtin route?',
   'allowed':'Reuse prepared narrow supplement and old graph proofs;avoid duplicate all-tensor logging and irrelevant branches.'}],
 'not_all_missing_collectors_are_prerequisites_for_any_profile':True,
 'all_missing_fields_remain_UNKNOWN':True,'no_new_independence_or_whole_closure':True,'overall_complete':False}
overlay={'version':163,'parent':pin(Path(old['evidence_overlays'][-1])),
 'changed_shared_facets':['E0','E1','E2','E3','E4','E8'],
 'builtin_entrypoint_delta':pin(O/'profile-field-route-readiness-v162.json'),
 'exact_graph_source_contract':pin(O/'capture-workspace-epoch-contract-v163.json'),
 'targeted_interfaces':pin(O/'targeted-observer-readiness-v163.json'),
 'primary_query_epoch_docs':pin(O/'primary-query-epoch-docs-v163.json'),
 'new_pair_clauses':0,'new_independence_proofs':0,'new_native_operand_or_phase_promotions':0,
 'new_whole_module_closures':0,'whole_modules_closed':0,'overall_complete':False}
queue=deepcopy(read(Path(old['work_queue'])));queue.update(version=163,parent=pin(Path(old['work_queue'])))
for row in queue['work']:
 if row['id']=='Q155-GRAPH-CUTS':
  row.update(state='EXACT_SOURCE_EPOCH_RESOURCE_CONTRACT_AND_SHARED_BUILTIN_ACQUISITION_PREPARATION',
   action='Prepare one bounded same full171-case builtin cohort with startup graph/guard observation. Record actual backend/use_compress/use_sparse before routing generic PA update work. Reuse old proofs and supplement only conclusion-changing missing facts.',
   remaining='All branch/graph/effect exhaustiveness and actual source/value/reader domains remain open;CPU observer fixtures are not actual object/guard evidence.')
  row['evidence']+=['profile-field-route-readiness-v162.json','capture-workspace-epoch-contract-v163.json','targeted-observer-readiness-v163.json']
 elif row['id']=='Q155-NATIVE-OPERANDS':
  row.update(state='FAIL_CLOSED_BRIDGE_CORE_PREPARED_REAL_WITNESS_PRODUCER_OPEN',
   action='Reuse one real own-case success. Native bridge core rejects absent/sentinel/ambiguous process/device/context/API keys. Do not block independent builtin cohort preparation on all native gaps;retain the eager/captured routes and later add only needed supplements.',
   remaining='Real explicit API/task/capture witness and all-path/error/bypass/reader coverage not implemented or tested;no original1352 backfill,new model window not authorized.')
  row['evidence']+=['primary-query-epoch-docs-v163.json','targeted-observer-readiness-v163.json']
queue['new_evidence']=['Exact171 capture/update/worker sources bound','Factory shape flag forwarding gap isolated',
 'Targeted supplementary adapter/core implemented;35 CPU contracts only','Primary explicit-stream and inference-version limits pinned']
recs=deepcopy(read(O/'current-obligation-reconciliations-v161.json')['reconciliations'])
for row in recs.values():
 f=row['current_evidence_facets'];d=[]
 if 'E0' in f or 'E1' in f:d.append('v163 actual-source guard snapshot entry prepared;current raw nested config is not actual backend/branch/loaded-object proof.')
 if 'E2' in f:d.append('v163 bounded offset/extent/version extras implemented and CPU-tested;version/weak-ref/object token are not allocation epoch,all-readers or GPU completion.')
 if 'E3' in f:d.append('v163 fail-closed task bridge core prepared;actual witness producer missing. Task getter explicit-stream scope noted;no1352 recovery.')
 if 'E4' in f:d.append('v163 capture begin/end success/failure and update/replay interfaces implemented;Python graph epochs are not native template/modelId bridge.')
 if 'E8' in f:d.append('v163 conditional resource/provider/PA workspace contracts refine routing;generic PA reachability unknown. Prefer one shared builtin cohort rather than all-field readiness or per-pair prerequisites.')
 row.update(v163_evidence_delta=d,whole_domain_closed=False,original_broad_remaining_and_history_preserved=True)
write(O/'current-obligation-reconciliations-v163.json',{'parent':pin(O/'current-obligation-reconciliations-v161.json'),
 'reconciliations':recs,'does_not_delete_or_narrow_original_requirements':True,
 'all_missing_evidence_remains_unknown':True,'overall_complete':False})
report='''# v163：共享取证入口与图代次合同

完整Goal已恢复active。上一轮真实联合探针、服务恢复及v162工厂入口核对属于进展。本轮读取171自有容器的三个精确源文件，完成有界补充观察接口和35项新的CPU合同检查。未启动新NPU上下文、未发模型请求、未重启服务，旧单次探针授权保持已消耗。全部G00–G14及跨算子/层/卡/迭代范围保留，整体未完成。

精确Ascend profiler工厂没有转发record_shapes；配置项存在不等于该工厂实际使用。此前两行临时补丁及with_memory配置要求继续有效，但尚未安装。下一采集方向保留相同完整C8输入32768输出1024/前缀90%的Case：共享builtin字段与必要的startup graph/guard入口，不要求全部native/读者采集器先就绪才允许准备任何profile。未能观察的原生参数、offset/epoch、读者等继续按缺口保留，不缩减整体目标。

新观察器只补storage offset/nbytes、view指针、可用的版本字段及图边界；按TensorImpl与内置记录关联。输入/输出张量内容不读取，内置全部基础字段不再复制成一套全量logger。Python graph对象用process UUID、弱身份token、成功capture_begin代次和独立begin/end状态记录；begin/end失败不伪造成功代次。Python scope结束、event.record、weak reference和对象token都不当GPU读者完成或allocator generation。native join核心要求独立核实的同run/process/device/context/API witness与有效task/stream，并拒绝缺失、哨兵、复用歧义及signature矛盾；这些字段目前没有真实producer，不能靠状态字符串自证。

官方[Task查询文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/API/runtimeapi/aclcppdevg_03_1829.html)将查询域限定在当前线程、用户显式创建stream上最后下发的任务。它是9.2beta2文档，不自动证明本机9.2.0全部内部/error/bypass路径。官方[PyTorch2.10推理模式文档](https://docs.pytorch.org/docs/2.10/generated/torch.autograd.grad_mode.inference_mode.html)说明版本计数可能缺失；补充接口将这种失败记为UNKNOWN，不记0，也不把版本计数当完整参数更新代次。两份原始页面均已下载并pin。

精确UpdatableGraph源码另表明：缺key且use_max_workspace时可先后调用两次factory；缓存可被更大候选替换，capture_end再改为weak references。不能把每次factory候选大小相加当同时存活工作空间。GraphUpdateTask.bind允许provider参数覆盖capture初始kwargs；只看初始参数不能代表所有replay。局部PA分支在每次update新建ws_buffer，同key可给多个task赋同一个workspace Tensor，但这个结论只在确实执行该PA/update分支时成立。当前DSv41是否走该路线尚无实际guard证据，不把通用框架分支当当前优化对象，也不据不同key推独立。

为此增加了实际runner.use_compress/use_sparse、backend MRO、wrapper运行模式及graph绑定的观察入口。当前检查点compress_ratios嵌在text_config里，不能仅凭原始JSON替代实际对象属性或继承关系。后续先取实际guard，再决定哪些分支值得补证。

35项CPU检查采用精确图Python源码与假的设备方法，验证原operation/factory/provider/replay调用次数、返回别名、异常对象、失败cleanup/epoch、哨兵/复用拒绝、元数据失败和配额footer等。它们不验证真实Torch dispatch、CANN、编译或NPU capture兼容性。可选TorchDispatchMode默认关闭；开启前仍需验证其是否改变编译路径。首次用系统Python3.9导入slots dataclass失败，已改用现有Python3.12；没有安装软件、改写目标兼容源码或新做设备工作。

下一步完成共享builtin完整Case采集包的全局产物/时间/占用护栏、profiler-off参考、phase和恢复，再对具体新窗口请求授权。不是逐模块重复采集，也不是逐对做383次验证。原9/1352、849PMU、16phase、3136归属、G14 copy/control、数值/读者/物理容量与最低MFU缺口继续open。新增配对、独立性、历史native回填和整模块闭合均为0。
'''
jobs=[
 {'kind':'review','status':'pass','description':'v162/v163 exact own171 profiler/graph source reads and conditional source/operator scope audit;no runtime calls.',
  'artifacts':[O/'profile-field-route-readiness-v162.json',O/'graph-entrypoint-sources-v163.json',O/'capture-workspace-epoch-contract-v163.json']},
 {'kind':'design-check','status':'pass','command':'python3.12 targeted-evidence-v163/test_targeted_cpu.py',
  'code_ref':'New bounded observer/core and exact pinned graph Python source;fake device methods;no real Torch/CANN.',
  'description':'35 new CPU contracts pass;default dispatch off;API/epoch ambiguity fail closed;not native/compiled/runtime coverage.',
  'artifacts':[N/'targeted-evidence-v163/cpu-contract-result.json',O/'targeted-observer-readiness-v163.json']},
 {'kind':'review','status':'error','description':'Initial local Python3.9 source import failed on dataclass slots before tests;existing Python3.12 used successfully;not native/model failure.',
  'artifacts':[O/'targeted-local-environment-correction-v163.json']},
 {'kind':'design-check','status':'pass','description':'Independent new-input/source/protocol/default/scope review passes;no unchanged operator suites rerun.',
  'artifacts':[O/'targeted-routes-preflight-v163.json',O/'primary-query-epoch-docs-v163.json']}]
persist_delta(version=163,previous=161,prefix='targeted-routes',
 changes=['Exact171 graph/worker sources bound;provider/resource/epoch source contracts scoped without PA reachability claims.',
 'Bounded targeted extras and graph/guard interfaces implemented;35 CPU contracts pass;dispatch default off and real native witness missing.',
 'Primary query/inference-version limits prevent sentinel/epoch misuse.',
 'Prepare one shared full-case builtin cohort without blocking on all unrelated native gaps;retain all scope and unknowns.'],
 report=report,action=action,
 remaining_delta='G163: Targeted supplementary interfaces and35CPUcontracts prepared only;real Torch/compiled/capture/native witness and all-reader domains open. Independent builtin shared full171 Case preparation can proceed with exact limits/rollback/new authorization. All original full-scope/physical gaps retained.',
 source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,
 scripts=[Path(__file__),N/'persist_delta_checkpoint_v163.py',N/'verify_targeted_routes_v163.py',N/'analyze_targeted_routes_v163.py'],
 obligation_reconciliations=recs,
 remote_status='Read-only exact own171 container file acquisitions terminal. Local35CPUcontracts and source review terminal. No new model/device work;original service not restarted or instrumented. New full-case acquisition not run or authorized.')
