"""Persist the actual shared cohort and operator evidence as one coverage delta."""
import collections,json,sys
from pathlib import Path
from copy import deepcopy
sys.path.insert(0,'/Users/wio/work/166-vllm-analysis-continuation-20261030')
from finalize_origin_coverage_v9 import N,O,T,read,write
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v163 import persist_delta

HERE=Path(__file__).resolve().parent
old=read(O/'REVIEW-CHECKPOINT-v163.json')
rank=read(HERE/'ALL32-RANKING.json');major=read(HERE/'major-op-summary.json')
zero=read(HERE/'zero24-complete-and-source.json')['zero24']
done=read(O/'shared-builtin-window-completion-v164b.json')
receipt=read(HERE/'derivation-receipt.json');offline=read(HERE/'STATUS.json')
records=rank['profiles'];total=sum(x['sum_duration_ns_across_ranks_windows'] for x in rank['operator_ranking'])
small=sum(x['sum_duration_ns_across_ranks_windows'] for x in rank['operator_ranking'] if x['op'] in ['PadV3','MemSet','ZerosLike'])
sites=collections.Counter()
for r in zero['records']:
 if r['phase']=='formal':
  ts=[t for t in r['input_tensors'] if t['dtype']=='bool' and t['shape']==[24]]
  assert len(ts)==1
  sites[(r['stack_excerpt'].split(';')[0],ts[0]['device_type'],'execute_dummy_batch' in r['stack_excerpt'])]+=1
site_rows=[{'first_frame':k[0],'device_type':k[1],'execute_dummy_batch':k[2],'CPU_scope_event_count':v} for k,v in sites.items()]
checks={
 'cohort_terminal_restored':done['status']=='SHARED_FIXED_CASE_COMPLETE_ORIGINAL_SERVICE_RESTORED' and done['health_http']==200,
 'full_fixed_case_valid':done['baseline_valid'] and done['profile_valid'],
 'eight_response_texts_equal':done['exact_response_text_matched_count']==8,
 'sixteen_worker_logs_closed':done['all16_NPU_worker_footers_present'] and all(not x.get('footer_failures') for x in done['NPU_worker_supplementary_footers']),
 'vendor_export_returned':offline['state']=='VENDOR_ANALYSE_RETURNED' and offline['returncode']==0,
 'all32_profiles_ranked':rank['status']=='ALL32_VENDOR_DATABASES_RANKED' and len(records)==32,
 'all48_have_builtin_inputs':receipt['raw_census']['counts']['profiles_with_input_detail']==48,
 'all_raw_parsers_closed_without_errors':receipt['raw_census']['counts']['profiles_with_parse_errors']==0,
 'copied_original_raw_inputs_verified':read(HERE/'INPUT-MANIFEST.json')['original_and_copy_verified_equal'],
 'exact_native_flow_counts_balance':sum(rank['exact_compute_task_to_torch_connection_cardinality'].values())==sum(x['compute_task_count'] for x in records)==288846,
 'zero_sample_cap_repaired':zero['records_count']==152 and len(zero['targeted_rescan_profiles'])==24,
 'PMU_replay_records_disambiguated':major['duplicate_after_pairing']==0 and major['PMU_pairing']=={'unique_containing_same_id_interval':239525},
 'target_MoE_CPU_stack_in_all16_first_windows':len(major['target_CPU_stack_examples'])==16,
 'no_whole_closure':rank['overall_complete'] is False and major['overall_complete'] is False,
}
assert all(checks.values()),checks
write(O/'shared171-profile-preflight-v165.json',{'status':'PASS','preflight':True,'checks':checks,'basis':'Actual completed cohort, exported databases and measured records;no synthetic runtime test claims.'})
evidence=[p for p in HERE.iterdir() if p.is_file()]
evidence += [p for p in O.glob('*v164*.json') if p.is_file()]
for folder in ['shared-builtin-cohort-v164','shared-builtin-cohort-v164b']:
 evidence += [p for p in (N/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
evidence += [N/name for name in ['read_shared_builtin_raw_v164.py','prepare_shared_builtin171_v164.py','submit_shared_builtin171_v164.py','observe_shared_builtin171_v164.py']]
write(O/'shared171-profile-inputs-v165.json',{'inputs':[pin(p) for p in sorted(set(evidence))],'overall_complete':False})
summary={
 'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,
 'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,
 'actual_shared_cohort':{'startup_CPU_profiles':16,'formal_CPU_NPU_profiles':32,'input_detail_profiles':48,'valid_fixed_case':True,'exact_text_matches':8,'restored_health_http':200},
 'new171_compute_task_executions':288846,
 'exact_vendor_compute_to_CPU_connection_cardinality':rank['exact_compute_task_to_torch_connection_cardinality'],
 'exposed_task_work_ranking':rank['operator_ranking'],
 'all_rank_window_task_duration_ns':total,
 'DispatchFFNCombine_exposed_task_work_percent':100*rank['operator_ranking'][0]['sum_duration_ns_across_ranks_windows']/total,
 'PadV3_MemSet_ZerosLike_exposed_task_work_percent':100*small/total,
 'formal_BOOL24_CPU_scope_roles':site_rows,
 'new171_PMU_pairing':major['PMU_pairing'],
 'historical_native_promotions':0,'new_independence_proofs':0,'whole_modules_closed':0,
 'runtime_authorization':'User now authorizes probes and operating this owned container inference/framework processes;do not request repetitive per-probe approval.',
 'timing_limit':'Instrumented windows include CPU overhead and dummy participation;task-work sums are not production latency or physical ceiling.',
 'vendor_memory_warnings':len(receipt['vendor_warnings']),
 'dynamic_attach':'Installed CLI rejects current namespace worker IDs (single and comma list);no collection or new inference in these idle checks;service health remains200.',
}
write(O/'shared171-profile-summary-v165.json',summary)
action=(
 '继续全部模块/分支和跨算子、层、卡、迭代优化目标；本轮已完成171相同完整C8参照、共享profile、原服务恢复和全部32设备窗口离线分析，属于PROGRESS，整体未完成。'
 '直接复用shared-profile-analysis-v165/ALL32-RANKING.json、MAJOR-OP-FOCUS-v2.json、major-op-summary.json，不重读框架或重复旧CPU契约测试。'
 '先解决F21主耗时复合算子内部计算/通信/等待及EP不均衡，结合已有原生协议/条件MAC下界；本次实际调用栈已定位当前171的aclnnDispatchFFNCombine及moe_comm_method591，无需重读整个框架。'
 '新32窗口累计任务工作时长中该边界约68.28%，Pad/清零三类合计约0.25%；这不是生产延迟占比或Amdahl加速比，不能据此删除其他必要范围。'
 '优先用低开销同完整Case设备时间线检查profiling引入的rank等待；已尝试当前容器msprof动态入口，namespacePID单个/列表均被CLI拒绝，两个失败保留，下一步只核对其PID/动态注册入口，避免逐卡反复探针。若动态路线不可用，再使用一次共享NPU为主的短窗口，而非逐模块重载模型。'
 '用户已明确授权这个容器的推理/框架/探针进程操作；本范围内不重复请求许可，不动他人任务。'
 '形式相同的BOOL24按真实栈分流：新窗口64条C2/Engram记录来自dummy，48条是CPU gather，8条SP记录来自实际模型execute；旧原始80模板不按shape/name/time回填。'
 '新171计算任务183242有唯一官方CPU连接、105604没有该连接；该连接只绑定已记录CPU范围，不能替代张量offset/epoch/数值/全部reader或Python图到原生模板映射。'
 '同一原生graph模板globalTaskId可重复3次；PMU按同vendorDB、同ID且唯一包含timestamp的执行区间配对，239525条所选PMU记录无重复或歧义；不做最近时间补配。'
 '实际guard compressTrue/sparseFalse/MLA-MRO用于减少未选中通用PA分支工作，只适用于观察域。'
 '35+7候选、26模块358行穷尽性和G00-G14继续保留；原9/1352、849PMU、16phase、3136归属、copy/control、all-reader/数值/容量和最低物理MFU仍open。')
gate={'version':165,'parent':pin(Path(old['source_gate'])),'policy':'Profile first,operator-directed;reuse all existing source/proofs;no Cartesian pair sweep.',
 'actual_answers':[
  {'question':'Can a full shared fixed-case cohort provide the missing builtin fields without per-pair probes?','answer':'Yes:16startup+32formal profiles,all48 have input detail;8response text matches and original service restored.'},
  {'question':'Are all BOOL24 observations one real-request role?','answer':'No:new171 formal120CPU-scope events separate64dummy DSA/Engram,48CPU gather and8real-execute SP;not a historical-native mapping.'},
  {'question':'Where should conclusion-changing analysis start?','answer':'Measured exposure concentrates in DispatchFFNCombine;current171 target source boundary is linked by official task-to-CPU connection and callchain,without a framework reread.'},
  {'question':'Can PMU be grouped only by template globalTaskId?','answer':'No:captured template IDs repeat. v2 pairs only same-ID,unique containing vendor execution intervals;unmatched/ambiguous rows are not promoted.'}],
 'next_targeted_questions':[
  {'id':'Q165-MOE-CRITICALITY','question':'How much of current MoE duration is internal useful work versus collective/rank waiting or instrumentation disturbance?','allowed':'Existing task/PMU/communication profiles first;one shared low-overhead full-case short window if necessary;only narrow operator/profiler entrypoint code if profile cannot decide.'},
  {'id':'Q165-PID-ATTACH','question':'Which PID namespace or runtime dynamic-registration contract does this installed CLI accept?','allowed':'One exact installed entrypoint check;retain both rejected idle attempts;do not launch16independent experiments.'}],
 'current_container_process_authorization':'Broad user authorization supersedes prior per-window permission language within owned container scope.',
 'all_missing_fields_remain_UNKNOWN':True,'no_new_independence_or_whole_closure':True,'overall_complete':False}
overlay={'version':165,'parent':pin(Path(old['evidence_overlays'][-1])),
 'actual_cohort':pin(O/'shared-builtin-window-completion-v164b.json'),'actual_ranking':pin(HERE/'ALL32-RANKING.json'),
 'actual_major_op_focus':pin(HERE/'MAJOR-OP-FOCUS-v2.json'),'actual_major_op_summary':pin(HERE/'major-op-summary.json'),
 'affected_candidates':['F21','F31','F32'],'affected_modules':['M04','M12','M14','M21','M22','M23','M24'],
 'F21_delta':'Current171 native DispatchFFNCombine selected boundary/target callchain,explicit shapes and pipeline ratios measured;internal stage timing,receiver counts and a matching physical upper bound not established.',
 'BOOL24_delta':site_rows,'actual_guard_variants':receipt['raw_census']['actual_runner_guard_variants'],
 'changed_shared_facets':['E0','E1','E2','E3','E4','E8'],
 'new_pair_clauses':0,'new_independence_proofs':0,'new_native_operand_or_phase_promotions':0,
 'new_whole_module_closures':0,'whole_modules_closed':0,'overall_complete':False}
queue=deepcopy(read(Path(old['work_queue'])));queue.update(version=165,parent=pin(Path(old['work_queue'])))
queue['work'].insert(0,{'id':'Q165-MOE-TIMING','state':'ALL32_PROFILE_RANKED_ACTUAL_TARGET_BOUNDARY_LINKED',
 'action':'Reuse new171 task/PMU/callchain and existing F21 protocol/MAC contracts. Resolve internal work/wait and cross-rank criticality with one shared low-overhead window if needed;first check exact installed dynamic PID entrypoint after two fast failures. No repeated permission inside authorized owned container.',
 'remaining':'Instrumented task duration is not production critical path;new171 receiver rows,per-stage readiness,physical capacity/traffic and legal rescheduling bounds open.',
 'evidence':[str(HERE/'ALL32-RANKING.json'),str(HERE/'MAJOR-OP-FOCUS-v2.json'),str(HERE/'major-op-summary.json')]})
for row in queue['work']:
 if row['id']=='Q155-GRAPH-CUTS':
  row.update(state='SHARED_ACTUAL48_PROFILE_FIELDS_GUARDS_AND32_DEVICE_WINDOWS_AVAILABLE',
   action='Reuse exact process/scope markers and actual compressTrue/sparseFalse/MLA guard;route only reached operator branches. Build dependency neighborhoods from profiler calls/tensors;missing offset/epoch/readers stay unknown.',
   remaining='Graph-template/Python identity bridge and effect/readers exhaustive coverage remain open;newcase does not backfill old16phase or80BOOL templates.')
  row['evidence']+=[str(HERE/'zero24-complete-and-source.json'),str(HERE/'ALL32-RANKING.json')]
 elif row['id']=='Q155-NATIVE-OPERANDS':
  row.update(state='NEW171_OFFICIAL_CONNECTION_CENSUS_HISTORICAL_OPERANDS_STILL_OPEN',
   action='Use183242unique official CPU connections in their exact new171 DB namespace;retain105604no-flow executions without shape/time guessing. Add operand/reader witnesses only when needed for a specific fusion decision;no original1352 backfill.',
   remaining='All-path/error/bypass operands,offset/allocator/update epochs,readers and historical task witnesses remain open.')
  row['evidence']+=[str(HERE/'ALL32-RANKING.json')]
queue['new_evidence']=['Full171 C8 reference and shared48profiles completed,8texts identical,service restored',
 '32officialDBs and288846compute executions ranked','F21actualtargetboundary/PMU/shape focus linked',
 'BOOL24 cap repaired and real/dummy/CPU roles split','Repeated-template PMU interval pairing repaired;dynamic PID errors retained']
recs=deepcopy(read(O/'current-obligation-reconciliations-v163.json')['reconciliations'])
for key,row in recs.items():
 row.update(v165_evidence_delta={'shared_actual_profile':str(HERE/'ALL32-RANKING.json'),'operator_overlay':str(O/'shared171-profile-operator-overlay-v165.json'),
  'scope':'New171 full fixed-case domain only;actual field/guard/correlation/exposure evidence advances routing. All original requirements and historical gaps retained.'},
  whole_domain_closed=False,original_broad_remaining_and_history_preserved=True)
write(O/'current-obligation-reconciliations-v165.json',{'parent':pin(O/'current-obligation-reconciliations-v163.json'),'reconciliations':recs,'does_not_delete_or_narrow_original_requirements':True,'overall_complete':False})
report='''# v165：171完整负载共享profile的实际结果

已完成同一完整C8、每请求32768输入/1024输出、前缀约90%的无profile参照和隔离采集。8个请求的文本与参照逐字相同；原服务已恢复、健康检查200。无profile单次观测501.15输出token/s，只是参照，不是稳定性能结论。首次诊断启动因过早施加文件大小限制而阻止Engram共享文件创建，错误和恢复均保留；修正为模型加载后生效，没有重写失败记录或重复跑参照。

得到16份创建期CPU记录、32份正式CPU/NPU窗口，48份均有张量输入detail。16个NPU worker补充日志都有正常footer。官方离线导出全部返回；32份设备数据库覆盖288846条计算任务执行。官方connection表可将183242条执行唯一连到记录的CPU范围，105604条没有该连接。连接域以各自数据库为界，不补猜旧profile，也不等价于原生参数、全部reader或graph对象映射。

跨卡、跨窗口的累计计算任务时长中，DispatchFFNCombine占68.28%；单列AllReduce、AllToAll、AllGather、ReduceScatter合计约15.08%；PadV3、MemSet、ZerosLike合计0.246%。这是采集窗口的工作暴露，不是请求延迟占比，不能套Amdahl给生产加速倍数。CPU栈和内存记录会扰动调度；第一窗口还包含dummy参与和前端空隙。先分解主要复合算子与卡间等待，避免继续围绕微小清零逐对做探针。

当前171目标模型的aclnnDispatchFFNCombine已通过官方连接及调用栈定位到moe_comm_method.py:591，16个第一窗口均有目标形状记录。无需重读整个框架。目标的本地专家数24、M=211/1023/818等与草稿8专家形状分开；这些形状不是实际receiver行数。M=211域640次的平均报告aic_mac_ratio约1.81%，M=1023域约7.89%；这支持进一步区分计算与等待，不能把其倒数当可实现加速或MFU。沿用已有每assignment35389440MAC的条件式；本次实际receiver分布、内部阶段耗时和匹配的硬件容量上界仍缺，不能把旧166行数与新171时间相除。

同一globalTaskId在图replay中可重复三次。首次聚合把重复PMU键保留为未知；修正版按同vendorDB、同ID且timestamp唯一落入某个任务执行区间配对，239525条所选PMU记录无重复或歧义，不使用最近时间。PMU仍有缺项；28种pipeline字段没有直接给出HBM字节、链路字节或认证物理峰值。

BOOL24样本上限导致的24个遗漏已只对相应24份原始文件定点补扫，未重采模型。全部152条CPU算子范围事件包括32条创建期SP清零、120条正式记录。正式记录中，64条DSA/C2/Engram事件栈明确含execute_dummy_batch；48条是CPU gather；8条SP mask来自真实模型execute。zeros_like与其内部zero_可为嵌套范围，不能当64次独立原生kernel。仅定点读取了profile指向的sp_padding_mask小函数及其本地分片函数：None输入会建立全false mask，TP整除时分片仍clone。可继续研究隐式全有效谓词/消费者融合，但未覆盖全部读者，不能据此删除缓冲区；也不把这些新角色回填到旧80个模板。

实际3728次guard观测compress=True、sparse=False、MLA backend MRO，足以减少本观察域内通用PA分支的无关工作；不宣称所有输入/分支穷尽。另核对本机msprof动态入口并在空闲服务上快速试过PID列表与单PID，均被CLI拒绝为no valid pid values，未形成采集数据，服务保持200。下一步只查其PID namespace/动态注册入口；不能因为help存在就声称已采集，也不需要因此停止现有profile分析。

用户已明确授权本容器的推理、框架与探针进程操作，本范围后续不重复请求许可。当前仍为PROGRESS：35+7候选、26模块358行的穷尽性、G00—G14、原9/1352、849PMU、16phase、3136归属、copy/control、数值/读者/物理容量与最低MFU均保留未闭合。没有新增独立性证明、历史native回填或整模块完成声明。
'''
jobs=[
 {'kind':'profile','status':'error','description':'Initial owned diagnostic load failed because early RLIMIT_FSIZE blocked Engram shared backing;original service restored;failure preserved.',
  'artifacts':[O/'shared-builtin-diagnostic-load-failure-v164.json',O/'shared-builtin-root-cause-contexts-v164.json']},
 {'kind':'profile','status':'pass','correctness':'pass','description':'Same full171 fixed-case reference and corrected shared16startup+32formal profiles complete;8response texts exactly match reference;all16worker footers and originalservice health200.',
  'artifacts':[O/'shared-builtin-window-completion-v164b.json',O/'new171-full-case-baseline-reference-v164.json',O/'shared-builtin-correction-submission-v164b.json']},
 {'kind':'review','status':'pass','description':'CPU offline vendor export and all32database ranking/target-MoE focus;BOOL24 sample cap filled;per-execution PMU pairing distinguishes repeated template IDs;not production latency/physical ceiling.',
  'artifacts':[HERE/'ALL32-RANKING.json',HERE/'MAJOR-OP-FOCUS-v2.json',HERE/'major-op-summary.json',HERE/'zero24-complete-and-source.json',HERE/'derivation-receipt.json']},
 {'kind':'profile','status':'error','description':'Two2s idle dynamic-attach entry checks rejected current container PID list and singlePID before data acquisition;no model request;service remained200.',
  'artifacts':[HERE/'dynamic-idle-result.json',HERE/'dynamic-idle-single-result.json',HERE/'msprof-installed-help.json']}]
persist_delta(version=165,previous=163,prefix='shared171-profile',changes=[
 'Completed actual full171 fixed-case shared cohort and restored service;all8texts equal.',
 'All32vendor DBs ranked288846compute executions;main current171 MoE boundary linked with task/shape/CPUstack/PMU evidence.',
 'Repaired only24BOOL24 sampling gaps;classified real request,dummy andCPU roles without historical promotions.',
 'Resolved repeated-template PMU record assignment in vendor timing scope;preserved failed initialization and dynamic PID checks.',
 'Prioritize shared evidence at major operator boundary;keep all original whole-scope gaps and broad container-process authorization.'],
 report=report,action=action,remaining_delta='G165:New171 full shared profile and actual current MoE boundary available;low-overhead production-timing validation,MoE internal stage/receiver counts,all-reader/native template mappings and physical resource upper remain open. All original broad domains retained.',
 source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=[Path(__file__),*HERE.glob('*.py')],obligation_reconciliations=recs,
 remote_status='Full171 cohort and all32offline DB analyses terminal;own original service restoredhealth200. Two idle dynamic checks rejected before acquisition;no residual profiler/model job from these checks.')
cp=read(O/'REVIEW-CHECKPOINT-v165.json');manifest=read(O/'checkpoint-manifest-v165.json')
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':165,'status':'PROFILE_EVIDENCE_AVAILABLE_NEXT_OPERATOR_ANALYSIS','next_action':action,'work_queue':str(O/'coverage-work-queue-v165.json'),'overall_complete':False})
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':165,'status':'TERMINAL_RESTORED_AND_ANALYZED','receipt':str(O/'shared-builtin-window-completion-v164b.json'),'offline':str(HERE/'STATUS.json'),'overall_complete':False})
post={'status':'PASS','checkpoint':165,'checks':checks,'registered_and_task_validated':True,'manifest_artifacts':len(manifest['artifacts']),
 'predecessor_pins_verified_before_shared_writes':True,'complete_original_scope_preserved':True,'overall_complete':False}
write(O/'shared171-profile-postcheck-result-v165.json',post);write(N/'CURRENT-POSTCHECK.json',post)
print(json.dumps(post,ensure_ascii=False))
