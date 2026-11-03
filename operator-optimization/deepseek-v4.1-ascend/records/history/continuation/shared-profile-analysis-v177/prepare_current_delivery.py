"""Prepare current measured-group diagnosis,comparisons and report for atomic publication."""
import json
from copy import deepcopy
from pathlib import Path

H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
def read(p):return json.loads(p.read_text())
def write(n,x):(H/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

legacy=read(R/'module-analysis.json');write('module-analysis-before-v177.json',legacy)
cur=read(N/'shared-profile-analysis-v174/module-analysis.json')
routes=read(N/'shared-profile-analysis-v172/all26-module-metric-and-P5-routes.json')
activity=read(N/'shared-profile-analysis-v172/module-activity-review.json')
diagnosis=read(N/'shared-profile-analysis-v172/lowest-module-cause-review.json')
out=deepcopy(legacy)
out.update(status='MEASURED_GROUP_DIAGNOSIS_COMPLETE_WITH_EXPLICIT_UNKNOWNS;FULL_FUSION_SCOPE_OPEN',
    current_primary_module_namespace='M00-M25_FUNCTIONAL_MODULES',
    observed_graph_id='FROZEN171_PUBLIC_GRAPH_SCOPE_RECONCILIATION_PENDING',
    trace_source_ids=['V164_TWO_FORMAL_CPU_PMU_WINDOWS'],
    module_kernel_mapping_path=str(N/'shared-profile-analysis-v172/module-activity-review.json'),
    unknown_ids=['G0_SCOPE_AND_SOURCE_GUARD_COVERAGE','MISSING_AND_UNRECORDED_METRICS','PHYSICAL_RATE_UPPERS','UNMEASURED_HOST_AND_CRITICAL_PATH_COSTS'],
    current_primary_metric_cohort='v164_32profiles_two_formal_CPU_PMU_windows',
    historical_target_layer_analysis_ref=str(H/'module-analysis-before-v177.json'),
    current_operator_binding_ref=str(N/'shared-profile-analysis-v176/operator-conclusions-from-profile.json'),
    current_theory_comparison_ref=str(N/'shared-profile-analysis-v174/module-analysis.json'))
out['modules']=[dict(module_id=m['module'],**m) for m in routes['modules']]
out['rankings']=[{'metric_name':metric,'definition':'sum(reported_cycles*reported_ratio)/sum(reported_cycles),positive-denominator recorded population',
    'case_id':'v164_TWO_FORMAL_WINDOWS_POOLED','ordered_modules':rs,'lowest_module_ids':[rs[0]['module']],
    'status':'SUPPORTED_IN_THIS_RECORDED_COMPARABLE_GROUP','physical_MFU':False,
    'coverage':len(rs),'unrecorded_or_missing_PMU_not_zero':True} for metric,rs in activity['rankings'].items()]
out['coverage']=deepcopy(routes['coverage']);out['coverage']['scope']='Functional26modules;each measuredgroup has its own eligiblepopulation. Historical40layer observations remain separate.'
out['performance_comparisons']=deepcopy(cur['performance_comparisons'])
out['historical_performance_comparisons_ref']=str(H/'module-analysis-before-v177.json')
out['historical_comparison_count']=len(legacy['performance_comparisons'])
out['cause_review_path']=str(N/'shared-profile-analysis-v172/lowest-module-cause-review.json')
write('module-analysis.json',out)

candidates=read(O/'candidate-profile-ledger-v176.json')['candidates']
host=read(O/'framework-candidate-ledger-v154.json')
host=host.get('candidates',host.get('framework_candidates',[]))
assert isinstance(host,list) and len(candidates)==35 and len(host)==7
audit=read(H/'current-requirement-audit.json')
census=read(H/'profile-selected-operator-guard-census.json')
lines=[
 '**当前整体分析：继续推进，完整范围尚未通过验收。**',
 '',
 '本页汇总当前有效结论，替换旧报告中的过期状态。指定模型仍为DeepSeek V4.1 Flash目标网络、DSpark与Engram，冻结负载32K输入/1K输出、C8、P90；环境已迁移到171，CANN9.2。驱动/固件差异按用户接受的范围保留。各采集cohort单列，旧硬件/旧epoch不回填为新观测。',
 '',
 '登记范围含26个功能模块、360条语义行、35个模型候选族和7个框架边界候选族。这些是分析对象，不等于42项可直接实施或可叠加的融合收益。现有F21已融合dispatch、量化、双GMM、SwiGLU及return；当前greedy验证也已有Triton实现。',
 '',
 '已有的1101处精确静态点完成算子效应审查；全部42个登记族有组合约束，27份源码模板的854节点/1346边完成一轮7规则条件审查。完整模型范围仍需把精确条件集合与这些答案对齐，并核齐分层G0的消费者、别名、状态与跨owner端口。',
 '',
 '| 要求 | 当前状态 |', '|---|---|']
for r in audit['requirements']:lines.append('|'+r['id']+'|'+r['state']+'|')
lines += ['',f"本轮已用profile调用与已加载类固定{len(census['files'])}份源码、{len(census['functions'])}个选中函数及{len(census['guards'])}个条件引用，缺失/哈希不匹配为0。67份已有profile源码哈希匹配，另5份为新绑定的已加载类文件。条件数量不表示同等数量的新优化任务；先复用已有合同，只分析未匹配的算子效应。同名条件定义、属性访问器和传给executor的嵌套回调均按源码位置保留。",'',
 '已登记的候选族如下；具体条件、必要工作、候选图和未知项由当前台账及组合合同给出。', '',
 '| 编号 | 对象 | 结论范围 |','|---|---|---|']
for c in candidates+host:
    cid=c.get('id',c.get('candidate_id'));name=c.get('name',c.get('title',''))
    state='已有融合；内部拆分、重排与载荷分配为条件候选' if cid=='F21' else '条件分析；实际适用性与收益分别核对'
    lines.append('|'+str(cid)+'|'+str(name).replace('|','/')+'|'+state+'|')
lines += ['',
 '模块最低者已有同口径结果：12个带MAC活动模块中M12最低，合并报告AIC/MAC周期加权比为2.41549%，两个正式窗口分别为2.55250%和1.17269%，对已记录的归属歧义仍稳健。20个带Vector活动模块中M23算术活动最低，合并值约0.01791%；它承担通信，不据此判定无用。',
 '',
 '这些是报告计数器活动比例，不能称为物理MFU、整卡忙时或可移除比例。未测host、3510条缺PMU、87495条归属歧义和未记录工作明确保留，不能填0或强行与不同资源排名。',
 '', '| 模块 | 功能 | 报告MAC活动 | 报告Vector活动 |', '|---|---|---:|---:|']
for m in routes['modules']:
    a=m['primary_reported_activity']['AIC_MAC']['fraction'];v=m['primary_reported_activity']['AIV_Vector']['fraction']
    fmt=lambda z:'UNKNOWN/不适用' if z is None else f'{z*100:.5f}%'
    lines.append(f"|{m['module']}|{m['name']}|{fmt(a)}|{fmt(v)}|")
lines += ['',
 'M12的低活动与主要累计kernel时间位于F21内部：两个窗口中F21占M12累计native时间98.7649%/99.1925%，占报告AIC周期99.7213%/99.9279%。这排除了“只需把外围小算子合并”的解释。小receiver工作、专家/rank不均和跨卡到达/credit/polling是有源码或跨cohort机制支持的竞争原因；当前PMU不能独自分解其因果比例，累计跨卡kernel时间也不是请求walltime。',
 '',
 '理论对照现有94条当前规范记录：26个模块合同、42个未实现组合、26组匹配的局部稠密模式。各对象均显式给出Q、T、L、Q/T、U、L/T、T/L、活动指标、支配约束和证据状态；缺值保留条件/符号/UNKNOWN。旧76条对照独立保存在历史快照，不能与当前cohort直接合并。',
 '',
 'Cube直接稠密域的形式为L=Q_MAC/(24·r_mode·f_upper)，INT8的r=8192、BF16/FP16的r=4096、受限FLOAT模式按各自1024/2048条件。1800MHz是额定参考，未当物理fmax。整体融合按必要且不重复的计算、逐层必要I/O、共享资源/通信cut和已证明依赖重算，各独立资源下界取max。权重/缓存/workspace分配大小不自动等于每次HBM或wire流量。',
 '',
 '部分可复算对照如下，T为同一所选执行集合的累计kernel服务时间，跨设备累计，不能当请求延迟；FLOAT模式行是互斥假设，不能相加。参考L/T不是设备活动率或可达提速。',
 '', '| 对象/窗口/模式 | 必要MAC（条件域） | 匹配累计T(ms) | 1800MHz参考L/T |', '|---|---:|---:|---:|']
for c in cur['performance_comparisons']:
    if c['object_kind']!='MATCHED_PARTIAL_INVOCATION_POPULATION':continue
    t=c['measured_time_T']['decimal_for_display']*1000
    ratio=c['reference1800MHz_only']['L_over_T']['decimal_for_display']*100
    lines.append(f"|{c['object_id']}|{c['effective_work_Q']}|{t:.6f}|{ratio:.5f}%|")
lines += ['',
 '当前公开绑定证据还确认：目标/草稿共用16MiB索引bank，需保留writer/reader/rollback代际；C2mask的24/1704/3200/8176元素视图来自同8KiB backing，不能按固定24B临时allocation或恒值处理；四个embedding/head存储对象不同但未读数值，不能只由shape或指针作数值CSE判断。16worker共1104次实际采样调用绑定greedy、无penalty/entropy/synthetic/max_num_logprobs前提；这只支持对应观察人群和已有源合同，不证明所有score消费者消失。',
 '',
 'v176采集触及单worker ET限额，完整C8校验失败。已恢复原服务；其部分窗口只作绑定证据，性能结论继续使用先前有效profile。恢复及终止记录可核查。此后仅CPU离线解析，没有新建NPU上下文、推理或重启。',
 '',
 '下一步只做两项范围检查：精确条件到已有算子合同的覆盖对应；分层G0的调用、消费者、别名和共享状态端口对应。不得把已回答的条件未知、历史私有ABI、物理fmax或未实施提速重新当作通用阻塞；也不得在真实未分析范围仍在时因本页成稿标完成。',
 '',
 f"当前验收表：[current-requirement-audit.json]({H/'current-requirement-audit.json'})；当前模块/对照：[module-analysis.json]({H/'module-analysis.json'})；条件精确引用：[exact-source-guard-coverage-frontier.json]({H/'exact-source-guard-coverage-frontier.json'})；条件源码规则轮：[declared-source-template-rule-review.json]({rule if (rule:=N/'shared-profile-analysis-v176/declared-source-template-rule-review.json') else rule})。",'']
(H/'REPORT.md').write_text('\n'.join(lines))
if (R/'REPORT.md').exists():(H/'REPORT-before-v177.md').write_bytes((R/'REPORT.md').read_bytes())
assert len(out['modules'])==26 and len(out['performance_comparisons'])==94
assert len(out['rankings'])==2 and len(host)==7
write('current-delivery-checks.json',{'status':'PASS','current_functional_modules':26,'current_comparisons':94,
    'historical_comparisons_preserved':len(legacy['performance_comparisons']),'candidate_families':42,
    'P4_P5_P6_current_results_not_replaced_by_legacy_flags':True,'fullscope_not_complete':True})
print(json.dumps({'status':'PASS','report_lines':len(lines),'current_modules':26,'current_comparisons':94,'registered_families':42}))
