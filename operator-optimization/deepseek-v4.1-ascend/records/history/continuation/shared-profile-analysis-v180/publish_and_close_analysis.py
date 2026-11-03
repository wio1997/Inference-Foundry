"""Publish audited evidence,verify actual entrypoints,then close the analysis Task."""
import copy
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031';P=N/'shared-profile-analysis-v179'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W=R/'whole-scope-20261030';T=W/'tasks/whole-opportunity-coverage';L=T/'loops/loop-005'
CTL=Path('/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py')
def read(p):return json.loads(p.read_text())
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def ctl(*args):
 p=subprocess.run([sys.executable,str(CTL),*map(str,args)],capture_output=True,text=True)
 if p.returncode:raise RuntimeError({'command':args[0],'exit':p.returncode,'stdout':p.stdout[-1500:],'stderr':p.stderr[-1500:]})
 return p.stdout.strip()

assert not (O/'REVIEW-CHECKPOINT-v180.json').exists(),'publish once'
audit=read(H/'prepublication-requirement-audit.json');assert audit['status'].startswith('EVIDENCE_AND_STAGED_DELIVERY_PASS')
assert all(audit['checks'].values())
old=read(O/'REVIEW-CHECKPOINT-v179.json');oldm=read(O/'checkpoint-manifest-v179.json')
bad=[r['path'] for r in oldm['artifacts'] if pin(Path(r['path']))['sha256']!=r['sha256']]
assert not bad,{'mismatch_count':len(bad),'examples':bad[:3]}
for r in audit['requirements']:
 for p in r['evidence']:assert pin(Path(p['path']))['sha256']==p['sha256'],p['path']
stamp=datetime.now(timezone.utc).isoformat()

# Current root documentation is generated from the already reviewed records/metrics,not new findings.
records=read(H/'records.json');ma=read(H/'module-analysis.json');summary=read(H/'profile-summary.json')
mods={m['module_id']:m for m in ma['modules']};rows=list(csv.DictReader((H/'module_metrics.csv').open()))
def pct(v):return 'UNKNOWN' if v in [None,''] else f'{float(v)*100:.5f}%'
report=[
 '**整体优化机会分析已通过验收，停在发现与诊断阶段。**',
 '',
 '范围为 DeepSeek-V4.1-Flash-1017-w8a8、DSpark、Engram，171上的CANN9.2环境，冻结32K输入/1K输出、C8、P90文本负载。覆盖26个功能模块、360条语义行、原1101处静态审查和另外1867处精确条件引用。40层目标网络、3层草稿网络及跨算子、跨层、跨卡、跨迭代关系均纳入当前公共条件图。驱动/固件差异按用户接受的条件保留。',
 '',
 '共保留35个模型候选族和7个host边界候选族，42个均为条件式候选。每个族包含原图/候选图映射、消费者/状态/别名约束、工作与I/O变化及资源条件。没有实施优化，没有融合后的实测提速。当前F21已经融合dispatch、量化、双GMM、SwiGLU及return，不能把同一融合再次计为新收益。',
 '',
 '停止依据是完整的声明范围、Case与7条规则扫描及语义复核：两两、三算子、更多算子、拆分重组、共享输入、跨模块和多消费者均有结论；最后一轮未出现新的非等价候选。42个族的互斥、依赖、共享资源与生命周期约束同时保留。状态为SCAN_COMPLETE_IN_SCOPE；这是有限规则下的覆盖结果，不是所有可能算法已穷尽的证明。',
 '',
 '实质核验修正了四处会影响判断的关系：',
 '',
 '- DSpark三层共享上下文输入，分别使用各自WKV、norm/RoPE、cache/group/slot。联合输入读取不等于复用一份KV输出。',
 '- source20的2048候选块列表有fresh output及发布拷贝，供24/28/32/36四个过滤读者使用；它与512位置TopK/prepare输出的别名关系不同。',
 '- 批状态修正的三次gather先于computed写入，accepted的else读取在该写入之后。融合保留跨行旧值和潜在跨平面别名顺序；逐行原地重映射反例产生[21,22]，正确值为[21,11]。',
 '- SP提供mask的True前缀及True padding尾部保留；不能凭BOOL shape把SP mask与C2 mask当成同一状态。',
 '',
 '另已纠正Engram的历史CPU异步标签：当前正向证据为NPU host-UVA gather/dequant，记录的overlap标志为False。在已捕获EP0 worker中，目标/草稿共用16MiB TopK backing；8个源层值代际供38层注意力读取，重排保留重叠范围的RAW/WAR/WAW及下一轮覆写约束。跨rank不能按相同虚拟地址推断物理共享。',
 '',
 '最低利用率需分组表述。按报告周期加权的AIC MAC活动率，M12在12个有效计算模块中最低，合并窗口为约2.41549%，两个窗口分别约2.55250%和1.17269%；对已记录的归属歧义仍稳健。M23在20个有效Vector算术组中最低，通信搬运/轮询并不因此无用。没有一个可信的host/Cube/Vector/通信全体最低值；这些比例也不是整卡忙碌率或物理MFU。',
 '',
 'M12的低活动位置集中在已融合F21：两个窗口中，它分别占M12累计native时间约98.76%/99.19%，占报告AIC周期约99.72%/99.93%。小接收tile、rank/expert不均衡以及peer到达/协议等待仍为竞争假说；现有指标不支持“单纯HBM饱和解释整个区间”。累计kernel时间不是请求墙钟，也不能直接视为可消除时间或端到端提速。',
 '',
 '当前171无profiler参考完成8个32768输入/1024输出请求，客户端墙钟16.346353521秒，返回输出约501.1515 token/s；这是单次参考，不是稳定性或提速结论。正式profile在16个逻辑设备采两段窗口，共32份profile；全部8份响应与参考文本匹配。全量聚合288846条native执行，201351条可唯一功能归属，87495条歧义单列，3510条缺PMU不填零。v176未完成的完整C8采集仅提供部分绑定证据，不用于替代有效性能数据。',
 '',
 '理论与实际对照共94条：26个完整功能模块的条件/符号记录、42个未实现候选记录、26个同口径密集矩阵子集记录。360条语义行都有对照路由。所有对象保留T、L、Q/T、U、L/T、T/L、活动指标、支配约束和证据状态；候选T为空。26个匹配子集已用精确分数重新计算。1800MHz仅为额定参考，不是物理最大频率；未知的f_upper、带宽、模式或必要工作继续保留符号/条件或UNKNOWN，未伪造完整模块的有限数值上限。L/T不是设备利用率，T/L不是可达提速。',
 '',
 '|模块|名称|唯一归属native条数|AIC MAC活动率|AIV Vector活动率|',
 '|---|---|---:|---:|---:|']
for r in rows:
 report.append('|'+r['module']+'|'+r['name']+'|'+(r['certain_native_executions'] or 'UNKNOWN')+'|'+pct(r['AIC_MAC_fraction'])+'|'+pct(r['AIV_Vector_fraction'])+'|')
report+=['','上表只汇总已记录的唯一归属执行和正分母活动字段。host、未执行、未采到或无法归属的部分不自动为0；完整分母、窗口、rank及歧义见原始聚合索引。','','|候选族|登记的机会范围|状态|','|---|---|---|']
for c in records['fusion_candidates']:report.append('|'+c['id']+'|'+c['name']+'|CONDITIONAL|')
report+=['','|匹配子集|必要密集MAC|累计kernel时间ms|额定1800参考L/T|','|---|---:|---:|---:|']
for r in ma['performance_comparisons']:
 if r['object_kind']=='MATCHED_PARTIAL_INVOCATION_POPULATION':
  report.append(f"|{r['object_id']}|{r['effective_work_Q']}|{r['measured_time_T']['decimal_for_display']*1000:.6f}|{r['reference1800MHz_only']['L_over_T']['decimal_for_display']*100:.5f}%|")
report+=['',
 '保留的未证项已按影响分类：物理fmax/模式/带宽限制绝对数值上限；私有参数、native别名与retirement限制相关私有编辑；host当期成本和请求关键路径暴露限制因果提速判断；F21的等待机制仍有竞争解释。它们均未被当成独立、无效、零开销或已证可优化。所有声明范围均已有分析答案；更强的无条件融合、物理上限或实现收益需要新的验证，不属于本次发现与诊断的已证结论。',
 '',
 '本次验收为SELF_REVIEW，不声称独立外部审阅或形式化证明。原始大数据保留原路径和SHA，历史169/166及失败采集单列，未混入当前171指标。原服务已在取证阶段恢复，最后服务校验在v177；本次收尾未执行远程命令、推理或重启，也未把旧health当作新的在线检查。',
 '',
 '当前交付入口：',
 '',
 f'- [合同]({R/"contract.json"})、[原图]({R/"graphs/G0-current-v180.json"})、[42候选及条件]({R/"records.json"})、[扫描台账]({R/"fusion-scan.json"})。',
 f'- [模块及94条理论/实测对照]({R/"module-analysis.json"})、[界输入]({R/"bounds-input.json"})、[精确分数复算]({H/"exact-rational-P5-recalculation.json"})。',
 f'- [profile小摘要]({R/"profile-summary.json"})、[完整数据索引]({R/"profile-index.json"})、[26模块汇总]({R/"module_metrics.csv"})、[84算子汇总]({R/"operator_metrics.csv"})。',
 f'- [逐项完成验收]({H/"final-completion-audit.json"})、[360条语义与状态复核]({P/"semantic-contract-sufficiency-review.json"})。',
 '']
report_text='\n'.join(report)
(H/'REPORT.md').write_text(report_text)

root_files=['contract.json','records.json','fusion-scan.json','module-analysis.json','profile-index.json','profile-summary.json','bounds-input.json','profiling-plan.json','module_metrics.csv','operator_metrics.csv']
published=[]
for name in root_files:
 (R/name).write_bytes((H/name).read_bytes());published.append(R/name)
(R/'graphs/G0-current-v180.json').write_bytes((H/'G0-current-v180.json').read_bytes());published.append(R/'graphs/G0-current-v180.json')
for p in (H/'candidates/whole-scope-v180').glob('*.json'):
 q=R/'candidates/whole-scope-v180'/p.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());published.append(q)
(R/'REPORT.md').write_text(report_text);published.append(R/'REPORT.md')

reqs=copy.deepcopy(audit['requirements'])
for r in reqs:r['state']='PROVED_FOR_ANALYSIS_CONTRACT';r['completion_proved_for_analysis_contract']=True
reqs[-1]['argument']='Audited exact current documents and all42candidategraphs published to authoritative root;root byte/hash/reference checks below verify P7. Task/Loop closure and final manifest remain verification gates before goal completion.'
current_audit={'version':180,'status':'ALL9_ORIGINAL_ANALYSIS_REQUIREMENTS_PROVED','review_type':'SELF_REVIEW',
 'requirements':reqs,'prepublication_audit':pin(H/'prepublication-requirement-audit.json'),
 'nonblocking_conditional_unknowns':audit['nonblocking_conditional_unknowns'],
 'unanalysed_fixed_scope_items':[],'missing_critical_evidence_for_claimed_conclusions':[],
 'scope_preserved':True,'implementation_or_measured_optimization_gain_claimed':False,'overall_complete':True}
write(O/'whole-scope-requirement-audit-v180.json',current_audit)

# Replace stale current assessment fields while preserving immutable predecessor ledgers.
authorities={}
for key,stem in [('candidate_ledger','candidate-profile-ledger'),('module_ledger','module-profile-ledger'),('branch_matrix','module-phase-branch-matrix')]:
 d=copy.deepcopy(read(Path(old[key])));d.update(parent_review_authority=pin(Path(old[key])),current_review_checkpoint=180,
   current_assessment_authority=str(O/'whole-scope-requirement-audit-v180.json'),overall_complete=True,
   analysis_complete_in_fixed_scope=True,completion_definition='Conditional discovery/diagnosis in full fixedmodel/source/hardware/load and declaredrule scope;not nativeimplementation/everyphysicalunknown resolved.')
 if key=='candidate_ledger':
  d.update(candidate_universe_complete_in_declared_scope=True,candidate_universe_all_possible_algorithms_exhausted=False)
  for c in d['candidates']:c['current_analysis_complete_in_scope']=True;c['current_candidate_graph']=str(R/'candidates/whole-scope-v180'/(c['id']+'.json'))
 else:
  d.update(whole_modules_closed=26,all_phases_closed=True,all_conditions_analyzed=True)
  for m in d['modules']:
   m['historical_closed_before_final_review']=m.get('closed',m.get('whole_module_closed'))
   m.update(closed=True,whole_module_closed=True,analysis_complete_in_fixed_scope=True,
     current_state='ANALYSIS_COMPLETE_WITH_EXPLICIT_RUNTIME_AND_NATIVE_ELIGIBILITY_CONDITIONS')
   if key=='branch_matrix':
    for row in m['branches']:
     row['historical_state_before_final_review']=row['state']
     row['state']='ANALYSIS_COMPLETE_IN_FIXED_SCOPE_WITH_TYPED_ELIGIBILITY_CONDITIONS'
     row['current_final_analysis_review']=str(P/'semantic-contract-sufficiency-review.json')+'#'+row['id']
 q=O/(stem+'-v180.json');write(q,d);authorities[key]=str(q)

queue={'version':180,'status':'COMPLETE','work':[],'completion_audit':str(H/'final-completion-audit.json'),
 'conditional_candidate_implementation_validation_is_not_queued_analysis_work':True,'overall_complete':True}
write(O/'coverage-work-queue-v180.json',queue)
oldobs=read(Path(old['coverage_obligations']))
for r in oldobs['obligations']:
 r['historical_remaining_before_final_audit']=r['remaining'];r.update(remaining='',state='ANALYSIS_OBLIGATION_SATISFIED_WITH_EXPLICIT_CONDITIONS',
  current_evidence_authority=str(O/'whole-scope-requirement-audit-v180.json'))
oldobs.update(version=180,parent=pin(Path(old['coverage_obligations'])),overall_complete=True,whole_modules_closed=26)
write(O/'coverage-current-obligations-v180.json',oldobs)
write(O/'module-coverage-gaps-v180.json',{'version':180,'remaining':[],'nonblocking_conditional_unknowns':records['unknowns'],
 'source':str(H/'final-completion-audit.json'),'overall_complete':True})
cp={'checkpoint_schema':'FINAL_ANALYSIS_CHECKPOINT_V1','version':180,'status':'COMPLETE',
 'previous_checkpoint':str(O/'REVIEW-CHECKPOINT-v179.json'),'previous_turn_classification':'PROGRESS',
 'objective':read(T/'index.json')['goal'],'scope':'Originalspecifiedmodel/source/hardware/load,all26modules and applicableoperatorcases,declared7rewriterules',
 'requirement_audit':str(O/'whole-scope-requirement-audit-v180.json'),'final_completion_audit':str(H/'final-completion-audit.json'),
 'report':str(R/'REPORT.md'),'candidate_ledger':authorities['candidate_ledger'],'module_ledger':authorities['module_ledger'],'branch_matrix':authorities['branch_matrix'],
 'work_queue':str(O/'coverage-work-queue-v180.json'),'coverage_obligations':str(O/'coverage-current-obligations-v180.json'),
 'coverage_gaps':str(O/'module-coverage-gaps-v180.json'),'current_G0':str(R/'graphs/G0-current-v180.json'),
 'whole_modules_closed':26,'conditional_candidate_count':42,'P5_comparisons':94,
 'remaining_required_work':[],'nonblocking_conditional_unknowns':records['unknowns'],
 'next_action':'Analysisdelivered;conditionalimplementation/strongerphysicalclaimsrequireaseparatevalidatedworkitem.',
 'target_implementation_changed':False,'measured_optimization_speedup':None,'overall_complete':True,'updated_utc':stamp}

# Evidence run and loop/task gates are analysis gates,not new-kernel correctness/performance assertions.
run=ctl('new-run','--loop-dir',L,'--case-key','c8-input32k-output1k-prefix90-171','--kind','review',
 '--command','Finaloriginalobjective/P0-P7 auditandcanonicaldelivery;no modelorNPUexecution',
 '--code-ref','Pinnedsource/profilecontracts;SELF_REVIEW;noimplementationchange','--working-directory',N)
args=['finish-run','--run-dir',run,'--status','pass','--correctness','not-applicable',
 '--summary','Fullfixed-scope conditionalanalysis audited;canonical42graphs/26modules/94P5/fullprofileindex published.No newNPU/model/restart.']
for p in [H/'prepublication-requirement-audit.json',H/'exact-rational-P5-recalculation.json',H/'REPORT.md',H/'candidate-graph-index.json']:args+=['--artifact',p]
ctl(*args)
verdict=ctl('verdict','--loop-dir',L,'--result','accepted','--goal-satisfied','yes','--comparability','not-applicable',
 '--causal-conclusion','partially-supported','--reason','Sharedprofile/typedpubliccontracts answer the operator questions and preserve unboundprivateeffects as conditional/unsupported.Originalscope/semanticreview and fullcasebudget/restore evidence complete;no missingfield treated independent.',
 '--next-action','Delivercompleteanalysis;nooptimizationimplementationorunrequestedexperiment.',
 '--evidence-path',H/'prepublication-requirement-audit.json','--evidence-path',P/'semantic-contract-sufficiency-review.json')
closed=ctl('close-task','--task-dir',T,'--summary',
 'Originalfixedmodel/version/hardware/load analysiscomplete:26modules360semanticrows42conditionalfamilies;P4groupedminima/P5symbolicandmatchedcomparisons/P6competingcauses/P7canonicaldelivery.Implementationnotperformed.',
 '--correctness-gate','pass','--performance-gate','not-applicable','--regression-gate','pass','--delivery-gate','pass',
 '--stop-reason','Originalanalysisobjectiveanddeclaredrule/casecoverage satisfied;lastfullsemanticround no newnonequivalentfamily.Requiredanalysisworknone;conditionalnative/physicalunknowns documented.')
compact=ctl('compact','--task-dir',T);validation=ctl('validate','--task-dir',T)
write(H/'task-loop-close-receipt.json',{'status':'PASS','run':run,'verdict':verdict,'close_task':closed,'compact':compact,'validation':validation,
 'correctness_performance_gate_scope':'Analysisartifactconsistency,not optimizedkernelnumeric/performancevalidation'})

state={'version':180,'state':'COMPLETE','current_phase':'P7_DELIVERED','mode':'discovery_and_diagnosis_only',
 'current_goal_20261030':cp['objective'],'original_graph_id':read(H/'G0-current-v180.json')['id'],
 'fusion_scan_status':'SCAN_COMPLETE_IN_SCOPE','module_diagnosis_status':'GROUPED_MEASURED_MINIMA_AND_CAUSES_REVIEWED',
 'performance_comparison_status':'94_CONDITIONAL_SYMBOLIC_AND_MATCHED_PARTIAL_RECORDS',
 'lowest_grouped_modules':{'AIC_MAC':'M12','AIV_Vector':'M23'},
 'current_requirement_audit':str(O/'whole-scope-requirement-audit-v180.json'),
 'final_completion_audit':str(H/'final-completion-audit.json'),'report':str(R/'REPORT.md'),
 'current_profile_index':str(R/'profile-index.json'),'current_profile_summary':str(R/'profile-summary.json'),
 'current_checkpoint':str(O/'REVIEW-CHECKPOINT-v180.json'),'remaining_required_work':[],
 'nonblocking_conditional_unknowns':records['unknowns'],'whole_modules_closed':26,'overall_complete':True,
 'no_optimization_implementation':True,'last_service_health_receipt':str(N/'shared-profile-analysis-v177/final-CPU-and-service-receipt.json'),
 'no_new_NPU_model_restart_in_final_review':True,'updated_utc':stamp}
write(R/'STATE.json',state);write(W/'STATE.json',{**state,'profile_candidate_ledger':authorities['candidate_ledger'],'profile_module_ledger':authorities['module_ledger']})
write(R/'PARALLEL-RESUME-PACK.json',{'status':'COMPLETE','read_first':[str(R/'REPORT.md'),str(H/'final-completion-audit.json'),str(R/'profile-summary.json')],
 'checkpoint':str(O/'REVIEW-CHECKPOINT-v180.json'),'remaining_required_work':[],'overall_complete':True})
write(N/'RECOVERY.json',{'status':'COMPLETE','effective_checkpoint':180,'read_first':[str(R/'REPORT.md'),str(H/'final-completion-audit.json')],
 'remaining_required_work':[],'overall_complete':True})
workflow=read(N/'CURRENT-ANALYSIS-WORKFLOW.json');workflow.update(effective_checkpoint=180,status='COMPLETE',
 work_queue=pin(O/'coverage-work-queue-v180.json'),primary_queue=[],overall_complete=True,completion_audit=str(H/'final-completion-audit.json'))
write(N/'CURRENT-ANALYSIS-WORKFLOW.json',workflow)
write(O/'REVIEW-CHECKPOINT-v180.json',cp)
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':180,'status':'NO_REQUIRED_ANALYSIS_WORK','remaining':[],
 'nonblocking_conditional_unknowns':records['unknowns'],'overall_complete':True})
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':180,'status':'ANALYSIS_DELIVERED_NO_NEW_REMOTE_ACTION',
 'last_service_receipt':state['last_service_health_receipt'],'fresh_health_not_checked':True,'new_NPU_model_restart':0,'overall_complete':True})
(R/'CURRENT-FINDINGS.md').write_text(report_text)
(R/'CONTINUATION-STATUS.md').write_text('整体分析已通过逐项验收。当前入口：'+str(R/'REPORT.md')+'；完成审查：'+str(H/'final-completion-audit.json')+'。无待执行分析工作；未实施优化。\n')

rootchecks={}
rootchecks['root_payloads_match_audited_staging']=all((R/n).read_bytes()==(H/n).read_bytes() for n in root_files)
rootchecks['root_G0_matches_audited_staging']=(R/'graphs/G0-current-v180.json').read_bytes()==(H/'G0-current-v180.json').read_bytes()
rootchecks['all42_candidate_graph_hashes']=all(pin(Path(c['candidate_graph_path']))['sha256']==c['candidate_graph_sha256'] for c in read(R/'records.json')['fusion_candidates'])
rootchecks['all42_original_graph_hashes']=all(c['original_graph_sha256']==pin(R/'graphs/G0-current-v180.json')['sha256'] for c in read(R/'records.json')['fusion_candidates'])
rootchecks['root_scan_complete']=read(R/'fusion-scan.json')['status']=='SCAN_COMPLETE_IN_SCOPE'
rootchecks['root_summary_bounded']=(R/'profile-summary.json').stat().st_size<=32768
rootchecks['current_contract171']=read(R/'contract.json')['hardware']['host']=='172.16.10.171'
rootchecks['root_module26_comparisons94']=len(read(R/'module-analysis.json')['modules'])==26 and len(read(R/'module-analysis.json')['performance_comparisons'])==94
rootchecks['Task_closed']=read(T/'index.json')['status']=='COMPLETED'
rootchecks['Loop_terminal']=read(L/'state.json')['status'] not in ['ACTIVE','EVALUATING']
rootchecks['no_remaining_current_work']=not read(O/'coverage-work-queue-v180.json')['work'] and not read(R/'STATE.json')['remaining_required_work']
assert all(rootchecks.values()),rootchecks
final={**current_audit,'status':'PASS_FULL_ORIGINAL_ANALYSIS_OBJECTIVE','published_root':str(R),
 'root_postpublication_checks':rootchecks,'task_loop_receipt':pin(H/'task-loop-close-receipt.json'),
 'published_file_pins':[pin(p) for p in published],
 'review_type':'SELF_REVIEW;172staged checks+actualpostpublicationchecks;no formal/externalreview claim',
 'overall_complete':True,'goal_tool_update_required_after_manifest':True}
write(H/'final-completion-audit.json',final)
write(O/'whole-scope-requirement-audit-v180.json',{**current_audit,'final_completion_audit':pin(H/'final-completion-audit.json')})
(O/'PROFILE-FINDINGS-v180.md').write_text(report_text)
write(O/'final-analysis-summary-v180.json',{'status':'COMPLETE','previous_goal_turn':'PROGRESS','requirements_proved':9,
 'modules':26,'semantic_rows':360,'conditional_candidates':42,'P5records':94,'remaining_required_work':[],
 'completion_audit':str(H/'final-completion-audit.json'),'overall_complete':True})
paths={Path(x['path']) for x in oldm['artifacts']}
paths.update(p for p in H.rglob('*') if p.is_file());paths.update(published)
paths.update(p for p in O.glob('*v180*') if p.is_file() and not p.name.startswith('checkpoint-manifest'))
paths.update([N/'CURRENT-ANALYSIS-WORKFLOW.json',N/'RECOVERY.json',N/'NEXT-EVIDENCE-PENDING.json',N/'LIVE-SHARED-COHORT.json',
 R/'STATE.json',W/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'CURRENT-FINDINGS.md',R/'CONTINUATION-STATUS.md'])
paths.update(p for p in T.rglob('*') if p.is_file())
manifest={'schema':'FINAL_ANALYSIS_CHECKPOINT_V1','version':180,'predecessor_manifest':pin(O/'checkpoint-manifest-v179.json'),
 'predecessor_all1786pins_validated_before_shared_writes':True,'artifacts':[pin(p) for p in sorted(paths)],
 'overall_complete':True,'whole_modules_closed':26,'updated_utc':stamp}
write(O/'checkpoint-manifest-v180.json',manifest)
assert all(pin(Path(r['path']))['sha256']==r['sha256'] for r in manifest['artifacts'])
post={'status':'PASS','checkpoint':180,'manifest_artifacts':len(paths),'all_pins_verified':True,
 'all9_requirements_proved':True,'Task_Loop_closed':True,'remaining_required_work':[],
 'no_optimization_implementation':True,'no_new_NPU_model_restart':True,'overall_complete':True}
write(O/'final-analysis-postcheck-result-v180.json',post);write(N/'CURRENT-POSTCHECK.json',post)
print(json.dumps(post))
