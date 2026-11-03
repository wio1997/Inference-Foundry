"""Integrate exact exported graph-submit links and bounded timing associations."""
from pathlib import Path
from statistics import mean
import csv,datetime,hashlib,json
T=Path(__file__).resolve().parents[1]
def read(n):return json.loads((T/n).read_text())
def write(n,v):(T/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(n):return hashlib.sha256((T/n).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
summary=read('graph-submit-candidate-analysis-v2/summary.json')
chains=read('model-execute-wait-chain-v2/summary.json')
calls=read('model-execute-wait-chain-v2/calls.json')
submit=list(csv.DictReader((T/'graph-submit-candidate-analysis-v2/rank_layer0_submit_candidates.csv').open()))
assert summary['candidate_count_distribution']=={'1':99,'0':5}
assert chains['calls_checked']==chains['one_execute_one_wait_count']==chains['also_unique_wait_contains_source_validated_layer_count']==99
assert chains['name_counts']=={'MODEL_EXECUTE':99,'MODEL_WAIT_COMPLETE':99}
tab=[]
for ordinal in [1,6,7]:
 cs=[x for x in calls if x['ordinal']==ordinal];assert len(cs)==16 and all(x['interior'] for x in cs)
 origin=min(x['cann_execute_api'][0][0] for x in cs);D={}
 for dp in ['dp0','dp1']:
  d=[x for x in cs if x['dp']==dp];assert len(d)==8;values=[]
  for x in d:
   r=[y for y in submit if y['rank']==x['rank'] and int(y['target_forward_ordinal'])==ordinal];assert len(r)==1
   exe=next(z for z in x['tasks'] if z[5]=='MODEL_EXECUTE');wait=next(z for z in x['tasks'] if z[5]=='MODEL_WAIT_COMPLETE')
   values.append({'api':x['cann_execute_api'][0][0], 'model_execute':exe[0], 'wait':wait[0], 'layer':int(r[0]['layer_start_ns'])})
  D[dp]={k:mean((x[k]-origin)/1000 for x in values) for k in ['api','model_execute','wait','layer']}
 tab.append({'ordinal':ordinal,**{k+'_dp0_minus_dp1_us':D['dp0'][k]-D['dp1'][k] for k in D['dp0']},
             'interpretation':'Conditional on offset0 rank-forward pairing and acceptable cross-process/device clock error.'})
trace=read('graph-trace-argument-inspection/summary.json');assert trace['events_scanned']>0
doc={'status':'EXPORTED_CONNECTION_CHAIN_AND_CONDITIONAL_DP_TIMING','date':now,
 'captured_layer0_calls':104,'unique_candidate_within_10ms':99,'no_candidate_within_10ms':5,
 'all_unmatched_scope':'First captured forward ordinal0 only',
 'exact_chain':'CANN aclmdlRIExecuteAsync connectionId -> same modelId TASKs MODEL_EXECUTE and MODEL_WAIT_COMPLETE; unique wait task contains source-validated captured layer0 bracket',
 'chain_count':99,'paired_interior_forward_groups':tab,
 'source_snapshots':['graph-submit-candidate-analysis-v2/summary.json','graph-submit-task-link-validation-v2/summary.json','model-execute-wait-chain-v2/summary.json',
                     'graph-trace-argument-inspection/summary.json','host-gap-tail-inspection/summary.json'],
 'limitations':['99/104 are within a chosen10ms candidate window; five initial capture calls are unmatched.',
   'Exporter connection ID and modelId are explicit. Identifying the layer inside the uniquely enclosing MODEL_WAIT_COMPLETE task also uses its recorded time interval.',
   'Offset0 cross-DP forward pairing, host/device clock conversion and clock-error bound are not independently certified.',
   'Similar API and layer-entry skews suggest a pre-layer or submission-timing component, but do not isolate the upstream cause, exact waiting time or removable end-to-end cost.',
   'MODEL_WAIT_COMPLETE is a profiler task name; its full duration cannot be equated with pure calculation or pure waiting.',
   'The trace example reports a connection ID but no explicit request or model handle argument.'],
 'no_device_work':True,'no_target_implementation_modified':True}
write('evidence/graph-submit-diagnosis.json',doc)
records=read('records.json')
if not any(s['id']=='GRAPH_SUBMIT_DIAGNOSIS' for s in records['sources']):
 records['sources'].append({'id':'GRAPH_SUBMIT_DIAGNOSIS','url_or_path':'evidence/graph-submit-diagnosis.json',
  'snapshot_path':'evidence/graph-submit-diagnosis.json','sha256':sha('evidence/graph-submit-diagnosis.json'),'registered_at':now})
cause=next(x for x in records['module_causes'] if x['id']=='C_ARRIVAL_IMBALANCE')
if 'evidence/graph-submit-diagnosis.json' not in cause['supporting_evidence']:cause['supporting_evidence'].append('evidence/graph-submit-diagnosis.json')
cause['missing_evidence']=[x for x in cause['missing_evidence'] if not x.startswith('No valid captured TASK->per-replay CPUsubmission link')]
cause['missing_evidence'].append('CANN API -> MODEL_EXECUTE/MODEL_WAIT_COMPLETE matches modelId in 99/104 calls; precise request identity, time calibration and the reason for late submission remain unknown.')
records['module_causes']=[x for x in records['module_causes'] if x['id']!='C_GRAPH_SUBMIT_SKEW']
records['module_causes'].append({'id':'C_GRAPH_SUBMIT_SKEW','module_id':'target.layers.0.forward','observed_graph_id':'G1_REFINED_MODULE_GRAPH',
 'affected_cases':['CAPTURED_TARGET_DECODE_INTERIOR_RUNS_1_2_3'],
 'observed_symptom':'Across3 conditionally aligned interior forward groups, DP0 graph-submit timestamps lag DP1 by approximately 1.348,0.546,1.240ms; first-layer entry skews are approximately 1.761,0.541,1.306ms.',
 'mechanism':'Observed prior inter-forward skew is already present around the linked graph submission and may propagate to early layer0 MC2 arrival.',
 'supporting_evidence':['evidence/graph-submit-diagnosis.json','evidence/inter-forward-gap-review.json','evidence/MC2-known-dependencies.md'],
 'counter_evidence':['Time proximity is supported by exact exporter connection and modelId but still uses a containing task interval and hypothesized cross-DP forward pairing.',
                     'Profiler host/device timestamp error and cross-device clock offset are unbounded; first interior group has approximately0.413ms additional entry skew after submit.',
                     'Current evidence does not identify which earlier task or scheduler action produced the late submission.'],
 'missing_evidence':['Clock/error calibration and request/replay identity', 'Unprofiled exposure and profiler-induced perturbation',
                     'Exact causality among Engram preparation, communication, prior draft/target work and host scheduling'],
 'status':'HYPOTHESIS_WITH_STRONGER_EXPORTED_ASSOCIATION',
 'confidence_and_reason':'Exact API->model-task exporter links and timing concordance support a candidate timing path; cause and removable time are unproved.',
 'end_to_end_impact':None})
write('records.json',records)
analysis=read('module-analysis.json');analysis['cause_ids']=list(dict.fromkeys(analysis['cause_ids']+['C_GRAPH_SUBMIT_SKEW']));write('module-analysis.json',analysis)
profile=read('profile-summary.json');profile['long_wait_or_idle_groups'].append({'id':'C_GRAPH_SUBMIT_SKEW','state':'EXACT_EXPORT_LINK_WITH_CONDITIONAL_CROSS_RANK_TIMING',
 'evidence':'evidence/graph-submit-diagnosis.json','not_a_causal_exposed_time':True})
assert len(json.dumps(profile).encode())<32768;write('profile-summary.json',profile)
index=read('profile-index.json');index['evidence_slices'].append({'id':'GRAPH_SUBMIT_CHAIN','path':'evidence/graph-submit-diagnosis.json',
 'source_data':'model-execute-wait-chain-v2/calls.json','read_policy':'Filter by rank and ordinal; no more than200rows/32KiB in context.'})
write('profile-index.json',index)
report=T/'REPORT.md';s=report.read_text()
old='这些证据与早到 rank 在 MC2 等对端一致。当前 TASK 连接到的是重复的捕获定义，104 次第 0 层精确 TASK 查找均成功，却未找到对应的每次 replay CANN 提交连接；不能靠 CPU 时间窗包含关系补造归属。因此，DP0 为什么更晚进入图——host 提交、前序设备队列、其他工作或 profiler 扰动——仍未区分。'
new='这些证据与早到 rank 在 MC2 等对端一致。最初按第 0 层 TASK 自身连接 ID 查询时，没有直接连到每次 CANN replay；随后沿提交 API 的连接 ID 找到了同一模型的 MODEL_EXECUTE 和 MODEL_WAIT_COMPLETE，并在 99/104 次中唯一关联到包含该层的模型任务。仍不能单靠时间窗认定请求身份，也未确定 DP0 更晚进入图的上游原因。'
assert old in s;s=s.replace(old,new)
s+='''

## 后续只读追查：图提交和目标层到达

对 104 次捕获的第 0 层调用，以前 10ms 为候选范围，99 次有唯一 `aclmdlRIExecuteAsync` 事件；其连接 ID 在导出数据库中均关联同一个 modelId 的一条 `MODEL_EXECUTE` 和一条 `MODEL_WAIT_COMPLETE`，后者唯一覆盖该层已验证的窗口。5 次未匹配均在首个捕获轮次，未强行归属。连接支持图提交与模型任务的对应；它不是请求身份或时钟同步的形式证明，也不能把 `MODEL_WAIT_COMPLETE` 直接解释成纯计算或纯等待。

在三个两组 DP 都有 16 rank 完整连接的内部样本中，候选提交与层入口的 DP0 减 DP1 时间差如下，单位为微秒：

| 条件配对序号 | 提交 API 差 | MODEL_EXECUTE 差 | 第 0 层入口差 |
|---|---:|---:|---:|
'''
for x in tab:s+=f"| {x['ordinal']} | {x['api_dp0_minus_dp1_us']:.3f} | {x['model_execute_dp0_minus_dp1_us']:.3f} | {x['layer_dp0_minus_dp1_us']:.3f} |\n"
s+='''
这些数需要 offset0 跨 DP 配对和可接受的跨时钟误差作为前提。它们支持“时间差在进入第 0 层之前就已出现”的观察，并提示图提交时序是需要追查的一段；第一个样本中入口差比提交差还大约 0.413ms。因缺少精确请求身份、时钟误差界及上游数据依赖，不能断言调度器、Engram、通信或前序草稿计算中的某一项造成了差异，更不能把差值当作可消除的端到端时间。
'''
report.write_text(s)
state=read('STATE.json');state.update(current_phase='P6',last_update=now,overall_work_paused=False,
 next_read_only_action='Trace upstream causes of the linked submit skew, and continue expanding module attribution without treating hypotheses or nominal capacity as proved.')
write('STATE.json',state)
print(json.dumps({'linked_calls':chains['calls_checked'],'unmatched_initial':5,'paired_interior':tab,
                  'causal_impact_proven':False,'whole_framework_minimum_proven':False},indent=2))
