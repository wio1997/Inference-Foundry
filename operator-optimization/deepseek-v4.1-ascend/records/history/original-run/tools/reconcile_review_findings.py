# -*- coding: utf-8 -*-
"""Reconcile independently identified scope, provenance and comparison issues."""
from pathlib import Path
from fractions import Fraction
import copy,datetime,hashlib,json
T=Path(__file__).resolve().parents[1];Q=Path('/Users/wio/work/173-sglang-matched-172-benchmark')
def load(p):return json.loads((T/p).read_text())
def save(p,x):(T/p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256((T/p).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat();m=load('module-analysis.json');c=load('evidence/conditional-performance-comparisons.json');old=[]
byid={x['object_id']:x for x in c['comparisons']}
for x in m['performance_comparisons']:
 if x.get('measured_over_ceiling_ratio') is None:continue
 assert x['scope_match_status']=='CONDITIONAL_DOMAIN_MEMBERSHIP_NOT_FULLY_ESTABLISHED_FOR_OBSERVED_RUN'
 y=byid[x['object_id']];assert y['measured_over_ceiling_ratio']==x['measured_over_ceiling_ratio'] and y['time_gap_ratio']==x['time_gap_ratio']
 old.append({'object_id':x['object_id'],'measured_over_conditional_ceiling_arithmetic':copy.deepcopy(x['measured_over_ceiling_ratio']),'observed_time_over_conditional_lower_arithmetic':copy.deepcopy(x['time_gap_ratio']),'scope_status':x['scope_match_status'],'interpretation':'Hypothetical sensitivity if the measured case is shown to satisfy the stated restricted domain and capacity premises; not a verified comparison and not an achievable speedup.'})
 for v in (x,y):v['conditional_arithmetic_sensitivity_ref']='evidence/conditional-ratio-sensitivity.json';v['measured_over_ceiling_ratio']=None;v['time_gap_ratio']=None;v['formal_ratio_status']='UNKNOWN_SCOPE_MEMBERSHIP_NOT_PROVEN'
assert len(old)==7
save('evidence/conditional-ratio-sensitivity.json',{'timestamp':now,'status':'UNVERIFIED_DOMAIN_SENSITIVITY_NOT_FORMAL_RATIO','objects':old,'formal_same_scope_ratio':'UNKNOWN','no_data_deleted':True})
new_missing=['Eager target full phase semantics and 720 windows without exported attention','Individual eager DSpark block identities and complete activity','Engram host-UVA NPU lookup full ownership, surrounding gate/collective/event and CPU work','Root/head/metadata']
m['coverage']['missing_module_ids']=new_missing;m['coverage']['eager_target_partial_ref']='eager-target-attribution-v2/summary.json';m['coverage']['engram_kernel_partial_ref']='engram-kernel-profile/summary.json';save('module-analysis.json',m);save('evidence/conditional-performance-comparisons.json',c)
contract=load('contract.json');contract['measurement_scope_refinement']['missing_scope']=new_missing;save('contract.json',contract)
r=load('records.json');ca=next(x for x in r['module_causes'] if x['id']=='C_ARRIVAL_IMBALANCE')
ca['supporting_evidence']=[x.replace('graph-launch-link-inspection/summary.json: exact captured TASKs found for104layer0calls, but no CANNreplay link recovered; capture-definition connection IDs repeat across replays','Historical first attempt graph-launch-link-inspection/summary.json: task IDs alone did not map replay; superseded by evidence/graph-submit-diagnosis.json, where 99/104 calls have an API-to-model-task link. Request identity and cross-clock error remain unresolved.') for x in ca['supporting_evidence']]
causes=[
 {'id':'C_LAYER0_SMALL_M','mechanism':'Small physical query rows or expert token fragments could limit parallel work; counts per expert and effective tile occupancy are not observed.','observed_symptom':'Target layer0 M=3 and low AIC MAC proxy','supporting_evidence':['evidence/q-projection-16rank-diagnosis.json'],'counter_evidence':['M=3 alone does not prove that limited parallel work caused the layer-level metric'],'missing_evidence':['per-expert effective rows and tile occupancy','unprofiled causal comparison']},
 {'id':'C_LAYER0_MEMORY','mechanism':'Memory-path pressure may limit useful work; physical bandwidth, residency and stall causality are unverified.','observed_symptom':'AIC MTE2 pipeline counter proxy is present','supporting_evidence':['evidence/captured-layer0-engine-diagnosis.json'],'counter_evidence':['Pipeline activity overlaps other pipes and is not HBM saturation'],'missing_evidence':['per-layer actual bytes','hard capacity and residency','causal memory stall evidence']},
 {'id':'C_LAYER0_EXPERT_SKEW','mechanism':'Expert or rank imbalance could prolong fused dispatch/GMM/combine.','observed_symptom':'MC2 dominates the sampled layer0 AIC-cycle denominator','supporting_evidence':['evidence/captured-layer0-engine-diagnosis.json'],'counter_evidence':['rank arrival skew alone does not establish expert-token imbalance'],'missing_evidence':['per-expert token distributions','rank-internal stage timestamps']},
 {'id':'C_LAYER0_PATH_OR_MEASUREMENT','mechanism':'Actual fused operator lowering, profiling perturbation, or module assignment may affect the comparison.','observed_symptom':'Graph capture and profiler-based source-role attribution are used','supporting_evidence':['captured-layer-attribution/source-mapping-validation.json','evidence/graph-submit-diagnosis.json'],'counter_evidence':['rank role checks and sample robustness support but do not prove attribution'],'missing_evidence':['unprofiled per-module timings','full instruction/internal path','explicit request and tensor identities']}]
known={x['id'] for x in r['module_causes']}
for x in causes:
 assert x['id'] not in known;x.update(module_id='target.layers.0.forward',observed_graph_id='G1_REFINED_MODULE_GRAPH',affected_cases=['FROZEN_E2E_C8_PROFILED_CAPTURED_TARGET_INTERIOR'],status='HYPOTHESIS',end_to_end_impact=None);r['module_causes'].append(x)
save('records.json',r)
raw=load('evidence/used-raw-source-sha256.json');assert raw['count']==97 and len(raw['records'])==97
idx=load('profile-index.json');h={x['relative_path']:x for x in raw['records']};matched=0
for f in idx['files']:
 if f['id'] in h:
  z=h[f['id']];assert f['size_bytes']==z['size_bytes'];f['sha256']=z['sha256'];matched+=1
assert matched==17,matched
idx['used_raw_hashes']={'path':'evidence/used-raw-source-sha256.json','scope':'16 used profiler DBs, 16 profiler info files, 64 host/device clock anchors and selected EP0 trace; kernel CSV hashes are separately indexed','indexed_file_matches':matched,'other_unreferenced_exports':'not fully hashed'};save('profile-index.json',idx)
scan=load('fusion-scan.json');scan['replay_verifier']='tools/verify_fusion_enumeration_readonly.py';scan['replay_evidence']='zcode/operator-ceiling169-v2-069-fusion-replay.output.txt';scan['per_node_edge_coverage_ledger']['interpretation']='G1 count indicates inclusion as an internal edge/node of at least one cross-module connected set; it does not assert the edge itself crosses a module or was semantically validated.';save('fusion-scan.json',scan)
p=T/'REPORT.md';text=p.read_text();text=text.replace('现有时间线支持跨 DP 到达不齐、早到 rank 在 MC2 中等待的机制；尚未证明其可消除时间或端到端影响。','现有时间线支持跨 DP 到达不齐，与早到 rank 在 MC2 中等待的假说一致；内部等待、可消除时间及端到端影响尚未直接证明。')
text=text.replace('覆盖缺口：eager prefill、3 个 eager DSpark block、decoder 外的 Engram CPU lookup/投影门控、root/head/metadata 未形成同口径模块排名。','覆盖缺口：eager target 完整阶段语义及 720 个未见 attention 窗口、3 个 eager DSpark block 的独立身份与完整活动、Engram host-UVA NPU lookup 的完整归属及投影门控/通信/CPU 工作、root/head/metadata 未形成同口径全框架排名。')
text=text.replace('| 对象 | 实测中位数/微秒 | 条件计算下界/微秒 | T/L（非可达提速） |','正式实测/界比值均为 UNKNOWN：样本属于下面的受限实现域尚未核实。下表仅展示假设样本满足这些条件时的算术敏感性，不是已验证比较或可达提速。\n\n| 对象 | 实测中位数/微秒 | 条件计算下界/微秒 | 假设同域的 T/L 算术 |')
text=text.replace('现有时间线支持跨 DP 到达不齐、早到 rank 在 MC2 中等待的机制','现有时间线支持跨 DP 到达不齐，与早到 rank 在 MC2 中等待的假说相符')
p.write_text(text)
with p.open('a') as f:f.write('''\n\n## 最终交付口径和复现入口\n\n融合图的逐节点、逐边结构覆盖台账在 `coverage/`。G1 的跨请求反馈边 `verify_accept → inputs` 明确排除在单轮 2–4 节点枚举之外；F31 有条件 metadata 局部图已保存，消费者和任务等待仍未知。`python3 tools/verify_fusion_enumeration_readonly.py` 可只读重算 G1 跨模块及 G2/G5 已声明的有限集合；它只验证结构枚举，不验证语义等价或全图穷尽。\n\n现有受限域的七项正式 T/L 与 L/T 保留 UNKNOWN；历史数值另存 `evidence/conditional-ratio-sensitivity.json`，仅表示假设输入及实现确属该域时的算术。16 个实际使用的 profiler DB、时间锚点及 EP0 trace 的原路径和哈希在 `evidence/used-raw-source-sha256.json`；kernel CSV 哈希在 `evidence/TRACE-EXPORT-SHA256.json`。未被分析引用的其他导出文件并未全部计算哈希。\n''')
state=load('STATE.json');state.update(current_phase='P7',state='PARTIAL_DELIVERY_REVIEW',last_update=now,checkpoint_status='AUDITED_PARTIAL_DELIVERY',next_read_only_action='Read final review and unresolved coverage; no implement/tuning work.');save('STATE.json',state)
replay=Q/'operator-ceiling169-v2-069-fusion-replay.output.txt';rr=json.loads(replay.read_text());assert rr['status']=='PASS_BOUNDED_STRUCTURAL_REPLAY_ONLY'
print(json.dumps({'formal_ratios_set_unknown':len(old),'raw_index_hashes_added':matched,'new_explicit_hypotheses':len(causes),'replay_status':rr['status'],'timestamp':now}))
