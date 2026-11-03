from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy
N=Path(__file__).parent;O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W=R/'whole-scope-20261030'
def read(p):return json.loads(p.read_text())
def write(p,d):
 assert not p.exists(),('refuse_overwrite',str(p));p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def receipt(name):
 d=read(N/name);assert d['returncode']==0,(name,d['stderr']);return json.loads(d['stdout'])
def main():
 now=datetime.now(timezone.utc).isoformat()
 q=receipt('profile-vector-graph-links166-v10.json');g=receipt('profile-vector-graph-blocks166-v10-v3.json')
 p=receipt('profile-graph-schedule-proxies166-v10.json');a=receipt('profile-allrank-moe-roots-read166-v10.json')
 assert g['summary']['six_field_vector_matches']==242 and g['summary']['complete_COMPUTE_CSV_overlay_records']==36521
 assert g['summary']['control_pair_container_check'] and g['summary']['graph_compute_records']==18150
 assert g['summary']['marker_sequence_pass']==11 and g['summary']['per_block_marker_attention_moe_pass']==440
 assert p['summary']['schedule_context_counts']=={'SCHEDULE_CONTEXT_UNBOUND':1,'BOUND_NONEMPTY_CACHED_DECODE_SCHEDULE':10}
 assert p['summary']['eligible_schedule_blocks']==400 and p['summary']['bound_decode_requests_subset_of_two_eager_cases']
 assert a['summary']['rank_count']==16 and a['summary']['retained_graph_MC2_rows']==7040 and a['summary']['all_retained_graph_MC2_are_M3_E24_K6']
 assert a['summary']['all_MC2_rows']==8944
 for f in ['profile-vector-graph-blocks166-v10.json','profile-vector-graph-blocks166-v10-v2.json','profile-allrank-moe-roots166-v10.json']:assert read(N/f)['returncode']==1
 source=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py'
 source_sha=hashlib.sha256(source.read_bytes()).hexdigest();assert source_sha=='101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91'
 overlay=dict(status='SCOPED_EXACT_RECORD_CORRELATION_WITH_MODEL_METADATA_CONFLICT',new_six_field_matches=242,
  full_four_field_matches=36279,total_COMPUTE_CSV_correspondences=36521,unpaired_COMPUTE_CSV_records_in_EP3=0,
  exact_fields=['name','type','stream','task','start','end'],conflicting_fields={'CSV.Model_ID':4294967295,'TASK.modelId':49},
  corrected_types=['_quantize_indexer_query_kernel','_prepare_indexer_indices_kernel','_pool_kernel','_cache_update_kernel'],
  evidence=[{k:r[k] for k in ['full_result','result_sha256']} for r in [q,g]],
  interpretation='Record correspondence is resolved for these242events; the model field disagreement is preserved. No blanket sentinel rewrite or exporter-cause claim is made.',
  CSV_partition=dict(compute_correspondences=36521,COMM_intervals=1849,AICPU_tasks=278,total=38648,durations_additive=False),
  applicability='This oneEP3original diagnostic capture only. Other ranks/captures need their own correlation before using this correction.')
 graphs=dict(summary=g['summary'],schedule=p['summary'],allrank_MC2=a['summary'],
  physical_regimes=a['rank_regimes'],first_scheduled_MC2_signature=p['first_MoE_signature'],
  graph_record_identity='TASK.rowid is an execution;1650definitions recur11times. StoredCANN connection IDs bind each MODEL_EXECUTE/WAIT pair.',
  container_correction='WAIT starts after some graph tasks begin. The directly connected EXECUTE-start to WAIT-end pair contains all1650records per replay, with0missed/overlapping assignments. This is an export association, not a causal wait-time estimate.',
  schedule_binding='10native submissions are inside same-thread annotations whose exactcall_id/pid/tid match paired sidecar execute_model entries. All4requestHMACs are shared with the two profiled eager cases; each request hascached>=32768,spec5,scheduled6,total24.',
  schedule_limits=['No direct PYTORCH connection exists for graph submission in this export. Custom annotation identity plus host containment is the stated bridge.',
    'Exact client request_id, accepted token values, masks, expert receiver counts and crossrank call cohorts remain separate.',
    'Initial graphordinal0has no scheduled-model annotation and is not labeled formal/dummy/warmup.'],
  resource_comparison=dict(metric='AIC cycle-weighted reported MAC counter proxy',population='EP3 40observed marker brackets x10bound cached-decode schedules',
    lowest=p['summary']['lowest_observed_AIC_proxy_blocks'][0],whole_NPU_lowest_proved=False,
    meaning='Reported AIC and AIV proxies remain separate. This ranking excludes outside-marker work and is not a global module ranking, physical-peak efficiency or a speedup bound.'),
  existing_large_M_evidence='Retained target eager MC2M643/1023 and draftM2/3 are now kept separate from target capturedM3; one frozen C8case has multiple resource regimes.',
  source_graph_ownership_answer=dict(path=str(source),sha256=source_sha,ranges=[[1510,1615]],
    facts=['Outer model loop records selected auxiliary means before Engram and before decoder layer execution(1556-1564).',
      'Outer loop performs readiness waits, shards lookups and applies layer.engram before calling layer(...)(1565-1592).',
      'Final fullHC MTP copy and collapse/norm are outer-model consumers(1593-1607).'],
    implications=['Outside the four-marker decoder brackets can be real Engram/auxiliary/embedding/final-state work, not idle.',
      'Cross-layer HCpost->HCpre fusion/reordering must preserve checkpoint reads before Engram, Engram-updated hidden states, readiness and pre_mix contracts.',
      'The111outside records still need per-definition ownership/consumer reconciliation; source control-flow ownership alone does not map every record or alias.']),
  outside_marker_inventory=dict(records_each=111,position_counts=a['summary']['first_outside_region_counts'],signature_groups=57,
    full_data=a['full_result'],full_data_sha256=a['result_sha256'],display_truncated=a['first_outside_groups_display_truncated']),
  whole_modules_closed=0,overall_complete=False)
 old=read(O/'candidate-profile-ledger-v9.json')
 for c in old['candidates']:
  if c['id']=='F21':
   c['profile_review_state']='ALL16RANK_MC2_REGIMES_AND_EP3_CACHED_DECODE_GRAPH_CONTEXT_REVIEWED; NATIVE_INTERNAL_PHASES_AND_ROUTING_VALUES_OPEN'
   c['scoped_detail']['captured_decode_v10']=dict(allrank=a['summary'],EP3_schedule=p['summary']['schedule_context_counts'],MC2_signature=p['first_MoE_signature'],
      numerical_or_traffic_bound='No end-to-end gain, actual routing load or necessary weight/HBM traffic is inferred from physicalM3; dispatch/compute/combine remains a compound task.')
   c['conclusion_changing_gap']='TargetM3captured decode is observed on16ranks; per-call phase bound onEP3only. Actual masks/receivers, native-stage cost, complete consumers and same-domain hardware/resource ceilings remain open.'
  if c['id']=='F03':
   c['scoped_detail']=c.get('scoped_detail') or {}
   c['scoped_detail']['outer_loop_consumers_v10']=graphs['source_graph_ownership_answer']
  if c['id'] in ['F09','F11','F12','F17','F18','F21','F24','F25']:
   c.setdefault('evidence_files',[]).append('graph-coverage-reconciliation-v10.json')
   c['captured_decode_profile_supplement']='Corrected242Triton record correspondences retain conflicting model fields;11graph executions have40marker blocks each,10bound cached-decode schedules onEP3. No global alias/phase closure.'
 old.update(generated_utc=now,parent_ledger='candidate-profile-ledger-v9.json',review_v10_utc=now,candidate_universe_complete=False,overall_complete=False)
 modules=read(O/'module-profile-ledger-v9.json');matrix=read(O/'module-phase-branch-matrix-v9.json')
 affected=['M05','M06','M07','M08','M09','M10','M11','M12','M13','M14','M15','M16','M21','M22','M23','M24']
 for m in modules['modules']:
  if m['id'] in affected:
   m['profile_entry_points'].append('graph-coverage-reconciliation-v10.json');m['graph_profile_update_v10']='EP3record identities/graph context and16rankMC2physical domains; partial consumer/phase/resource closure.'
  m['phase_branch_record']='module-phase-branch-matrix-v10.json#'+m['id'];m['whole_module_closed']=False
 modules.update(generated_utc=now,parent_ledger='module-profile-ledger-v9.json',whole_modules_closed=0,overall_complete=False)
 for m in matrix['modules']:
  if m['id'] in affected:m['graph_profile_update_v10']=dict(evidence='graph-coverage-reconciliation-v10.json',schedule_class='10bound cached-decode schedules onEP3;1initialgraph unbound',scope='40marker blocks, partial per-definition ownership; full16rank physical MC2regime census')
  if m['id']=='M12':m['physical_M_regimes_v10']=dict(target_graph=3,target_eager=[643,1023],draft_eager=[2,3],actual_masks_and_EP_cohorts='NOT_ATTESTED')
  if m['id'] in ['M08','M09','M22']:m['Triton_model_field_overlay']='242EP3record correlations resolved by six exact nonmodel fields; originalCSV sentinel conflict preserved.'
  if m['id'] in ['M06','M14','M15','M24']:m['outer_model_consumer_contract_v10']=graphs['source_graph_ownership_answer']
 matrix.update(generated_utc=now,parent_matrix='module-phase-branch-matrix-v9.json',overall_complete=False)
 summary=dict(generated_utc=now,status='PROGRESS',vector_overlay=overlay,graph_and_schedule=g['summary'],schedule_and_counter_proxies=p['summary'],allrank_MC2=a['summary'],
  source_ownership=graphs['source_graph_ownership_answer'],remaining=['111outside-marker records per replay/57signatures need per-definition module+consumer ownership, using v6root/Engram/tail evidence and current outer-loop contract.',
   'Initialgraph0lacks scheduled-model context. No automatic warmup/dummy/formal label; inspect only conclusion-changing phase evidence.',
   'Other15ranks need schedule/cohort association before routing-volume/timing joins; actual masks/receiver counts and nativeMC2stages remain absent.',
   'Whole26module branch/consumer/alias closure, framework CPU/ready/frontend/hash costs, same-domain resource bounds and global lowest-utilization assessment.'],
  no_new_device_or_inference_work=True,whole_modules_closed=0,overall_complete=False,
  saved_server_results=[{k:r[k] for k in ['full_result','result_sha256']} for r in [q,g,p,a]],
  errors_retained=['Initial WAIT-only container guard omitted tasks that begin before WAIT. Paired control span was verified without a guessed time offset.',
    'v2metadata path used one wrong directory level;v3fixed only the path.',
    'Allrank CSVwork completed and saved; initialstdout display guard failed. Bounded saved-result retrieval succeeded without a CSVrequery.'])
 gate=read(O/'PROFILE-CODE-QUESTION-GATE-v10.json');gate['review_answer']=graphs['source_graph_ownership_answer']
 write(O/'PROFILE-CODE-QUESTION-REVIEW-v10.json',gate)
 write(O/'vector-model-field-overlay-v10.json',overlay);write(O/'graph-coverage-reconciliation-v10.json',graphs)
 write(O/'graph-coverage-summary-v10.json',summary);write(O/'candidate-profile-ledger-v10.json',old)
 write(O/'module-profile-ledger-v10.json',modules);write(O/'module-phase-branch-matrix-v10.json',matrix)
 write(O/'module-coverage-gaps-v10.json',dict(generated_utc=now,remaining=summary['remaining'],
  next_action='First reconcile all57outside-marker signatures/111records to outer-model embedding,aux,Engram,HCinitialization and finalstate consumers. Keep graph0unbound and crossrank phase limits. Then address remaining framework causal profiles with a finite authorized collection if original data cannot answer, after fresh166health/occupancy check.',
  whole_modules_closed=0,overall_complete=False))
 report='''# 图执行、字段冲突与 decode 范围审查 v10

本轮补齐了一个原始 EP3 capture 的 CSV/计算记录对应关系，并把图提交关联到 replay 控制对和已记录调度；同时完整核对同次16卡的 MC2物理形状。整体仍未完成，0个整模块关闭。

## 242条记录的对应已解决，冲突保留

242条Triton记录的name、type、stream、task、start、end六字段全部严格一致，且一一对应。CSV把Model ID写为4294967295，数据库TASK写为49；此差异保留在独立overlay，没有改原始数据，也没有对所有sentinel统一替换。当前EP3的36521条计算执行均已找到CSV对应：36279条原四字段对应，加242条显式冲突对应。

完整CSV仍按层级分账：36521条计算对应、1849条COMM区间、278条AICPU任务，共38648条。COMM区间与内部任务可能重叠，不能把两层时长相加。该修正只验证了此EP3；其他rank或capture不能自动套用。

## Replay与调度证据

11次原生graph execute各有同connection的MODEL_EXECUTE/WAIT控制对。仅从WAIT开始取区间会遗漏先启动的图任务；使用同一控制对的EXECUTE起点至WAIT终点后，每次1650条计算均唯一归属，合计18150条，0遗漏/重复。1650个定义在11次执行中保持一致，各stream的定义顺序签名也全部一致。

每次均有40组HcPre/HcPost/HcPre/HcPost，且每组各1个attention、1个MC2，共440个结构检查通过的块。原生PyTorch connection没有记录到提交父级；另通过同线程自定义annotation的确切call_id与配对sidecar，绑定了其中10次提交的调度：同4个请求均已缓存至少32768token，各调度6token、spec5，总24。请求HMAC集合与两次已分析eager调用一致。该关联由call ID与同线程包含域组成，不是时间最近匹配，也不是客户端request_id证明。

首次图提交没有对应的scheduled-model annotation，继续保留阶段未知。调度逻辑token数不能替代物理有效mask、接受数或专家接收计数。

## MC2的不同物理规模

16卡共8944条MC2记录，其中捕获图内7040条，每卡440条，物理M均为3、local experts24、K6。eager target M643/1023和draft M2/3分别保留；不能用一种形状的瓶颈解释整个C8负载。

这一结果补充了F21的小M decode范围，但没有展开原生dispatch/FFN/combine内部阶段。实际有效mask、专家负载、同调用EP参与者、权重流量和净收益仍缺证据。166observer计数与这些169历史时延不作同调用拼接。

## 计数器与模块边界

EP3的10次已绑定decode提交有400个marker块。按同域AIC周期加权MAC代理，块20最低，约0.0681534；这是这个单卡、这个decode子集的记录代理，不是全模型最低NPU利用率、MFU或硬件峰值效率。AIV向量代理单独保留，不能与AIC或CPU占核混为一个利用率。

每次1650条计算中，1539条位于marker块内；余111条分为块前21、块间/跨块82、块后8，共57种形状/类型签名。它们是实际工作。定点复用外层model.forward确认：辅助hidden mean先于Engram，外层循环完成lookup就绪等待及Engram，再调用decoder layer；最后还有fullHC MTP复制、collapse和norm。因此跨层HC融合必须保留这些前后消费者，块外不能直接称为idle或并入某一decoder。111条仍需逐definition归属；源码的控制流回答不能替代实际alias/代际证据。

下一步先利用已经保存的57签名及v6 Engram/tail合同，完成块外工作与消费者对应，再处理未定阶段和需要有界补采的框架成本。未改目标实现、未操作服务、未发新推理/NPU负载。

本轮三个分析错误均保留：WAIT-only容器假设不足、metadata目录层级错误、全16卡结果保存后的stdout展示预算错误。最后一项仅重新读取已完成结果，没有重查CSV。
'''
 target=O/'PROFILE-FINDINGS-v10.md';assert not target.exists();target.write_text(report)
 print(json.dumps(dict(status='PROGRESS',resolved_vector_records=242,COMPUTE_correspondences=36521,graph_records=18150,bound_decode_replays=10,allrank_graph_MC2=7040,
  remaining_outside_records_per_replay=111,whole_modules_closed=0,overall_complete=False,summary=str(O/'graph-coverage-summary-v10.json')),ensure_ascii=False))
if __name__=='__main__':main()
