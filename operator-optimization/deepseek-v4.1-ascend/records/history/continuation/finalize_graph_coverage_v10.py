from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl

def main():
 assert not (O/'REVIEW-CHECKPOINT-v10.json').exists(),'refuse_duplicate_finalization'
 now=datetime.now(timezone.utc);s=read(O/'graph-coverage-summary-v10.json')
 assert s['vector_overlay']['total_COMPUTE_CSV_correspondences']==36521 and s['graph_and_schedule']['control_pair_container_check']
 assert s['allrank_MC2']['retained_graph_MC2_rows']==7040 and not s['overall_complete']
 assert len(read(O/'candidate-profile-ledger-v10.json')['candidates'])==32 and len(read(O/'module-profile-ledger-v10.json')['modules'])==26
 jobs=[
  ('profile','pass','242vector original-key diagnostics expose ModelID-only disagreement;11CANN execute APIs directly associate with11MODEL_EXECUTE/WAIT control pairs;no directPYTORCH submission links.',[N/'profile-vector-graph-links166-v10.json',N/'query_vector_identity_and_graph_links_v10_166.py']),
  ('review','error','First six-field overlay passed but WAIT-only containment guard failed; graph tasks can begin before WAIT. Preserved without offset guessing or forced assignment.',[N/'profile-vector-graph-blocks166-v10.json',N/'reconcile_vector_graph_records_v10_166.py']),
  ('review','error','v2control-span diagnostic reached a wrong metadata directory level and stopped. v3changed only runtime metadata path;original script/receipt preserved.',[N/'profile-vector-graph-blocks166-v10-v2.json',N/'reconcile_vector_graph_records_v10_166_v2.py']),
  ('profile','pass','v3resolves242exact6nonmodel-field correspondences preserving model conflict;all36521compute records haveCSVoverlay.11connected control spans each1650tasks,440four-marker blocks,0missed/overlapping graph records.',[N/'profile-vector-graph-blocks166-v10-v3.json',N/'reconcile_vector_graph_records_v10_166_v3.py']),
  ('profile','pass','Exactannotation call_id/pid/tid to pairedsidecar plus same-thread CANNcontainment binds10cached-decode schedules(total24/fourrequests/spec5);initialgraph remainsunbound.400eligible blocks have separateAIC/AIVPMUproxies;not globalNPUutilization.',[N/'profile-graph-schedule-proxies166-v10.json',N/'query_graph_schedule_and_layer_proxies_v10_166.py']),
  ('profile','error','All16CSVcensus and111outside-marker inventory completed and fullresult saved;only finalstdout sizeguard failed. No input or data failure; bounded saved-result reader follows without CSVrequery.',[N/'profile-allrank-moe-roots166-v10.json',N/'query_allrank_moe_regimes_and_graph_roots_v10_166.py']),
  ('review','pass','Read saved allrank result only,verify structure/stat/SHA.8944totalMC2rows,7040capturedtargetM3/E24/K6;all16CSVpins retained.111outside-marker rows=21before+82between/crossing+8after,57signatures;displaytruncation explicit.',[N/'profile-allrank-moe-roots-read166-v10.json',N/'read_saved_allrank_moe_roots_v10_166.py']),
  ('review','pass','One question-gated cached model.forward slice(1510-1615,SHA101f96b...) settles outer-loop auxiliary/Engram/readiness and finalMTP/collapse/norm ownership. Does not assign every graph definition or recover aliases/timing.',[O/'PROFILE-CODE-QUESTION-GATE-v10.json',O/'PROFILE-CODE-QUESTION-REVIEW-v10.json']),
  ('review','pass','v10evidence grades and model/phase/candidate ledgers updated.242record gap resolvedinEP3;111outside-marker consumers,initialgraph context,other-rank call phases,native internal costs/framework evidence remain open.0whole modules closed.',[O/'graph-coverage-summary-v10.json',O/'graph-coverage-reconciliation-v10.json',O/'vector-model-field-overlay-v10.json',O/'candidate-profile-ledger-v10.json',O/'module-profile-ledger-v10.json',O/'module-phase-branch-matrix-v10.json',O/'module-coverage-gaps-v10.json',O/'PROFILE-FINDINGS-v10.md',N/'zcode-graph-coverage-reconcile-v10-output-20261031.json'])]
 registered=[]
 for i,(kind,status,description,artifacts) in enumerate(jobs):
  rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ')
  run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90','--kind',kind,'--command','Preparedv10evidence execution;retrospective registration: '+description,
   '--code-ref','Frozen target unchanged;old169archive read on166;profile/source hashes recorded;no new inference/NPU operation','--working-directory',str(N))
  args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
  for p in artifacts:assert p.exists(),p;args.extend(['--artifact',str(p)])
  ctl(*args);registered.append(dict(run_dir=run,status=status,description=description))
 action=('继续Loop004，上一轮v10是实质进展。242向量记录已通过name/type/stream/task/start/end六字段一一对应，保留CSVmodel4294967295与TASKmodel49冲突；'
  'EP3全部36521计算记录对应已闭合，不再重跑这项。11图控制对覆盖18150执行，40marker块×11检查通过；10图用annotation确切call_id/pid/tid+sidecar绑定4cached请求各6token/spec5/total24，初次图仍unbound。'
  '下一步先补每次111块外计算/57签名的逐definition模块与消费者归属：21before、82between/crossing、8after。'
  '完整saved结果allrank-moe-regimes-and-ep3-graph-roots-v10.json SHAda30720a8a2b39c4818f675725fbb98e3c01390385bc131330b6dbe7d477b3c5；读取衍生文件，不重查16CSV，公示root列表有展示截断。'
  '用原有v6Engram/tail合同及v10model.forward1510-1615归属：auxmean先于Engram，outerloop等待lookup/应用Engram后调用decoder；尾部fullHCcopy/collapse/norm另有消费者。'
  '对82条不要先假定全部Engram或归前一decoder；按邻近marker位置、原definition/shape和精确source合同分账，跨层HC候选保留aux-before-Engram读取和ready/alias代际。'
  'EP3已有每图40块metrics及全部outside task records，source overlay ep3-vector-overlay-and-graph-blocks-v10-v3.json SHA587b81f07b1b8c3057b9b7486bec6d58ce7aa946909d9385d8d4915bd6012c84；'
  'schedule/proxy ep3-graph-schedule-and-block-proxies-v10.json SHA747f6d1b71ae3c957d328419b2b137c8ad26a2c4608fe9bc2a5b565a3b8065b5。'
  '16卡MC2完整8944条，其中7040graph(targetM3/E24/K6)，eager targetM643/1023与draftM2/3分开。物理M不证明validmask/专家接收量；其他15rank未做逐call调度绑定，不能按ordinal自动跨卡同调用。'
  'AICproxy块20在EP3十次cached-decode子集最低0.0681534，只是marker bracket counter proxy，既不等于全模块/全NPU最低，也不与旧不同capture排名直接合并。'
  '后续查初次unbound图只围绕影响结论的phase证据；source不能补历史值。然后解决H01onCPU/ready、H02field/generation、H03hash成本与前端函数profile：优先既有数据，缺失时准备并执行有界166隔离诊断，先核验当前健康/占用，不重复问已有授权。'
  '同域资源上界/原因和26模块全条件/所有消费者未闭合。Goal保持active，0整模块关闭；不要因记录、报告或单项对应完成而关闭整体。')
 checkpoint=dict(status='PROGRESS',previous_turn_classification='PROGRESS:v9fullone-rankorigin/CSVcensus andF30postprocess contracts',
  current_verified_changes=['242Triton six-field correspondences resolve EP3computeCSVrecord gap while preserving ModelIDmetadata conflict.',
   '11nativegraph submissions associate to control pairs;18150repeated compute executions preserved and440markerblock invariants pass.',
   '10submissions bind exactannotation/sidecar callIDs and cached-decode schedules;firstgraph remainsunbound;400blocks have scopedPMUproxies.',
   'Complete16rankMC2regime census:8944rows/7040capturedtargetM3/E24/K6;separate eager/draft regimes.',
   '111outside-marker records per replay/57signatures inventoried;one narrow model.forward slice clarifies auxiliary,Engram,readiness and finalstate ownership without closing aliases.'],
  summary=str(O/'graph-coverage-summary-v10.json'),candidate_ledger=str(O/'candidate-profile-ledger-v10.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v9.json'),
  module_ledger=str(O/'module-profile-ledger-v10.json'),branch_matrix=str(O/'module-phase-branch-matrix-v10.json'),report=str(O/'PROFILE-FINDINGS-v10.md'),
  source_gate=str(O/'PROFILE-CODE-QUESTION-REVIEW-v10.json'),coverage_gaps=str(O/'module-coverage-gaps-v10.json'),
  pending_full_scope_domains=['111outside-marker module/consumer definitions','initialgraph andother-rank call phase evidence','allmodule/branch/alias/generation closure','framework CPUready/frontend/hash costs','same-domain ceilings/ranking/causal reasons'],
  remote_job_status='All v10readonly jobs terminal;three failures preserved and corrected/recovered, including display-only failure after fullresult save. No new device/inference/service operation.',
  next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=now.isoformat())
 write(O/'REVIEW-CHECKPOINT-v10.json',checkpoint)
 # Keep the historical summary intact and append a separate population.
 if not (O/'profile-summary-before-v10.json').exists():
  (O/'profile-summary-before-v10.json').write_bytes((R/'profile-summary.json').read_bytes())
 prof=read(R/'profile-summary.json');prof['supplementary_graph_schedule_v10']=dict(summary=checkpoint['summary'],
  scope='Old169diagnostic capture,EP3tenbound cached-decode schedules and16rankMC2physical census;not crosscapturemerge.',
  captured_target_MC2_physical_M=3,allrank_captured_MC2_rows=7040,lowest_observed_marker_AIC_proxy=s['schedule_and_counter_proxies']['lowest_observed_AIC_proxy_blocks'][0],
  global_lowest_NPU_utilization_proved=False,whole_modules_closed=0,overall_complete=False)
 write(R/'profile-summary.json',prof)
 for p in [L/'state.json',T/'index.json']:
  d=read(p);assert d['status'] in ['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=now.isoformat());write(p,d)
 for p in [L/'status.md',T/'status.md']:
  ls=p.read_text().splitlines();ls=[('- 下一步：'+action) if x.startswith('- 下一步：') else ('- 更新时间：`'+now.isoformat()+'`') if x.startswith('- 更新时间：') else x for x in ls];p.write_text('\n'.join(ls)+'\n')
 for p in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
  d=read(p);d.update(latest_profile_first_review=checkpoint,overall_complete=False,whole_modules_closed=0,updated_utc=now.isoformat());d['next_read_only_action' if p==R/'STATE.json' else 'next_action']=action
  if p==W/'STATE.json':d.update(profile_candidate_ledger=checkpoint['candidate_ledger'],profile_module_ledger=checkpoint['module_ledger'])
  if p in [N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[checkpoint['summary'],str(O/'REVIEW-CHECKPOINT-v10.json'),checkpoint['coverage_gaps']]
  if p==N/'RECOVERY.json':d['original_service']='Last verified during prior166 restoration;no freshhealth/occupancy check inv10,no inference/NPU/service mutation thispass.'
  write(p,d)
 compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T))
 write(O/'task-run-registration-v10.json',dict(registered_utc=now.isoformat(),runs=registered,compact_output=compact,validation_output=validation,overall_complete=False))
 paths=[O/n for n in ['PROFILE-FINDINGS-v10.md','graph-coverage-summary-v10.json','graph-coverage-reconciliation-v10.json','vector-model-field-overlay-v10.json',
  'candidate-profile-ledger-v10.json','module-profile-ledger-v10.json','module-phase-branch-matrix-v10.json','module-coverage-gaps-v10.json','PROFILE-CODE-QUESTION-GATE-v10.json',
  'PROFILE-CODE-QUESTION-REVIEW-v10.json','REVIEW-CHECKPOINT-v10.json','task-run-registration-v10.json','profile-summary-before-v10.json']]
 paths += [N/n for n in ['profile-vector-graph-links166-v10.json','profile-vector-graph-blocks166-v10.json','profile-vector-graph-blocks166-v10-v2.json','profile-vector-graph-blocks166-v10-v3.json',
  'profile-graph-schedule-proxies166-v10.json','profile-allrank-moe-roots166-v10.json','profile-allrank-moe-roots-read166-v10.json']]
 paths += [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
 write(O/'checkpoint-manifest-v10.json',dict(generated_utc=now.isoformat(),artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],whole_modules_closed=0,overall_complete=False))
 print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v10.json'),registered_runs=len(registered),statuses=[r['status'] for r in registered],validation_output=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))
if __name__=='__main__':main()
