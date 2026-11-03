"""Resume only the postpublication check after the Task enum was observed asCOMPLETED."""
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;N=H.parent;O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W=R/'whole-scope-20261030';T=W/'tasks/whole-opportunity-coverage';L=T/'loops/loop-005'
def read(p):return json.loads(p.read_text())
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
assert read(T/'index.json')['status']=='COMPLETED'
assert read(L/'state.json')['status']=='ACCEPTED'
assert read(H/'task-loop-close-receipt.json')['status']=='PASS'
write(H/'postpublication-enum-correction.json',{'status':'CORRECTED_VERIFIER_ONLY',
 'expected_Task_enum':'COMPLETED','incorrect_expected_enum':'COMPLETE',
 'actual_close_task_and_validate':'PASS','source_or_NPU_work_reexecuted':False})
root_files=['contract.json','records.json','fusion-scan.json','module-analysis.json','profile-index.json','profile-summary.json','bounds-input.json','profiling-plan.json','module_metrics.csv','operator_metrics.csv']
published=[R/n for n in root_files]+[R/'graphs/G0-current-v180.json',R/'REPORT.md']+sorted((R/'candidates/whole-scope-v180').glob('*.json'))
rec=read(R/'records.json')
checks={
 'root_payloads_match_audited_staging':all((R/n).read_bytes()==(H/n).read_bytes() for n in root_files),
 'root_G0_matches_audited_staging':(R/'graphs/G0-current-v180.json').read_bytes()==(H/'G0-current-v180.json').read_bytes(),
 'all42candidate_graph_hashes':len(rec['fusion_candidates'])==42 and all(pin(Path(c['candidate_graph_path']))['sha256']==c['candidate_graph_sha256'] for c in rec['fusion_candidates']),
 'all42original_graph_hashes':all(c['original_graph_sha256']==pin(R/'graphs/G0-current-v180.json')['sha256'] for c in rec['fusion_candidates']),
 'root_scan_complete':read(R/'fusion-scan.json')['status']=='SCAN_COMPLETE_IN_SCOPE',
 'root_summary_bounded':(R/'profile-summary.json').stat().st_size<=32768,
 'current_contract171':read(R/'contract.json')['hardware']['host']=='172.16.10.171',
 'root_module26_comparisons94':len(read(R/'module-analysis.json')['modules'])==26 and len(read(R/'module-analysis.json')['performance_comparisons'])==94,
 'Task_completed':read(T/'index.json')['status']=='COMPLETED',
 'Loop_accepted':read(L/'state.json')['status']=='ACCEPTED',
 'no_remaining_current_work':not read(O/'coverage-work-queue-v180.json')['work'] and not read(R/'STATE.json')['remaining_required_work']}
assert all(checks.values()),checks
current=read(O/'whole-scope-requirement-audit-v180.json')
for r in current['requirements']:
 if r['id']=='CRITICAL_EVIDENCE_AND_FINAL_CLOSURE':r['argument']='Audited exactcurrent contract/G0/42candidategraphs/scan/94comparisons/records/index/boundedsummary/aggregates/bounds/rawcalculations/report are published. Actualroot byte/hash/referencechecks pass;TaskCOMPLETED,LoopACCEPTED andtaskvalidatePASS. No requiredanalysiswork remains.'
final={**current,'status':'PASS_FULL_ORIGINAL_ANALYSIS_OBJECTIVE','published_root':str(R),
 'root_postpublication_checks':checks,'task_loop_receipt':pin(H/'task-loop-close-receipt.json'),
 'published_file_pins':[pin(p) for p in published],
 'review_type':'SELF_REVIEW;172staged checks+actualpostpublicationchecks;noformal/externalreviewclaim',
 'overall_complete':True,'goal_tool_update_required_after_manifest':True}
write(H/'final-completion-audit.json',final)
write(O/'whole-scope-requirement-audit-v180.json',{**current,'final_completion_audit':pin(H/'final-completion-audit.json')})
(O/'PROFILE-FINDINGS-v180.md').write_bytes((R/'REPORT.md').read_bytes())
write(O/'final-analysis-summary-v180.json',{'status':'COMPLETE','previous_goal_turn':'PROGRESS','requirements_proved':9,
 'modules':26,'semantic_rows':360,'conditional_candidates':42,'P5records':94,'remaining_required_work':[],
 'completion_audit':str(H/'final-completion-audit.json'),'overall_complete':True})

# Align other user-facing recovery/draft entrypoints; their old contents remain in the archive.
for name in ['PUBLIC-REPORT-DRAFT.md','RESEARCH-HANDOFF.md','CHECKPOINT-MANIFEST.json','publication-draft/FUSION-INVENTORY.md']:
 p=R/name;q=H/'historical-other-root'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
(R/'PUBLIC-REPORT-DRAFT.md').write_bytes((R/'REPORT.md').read_bytes())
(R/'RESEARCH-HANDOFF.md').write_text('分析已完成。恢复入口：'+str(R/'REPORT.md')+'；完成验收：'+str(H/'final-completion-audit.json')+'；当前状态：'+str(R/'STATE.json')+'。剩余条件项是候选适用性/物理数值/因果验证限制，不存在待执行的范围分析任务。未实施优化。\n')
write(R/'CHECKPOINT-MANIFEST.json',{'schema':'CURRENT_DELIVERY_POINTER','version':180,'status':'COMPLETE',
 'manifest':str(O/'checkpoint-manifest-v180.json'),'completion_audit':pin(H/'final-completion-audit.json')})
pub=R/'publication-draft/FUSION-INVENTORY.md'
pub.write_text(pub.read_text()+'\n\nv180最终验收：当前42个条件候选及原图映射以 '+str(R/'records.json')+' 为准；完整报告 '+str(R/'REPORT.md')+'；最终完成验收 '+str(H/'final-completion-audit.json')+'。原固定范围分析已完成，未实施优化。以上旧轮次的未完成措辞均保留为历史。\n')
oldm=read(O/'checkpoint-manifest-v179.json');paths={Path(r['path']) for r in oldm['artifacts']}
paths.update(p for p in H.rglob('*') if p.is_file());paths.update(published)
paths.update(p for p in O.glob('*v180*') if p.is_file() and not p.name.startswith('checkpoint-manifest'))
paths.update(p for p in T.rglob('*') if p.is_file())
paths.update([N/'CURRENT-ANALYSIS-WORKFLOW.json',N/'RECOVERY.json',N/'NEXT-EVIDENCE-PENDING.json',N/'LIVE-SHARED-COHORT.json',
 R/'STATE.json',W/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'CURRENT-FINDINGS.md',R/'CONTINUATION-STATUS.md',
 R/'PUBLIC-REPORT-DRAFT.md',R/'RESEARCH-HANDOFF.md',R/'CHECKPOINT-MANIFEST.json',pub])
manifest={'schema':'FINAL_ANALYSIS_CHECKPOINT_V1','version':180,'predecessor_manifest':pin(O/'checkpoint-manifest-v179.json'),
 'predecessor_all1786pins_validated_before_shared_writes':True,'artifacts':[pin(p) for p in sorted(paths)],
 'overall_complete':True,'whole_modules_closed':26}
write(O/'checkpoint-manifest-v180.json',manifest)
assert all(pin(Path(r['path']))['sha256']==r['sha256'] for r in manifest['artifacts'])
post={'status':'PASS','checkpoint':180,'manifest_artifacts':len(paths),'all_pins_verified':True,
 'all9_requirements_proved':True,'Task_Loop_closed':True,'remaining_required_work':[],
 'no_optimization_implementation':True,'no_new_NPU_model_restart':True,'overall_complete':True}
write(O/'final-analysis-postcheck-result-v180.json',post);write(N/'CURRENT-POSTCHECK.json',post)
print(json.dumps(post))
