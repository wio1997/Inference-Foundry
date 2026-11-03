from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,subprocess,hashlib
N=Path(__file__).parent;O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W=R/'whole-scope-20261030';T=W/'tasks/whole-opportunity-coverage';L=T/'loops/loop-004'
CTL='/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def ctl(*args):
 p=subprocess.run(['python3',CTL,*args],capture_output=True,text=True);assert p.returncode==0,(args,p.stdout,p.stderr);return p.stdout.strip()
def main():
 assert not (O/'REVIEW-CHECKPOINT-v9.json').exists(),'refuse_duplicate_finalization'
 now=datetime.now(timezone.utc);s=read(O/'origin-coverage-summary-v9.json')
 assert s['CSV_partition']['partition_conserved'] and not s['overall_complete']
 assert len(read(O/'module-profile-ledger-v9.json')['modules'])==26
 assert len(read(O/'candidate-profile-ledger-v9.json')['candidates'])==32
 for name in ['profile-task-identity166-v9.json','profile-all-compute-origins166-v9-v2.json','profile-origin-source-audit166-v9-v2.json','profile-postprocess-csv-kinds166-v9.json']:assert read(N/name)['returncode']==0,name
 for name in ['profile-all-compute-origins166-v9.json','profile-origin-source-audit166-v9.json']:assert read(N/name)['returncode']==1,name
 # Correct a source-record description: the profile does not establish11
 # distinct code sites. Counts at individual lines are API records.
 p=O/'module-phase-branch-matrix-v9.json';d=read(p);m=next(m for m in d['modules'] if m['id']=='M24')
 m['branches'][0]['branch']='source-linked acceptance-count correction and existing persistent hash/cache paths';write(p,d)
 jobs=[
  ('profile','error','Initial whole-origin query stopped at unsupported globalTaskId-as-execution-unique guard. No derived full result; preserve script and receipt.',[N/'query_profile_origin_coverage_v9_166.py',N/'profile-all-compute-origins166-v9.json']),
  ('review','pass','Raw record/schema audit proves36521TASK executions share20021COMPUTE definitions; example repeated definition has11execution times. No deduplication or source/phase guessing.',[N/'profile-task-identity166-v9.json',N/'query_profile_task_identity_v9_166.py']),
  ('profile','pass','Complete one-EP3 TASK/COMPUTE origin census retains36521executions,18371source-linked and18150unlinked graph records,14821full callchains/356826frames/65project files/228leaf locations;SHA/stat verified.',[N/'profile-all-compute-origins166-v9-v2.json',N/'query_profile_origin_coverage_v9_166_v2.py']),
  ('review','error','Initial CSV identity parser stopped on N/A. Original script/receipt retained. No sentinel/nearest-time repair was made.',[N/'profile-origin-source-audit166-v9.json',N/'audit_saved_origin_sources_v9_166.py']),
  ('review','pass','Saved-origin/CSV v2 accepts36279full4field matches with0duration deltas, exposes missing CSVstream/task fields and2369unmatched rows;IndexFillfullstack and228leaf locations retained.',[N/'profile-origin-source-audit166-v9-v2.json',N/'audit_saved_origin_sources_v9_166_v2.py']),
  ('profile','pass','Full selected proposer/sampler ancestry yields5131compute records/462signature groups, including65MarkovAdd/ArgMax/Cast steps.2369remainingCSV rows split into1849exactCOMM,278exactAICPU TASK and242unresolved vector rows.',[N/'profile-postprocess-csv-kinds166-v9.json',N/'query_saved_postprocess_and_csv_kinds_v9_166.py']),
  ('review','pass','Profile-gated existing-source review: two metadata snapshots show batch/group sharing and CP RoPE view reuse already implemented; two frozen proposer method/init slices bind greedy/recurrence/probability/confidence and eager contracts. No target implementation change.',[O/'PROFILE-CODE-QUESTION-GATE-v9.json',O/'F30-proposer-review-v9.json',O/'input-metadata-review-v9.json']),
  ('review','pass','v9reconciliation establishes a conserved CSVrecord-kind partition, finite26module phase/condition requirements and threeF30conditional variants. Source-path overlap and unlinked replay contexts remain explicit;0whole modules closed.',[O/'origin-coverage-summary-v9.json',O/'module-phase-branch-matrix-v9.json',O/'module-profile-ledger-v9.json',O/'candidate-profile-ledger-v9.json',O/'framework-candidate-ledger-v9.json',O/'module-coverage-gaps-v9.json',O/'PROFILE-FINDINGS-v9.md',N/'zcode-origin-reconcile-v9-output-20261031.json'])]
 registered=[]
 for i,(kind,status,description,artifacts) in enumerate(jobs):
  rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ')
  run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90','--kind',kind,'--command','Prepared v9evidence execution; retrospective registration: '+description,
   '--code-ref','Frozen target unchanged; old169data read from own166archive, SHA pinned; no new NPU/inference workload','--working-directory',str(N))
  args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
  for p in artifacts:assert p.exists(),p;args.extend(['--artifact',str(p)])
  ctl(*args);registered.append(dict(run_dir=run,status=status,description=description))
 action=('继续Loop004。v9是实质进展，不能当无进展：完整单EP3原DB36521计算执行、14821API/fullchains、228最内层位置及CSV38648条记录已审计。'
  '先解决原数据可查的242条vector CSV身份缺口：源结果ep3-all-compute-origins-v9-v2.json SHAfdce699c6cf2fdf15b9ba85c5e9e5a5cd7893d2b34a0f69b0321e719a9ef6b9d；'
  'CSV审计ep3-origin-source-and-csv-audit-v9-v2.json SHA7124c7e900ed2bfb4d1ed9a29eb335096a5d8f9a6bd2be85b1f43b9d91453541。'
  '242来自quantize_indexer_query88/prepare_indexer_indices88/pool33/cache_update33；看完整stream/task/model/start/end原键及DB同字段候选/graph执行ordinal和原flow，不能最近时间补配。'
  '36521执行共享20021定义，其中1650定义×11=18150图执行缺原生API来源，不能按globalTaskId去重，也不能用dummy提交栈当实际replay阶段。随后核对这些图定义到模块/阶段及消费者的已有capture/graph元数据。'
  'CSV已完整分账：36279计算执行、1849COMM区间、278AICPU任务、242未定向量；不同层级时长不可相加。另1条IndexFill已在proposal上下文绑定。'
  '26模块已有有限phase/condition要求表，但source路径或表行不是语义穷尽。继续按原始完整调用栈/模块标记复核，勿整库AST/源码审计。'
  'F30已绑定65greedy Markov Add/ArgMax/Cast及Slice/ViewCopy，3条件变体是strided加法归约保留舍入/回写、词表所有权重组、静态Markov行跨迭代复用。'
  '全BF16dense转移表逻辑33426636800B、每行258560B，不代表显存可行或收益。保留actual dynamic/parallel/vocab flags、别名、概率/置信度消费者与数值合同。'
  'H02已有batch/group共享、C2length共享、staged metadata和CP RoPE切片复用；只查剩余CP/slot/ring字段的代际与就绪问题，勿重复报缺失优化。'
  '再把缺口区分已有数据可再查和需有界补采；补采前核验166当前独占/健康，不重复请求已有诊断授权。前端函数、H01onCPU/ready、H03hash成本仍缺，代码不能补历史timing。'
  '同域利用率/理论界、所有消费者/跨层卡迭代未闭合，Goal仍active，0整模块关闭；不重跑已成功v9查询，不因报告成稿关闭。')
 checkpoint=dict(status='PROGRESS',previous_turn_classification='PROGRESS: v8framework/request/full-prepare evidence and graded hypotheses',
  current_verified_changes=['36521device compute executions/20021definitions audited without replay deduplication;18371linked execution records and14821complete callchains retained.',
    'Complete CSV partition38648=36279compute+1849COMM+278AICPU+242unresolved vector rows, with exact identities and0duration deltas for matched compute.',
    'New observed IndexFill branch,65source-bound greedy Markov postprocess steps, recurrence/rounding/probability/confidence constraints and threeconditional F30variants.',
    'Existing batch/group metadata sharing and CP RoPE view reuse confirmed;H02 narrowed to remaining field/readiness/lifetime work.',
    '26module finite phase/branch requirements preserve active, unrequested, partial and unknown evidence; no module-universe or consumer closure claimed.'],
  summary=str(O/'origin-coverage-summary-v9.json'),candidate_ledger=str(O/'candidate-profile-ledger-v9.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v9.json'),
  module_ledger=str(O/'module-profile-ledger-v9.json'),branch_matrix=str(O/'module-phase-branch-matrix-v9.json'),report=str(O/'PROFILE-FINDINGS-v9.md'),
  source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v9.json'),coverage_gaps=str(O/'module-coverage-gaps-v9.json'),
  pending_full_scope_domains=['242vector record identities','18150graph execution source/phase/module relation','fullmodule/allconsumer/alias/generation and finite branch closure','framework causal/function profiles','same-domain resource bounds/ranking/critical-path causes'],
  remote_job_status='All v9readonly/derived jobs terminal:4successful remote receipts rc0,2failed guards/parsers preserved and corrected. No fresh device/inference/service operation.',
  next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=now.isoformat())
 write(O/'REVIEW-CHECKPOINT-v9.json',checkpoint)
 for p in [L/'state.json',T/'index.json']:
  d=read(p);assert d['status'] in ['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=now.isoformat());write(p,d)
 for p in [L/'status.md',T/'status.md']:
  ls=p.read_text().splitlines();ls=[('- 下一步：'+action) if x.startswith('- 下一步：') else ('- 更新时间：`'+now.isoformat()+'`') if x.startswith('- 更新时间：') else x for x in ls];p.write_text('\n'.join(ls)+'\n')
 for p in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
  d=read(p);d.update(latest_profile_first_review=checkpoint,overall_complete=False,whole_modules_closed=0,updated_utc=now.isoformat());d['next_read_only_action' if p==R/'STATE.json' else 'next_action']=action
  if p==W/'STATE.json':d.update(profile_candidate_ledger=checkpoint['candidate_ledger'],profile_module_ledger=checkpoint['module_ledger'])
  if p in [N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[checkpoint['summary'],str(O/'REVIEW-CHECKPOINT-v9.json'),checkpoint['coverage_gaps']]
  if p==N/'RECOVERY.json':d['original_service']='Last verified during prior166 restoration; no fresh health/occupancy check in v9, no NPU/inference/service mutation this pass.'
  write(p,d)
 compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T))
 write(O/'task-run-registration-v9.json',dict(registered_utc=now.isoformat(),runs=registered,compact_output=compact,validation_output=validation,overall_complete=False))
 paths=[O/n for n in ['PROFILE-FINDINGS-v9.md','origin-coverage-summary-v9.json','F30-proposer-review-v9.json','input-metadata-review-v9.json','module-phase-branch-matrix-v9.json',
   'module-profile-ledger-v9.json','candidate-profile-ledger-v9.json','framework-candidate-ledger-v9.json','module-coverage-gaps-v9.json','PROFILE-CODE-QUESTION-GATE-v9.json','REVIEW-CHECKPOINT-v9.json','task-run-registration-v9.json']]
 paths += [N/n for n in ['profile-all-compute-origins166-v9.json','profile-task-identity166-v9.json','profile-all-compute-origins166-v9-v2.json','profile-origin-source-audit166-v9.json','profile-origin-source-audit166-v9-v2.json','profile-postprocess-csv-kinds166-v9.json']]
 paths += [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',L/'state.json',T/'index.json']
 write(O/'checkpoint-manifest-v9.json',dict(generated_utc=now.isoformat(),artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],whole_modules_closed=0,overall_complete=False))
 print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v9.json'),registered_runs=len(registered),statuses=[r['status'] for r in registered],validation_output=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))
if __name__=='__main__':main()
