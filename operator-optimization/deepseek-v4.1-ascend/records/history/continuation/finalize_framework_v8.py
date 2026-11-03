from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, subprocess

N=Path(__file__).parent; O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W=R/'whole-scope-20261030'; T=W/'tasks/whole-opportunity-coverage'; L=T/'loops/loop-004'
CTL='/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py'
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def ctl(*args):
    p=subprocess.run(['python3',CTL,*args],capture_output=True,text=True)
    assert p.returncode==0,(args,p.stdout,p.stderr)
    return p.stdout.strip()

def main():
    assert not (O/'REVIEW-CHECKPOINT-v8.json').exists(),'refuse_duplicate_finalization'
    now=datetime.now(timezone.utc)
    s=read(O/'framework-summary-v8.json'); candidates=read(O/'framework-candidate-ledger-v8.json')
    modules=read(O/'module-profile-ledger-v8.json'); branches=read(O/'framework-module-branch-matrix-v8.json')
    assert len(candidates['candidates'])==3 and len(modules['modules'])==26 and len(branches['modules'])==6
    assert len(read(O/'candidate-profile-ledger-v8.json')['candidates'])==32
    assert all(not m['whole_module_closed'] for m in modules['modules'])
    assert s['prepare']['partition_checks_pass']==24 and s['prepare']['independent_trace_gap_prepare_pass']==22
    assert s['request_contract']['input_transport']=='text' and s['request_contract']['hash_binds_original_framework_run']
    for name in ['archived-framework-speedscope-callsites166-v8.json','framework-source-slices166-v8.json','framework-request-contract166-v8-v2.json']:
        assert read(N/name)['returncode']==0,name
    assert read(N/'framework-request-contract166-v8.json')['returncode']==1
    # Two review annotations only: correct the client request-construction line
    # range and avoid an unsupported candidate-prerequisite edge.
    for p in [O/'framework-summary-v8.json',O/'FRAMEWORK-CODE-QUESTION-REVIEW-v8.json']:
        d=read(p)
        contract=d['request_contract'] if 'request_contract' in d else d['case_transport_answer']
        contract['client_contract_sources'][0]['lines']='36-41,133-135'
        write(p,d)
    candidates['relationships'][2].update(relation='interacts',scope='Both touch KV lookup/ownership and subsequent metadata. Neither candidate requires implementing the other; not a demonstrated performance interaction.')
    write(O/'framework-candidate-ledger-v8.json',candidates)
    jobs=[
      ('review','invalid','First local gap comparison omitted gap_ns and falsely reported22 full-dict mismatches. Counts/partitions unaffected; invalid comparison preserved and superseded, not source-data conflict.',[O/'framework-v8-invalid-comparison-receipt.json',O/'framework-profile-summary-v8-first-invalid.json',N/'query_framework_profiles_v8_first_invalid.py']),
      ('profile','pass','CPU-only full24 prepare root/166944 descendant/113name audit;24 partitions and22 same-trace gap checks pass,0 old/new disagreements; EngineCore and worker runs kept separate.',[O/'framework-profile-audit-v8.json',O/'framework-profile-summary-v8.json',N/'query_framework_profiles_v8.py']),
      ('profile','pass','Read four SHA-pinned original EngineCore raw files on166; complete response stack groups bind source callsites. No absolute per-sample crop or per-function CPU attribution claimed.',[N/'archived-framework-speedscope-callsites166-v8.json',N/'query_framework_speedscope_v8_166.py']),
      ('review','pass','Question-gated docker cp of three fixed-image files, no exec/start/import. Existing hybrid yield/poll/notification/cancel and Future FIFO contracts reviewed; existing worker/hash snapshots reused narrowly.',[N/'framework-source-slices166-v8.json',O/'FRAMEWORK-CODE-QUESTION-GATE-v8.json',O/'FRAMEWORK-CODE-QUESTION-REVIEW-v8.json']),
      ('review','error','Initial request-contract script assumed8 seeds; actual frozen fixture has8 formal requests and2 seeds. Stopped at guard before writing derived result; original script/receipt preserved.',[N/'framework-request-contract166-v8.json',N/'query_framework_request_contract_v8_166.py']),
      ('review','pass','Corrected seed count guard; frozen8 text prompt hash binds original framework result.8 distinct full prompts share117286 characters; no token/extra-key identity or hash-cost claim.',[N/'framework-request-contract166-v8-v2.json',N/'query_framework_request_contract_v8_166_v2.py']),
      ('review','pass','Framework profile-first reconciliation:6 partially reviewed domains,3 graded hypotheses,8 consumer boundaries; existing32 model intakes preserved. New/current and existing optimizations distinguished;26module universe and all closures remain open.',[O/'framework-summary-v8.json',O/'framework-candidate-ledger-v8.json',O/'framework-module-branch-matrix-v8.json',O/'framework-consumer-edges-v8.json',O/'candidate-profile-ledger-v8.json',O/'module-profile-ledger-v8.json',O/'module-coverage-gaps-v8.json',O/'PROFILE-FINDINGS-v8.md',N/'zcode-framework-reconcile-v8-output-20261031.json'])
    ]
    registered=[]
    for i,(kind,status,description,artifacts) in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90','--kind',kind,
                '--command','ZCode prepared v8 mechanical evidence execution; retrospective receipt registration: '+description,
                '--code-ref','Frozen target unchanged; source/profile/input hashes in evidence; no NPU/inference workload','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
        for p in artifacts: assert p.exists(),p; args.extend(['--artifact',str(p)])
        ctl(*args); registered.append(dict(run_dir=run,status=status,description=description))
    action=('继续Loop004。v8已部分审查M00/M01/M02/M03/M04/M25，并保留3个框架假设；六域不是已关闭模块。'
            '下一轮先用module-profile-ledger-v8及既有cohort-and-gap-bindings、projection-pipeline-summary、compressor-gap-v4、cache-merge-v5、hc-engram-v6、moe-draft-v7证据，'
            '建立26模块在冻结负载内的有限phase/condition要求与实际active/inactive/unknown对应，区分startup、seed、formal、dummy、capture、proposal；核验模块集合是否遗漏。'
            '先利用已有全量API/trace/stack汇总补阶段和消费者，不做整库源码/AST审计。'
            '按缺口标清已有原始数据可查询或需有限补采：H01同窗on-CPU及producer-ready，H02精确field/copy/event和graph代际，H03真实token/hash-extra/H与成本；前端两条trace仅窗口标记，不能由代码还原历史时延。'
            '请求传输已核实为8条不同文本，输入hash绑定原EngineCore运行；Task token-ID shape是语义输入而非HTTP格式。2条prefix seed不计正式统计。'
            'SHM已有yield/poll通知取消、worker已有持久buffer和异步修正、hash已有按请求增量；勿重复列为缺失能力。'
            '全模型32候选仍需条件/所有消费者/跨层卡迭代闭合与同域利用率及上界。CPU0.98核不等于NPU利用率，也不能当推理瓶颈；保持整体未完成和0整模块关闭。'
            '不重跑v8查询，不把已改正的gap_ns比较误报或seed数量guard当原始数据冲突。')
    checkpoint=dict(status='PROGRESS',previous_turn_classification='PROGRESS: v7 native/COMM/phase and MoE count reconciliation',
        current_verified_changes=['Full24 prepare-root aggregate/113names/166944descendants;24 partitions and22 original/recomputed trace gap intersections pass.',
                                  'Original raw EngineCore stacks bind sampled SHM busy/yield/read protocol; both leader threads ~0.98 CPUcore, no causal gain attribution.',
                                  'Three question-gated files show existing hybrid wait/cancel/Future FIFO; existing worker/hash snapshots prevent duplicate-optimization claims.',
                                  'Frozen input hash matches original run:8 distinct text prompts,2 prefix seeds; common117286characters is not token/hash identity.',
                                  'Six-domain phase/branch matrix,3 graded framework hypotheses and8 consumer boundaries complement existing32 model candidates without module closure.'],
        summary=str(O/'framework-summary-v8.json'),candidate_ledger=str(O/'candidate-profile-ledger-v8.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v8.json'),
        module_ledger=str(O/'module-profile-ledger-v8.json'),branch_matrix=str(O/'framework-module-branch-matrix-v8.json'),consumer_edges=str(O/'framework-consumer-edges-v8.json'),
        report=str(O/'PROFILE-FINDINGS-v8.md'),source_gate=str(O/'FRAMEWORK-CODE-QUESTION-REVIEW-v8.json'),coverage_gaps=str(O/'module-coverage-gaps-v8.json'),
        pending_full_scope_domains=['26module universe verification','allmodule finite phase/branch/consumer closure','framework missing causal/profile costs','same-domain resource bounds/ranking and critical-path causes'],
        remote_job_status='All v8 bounded read/derived jobs terminal;3 remote successful receipts rc0;1 failed seed-count guard preserved. One invalid local comparison corrected. No fresh inference/NPU/service operation.',
        next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=now.isoformat())
    write(O/'REVIEW-CHECKPOINT-v8.json',checkpoint)
    for p in [L/'state.json',T/'index.json']:
        d=read(p);assert d['status'] in ['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=now.isoformat());write(p,d)
    for p in [L/'status.md',T/'status.md']:
        lines=p.read_text().splitlines()
        lines=[('- 下一步：'+action) if x.startswith('- 下一步：') else ('- 更新时间：`'+now.isoformat()+'`') if x.startswith('- 更新时间：') else x for x in lines]
        p.write_text('\n'.join(lines)+'\n')
    for p in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(p);d.update(latest_profile_first_review=checkpoint,overall_complete=False,whole_modules_closed=0,updated_utc=now.isoformat())
        d['next_read_only_action' if p==R/'STATE.json' else 'next_action']=action
        if p==W/'STATE.json':d.update(profile_candidate_ledger=checkpoint['candidate_ledger'],profile_module_ledger=checkpoint['module_ledger'])
        if p in [N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[checkpoint['summary'],str(O/'REVIEW-CHECKPOINT-v8.json'),checkpoint['coverage_gaps']]
        if p==N/'RECOVERY.json':d['original_service']='Last verified during prior166 restoration. No fresh health/occupancy check in v8; no NPU/inference/service mutation in this pass.'
        write(p,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T))
    write(O/'task-run-registration-v8.json',dict(registered_utc=now.isoformat(),runs=registered,compact_output=compact,validation_output=validation,overall_complete=False))
    paths=[O/n for n in ['PROFILE-FINDINGS-v8.md','framework-summary-v8.json','framework-profile-audit-v8.json','framework-profile-summary-v8.json',
        'framework-candidate-ledger-v8.json','framework-module-branch-matrix-v8.json','framework-consumer-edges-v8.json','candidate-profile-ledger-v8.json',
        'module-profile-ledger-v8.json','module-coverage-gaps-v8.json','REVIEW-CHECKPOINT-v8.json','FRAMEWORK-CODE-QUESTION-GATE-v8.json',
        'FRAMEWORK-CODE-QUESTION-REVIEW-v8.json','framework-v8-invalid-comparison-receipt.json','task-run-registration-v8.json']]
    paths += [N/n for n in ['framework-request-contract166-v8.json','framework-request-contract166-v8-v2.json','framework-source-slices166-v8.json','archived-framework-speedscope-callsites166-v8.json']]
    paths += [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',L/'state.json',T/'index.json']
    write(O/'checkpoint-manifest-v8.json',dict(generated_utc=now.isoformat(),artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v8.json'),registered_runs=len(registered),statuses=[r['status'] for r in registered],validation_output=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
