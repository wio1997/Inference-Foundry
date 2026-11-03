from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, subprocess

N = Path(__file__).parent
O = N/'profile-first-20261031'
R = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W = R/'whole-scope-20261030'
T = W/'tasks/whole-opportunity-coverage'
L = T/'loops/loop-004'
CTL = '/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py'


def read(p):
    return json.loads(p.read_text())


def write(p,d):
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def ctl(*args):
    p = subprocess.run(['python3',CTL,*args],capture_output=True,text=True)
    assert p.returncode == 0,(args,p.stdout,p.stderr)
    return p.stdout.strip()


def main():
    assert not (O/'REVIEW-CHECKPOINT-v6.json').exists(), 'refuse_duplicate_finalization'
    summary = read(O/'hc-engram-tail-summary-v6.json')
    ledger = read(O/'candidate-profile-ledger-v6.json')
    modules = read(O/'module-profile-ledger-v6.json')
    assert summary['retained_scopes'] == 1280 and summary['crosslayer_HC_pairs'] == 1248
    assert len(ledger['candidates']) == 32 and len(modules['modules']) == 26
    assert not summary['overall_complete'] and not ledger['overall_complete']
    assert all(not c['candidate_closed'] and not c['legal_fusion_proved'] for c in ledger['candidates'])
    for name in ['archived-ep3-hc-query-status166.json','archived-ep3-hc-boundaries-saved166.json','archived-ep3-hc-tail-evidence166.json','archived-ep3-hc-tail-saved166.json','archived-ep3-hc-comm-links166.json']:
        assert read(N/name)['returncode'] == 0,name
    now = datetime.now(timezone.utc)
    jobs = [
      ('profile','error','Initial sixteen-window archived HC query controller cancelled124 before local receipt; later independent recovery proved saved remote result complete. This is a controller failure, not invalid profiler data.',[N/'zcode-hc-archived-stderr-20261031.txt',N/'zcode-hc-archived-task-20261031.txt']),
      ('profile','pass','Independent rc0 status and saved JSON read recover completed16-window HC query; DB SHA/stat verified.160 compute records across4Engram boundaries; selected ordinary gate34, WKV1, parameter3, copies2.',[N/'archived-ep3-hc-query-status166.json',N/'archived-ep3-hc-boundaries-saved166.json']),
      ('profile','pass','Two selected tail windows extended3ms to20ms with batch API/callchain lookup; finalnorm/Index/LMhead and MTPcopy bound. No device work; no claim that all39/38records are pure tail.',[N/'archived-ep3-hc-tail-evidence166.json',N/'archived-ep3-hc-tail-saved166.json']),
      ('profile','pass','Four exact COMM→CONNECTION_IDS→PYTORCH_API/callchain links prove selected TP8 auxiliary, MTP residual, premix and DSpark FC submission locations; one rank only.',[N/'archived-ep3-hc-comm-links166.json']),
      ('review','error','First v6 consolidation failed before ledger write because common.py excerpt end75 exceeded actual EOF72. Main corrected excerpt to54-72; raw profile unchanged; failure preserved.',[N/'zcode-hc-v6-output-20261031.json',N/'zcode-hc-v6-task-20261031.txt']),
      ('review','pass','Final v6 consolidation reconciles F02/F03/F24/F25/F26/F27 with six narrowly gated source snippets. Conditional row selection/raw-MTP variants, complete consumer obligations and no-gain claims retained. Overall Goal and module closure remain open.',[O/'hc-engram-tail-reconciliation-v6.json',O/'hc-engram-tail-summary-v6.json',O/'candidate-profile-ledger-v6.json',O/'module-profile-ledger-v6.json',O/'PROFILE-FINDINGS-v6.md']),
    ]
    runs = []
    for i,(kind,status,description,artifacts) in enumerate(jobs):
        rid = 'run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ')
        run = ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90','--kind',kind,
                  '--command','ZCode prepared HC/Engram/tail read-only evidence and consolidation; retrospective receipt registration; '+description,
                  '--code-ref','Frozen target unchanged; source/DB/artifact SHA retained in v6 evidence; no implementation or NPU work',
                  '--working-directory',str(N))
        args = ['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
        for p in artifacts:
            assert p.exists(),p
            args += ['--artifact',str(p)]
        ctl(*args)
        runs.append(dict(run_dir=run,status=status,description=description))
    next_action = ('继续Loop004：先对F21旧169 profile/166同调用计数、F28 first-draft初始化、F29草稿logits/Markov做剩余profile/栈细查；F27所选context FC/TP8已绑定。'
                   '然后按26模块/阶段/条件分支覆盖矩阵核对未分析范围和真正影响结论的证据，不以32候选当前复核遍历代替整体闭合。'
                   '优先复用v6已存两尾窗和既有draft/Markov局部材料；仅为无法定案的明确问题补代码。'
                   'F26按完整消费者行并集、raw-MTP在aux分支的可观察性两条件域保留候选，不把复制出现当不可删除，也不先宣称它无用。'
                   '不重跑已完成16窗/20ms/4连接查询；本轮无新设备采集。')
    checkpoint = dict(status='PROGRESS',previous_turn_classification='PROGRESS: v5 key cache, runner merge and CP layout contracts',
                      current_verified_changes=['1280HC intra pairs and1248crosslayer edges queried with outside-scope Engram/aux recovered',
                                                'Selected34-record ordinary rotated gate plus WKV/parameter/copy roles bound',
                                                'Selected target repeat, MTPcopy, collapse, finalnorm, logits row selection and LMhead directly bound',
                                                'Four exact TP8 collective links bind aux, rawHC, premix and DSpark contextFC',
                                                'Targeted consumer review rejects unconditional logits-only rows and registers conditional raw-MTP/aux dead-production variant'],
                      summary=str(O/'hc-engram-tail-summary-v6.json'),candidate_ledger=str(O/'candidate-profile-ledger-v6.json'),
                      module_ledger=str(O/'module-profile-ledger-v6.json'),report=str(O/'PROFILE-FINDINGS-v6.md'),
                      source_gate=str(O/'HC-ENGRAM-CODE-QUESTION-GATE.json'),
                      pending_fine_profile_reconciliation_in_current_pass=['F21','F28','F29'],
                      remote_job_status='All bounded queries have rc0 saved-result receipts; initial controller124 recovered without16-window retry. No NPU/inference job launched.',
                      source_review='Six files, explicit snippets after profile question gate; no repository-wide source audit',
                      next_action=next_action,whole_modules_closed=0,overall_complete=False,updated_utc=now.isoformat())
    write(O/'REVIEW-CHECKPOINT-v6.json',checkpoint)
    state = read(L/'STATE.json')
    state.update(next_action=next_action,updated_at=now.isoformat())
    assert state['status'] == 'EVALUATING'
    write(L/'STATE.json',state)
    for p in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d = read(p)
        d.update(latest_profile_first_review=checkpoint,overall_complete=False,whole_modules_closed=0,updated_utc=now.isoformat())
        if p == R/'STATE.json':
            d['next_read_only_action'] = next_action
        else:
            d['next_action'] = next_action
        if p == W/'STATE.json':
            d.update(profile_candidate_ledger=str(O/'candidate-profile-ledger-v6.json'),profile_module_ledger=str(O/'module-profile-ledger-v6.json'))
        if p in [N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:
            d['read_first'] = [checkpoint['summary'],str(O/'REVIEW-CHECKPOINT-v6.json'),checkpoint['report']]
        if p == N/'RECOVERY.json':
            d['original_service'] = 'Last verified in prior166 observer restoration: original base healthy/idle and own17processes. No fresh service health/occupancy check in v6; no NPU/inference job launched this turn.'
        write(p,d)
    compact = ctl('compact','--task-dir',str(T))
    validation = ctl('validate','--task-dir',str(T))
    registration = dict(registered_utc=now.isoformat(),runs=runs,compact_output=compact,validation_output=validation,overall_complete=False)
    write(O/'task-run-registration-v6.json',registration)
    paths = [O/'PROFILE-FINDINGS-v6.md',O/'hc-engram-tail-summary-v6.json',O/'hc-engram-tail-reconciliation-v6.json',O/'candidate-profile-ledger-v6.json',O/'module-profile-ledger-v6.json',O/'REVIEW-CHECKPOINT-v6.json',O/'HC-ENGRAM-CODE-QUESTION-GATE.json',O/'task-run-registration-v6.json',N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']
    write(O/'checkpoint-manifest-v6.json',dict(generated_utc=now.isoformat(),artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v6.json'),registered_runs=len(runs),pending=['F21','F28','F29'],validation_output=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))


if __name__ == '__main__':
    main()
