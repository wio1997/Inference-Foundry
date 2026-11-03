from pathlib import Path
from datetime import datetime, timezone, timedelta
import json, hashlib, subprocess

N=Path(__file__).parent
O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W=R/'whole-scope-20261030'
T=W/'tasks/whole-opportunity-coverage'
L=T/'loops/loop-004'
CTL='/Users/wio/.codex/skills/operator-task-loop/scripts/taskctl.py'


def read(p):return json.loads(p.read_text())


def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')


def ctl(*args):
    p=subprocess.run(['python3',CTL,*args],capture_output=True,text=True)
    assert p.returncode==0,(args,p.stdout,p.stderr)
    return p.stdout.strip()


def main():
    assert not (O/'REVIEW-CHECKPOINT-v7.json').exists(),'refuse_duplicate_finalization'
    summary=read(O/'moe-draft-summary-v7.json')
    ledger=read(O/'candidate-profile-ledger-v7.json')
    modules=read(O/'module-profile-ledger-v7.json')
    assert len(ledger['candidates'])==32 and len(modules['modules'])==26 and summary['dspark_linked_tasks']==3307
    assert summary['explicit_HCCL_COMM_links']==165 and summary['pending_family_only_current_pass']==[]
    assert not summary['overall_complete'] and all(not m['whole_module_closed'] for m in modules['modules'])
    receipts=['archived-ep3-dspark-moe-evidence166-v7-v2.json','observed-moe-counts-evidence166-v7.json','archived-ep3-dspark-pipeline-audit166-v7-v2.json','archived-ep3-dspark-communications166-v7.json','archived-ep3-dspark-phase-ancestry166-v7.json','archived-ep3-embedding-frames166-v7.json']
    for name in receipts:assert read(N/name)['returncode']==0,name
    assert read(N/'archived-ep3-dspark-moe-evidence166-v7.json')['returncode']==1
    assert read(N/'archived-ep3-dspark-pipeline-audit166-v7.json')['returncode']==1
    now=datetime.now(timezone.utc)
    jobs=[
      ('profile','pass','Local full1280target scope MC2 query;onecompound each, physicalM643/1023,24local experts,K6;8anchor labels, not8distinct tasks.',[O/'moe-profile-query-v7.json',O/'moe-profile-summary-v7.json',N/'zcode-moe-profile-v7-output-20261031.json']),
      ('profile','error','OriginalDB source-chain count118287 exceeded initial50000guard; stopped before task association and wrote no full result. Fixed by separating allhost source chains from linked-device APIs, not by adding device work.',[N/'archived-ep3-dspark-moe-evidence166-v7.json',N/'zcode-dspark-moe-v7-output-20261031.json']),
      ('profile','pass','OriginalDB read-only v2 maps2671linkedDSpark APIs to3307tasks and8selectedMC2 anchor checks/6unique tasks; SHA/stat unchanged. Full data saved server-side, no inference/NPU job.',[N/'archived-ep3-dspark-moe-evidence166-v7-v2.json']),
      ('review','pass','CPU-only recheck4736saved166snapshots/296cohorts; phase-separated87formal cohorts(target80,draft7); perexpert and perrank count populations retained without169timing join.',[N/'observed-moe-counts-evidence166-v7.json']),
      ('review','error','Saved-JSON parser assumed Tile source258 and stopped; actual Tile is in combined expression249. Removed unsupported line assumption; no database requery needed for parser retry.',[N/'archived-ep3-dspark-pipeline-audit166-v7.json',N/'zcode-dspark-pipeline-v7-output-20261031.json']),
      ('review','pass','Saved-JSON v2 audits13repeat groups/39context KV chains,6HCpre/post perordered group and156context Index; records actual operator and source signatures, not tensor/request identity.',[N/'archived-ep3-dspark-pipeline-audit166-v7-v2.json']),
      ('profile','pass','165exactHCCL API→connection→COMM matches including13basehead gathers; TASK/COMPUTE omissions corrected rather than called absent communication.',[N/'archived-ep3-dspark-communications166-v7.json']),
      ('profile','pass','Full65known parent callchains distinguish1dummy/12proposal repeats andheads,3dummy/36proposal context projections. Separate savedembedding frames expose dummy ancestry; no claim of formal client identity.',[N/'archived-ep3-dspark-phase-ancestry166-v7.json',N/'archived-ep3-embedding-frames166-v7.json']),
      ('review','pass','v7profile-first reconciliation ofF21/F28/F29 andF30subset,6narrow source files. Copy-onlySP-before-repeat relation and conditionalcollective/contextKV variants recorded; framework/module/branch coverage remains open.',[O/'moe-draft-reconciliation-v7.json',O/'moe-draft-summary-v7.json',O/'candidate-profile-ledger-v7.json',O/'module-profile-ledger-v7.json',O/'module-coverage-gaps-v7.json',O/'PROFILE-FINDINGS-v7.md']),
    ]
    registered=[]
    for i,(kind,status,description,artifacts) in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90','--kind',kind,
                '--command','ZCode prepared v7mechanical evidence execution; retrospective receipt registration: '+description,
                '--code-ref','Frozen target implementation unchanged;source/profile hashes in v7evidence; no NPU/inference work','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
        for p in artifacts:assert p.exists(),p;args.extend(['--artifact',str(p)])
        ctl(*args);registered.append(dict(run_dir=run,status=status,description=description))
    next_action=('继续Loop004，进入全模块/阶段/条件分支覆盖审计；原32候选已有分范围profile复核，但不是完整机会宇宙。'
                 '先核对M00注册选择、M01请求/采样合同、M02前缀/KV、M03调度、M04worker输入、M25响应这6个当前无candidate_links范围。'
                 '优先复用publication-draft/framework-cpu.json、continuation-20261030的worker-validation/prepare-input/graph-executions/iteration-gaps及R/STATE.json历史条目的确切产物路径，避免重放全部历史。'
                 '建立26模块×实际阶段/条件的active/inactive/unknown证据、跨算子/层/卡/迭代消费者边与剩余最小问题；同时验证26模块是否完整。空链接不能当无优化或无历史分析。'
                 'F29是contextKV；F30才是logits/Markov。v7已区分dummy/proposal，proposal不等于正式客户端阶段。'
                 'F28 SP-before-repeat有copy/pad值映射依据；AllReduce→ReduceScatter仍需group/数值合同，F29slot/position/alias、F30proposer端和MC2内部阶段因果仍未闭合。'
                 '只对影响结论且profile不能定案的问题定点补代码，保留同域资源/符号上界与最低利用率口径。不要因32入口或报告完成而关闭Goal，不重跑已保存v7查询。')
    checkpoint=dict(status='PROGRESS',previous_turn_classification='PROGRESS: v6HC/Engram/tail/aux context bindings and conditional consumer variants',
                    current_verified_changes=['All1280retainedtargetMC2scopes and39draftcompound tasks reviewed;6unique selected native tasks bindnonMega callsite.',
                                              'Fullone-rankDSpark source/API query:2671linkedAPI/3307tasks; all13repeat and39contextchains retained.',
                                              '65fullparentstacks prove1dummy+12proposal and3dummy+36proposal context;formalclient phase remains separate.',
                                              '165explicitHCCL/COMM links correctbaseheadTASKassociation gap.',
                                              'Separate166counts preserveformal80target/7draftcohorts; observedrank/expert count skew is not elapsed-time attribution.',
                                              'Copy-onlySP/shard relation supportsdraft reorder;conditionalembeddingReduceScatter andcontextRoPE/cache/index variants refined with6targeted source files.'],
                    summary=str(O/'moe-draft-summary-v7.json'),candidate_ledger=str(O/'candidate-profile-ledger-v7.json'),module_ledger=str(O/'module-profile-ledger-v7.json'),
                    report=str(O/'PROFILE-FINDINGS-v7.md'),source_gate=str(O/'MOE-DRAFT-CODE-QUESTION-GATE-v7.json'),coverage_gaps=str(O/'module-coverage-gaps-v7.json'),
                    pending_family_only_current_pass=[],pending_full_scope_domains=['M00','M01','M02','M03','M04','M25','allmodule phase/branch/resource/consumer audit'],
                    remote_job_status='All v7bounded jobs terminal; six successful saved-data/profile receipts rc0; two early analysis errors preserved and corrected; no new device/inference workload.',
                    next_action=next_action,whole_modules_closed=0,overall_complete=False,updated_utc=now.isoformat())
    write(O/'REVIEW-CHECKPOINT-v7.json',checkpoint)
    state=read(L/'state.json');assert state['status']=='EVALUATING';state.update(next_action=next_action,updated_at=now.isoformat());write(L/'state.json',state)
    for p in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(p);d.update(latest_profile_first_review=checkpoint,overall_complete=False,whole_modules_closed=0,updated_utc=now.isoformat())
        d['next_read_only_action' if p==R/'STATE.json' else 'next_action']=next_action
        if p==W/'STATE.json':d.update(profile_candidate_ledger=checkpoint['candidate_ledger'],profile_module_ledger=checkpoint['module_ledger'])
        if p in [N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[checkpoint['summary'],str(O/'REVIEW-CHECKPOINT-v7.json'),checkpoint['coverage_gaps']]
        write(p,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T))
    write(O/'task-run-registration-v7.json',dict(registered_utc=now.isoformat(),runs=registered,compact_output=compact,validation_output=validation,overall_complete=False))
    paths=[O/n for n in ['PROFILE-FINDINGS-v7.md','moe-draft-summary-v7.json','moe-draft-reconciliation-v7.json','candidate-profile-ledger-v7.json','module-profile-ledger-v7.json','module-coverage-gaps-v7.json','REVIEW-CHECKPOINT-v7.json','MOE-DRAFT-CODE-QUESTION-GATE-v7.json','task-run-registration-v7.json']]+[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']
    write(O/'checkpoint-manifest-v7.json',dict(generated_utc=now.isoformat(),artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v7.json'),registered_runs=len(registered),validation_output=validation,remaining_scope=checkpoint['pending_full_scope_domains'],whole_modules_closed=0,overall_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
