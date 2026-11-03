"""Reusable compact persistence: inherit operator authorities, seal evidence delta."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from finalize_origin_coverage_v9 import N, O, R, W, T as TASK_DIR, L as LOOP_DIR, read, write, ctl
from analyze_evidence_batches_v155 import pin


def persist_delta(*, version, previous, prefix, changes, report, action, remaining_delta,
                  source_gate, operator_overlay, queue, jobs, scripts, obligation_reconciliations=None, remote_status=None, case_registrations=None):
    global LOOP_DIR
    suffix='v'+str(version);old_suffix='v'+str(previous);checkpoint_path=O/('REVIEW-CHECKPOINT-'+suffix+'.json');assert not checkpoint_path.exists(),'seal once'
    old_manifest_path=O/('checkpoint-manifest-'+old_suffix+'.json');old_manifest=read(old_manifest_path);old=read(O/('REVIEW-CHECKPOINT-'+old_suffix+'.json'))
    preflight=read(O/(prefix+'-preflight-'+suffix+'.json'));assert preflight['status']=='PASS'and preflight['preflight']
    for source in old_manifest['artifacts']:assert pin(source['path'])==source,source['path']
    # Registration mutates shared Task files only after predecessor pins are checked.
    for case in case_registrations or []:
        existing=read(TASK_DIR/'cases.json')['cases'].get(case['id'])
        if existing:
            assert all(existing[k]==case[k] for k in ['id','shape','dtype','layout','attributes','group','reason'])
            continue
        ctl('add-case','--task-dir',str(TASK_DIR),'--case-id',case['id'],
            '--shape',json.dumps(case['shape']),'--dtype',case['dtype'],
            '--layout',case['layout'],'--attributes-json',json.dumps(case['attributes'],ensure_ascii=False),
            '--group',case['group'],'--weight','1','--protected','yes','--reason',case['reason'])
    old_loop=LOOP_DIR
    assert old_loop.name=='loop-004' and read(old_loop/'decision.json')['result']=='PENDING'
    old_contracts=[pin(old_loop/name) for name in ['goal.md','hypotheses.md']] if (old_loop/'goal.md').exists() else [pin(old_loop/'hypotheses.md')]
    ctl('verdict','--loop-dir',str(old_loop),'--result','pivoted','--goal-satisfied','no',
        '--comparability','not-applicable','--causal-conclusion','partially-supported',
        '--reason','Migration changes the active case/hardware to171. Loop004 preserves its frozen169/166 case domain and all scoped profile/cut evidence;full-scope completeness is not achieved. Continue in a derived171 evidence Loop,not by relabeling171 probe as166.',
        '--next-action',action,'--evidence-path',str(O/'partial-persistence-correction-v161.json'),
        '--evidence-path',str(O/'joint-runtime-summary-v161.json'))
    loop_args=['new-loop','--task-dir',str(TASK_DIR),'--loop-id','loop-005',
        '--title','171共享算子证据与profile入口对齐','--mode','spec',
        '--hypothesis','复用实际内置profile字段并定点补offset/extent/epochs/values/native桥和读者，可减少无关逐对取证；未补齐证据不判独立或整体完成。',
        '--success','共享证据路由可审查，明确每个字段的实际producer/作用域；171全Case采集入口和缺口有具体预算/回滚，原全部未闭合范围保留。',
        '--falsification','字段语义或task/capture桥不成立，或重复logger/逐对工作无法回答结论变化；保留未知并转向有证据的入口。',
        '--allowed-path',str(N),'--next-action',action]
    for case_id in read(TASK_DIR/'cases.json')['cases']:loop_args+=['--case',case_id]
    LOOP_DIR=Path(ctl(*loop_args))
    for contract in old_contracts:assert pin(contract['path'])==contract
    write(O/'derived-loop-transition-v161.json',dict(previous_loop=str(old_loop),previous_result='PIVOTED',
        new_loop=str(LOOP_DIR),retrospective_registration=True,
        is_not_a_claim_of_experiment_preregistration=True,old_frozen_contracts_preserved=old_contracts,
        original_goal_not_achieved=True,new_native_execution=False))
    now=datetime.now(timezone.utc);stamp=now.isoformat();summary_path=O/(prefix+'-summary-'+suffix+'.json');summary=read(summary_path)
    assert summary['overall_complete']is False and summary.get('new_native_operand_or_phase_promotions',0)==0 and summary.get('new_whole_module_closures',0)==0
    report_path=O/('PROFILE-FINDINGS-'+suffix+'.md');report_path.write_text(report)
    gate_path=O/('PROFILE-CODE-QUESTION-GATE-'+suffix+'.json');write(gate_path,source_gate)
    overlay_path=O/(prefix+'-operator-overlay-'+suffix+'.json');write(overlay_path,operator_overlay)
    queue_path=O/('coverage-work-queue-'+suffix+'.json');write(queue_path,queue)
    remaining=deepcopy(old['pending_full_scope_domains'])+[remaining_delta]
    obligations_path=O/('coverage-current-obligations-'+suffix+'.json');base_obligations=Path(old['coverage_obligations']);base=read(base_obligations)
    current_obligations=deepcopy(base['obligations'])
    for row in current_obligations:
        if obligation_reconciliations and row['id'] in obligation_reconciliations:
            row['current_evidence_routing_'+suffix]=deepcopy(obligation_reconciliations[row['id']])
    if obligation_reconciliations:
        assert set(obligation_reconciliations)=={row['id']for row in current_obligations}
    write(obligations_path,dict(schema='OVERLAY_AND_INHERITED_OBLIGATIONS_V1',parent=pin(base_obligations),obligations=current_obligations,operator_delta=str(overlay_path),all_original_broad_gaps_and_history_inherited=True,no_missing_evidence_independence_inference=True,current_evidence_routing_supersedes_legacy_dispatch_wording=True,new_native_or_whole_closures=0,overall_complete=False))
    gaps_path=O/('module-coverage-gaps-'+suffix+'.json');write(gaps_path,dict(parent=pin(Path(old['coverage_gaps'])),remaining=remaining,operator_delta=str(overlay_path),work_queue=str(queue_path),whole_modules_closed=0,overall_complete=False))
    audit_path=O/('whole-scope-requirement-audit-'+suffix+'.json');base_audit=Path(old['requirement_audit']);base=read(base_audit)
    write(audit_path,dict(schema='OVERLAY_AND_INHERITED_REQUIREMENTS_V1',parent=pin(base_audit),requirements=[dict(id=row['id'],state=row['state'],base_requirement=str(base_audit)+'#'+row['id'],delta='No completion promotion;scoped operator evidence overlay only.')for row in base['requirements']],operator_delta=str(overlay_path),complete_original_scope_preserved=True,legacy_pair_counts_are_not_completion_metric=True,new_native_or_whole_closures=0,whole_modules_closed=0,overall_complete=False))
    summary.update(generated_utc=stamp,remaining=remaining,next_action=action,unchanged_operator_authorities_reused=True,operator_authority_baseline_checkpoint=old.get('operator_authority_baseline_checkpoint',previous),checkpoint_schema='DELTA_CHECKPOINT_V1',remote_job_status=remote_status or 'All v159 archive/source acquisitions terminal;no new device work within this evidence acquisition scope.');write(summary_path,summary)
    registered=[]
    for index,job in enumerate(jobs):
        runid='run-'+(now-timedelta(seconds=len(jobs)-index)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(LOOP_DIR),'--run-id',runid,'--case-key',job.get('case_key','c8-input32k-output1k-prefix90-171'),'--kind',job['kind'],'--command',job.get('command','Prepared'+suffix+' retrospective registration: '+job['description']),'--code-ref',job.get('code_ref','Pinned read-only evidence analysis; no target implementation changes'),'--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',job['status'],'--correctness',job.get('correctness','not-applicable'),'--summary',job['description']]
        for artifact in job['artifacts']:assert Path(artifact).exists();args+=['--artifact',str(artifact)]
        ctl(*args);registered.append(dict(run_dir=run,status=job['status'],description=job['description'],case_key=job.get('case_key','c8-input32k-output1k-prefix90-171'),correctness=job.get('correctness','not-applicable')))
    cp=deepcopy(old);cp.update(checkpoint_schema='DELTA_CHECKPOINT_V1',previous_checkpoint=str(O/('REVIEW-CHECKPOINT-'+old_suffix+'.json')),status='PROGRESS',active_task_loop=str(LOOP_DIR),previous_turn_classification=summary['previous_goal_turn'],current_verified_changes=changes,summary=str(summary_path),report=str(report_path),source_gate=str(gate_path),coverage_gaps=str(gaps_path),coverage_obligations=str(obligations_path),requirement_audit=str(audit_path),pending_full_scope_domains=remaining,next_action=action,work_queue=str(queue_path),remote_job_status=summary['remote_job_status'],updated_utc=stamp,unchanged_operator_authorities_reused=True,operator_authority_baseline_checkpoint=summary['operator_authority_baseline_checkpoint'],evidence_overlays=list(old.get('evidence_overlays',[]))+[str(overlay_path)],whole_modules_closed=0,overall_complete=False)
    write(checkpoint_path,cp)
    workflow_path=N/'CURRENT-ANALYSIS-WORKFLOW.json';workflow=read(workflow_path);workflow.update(effective_checkpoint=version,work_queue=pin(queue_path),previous_shared_state_validated_before_authority_advance=True,primary_queue=[x['id']for x in queue['work']]);write(workflow_path,workflow)
    shared=[workflow_path]
    for path in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        data=read(path);data.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);data['next_read_only_action'if path==R/'STATE.json'else'next_action']=action
        if path==W/'STATE.json':data.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
        if path in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:data['read_first']=[str(summary_path),str(queue_path),str(overlay_path),str(obligations_path),cp['analysis_workflow']]
        write(path,data);shared.append(path)
    for path in[LOOP_DIR/'state.json',TASK_DIR/'index.json']:
        data=read(path);assert data['status']in['ACTIVE','EVALUATING'];data.update(next_action=action,updated_at=stamp);write(path,data);shared.append(path)
    for path in[LOOP_DIR/'status.md',TASK_DIR/'status.md']:
        path.write_text('\n'.join(('- 下一步：'+action)if line.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if line.startswith('- 更新时间：')else line for line in path.read_text().splitlines())+'\n');shared.append(path)
    pub=R/'publication-draft/FUSION-INVENTORY.md';(O/('FUSION-INVENTORY-before-'+suffix+'.md')).write_bytes(pub.read_bytes());pub.write_text(pub.read_text()+'\n\n'+suffix+'算子证据增量：'+str(report_path)+'；当前权威 '+str(checkpoint_path)+'。保留原算子台账路径和全部未闭合范围；未新增整模块或物理上限完成声明。\n');shared.append(pub)
    profile=read(R/'profile-summary.json');profile['supplementary_'+suffix]=dict(summary=str(summary_path),operator_evidence_overlay=str(overlay_path),work_queue=str(queue_path),unchanged_operator_authorities_reused=True,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',profile);shared.append(R/'profile-summary.json')
    compact=ctl('compact','--task-dir',str(TASK_DIR));validation=ctl('validate','--task-dir',str(TASK_DIR));write(O/('task-run-registration-'+suffix+'.json'),dict(runs=registered,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    paths={Path(x['path'])for x in old_manifest['artifacts']};paths.update(Path(x['path'])for x in read(O/(prefix+'-inputs-'+suffix+'.json'))['inputs']);paths.update(shared);paths.update(scripts);paths.add(Path(__file__))
    paths.update(p for p in O.glob('*'+suffix+'*')if p.is_file()and not p.name.startswith('checkpoint-manifest')and'postcheck'not in p.name)
    paths.update(p for p in TASK_DIR.iterdir() if p.is_file())
    paths.update(p for p in old_loop.iterdir() if p.is_file())
    paths.update(p for p in LOOP_DIR.iterdir() if p.is_file())
    for registered_run in registered:paths.update(p for p in Path(registered_run['run_dir']).rglob('*') if p.is_file())
    for job in jobs:paths.update(Path(p)for p in job['artifacts'])
    manifest=O/('checkpoint-manifest-'+suffix+'.json');write(manifest,dict(schema='DELTA_CHECKPOINT_V1',generated_utc=stamp,predecessor_manifest=pin(old_manifest_path),predecessor_current_state_validated_before_shared_writes=True,artifacts=[pin(p)for p in sorted(paths)],old_unchanged_proofs_reused=True,whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(status='PROGRESS',checkpoint=version,manifest_files=len(paths),registered_runs=len(registered),validation=validation,unchanged_operator_authorities_reused=True,old_unchanged_operator_cases_reexecuted=False,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))
