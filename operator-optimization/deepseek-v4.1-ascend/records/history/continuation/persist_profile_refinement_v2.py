"""Persist a source/profile refinement; this does not implement a model change."""
from copy import deepcopy
from datetime import datetime,timezone,timedelta
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl

def persist_refinement(*,version,previous,title,summary,changes,candidate_updates,module_updates,
                       remaining,action,jobs,report,read_first,publication_rows,gate_updates,
                       scripts,executor_prefix,main_script,framework_updates=None):
    v='v'+str(version);pv='v'+str(previous);now=datetime.now(timezone.utc);stamp=now.isoformat()
    assert not(O/('REVIEW-CHECKPOINT-'+v+'.json')).exists(),'refuse duplicate checkpoint'
    prev=read(O/('REVIEW-CHECKPOINT-'+pv+'.json'));assert prev['status']=='PROGRESS'
    # The previous checkpoint's shared state must still be the state being refined.
    for a in read(O/('checkpoint-manifest-'+pv+'.json'))['artifacts']:
        from pathlib import Path
        q=Path(a['path']);b=q.read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],str(q)
    model=deepcopy(read(O/('candidate-profile-ledger-'+pv+'.json')));fw=deepcopy(read(O/('framework-candidate-ledger-'+pv+'.json')));mods=deepcopy(read(O/('module-profile-ledger-'+pv+'.json')));matrix=deepcopy(read(O/('module-phase-branch-matrix-'+pv+'.json')))
    cm={c['id']:c for c in model['candidates']};mm={m['id']:m for m in mods['modules']};bm={m['id']:m for m in matrix['modules']}
    fm={c['id']:c for c in fw['candidates']}
    assert set(framework_updates or {})<=fm.keys()
    for cid,update in (framework_updates or {}).items():
        fm[cid].update(deepcopy(update));fm[cid].update(candidate_closed=False,critical_path_gain_us=None)
    assert set(candidate_updates)<=cm.keys()
    for cid,update in candidate_updates.items():
        cm[cid].update(deepcopy(update));cm[cid].update(legal_fusion_proved=False,candidate_closed=False,critical_path_gain_us=None)
        q=R/'candidates'/(cid+'.json');master=read(q);write(O/(cid+'-master-before-'+v+'.json'),master);master['review_'+v]=deepcopy(update.get('review_'+v,update));write(q,master)
    for mid,u in module_updates.items():
        for cid in u.get('candidates',[]):
            assert cid in cm
            if cid not in mm[mid]['candidate_links']:mm[mid]['candidate_links'].append(cid)
        mm[mid]['review_'+v]=dict(finding=u['finding'],evidence=u['evidence'],whole_module_closed=False)
        bm[mid]['branches'].append(dict(branch=u.get('branch',title),state='SCOPED_MEASURED_DOCUMENTED_DERIVED;WHOLE_MODULE_OPEN',evidence=u['finding'],gap=u['gap']))
    for m in mods['modules']:m['phase_branch_record']='module-phase-branch-matrix-'+v+'.json#'+m['id'];bm[m['id']]['candidate_links']=list(m['candidate_links'])
    for obj,base in[(model,'candidate-profile-ledger'),(fw,'framework-candidate-ledger'),(mods,'module-profile-ledger')]:
        obj.update(generated_utc=stamp,parent_ledger=base+'-'+pv+'.json',whole_modules_closed=0,overall_complete=False,new_source_read=True,new_target_source_read=True,repository_wide_audit=False,new_device_work=False)
        if obj is model:obj['framework_candidate_ledger']='framework-candidate-ledger-'+v+'.json'
        if obj is mods:obj.update(existing_model_candidates=len(cm),framework_candidates=len(fw['candidates']))
        write(O/(base+'-'+v+'.json'),obj)
    matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-'+pv+'.json');write(O/('module-phase-branch-matrix-'+v+'.json'),matrix)
    registry=[]
    for c in model['candidates']:
        q=R/'candidates'/(c['id']+'.json');registry.append(dict(id=c['id'],master_record=str(q),master_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),current_authority=str(O/('candidate-profile-ledger-'+v+'.json'))+'#'+c['id'],modules=[m['id']for m in mods['modules']if c['id']in m['candidate_links']]))
    assert all(c['modules']for c in registry)
    write(O/('candidate-record-index-'+v+'.json'),dict(generated_utc=stamp,model_candidates=registry,framework_authority=str(O/('framework-candidate-ledger-'+v+'.json')),unlinked_candidate_ids=[],whole_modules_closed=0,overall_complete=False))
    write(O/('module-coverage-gaps-'+v+'.json'),dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
    summary=dict(summary,generated_utc=stamp,status='PROGRESS',model_candidates=len(cm),framework_candidates=len(fw['candidates']),whole_modules_closed=0,overall_complete=False,remaining=remaining)
    summary_path=O/summary.pop('filename');write(summary_path,summary)
    gate=read(O/('PROFILE-CODE-QUESTION-GATE-'+v+'.json'));gate.update(gate_updates);write(O/('PROFILE-CODE-QUESTION-GATE-'+v+'.json'),gate)
    report_path=O/('PROFILE-FINDINGS-'+v+'.md');report_path.write_text(report)
    pub=R/'publication-draft/FUSION-INVENTORY.md';(O/('FUSION-INVENTORY-before-'+v+'.md')).write_bytes(pub.read_bytes());lines=pub.read_text().splitlines()
    lines=[x.replace('当前持续分析台账（'+pv+'）','当前持续分析台账（'+v+'）').replace(str(O/('candidate-profile-ledger-'+pv+'.json')),str(O/('candidate-profile-ledger-'+v+'.json'))).replace(str(O/('PROFILE-FINDINGS-'+pv+'.md')),str(report_path))for x in lines]
    for j,line in enumerate(lines):
        for cid,new_line in publication_rows.items():
            if line.startswith('| '+cid+' |'):lines[j]=new_line
    lines+=['',v+'持续分析台账：'+str(len(cm))+'项模型候选、'+str(len(fw['candidates']))+'项框架候选，整模块仍open。当前记录 '+str(summary_path)+'；局部证据/报告成稿不证明全范围完成。'];pub.write_text('\n'.join(lines)+'\n')
    runs=[]
    for j,job in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-j)).strftime('%Y%m%dT%H%M%SZ');run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind',job['kind'],'--command','Prepared'+v+'retrospectiveregistration: '+job['description'],'--code-ref','Savedprofile/exactsource;targetunchanged;noNPU/model/servicework','--working-directory',str(N));args=['finish-run','--run-dir',run,'--status',job['status'],'--correctness','not-applicable','--summary',job['description']]
        for q in job['artifacts']:assert q.exists(),q;args+=['--artifact',str(q)]
        ctl(*args);runs.append(dict(run_dir=run,status=job['status'],description=job['description']))
    cp=dict(status='PROGRESS',previous_turn_classification=summary['previous_turn_classification'],current_verified_changes=changes,summary=str(summary_path),candidate_ledger=str(O/('candidate-profile-ledger-'+v+'.json')),framework_candidate_ledger=str(O/('framework-candidate-ledger-'+v+'.json')),module_ledger=str(O/('module-profile-ledger-'+v+'.json')),branch_matrix=str(O/('module-phase-branch-matrix-'+v+'.json')),report=str(report_path),source_gate=str(O/('PROFILE-CODE-QUESTION-GATE-'+v+'.json')),coverage_gaps=str(O/('module-coverage-gaps-'+v+'.json')),pending_full_scope_domains=remaining,remote_job_status=summary['remote_job_status'],model_candidates=len(cm),framework_candidates=len(fw['candidates']),next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp)
    write(O/('REVIEW-CHECKPOINT-'+v+'.json'),cp)
    for q in[L/'state.json',T/'index.json']:
        d=read(q);assert d['status']in['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(q,d)
    for q in[L/'status.md',T/'status.md']:q.write_text('\n'.join(('- 下一步：'+action)if x.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if x.startswith('- 更新时间：')else x for x in q.read_text().splitlines())+'\n')
    (O/('profile-summary-before-'+v+'.json')).write_bytes((R/'profile-summary.json').read_bytes());prof=read(R/'profile-summary.json');prof['supplementary_'+v]=dict(summary=cp['summary'],model_candidates=len(cm),framework_candidates=len(fw['candidates']),whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
    for q in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);d.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action'if q==R/'STATE.json'else'next_action']=action
        if q==W/'STATE.json':d.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
        if q in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[str(summary_path)]+[str(O/q)for q in read_first]+[cp['coverage_gaps']]
        write(q,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/('task-run-registration-'+v+'.json'),dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    artifacts=[q for q in O.glob('*'+v+'*')if q.is_file()and not q.name.startswith('checkpoint-manifest')]+scripts+[main_script,N/'persist_profile_refinement.py',N/'persist_profile_refinement_v2.py']
    artifacts += [q for q in N.glob(executor_prefix+'*'+v+'*20261031*')if q.is_file()]+[R/'candidates'/(cid+'.json')for cid in candidate_updates]+[pub,N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
    for job in jobs:artifacts+=job['artifacts']
    artifacts=sorted(set(artifacts));write(O/('checkpoint-manifest-'+v+'.json'),dict(generated_utc=stamp,artifacts=[dict(path=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())for q in artifacts],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/('REVIEW-CHECKPOINT-'+v+'.json')),manifest_files=len(artifacts),runs=len(runs),pass_runs=sum(r['status']=='pass'for r in runs),retained_error_runs=sum(r['status']=='error'for r in runs),model_candidates=len(cm),framework_candidates=len(fw['candidates']),validation=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))
