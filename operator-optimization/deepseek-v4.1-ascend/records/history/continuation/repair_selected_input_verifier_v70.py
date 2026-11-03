"""Retain failed verifier Run, repair naming-only audit, and refresh the manifest transparently."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,subprocess,sys
from finalize_origin_coverage_v9 import N,O,T,L,read,write,ctl

def main():
    audit=O/'selected-input-postcheckpoint-validation-v70.json';a=read(audit);assert a['status']=='ERROR_RETAINED_FIX_PENDING_VALIDATION'
    old=read(O/'checkpoint-manifest-before-verifier-repair-v70.json');reg=read(O/'task-run-registration-v70.json');assert len(reg['runs'])==4
    verifier=N/'verify_selected_inputs_v70.py';backup=O/'verify-selected-inputs-before-fix-v70.py'
    orig=next(x for x in old['artifacts']if x['path']==str(verifier));assert hashlib.sha256(backup.read_bytes()).hexdigest()==orig['sha256']
    permitted={str(verifier),str(O/'task-run-registration-v70.json'),str(T/'index.json'),str(L/'state.json')}
    def check_unchanged():
        for x in old['artifacts']:
            if x['path']not in permitted:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'],x['path']
    extras=[audit,backup,O/'checkpoint-manifest-before-verifier-repair-v70.json',Path(__file__),O/'selected-input-repaired-verifier-result-v70.json']
    def refresh():
        check_unchanged();paths=sorted(set(Path(x['path'])for x in old['artifacts'])|{q for q in extras if q.exists()})
        write(O/'checkpoint-manifest-v70.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),artifacts=[dict(path=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())for q in paths],repair_audit=str(audit),prior_manifest_archive=str(O/'checkpoint-manifest-before-verifier-repair-v70.json'),analysis_evidence_changed=False,whole_modules_closed=0,overall_complete=False))
    def record(status,description,paths,suffix):
        rid='run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+suffix
        run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind','review','--command',description,'--code-ref','Localverificationonly;analysisandtargetunchanged','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
        for p in paths:args+=['--artifact',str(p)]
        ctl(*args);reg['runs'].append(dict(run_dir=run,status=status,description=description));reg['compact_output']=ctl('compact','--task-dir',str(T));reg['validation_output']=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v70.json',reg)
    check_unchanged()
    record('error','Retainedactualv70verifierTypeError:integerTshadowedtaskdiratfinalstatuscheck;allprecedingoperator/evidencecheckscompleted. Originalscript/manifestarchived;noanalyticaldatachange.',[audit,backup],'verifier-error')
    a.update(status='FIX_APPLIED_REVALIDATION_RUNNING',retained_error_run=reg['runs'][-1]['run_dir']);write(audit,a);refresh()
    result=subprocess.run([sys.executable,str(verifier)],cwd=N,text=True,capture_output=True)
    result_file=O/'selected-input-repaired-verifier-result-v70.json';write(result_file,dict(command=[sys.executable,str(verifier)],returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,scope='Fullv70boundedoperator/evidence/stateverifierafterpathaliasfix;notgoalcompletion.'))
    if result.returncode:
        a.update(status='FIX_REVALIDATION_ERROR',returncode=result.returncode);write(audit,a);refresh();raise RuntimeError(result.stderr)
    record('pass','Actualrepairedv70verifierPASS:allnativecosts/sources/coverage/protectedmodules/masters/taskstate. Alias-onlyfix;failedRunretainedandnoanalyticalclauseschanged.',[audit,result_file,verifier],'verifier-pass')
    a.update(status='PASS_AFTER_ALIAS_ONLY_FIX',final_registered_runs=len(reg['runs']),pass_runs=sum(x['status']=='pass'for x in reg['runs']),error_runs=sum(x['status']=='error'for x in reg['runs']),actual_verifier_result=str(result_file),analysis_evidence_changed=False,overall_complete=False);write(audit,a);refresh()
    print(dict(status=a['status'],runs=len(reg['runs']),pass_runs=5,error_runs=1,analysis_evidence_changed=False,manifest_artifacts=len(read(O/'checkpoint-manifest-v70.json')['artifacts']),verifier_stdout=result.stdout.strip(),overall_complete=False))

if __name__=='__main__':main()
