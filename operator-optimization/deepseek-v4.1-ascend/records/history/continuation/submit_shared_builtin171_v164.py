"""Submit only after the separate user-approved full-model receipt exists."""
import sys,json,hashlib
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
authpath=O/'shared-builtin-authorization-v164.json'
assert authpath.exists(),'Full-model window is not authorized by the consumed small-probe grant'
auth=read(authpath);review=read(O/'shared-builtin-preparation-review-v164.json');prep=read(O/'shared-builtin-prepared-v164.json')
assert auth['status']=='AUTHORIZED_ONCE' and auth['user_reply']
assert auth['plan_sha256']==review['plan_sha256'] and auth['prepared_sha256']==review['prepared_sha256']
submission=O/'shared-builtin-submission-v164.json';assert not submission.exists(),'Inspect the one existing submission/handle;do not resubmit on observation timeout'
write(submission,{'state':'ONE_FULL_COHORT_SUBMISSION_ATTEMPT','authorization':pin(authpath),
 'root':prep['root'],'observation_timeout_does_not_authorize_resubmission':True})
code='ROOT='+repr(prep['root'])+'\nAUTH='+repr(auth)+'\n'+r'''import json,hashlib,os,subprocess,socket
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
r=Path(ROOT);c=r/'control';assert not (r/'STATUS.json').exists() and not (c/'SUBMITTED.json').exists()
assert hashlib.sha256((c/'PLAN.json').read_bytes()).hexdigest()==AUTH['plan_sha256']
assert hashlib.sha256((c/'PREPARED.json').read_bytes()).hexdigest()==AUTH['prepared_sha256']
p=c/'APPROVAL.json';assert not p.exists();p.write_text(json.dumps(AUTH,ensure_ascii=False,indent=2)+'\n')
fd=os.open(c/'SUBMITTED.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
with (r/'controller.log').open('xb') as log:
 child=subprocess.Popen(['python3','-I','-S',str(r/'controller.py'),'--root',str(r),'--execute'],
  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
ticks=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
out={'state':'SUBMITTED_ONE_FULL_COHORT','pid':child.pid,'proc_start_ticks':ticks,'root':ROOT,
 'status_file':str(r/'STATUS.json'),'controller_source':str(r/'controller.py'),
 'poll_same_handle_never_resubmit_from_timeout':True}
(c/'SUBMITTED.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))'''
result=invoke('target',code,30)
write(O/'shared-builtin-controller-submission-v164.json',result)
print(json.dumps(result,ensure_ascii=False))
