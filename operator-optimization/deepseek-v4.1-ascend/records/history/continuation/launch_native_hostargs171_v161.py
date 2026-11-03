"""Single explicit user-authorized joint probe; never resubmit an existing window."""
import json,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke

prep=read(O/'native-hostargs-probe-preparation-v161.json')
authpath=O/'native-hostargs-authorization-v161.json';submission=O/'native-hostargs-submission-v161.json'
assert not authpath.exists() and not submission.exists(),'Existing authorization/submission must be inspected;do not repeat'
grant={'status':'AUTHORIZED_ONCE','user_reply':'允许本次探针并恢复服务','approval_id':'native-pad-171-'+uuid.uuid4().hex,'bundle':prep['bundle_inside'],'probe_source_sha256':prep['probe_source_sha256'],'controller_source_sha256':prep['controller_source_sha256'],'authorized_utc':datetime.now(timezone.utc).isoformat(),'plan':prep['plan'],'own_service_stop_and_restore_authorized':True,'other_tasks_must_not_be_changed':True,'at_most_one_native_attempt':True}
write(authpath,dict(grant,prepared=pin(O/'native-hostargs-probe-preparation-v161.json')))
write(submission,{'state':'ONE_WINDOW_SUBMISSION_ATTEMPT','authorization':pin(authpath),'server_status':'/data/tiankuan/wio/migration-171-20261102/operator-evidence-v161/NATIVE-STATUS.json','observation_timeout_is_not_terminal_or_resubmission_permission':True})
code='GRANT = '+repr(grant)+'\n'+r'''import json,subprocess,hashlib,os,socket
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
r=Path('/data/tiankuan/wio/migration-171-20261102/operator-evidence-v161')
assert not (r/'APPROVAL.json').exists() and not (r/'NATIVE-STATUS.json').exists()
assert hashlib.sha256((r/'controller.py').read_bytes()).hexdigest()==GRANT['controller_source_sha256']
p=json.loads((r/'PREPARED-PROBE.json').read_text());assert p['probe_source_sha256']==GRANT['probe_source_sha256'] and p['bundle_inside']==GRANT['bundle']
assert hashlib.sha256((Path(p['bundle_host'])/'probe.py').read_bytes()).hexdigest()==GRANT['probe_source_sha256']
(r/'APPROVAL.json').write_text(json.dumps(GRANT,indent=2))
with (r/'controller.log').open('xb') as log:
 child=subprocess.Popen(['python3','-I','-S',str(r/'controller.py'),'--execute'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'state':'OWNED_CONTROLLER_STARTED','pid':child.pid,'status':str(r/'NATIVE-STATUS.json'),'log':str(r/'controller.log'),'approval_id':GRANT['approval_id']}
(r/'CONTROLLER-SUBMISSION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
d=invoke('target',code,30);write(O/'native-hostargs-controller-submission-v161.json',d);print(json.dumps(d))
