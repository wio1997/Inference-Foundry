"""Renew only the same UID-owned finite reservations for the publication prototype."""
from pathlib import Path
import datetime
import importlib.util
import json
import shlex
import subprocess
r=Path(__file__).resolve().parent;repo=r.parents[3]
assert json.loads((r/'early-sample-shadow-output-fence-v1.json').read_text())['completed']
ssh=['ssh','-S','/tmp/glm53-active-20261019.sock','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
spec=importlib.util.spec_from_file_location('guard',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
out=r/'early-sample-publication-holder-renewal-v1.json';assert not out.exists();rows=[]
for role,node in [('D',33),('P',32)]:
    plan=json.loads((r/('early-sample-shadow-'+role+'-plan-v1.json')).read_text());res=plan['reservation']
    def read(command):return json.loads(subprocess.check_output(ssh+[command],timeout=30))
    pod=read('kubectl get pod -n default '+res['name']+' -o json')
    assert pod['metadata']['uid']==res['expected_uid'] and pod['metadata']['labels']['owner']==res['owner_label']
    assert pod['spec']['nodeName']==res['node_name'] and pod['status']['phase']=='Running' and not pod['metadata'].get('ownerReferences')
    assert pod['spec']['activeDeadlineSeconds']==10800
    pods=read('kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json')['items']
    active=[z['metadata']['uid'] for z in pods if z['status']['phase'] not in ['Failed','Succeeded'] and any(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in z['spec']['containers'])]
    assert active==[res['expected_uid']]
    patch=[dict(op='test',path='/metadata/uid',value=res['expected_uid']),dict(op='test',path='/spec/activeDeadlineSeconds',value=10800),dict(op='replace',path='/spec/activeDeadlineSeconds',value=18000)]
    q=subprocess.run(ssh+['kubectl patch pod -n default '+res['name']+' --type=json -p '+shlex.quote(json.dumps(patch))+' -o json'],capture_output=True,text=True,timeout=30)
    row=dict(role=role,node=node,UID=res['expected_uid'],before_deadline_s=10800,requested_deadline_s=18000,rc=q.returncode)
    if q.returncode:
        row['error']=q.stderr;rows.append(row);out.write_text(json.dumps(dict(completed=False,rows=rows),indent=2)+'\n');raise AssertionError(q.stderr)
    updated=json.loads(q.stdout);assert updated['metadata']['uid']==pod['metadata']['uid'] and updated['spec']['activeDeadlineSeconds']==18000
    uid,allowed,expiry=g.fresh_lease(updated,res,datetime.datetime.now(datetime.timezone.utc));assert allowed>7200
    row.update(after_deadline_s=18000,current_allowed_model_s=allowed,expiry=expiry)
    (r/('early-sample-publication-'+role+'-lease'+str(node)+'-v1.json')).write_text(json.dumps(updated,indent=2)+'\n')
    rows.append(row)
out.write_text(json.dumps(dict(completed=True,rows=rows,scope='Same task-owned holder UID, fixed5h total deadline from original start. No resources added, no model timer extended, no other job or access/system pod touched.'),indent=2)+'\n')
print(json.dumps(rows))
