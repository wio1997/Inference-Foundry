"""Rebuild only task-owned finite holders after exact model snapshot and stop."""
from pathlib import Path
import copy
import datetime
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys

r=Path(__file__).resolve().parent;repo=r.parents[3]
control='/tmp/glm53-active-20261019.sock'
ssh=['ssh','-S',control,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']
spec=importlib.util.spec_from_file_location('transport',r/'run_prefix_count_service_v1.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec=importlib.util.spec_from_file_location('guard',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def read(command):return json.loads(subprocess.check_output(ssh+[command],timeout=30))
def active(node):
    pods=read('kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-'+str(node)+' -o json')['items']
    return [z for z in pods if z['status']['phase'] not in ['Failed','Succeeded'] and any(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in z['spec']['containers'])]
check=json.loads((r/'early-sample-publication-build-check-v3.json').read_text());assert check['passed']==22
build=json.loads((r/'early-sample-publication-build-manifest-v3.json').read_text());assert build['identity']==check['build']
assert json.loads(next((r/'early-sample-publication-stage-v3-33').glob('*/metrics.json')).read_text())['root']==build['remote_root']
assert json.loads((r/'early-sample-shadow-reviewed-v1.json').read_text())['bitwise_all']
assert json.loads((r/'early-sample-shadow-output-fence-v1.json').read_text())['observed_shadow_steps']==7
full=json.loads(next((r/'early-sample-shadow-c4-export-v1').glob('*/artifacts-backup.json')).read_text())
assert hashlib.sha256(Path(full['local_archive']).read_bytes()).hexdigest()==full['sha256']
old_pods={}
for role,node in [('D',33),('P',32)]:
    run=next((r/('early-sample-shadow-snapshot-v1-'+str(node))).glob('*/result.json')).parent
    subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(run),'--control',control,
                    '--backup-root','/Users/wio/work/glm53-execution-evidence-backup/20261019-early-sample-shadow-final-snapshot'],check=True)
    receipt=json.loads((run/'artifacts-backup.json').read_text());assert hashlib.sha256(Path(receipt['local_archive']).read_bytes()).hexdigest()==receipt['sha256']
    plan=json.loads((r/('early-sample-shadow-'+role+'-plan-v1.json')).read_text());res=plan['reservation']
    pod=read('kubectl get pod -n default '+res['name']+' -o json')
    assert pod['metadata']['uid']==res['expected_uid'] and pod['metadata']['labels']['owner']==res['owner_label']
    assert not pod['metadata'].get('ownerReferences') and [z['metadata']['uid'] for z in active(node)]==[res['expected_uid']]
    old_pods[node]=pod

for node in [33,32]:
    x=m.collect(r/'stop_early_shadow_for_publication_v1.py','early_sample_shadow_stop_for_publication_v1',
                'early-sample-shadow-stop-for-publication-v1-'+str(node),timeout=120,node=node)
    assert x['empty_physical_cards']==8
    pod=old_pods[node];name=pod['metadata']['name'];uid=pod['metadata']['uid']
    patch=[dict(op='test',path='/metadata/uid',value=uid),dict(op='test',path='/spec/activeDeadlineSeconds',value=10800),
           dict(op='replace',path='/spec/activeDeadlineSeconds',value=1)]
    q=subprocess.run(ssh+['kubectl patch pod -n default '+name+' --type=json -p '+shlex.quote(json.dumps(patch))+' -o json'],capture_output=True,text=True,timeout=30)
    assert q.returncode==0,q.stderr
    subprocess.run(ssh+["kubectl wait -n default --for=jsonpath='{.status.phase}'=Failed pod/"+name+' --timeout=45s'],check=True,capture_output=True,timeout=55)
    expired=read('kubectl get pod -n default '+name+' -o json')
    assert expired['metadata']['uid']==uid and expired['status']['phase']=='Failed' and not active(node)
    (r/('early-sample-publication-old-holder-expired-'+str(node)+'-v1.json')).write_text(json.dumps(expired,indent=2)+'\n')

for role,node in [('D',33),('P',32)]:
    old=old_pods[node];name='glm53-early-pub-'+str(node)+'-live1-20261019';owner='codex-glm53-early-pub-live1-20261019'
    assert not active(node)
    assert not subprocess.check_output(ssh+['kubectl get pod -n default '+name+' --ignore-not-found -o json'],timeout=25).strip()
    container={k:copy.deepcopy(old['spec']['containers'][0][k]) for k in ['name','image','command','args','resources','securityContext'] if k in old['spec']['containers'][0]}
    container['imagePullPolicy']='Never'
    spec=dict(nodeName=old['spec']['nodeName'],restartPolicy='Never',activeDeadlineSeconds=10800,terminationGracePeriodSeconds=10,
              automountServiceAccountToken=False,containers=[container])
    for key in ['tolerations','securityContext','schedulerName']:
        if key in old['spec']:spec[key]=copy.deepcopy(old['spec'][key])
    doc=dict(apiVersion='v1',kind='Pod',metadata=dict(name=name,namespace='default',labels=dict(app='glm53-early-pub',owner=owner)),spec=spec)
    q=subprocess.run(ssh+['kubectl create -f - -o json'],input=json.dumps(doc),capture_output=True,text=True,timeout=30);assert q.returncode==0,q.stderr
    subprocess.run(ssh+['kubectl wait -n default --for=condition=Ready pod/'+name+' --timeout=45s'],check=True,capture_output=True,timeout=55)
    lease=read('kubectl get pod -n default '+name+' -o json')
    assert lease['status']['hostIP']=='172.16.12.'+str(node) and not lease['metadata'].get('ownerReferences')
    (r/('early-sample-publication-'+role+'-lease'+str(node)+'-v1.json')).write_text(json.dumps(lease,indent=2)+'\n')
    plan=json.loads((r/('early-sample-shadow-'+role+'-plan-v1.json')).read_text());res=plan['reservation']
    res.update(name=name,owner_label=owner,expected_uid=lease['metadata']['uid'])
    uid,allowed,expiry=g.fresh_lease(lease,res,datetime.datetime.now(datetime.timezone.utc));assert allowed>=7200
    assert [z['metadata']['uid'] for z in active(node)]==[uid]
    access={32:'sglang-node-access-32',33:'sglang-tuning-access-33-20261013'}[node]
    smi=subprocess.check_output(ssh+['kubectl exec -n default '+access+' -- nsenter -t 1 -m -- npu-smi info'],text=True,timeout=40)
    assert smi.count('No running processes found in NPU')==8
    plan['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live1-20261019/node'+str(node)
    plan['container'].update(name='glm53-early-publication-'+str(node)+'-live1-20261019',lifetime_seconds=7200)
    plan['env'].update(GLM53_EARLY_SAMPLE_COPY_SHADOW='0',GLM53_EARLY_SAMPLE_PUBLICATION='1' if node==33 else '0')
    if node==33:plan['runtime_patches']=build['patches']
    argv=plan['serve_argv'];i=argv.index('--kv-transfer-config')+1;kv=json.loads(argv[i]);kv['engine_id']='glm53-early-publication-'+role.lower()+'-live1-20261019';argv[i]=json.dumps(kv,separators=(',',':'))
    plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,
                objective='Finite native PD comparison of early verified sample publication with separate all16 GPU retirement',
                notes=['Existing model/operators/precision/SDMA and accepted RoPE in both arms; no parameter search.',
                       'New owned10800s holder and7200s model. Original holders expired by UID-tested deadline reduction after exact snapshot/idle stop.',
                       'D immutable publication build4bfd, native MoE graph. Native InputBatch late event and GPU source retained; zero-token and CoW drain acknowledged by all16.'])
    target=r/('early-sample-publication-'+role+'-plan-v1.json');assert not target.exists();target.write_text(json.dumps(plan,indent=2)+'\n')
    subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/start_capacity.py'),'--plan',str(target),'--control',control,
                    '--output',str(r/('early-sample-publication-'+role+'-launch-v1'))],check=True)
    print(json.dumps(dict(role=role,node=node,holder_uid=uid,model_lifetime_s=7200)),flush=True)
