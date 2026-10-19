"""Finite fresh native P and guarded K-graph D, only owned 32/33 resources."""
from pathlib import Path
import copy
import datetime
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys

r=Path(__file__).resolve().parent
repo=r.parents[3]


def module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


m=module('transport',r/'run_prefix_count_service_v1.py')
g=module('guard',repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
ssh=['ssh','-S',m.control,'-oBatchMode=yes','-oStrictHostKeyChecking=yes','-p','60011','root@61.241.77.34']


def read(command):
    return json.loads(subprocess.check_output(ssh+[command],timeout=30))


def active(node):
    return [p for p in read('kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-'+str(node)+' -o json')['items']
            if p['status']['phase'] not in ['Failed','Succeeded'] and any(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in p['spec']['containers'])]


build=json.loads((r/'indexer-k-graph-overlap-build-manifest-v2.json').read_text())
check=json.loads((r/'indexer-k-graph-overlap-build-check-v2.json').read_text())
assert build['identity']==check['build'] and check['CPU_checks']==16
assert json.loads((r/'indexer-K-graph-copy-SDMA-fragment-v2.json').read_text())['all_tagged_copies_SDMA']
assert json.loads((r/'indexer-K-graph-overlap-fragment-reviewed-v1.json').read_text())['all_changing_input_checks']
assert (r/'indexer-k-graph-overlap-service-contract-v1.json').exists()
for node in [32,33]:
    p=next((r/f'rope-step-reuse-closed-node{node}-v1').glob('*/artifacts-backup.json'))
    rec=json.loads(p.read_text())
    assert hashlib.sha256(Path(rec['local_archive']).read_bytes()).hexdigest()==rec['sha256']

# P has naturally reached its original watchdog. Capture final logs/state before renewal.
x=m.collect(r/'snapshot_rope_step_reuse_closed_v1.py','indexer_graph_P_final_snapshot_v1','indexer-graph-P-final-snapshot-v1',timeout=120,node=32)
run=next((r/'indexer-graph-P-final-snapshot-v1').glob('*/result.json')).parent
subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/backup_artifacts.py'),'--run',str(run),'--control',m.control,'--backup-root','/Users/wio/work/glm53-execution-evidence-backup/20261019-indexer-K-graph-before-launch'],check=True)
rec=json.loads((run/'artifacts-backup.json').read_text())
assert hashlib.sha256(Path(rec['local_archive']).read_bytes()).hexdigest()==rec['sha256']

for node in [32,33]:
    x=m.collect(r/'check_indexer_launch_idle_v1.py','indexer_graph_idle_node'+str(node)+'_v1','indexer-graph-idle-node'+str(node)+'-v1',timeout=90,node=node)
    assert x['empty_physical_cards']==8 and not x['model_running']

P=json.loads((r/'rope-step-reuse-P-plan-v1.json').read_text())
old_res=P['reservation']
old=read('kubectl get pod -n default '+old_res['name']+' -o json')
assert old['metadata']['uid']==old_res['expected_uid'] and old['metadata']['labels']['owner']==old_res['owner_label']
assert not old['metadata'].get('ownerReferences')
assert [p['metadata']['uid'] for p in active(32)] in [[],[old_res['expected_uid']]]
if old['status']['phase']=='Running':
    patch=[dict(op='test',path='/metadata/uid',value=old_res['expected_uid']),dict(op='test',path='/spec/activeDeadlineSeconds',value=10800),dict(op='replace',path='/spec/activeDeadlineSeconds',value=1)]
    subprocess.run(ssh+['kubectl patch pod -n default '+old_res['name']+' --type=json -p '+shlex.quote(json.dumps(patch))],check=True,capture_output=True,timeout=30)
    subprocess.run(ssh+["kubectl wait -n default --for=jsonpath='{.status.phase}'=Failed pod/"+old_res['name']+' --timeout=45s'],check=True,capture_output=True,timeout=55)
else:
    assert old['status']['phase'] in ['Failed','Succeeded']
assert not active(32)
name='glm53-indexer-k-graph-32-live1-20261019'
owner='codex-glm53-indexer-prolog-20261019'
assert not subprocess.check_output(ssh+['kubectl get pod -n default '+name+' --ignore-not-found -o json'],timeout=25).strip()
container={k:copy.deepcopy(old['spec']['containers'][0][k]) for k in ['name','image','command','args','resources','securityContext'] if k in old['spec']['containers'][0]}
container['imagePullPolicy']='Never'
pod_spec=dict(nodeName=old['spec']['nodeName'],restartPolicy='Never',activeDeadlineSeconds=10800,terminationGracePeriodSeconds=10,automountServiceAccountToken=False,containers=[container])
for key in ['tolerations','securityContext','schedulerName']:
    if key in old['spec']:
        pod_spec[key]=copy.deepcopy(old['spec'][key])
doc=dict(apiVersion='v1',kind='Pod',metadata=dict(name=name,namespace='default',labels=dict(app='glm53-indexer-k-graph',owner=owner)),spec=pod_spec)
subprocess.run(ssh+['kubectl create -f - -o json'],input=json.dumps(doc),check=True,capture_output=True,text=True,timeout=30)
subprocess.run(ssh+['kubectl wait -n default --for=condition=Ready pod/'+name+' --timeout=45s'],check=True,capture_output=True,timeout=55)
lease=read('kubectl get pod -n default '+name+' -o json')
assert lease['status']['hostIP']=='172.16.12.32' and not lease['metadata'].get('ownerReferences')
P['reservation']=dict(old_res,name=name,owner_label=owner,expected_uid=lease['metadata']['uid'])
D=json.loads((r/'rope-step-reuse-D-plan-v1.json').read_text())
D['reservation']=json.loads((r/'indexer-prolog-fragment-ownership-v1.json').read_text())['reservation']
D['runtime_patches']=build['patches']
stage=m.collect(Path('/Users/wio/work/glm53-execution-evidence-backup/20261019-indexer-K-graph-build-v2/stage_overlay.py'),'indexer_k_graph_stage_v2','indexer-k-graph-stage-v2',timeout=90,node=33)
assert stage['root']==build['remote_root']

for role,node,plan in [('D',33,D),('P',32,P)]:
    res=plan['reservation']
    lease=read('kubectl get pod -n default '+res['name']+' -o json')
    uid,allowed,expiry=g.fresh_lease(lease,res,datetime.datetime.now(datetime.timezone.utc))
    assert not lease['metadata'].get('ownerReferences') and [p['metadata']['uid'] for p in active(node)]==[uid]
    plan['remote']['run_dir']='/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live1-20261019/node'+str(node)
    plan['container'].update(name='glm53-indexer-k-graph-'+str(node)+'-live1-20261019',lifetime_seconds=min(7200,allowed-60))
    assert plan['container']['lifetime_seconds']>=6600
    plan['env'].update(GLM53_ROPE_STEP_REUSE_PROBE='0',GLM53_INDEXER_K_GRAPH_OVERLAP='1' if node==33 else '0',GLM53_EARLY_SAMPLE_PUBLICATION='0',GLM53_EARLY_SAMPLE_COPY_SHADOW='0')
    argv=plan['serve_argv'];i=argv.index('--kv-transfer-config')+1;kv=json.loads(argv[i]);kv['engine_id']='glm53-indexer-k-graph-'+role.lower()+'-live1-20261019';argv[i]=json.dumps(kv,separators=(',',':'))
    plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),holder_expiry_utc=expiry,
                objective='Finite PD test of native MTP eager K-only graph overlap; target FULL and all cache/publication fences retained',
                notes=['Same W8A8C8 and original per-layer int8/BF16 Indexer, operators and SDMA. No parameter sweep.',
                       'D new immutable helper build705f9e704c26, default marker absent. P same original native immutable overlays.',
                       'Original P/proxy reached finite watchdog; D snapshot-checked exact stop for independent fragment. Only owned32/33 holders renewed; access pods and other jobs untouched.'])
    target=r/('indexer-k-graph-'+role+'-plan-v1.json')
    assert not target.exists()
    target.write_text(json.dumps(plan,indent=2)+'\n')
    subprocess.run([sys.executable,str(repo/'glm5-3-vllm/tools/start_capacity.py'),'--plan',str(target),'--control',m.control,'--output',str(r/('indexer-k-graph-'+role+'-launch-v1'))],check=True)
    print(json.dumps(dict(role=role,node=node,holder_UID=uid,model_lifetime_s=plan['container']['lifetime_seconds'],build=build['identity'] if node==33 else 'unchanged-native-P')),flush=True)
