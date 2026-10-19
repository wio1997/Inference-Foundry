"""Snapshot-checked exact D stop and renewal of only this task's D holder."""
from pathlib import Path
import copy
import datetime
import hashlib
import importlib.util
import json
import shlex
import subprocess

r = Path(__file__).resolve().parent
repo = r.parents[3]
spec = importlib.util.spec_from_file_location('transport', r/'run_prefix_count_service_v1.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ssh = ['ssh', '-S', m.control, '-oBatchMode=yes', '-p', '60011', 'root@61.241.77.34']


def read(command):
    return json.loads(subprocess.check_output(ssh+[command], timeout=30))


def active():
    pods = read('kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-33 -o json')['items']
    return [p for p in pods if p['status']['phase'] not in ['Failed', 'Succeeded'] and any(
        int(c.get('resources', {}).get('requests', {}).get('huawei.com/Ascend910', 0))
        for c in p['spec']['containers'])]


for node in [32, 33]:
    receipt = json.loads(next((r/f'rope-step-reuse-closed-node{node}-v1').glob('*/artifacts-backup.json')).read_text())
    data = Path(receipt['local_archive']).read_bytes()
    assert len(data) == receipt['bytes'] and hashlib.sha256(data).hexdigest() == receipt['sha256']

plan = json.loads((r/'rope-step-reuse-D-plan-v1.json').read_text())
res = plan['reservation']
old = read('kubectl get pod -n default '+res['name']+' -o json')
assert old['metadata']['uid'] == res['expected_uid']
assert old['metadata']['labels']['owner'] == res['owner_label']
assert not old['metadata'].get('ownerReferences')
assert old['status']['phase'] == 'Running' and old['status']['hostIP'] == '172.16.12.33'
assert [p['metadata']['uid'] for p in active()] == [res['expected_uid']]
start = datetime.datetime.fromisoformat(old['status']['startTime'].replace('Z', '+00:00'))
left = old['spec']['activeDeadlineSeconds'] - (datetime.datetime.now(datetime.timezone.utc)-start).total_seconds()
assert left > 180
stop = m.collect(r/'stop_reuse_D_for_indexer_fragment_v1.py', 'stop_reuse_D_for_indexer_fragment_v1', 'indexer-prolog-D-stop-v1', timeout=110)
assert stop['completed'] and stop['empty_physical_cards'] == 8
print(json.dumps(stop), flush=True)

patch = [dict(op='test', path='/metadata/uid', value=res['expected_uid']),
         dict(op='test', path='/spec/activeDeadlineSeconds', value=10800),
         dict(op='replace', path='/spec/activeDeadlineSeconds', value=1)]
subprocess.run(ssh+['kubectl patch pod -n default '+res['name']+' --type=json -p '+shlex.quote(json.dumps(patch))], check=True, capture_output=True, timeout=30)
subprocess.run(ssh+["kubectl wait -n default --for=jsonpath='{.status.phase}'=Failed pod/"+res['name']+' --timeout=45s'], check=True, capture_output=True, timeout=55)
expired = read('kubectl get pod -n default '+res['name']+' -o json')
assert expired['metadata']['uid'] == res['expected_uid'] and expired['status']['phase'] == 'Failed'
assert not active()
name = 'glm53-indexer-prolog-33-live1-20261019'
owner = 'codex-glm53-indexer-prolog-20261019'
assert not subprocess.check_output(ssh+['kubectl get pod -n default '+name+' --ignore-not-found -o json'], timeout=25).strip()
container = {k: copy.deepcopy(old['spec']['containers'][0][k]) for k in ['name', 'image', 'command', 'args', 'resources', 'securityContext'] if k in old['spec']['containers'][0]}
container['imagePullPolicy'] = 'Never'
pod_spec = dict(nodeName=old['spec']['nodeName'], restartPolicy='Never', activeDeadlineSeconds=10800,
                terminationGracePeriodSeconds=10, automountServiceAccountToken=False, containers=[container])
for key in ['tolerations', 'securityContext', 'schedulerName']:
    if key in old['spec']:
        pod_spec[key] = copy.deepcopy(old['spec'][key])
doc = dict(apiVersion='v1', kind='Pod', metadata=dict(name=name, namespace='default', labels=dict(app='glm53-indexer-prolog', owner=owner)), spec=pod_spec)
subprocess.run(ssh+['kubectl create -f - -o json'], input=json.dumps(doc), check=True, capture_output=True, text=True, timeout=30)
subprocess.run(ssh+['kubectl wait -n default --for=condition=Ready pod/'+name+' --timeout=45s'], check=True, capture_output=True, timeout=55)
lease = read('kubectl get pod -n default '+name+' -o json')
assert lease['status']['hostIP'] == '172.16.12.33' and not lease['metadata'].get('ownerReferences')
new_res = dict(res, name=name, owner_label=owner, expected_uid=lease['metadata']['uid'])
spec = importlib.util.spec_from_file_location('guard', repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
uid, allowed, expiry = g.fresh_lease(lease, new_res, datetime.datetime.now(datetime.timezone.utc))
assert allowed >= 7200 and [p['metadata']['uid'] for p in active()] == [uid]
receipt = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), node=33,
               old_holder_UID=res['expected_uid'], old_holder_phase=expired['status']['phase'],
               reservation=new_res, holder_remaining_s=allowed, holder_expiry_utc=expiry,
               holder_lifetime_s=10800, exclusive=True, model_lifetime_not_extended=True,
               stopped_exact_D_CID=stop['stopped_exact_D_CID'], P_untouched=True,
               contract_sha256=hashlib.sha256((r/'indexer-prolog-overlap-fragment-contract-v1.json').read_bytes()).hexdigest())
(r/'indexer-prolog-fragment-ownership-v1.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt), flush=True)
x = m.collect(r/'run_indexer_prolog_fragment_v1.py', 'indexer_prolog_fragment_v1', 'indexer-prolog-fragment-v1', timeout=440)
print(json.dumps(x), flush=True)
