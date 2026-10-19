"""Fresh finite own P/D, after successful full NZ fragment; immutable overlays."""
from pathlib import Path
import copy, datetime, hashlib, importlib.util, json, subprocess, sys

r = Path(__file__).resolve().parent
repo = r.parents[3]
control = '/tmp/glm53-active-20261019.sock'
ssh = ['ssh', '-S', control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-p', '60011', 'root@61.241.77.34']
assert Path(control).is_socket()
fragment = json.loads((r/'prefix-count-fragment-reviewed-v1.json').read_text())
assert fragment['passed'] and fragment['cases'] == 8 and fragment['replays'] == 96 and fragment['all_cases_median_improved']
backup = json.loads(next((r/'prefix-count-full-fragment-v1').glob('*/artifacts-backup.json')).read_text())
assert hashlib.sha256(Path(backup['local_archive']).read_bytes()).hexdigest() == backup['sha256']
build = json.loads((r/'prefix-count-build-manifest-v1.json').read_text())
assert json.loads(next((r/'prefix-count-stage-v1-33').glob('*/metrics.json')).read_text())['root'] == build['remote_root']
lease_spec = importlib.util.spec_from_file_location('lease_guard', repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
lease_guard = importlib.util.module_from_spec(lease_spec); lease_spec.loader.exec_module(lease_guard)

def read(command):
    return json.loads(subprocess.check_output(ssh+[command], timeout=30))

def active(node):
    return [z for z in read('kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-'+str(node)+' -o json')['items']
            if z['status']['phase'] not in ['Failed', 'Succeeded'] and any(int(c.get('resources', {}).get('requests', {}).get('huawei.com/Ascend910', 0)) for c in z['spec'].get('containers', []))]

# No eviction is needed: old holders and exact old model containers have expired.
assert not active(32)
assert subprocess.check_output(ssh+['kubectl exec -n default sglang-node-access-32 -- nsenter -t 1 -m -- npu-smi info'], text=True, timeout=40).count('No running processes found in NPU') == 8
old = json.loads((r/'cumsum-i32-P-live2-lease32.json').read_text())
name = 'glm53-prefix-count-32-live1-20261019'
owner = 'codex-glm53-prefix-count-P-live1-20261019'
assert not (r/'prefix-count-P-lease32-v1.json').exists()
assert not subprocess.check_output(ssh+['kubectl get pod -n default '+name+' --ignore-not-found -o json'], timeout=25).strip()
container = {k: copy.deepcopy(old['spec']['containers'][0][k]) for k in ['name', 'image', 'command', 'args', 'resources', 'securityContext'] if k in old['spec']['containers'][0]}
container['imagePullPolicy'] = 'Never'
spec = dict(nodeName=old['spec']['nodeName'], restartPolicy='Never', activeDeadlineSeconds=10800, terminationGracePeriodSeconds=10, automountServiceAccountToken=False, containers=[container])
for key in ['tolerations', 'securityContext', 'schedulerName']:
    if key in old['spec']: spec[key] = copy.deepcopy(old['spec'][key])
doc = dict(apiVersion='v1', kind='Pod', metadata=dict(name=name, namespace='default', labels=dict(app='glm53-prefix-count', owner=owner)), spec=spec)
q = subprocess.run(ssh+['kubectl create -f - -o json'], input=json.dumps(doc), text=True, capture_output=True, timeout=30)
assert q.returncode == 0, q.stderr
subprocess.run(ssh+['kubectl wait --for=condition=Ready pod/'+name+' -n default --timeout=60s'], check=True, capture_output=True, timeout=75)
pod = read('kubectl get pod -n default '+name+' -o json')
assert pod['status']['hostIP'] == '172.16.12.32' and not pod['metadata'].get('ownerReferences')
(r/'prefix-count-P-lease32-v1.json').write_text(json.dumps(pod, indent=2)+'\n')

for role, node, template in [('D', 33, 'cumsum-i32-D-plan.json'), ('P', 32, 'cumsum-i32-P-live2-plan.json')]:
    plan = json.loads((r/template).read_text())
    lease = json.loads((r/('prefix-count-'+role+'-lease'+str(node)+'-v1.json')).read_text())
    current = read('kubectl get pod -n default '+lease['metadata']['name']+' -o json')
    assert current['metadata']['uid'] == lease['metadata']['uid'] and not current['metadata'].get('ownerReferences')
    res = plan['reservation']; res.update(name=current['metadata']['name'], expected_uid=current['metadata']['uid'], owner_label=current['metadata']['labels']['owner'])
    uid, allowed, expiry = lease_guard.fresh_lease(current, res, datetime.datetime.now(datetime.timezone.utc))
    assert allowed >= 7200 and [z['metadata']['uid'] for z in active(node)] == [uid]
    plan['remote']['run_dir'] = '/data/wio/glm5-3-vllm-execution/20261018-state-tail/prefix-count-live1-20261019/node'+str(node)
    plan['container'].update(name='glm53-prefix-count-'+str(node)+'-live1-20261019', lifetime_seconds=7200)
    for key in ['GLM53_MOE_CUMSUM_I32_PROBE', 'GLM53_MOE_PREFIX_HISTOGRAM_PROBE', 'GLM53_MTP_MOE_GRAPH_PROBE', 'GLM53_MTP_FFN_GRAPH_PROBE', 'GLM53_SFA_REPLAY_ORDER_PROBE', 'GLM53_PREFILL_PADDING_PROBE', 'GLM53_PREFILL_ATTENTION_PROBE']:
        plan['env'][key] = '0'
    plan['env']['GLM53_MOE_PREFIX_GMM2_COUNT_PROBE'] = '1' if role == 'D' else '0'
    plan['env']['GLM53_ROPE_CACHE_LAYOUT_PROBE'] = '1' if role == 'D' else '0'
    if role == 'D': plan['runtime_patches'] = build['patches']
    argv = plan['serve_argv']; index = argv.index('--kv-transfer-config')+1; kv = json.loads(argv[index]); kv['engine_id'] = 'glm53-prefix-count-'+role.lower()+'-live1-20261019'; argv[index] = json.dumps(kv, separators=(',', ':'))
    plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), holder_expiry_utc=expiry, objective='Finite native PD test of routing prefix0 with native count recovery only at GMM2; MTP native, accepted RoPE fixed', notes=['All8 full NZ chain cases and96 graph replays bitwise; median full-layer ratios0.809-0.929, not a service forecast.', 'New own10800s holder and7200s model; no other job stopped. Existing pinned image, weights, operators, precision and SDMA transport retained.', 'D new immutable buildc965; P same immutable build9886 and original PIECEWISE[256,1024]/K1/sharedFalse. No parameter sweep.'])
    path = r/('prefix-count-'+role+'-plan-v1.json'); assert not path.exists(); path.write_text(json.dumps(plan, indent=2)+'\n')
    q = subprocess.run([sys.executable, str(repo/'glm5-3-vllm/tools/start_capacity.py'), '--plan', str(path), '--control', control, '--output', str(r/('prefix-count-'+role+'-launch-v1'))], cwd=repo, capture_output=True, text=True, timeout=240)
    (r/('prefix-count-'+role+'-launch-v1-collector.txt')).write_text(q.stdout+q.stderr)
    print(json.dumps(dict(role=role, node=node, rc=q.returncode, lifetime=7200, output=q.stdout)), flush=True)
    assert q.returncode == 0, q.stderr
