from pathlib import Path
import datetime, importlib.util, json, subprocess

r = Path(__file__).resolve().parent; repo = r.parents[3]
spec = importlib.util.spec_from_file_location('control', r/'run_prefix_count_service_v1.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
lease_spec = importlib.util.spec_from_file_location('lease_guard', repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
lease_guard = importlib.util.module_from_spec(lease_spec); lease_spec.loader.exec_module(lease_guard)
assert json.loads((r/'prefix-count-reviewed-verdict-v1.json').read_text())['decision'] == 'MIXED_DEFAULT_OFF_PREFIX_COUNT_NO_JOINT_GAIN'
ssh = ['ssh', '-S', m.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-p', '60011', 'root@61.241.77.34']
leases = []
for role, node in [('P', 32), ('D', 33)]:
    plan = json.loads((r/('prefix-count-'+role+'-plan-v1.json')).read_text()); res = plan['reservation']
    current = json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'], timeout=25))
    assert not current['metadata'].get('ownerReferences')
    uid, allowed, expiry = lease_guard.fresh_lease(current, res, datetime.datetime.now(datetime.timezone.utc))
    assert uid == res['expected_uid'] and allowed > 1800
    pods = json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'], timeout=25))['items']
    active = [z['metadata']['uid'] for z in pods if z['status']['phase'] not in ['Failed', 'Succeeded'] and any(int(c.get('resources', {}).get('requests', {}).get('huawei.com/Ascend910', 0)) for c in z['spec'].get('containers', []))]
    assert active == [uid]
    leases.append(dict(node=node, UID=uid, remaining_holder_allowed_s=allowed, expiry=expiry))
(r/'prefix-count-fixed-arrival-lease-preflight-v1.json').write_text(json.dumps(leases, indent=2)+'\n')
clock=json.loads(next((r/'prefix-count-clock-probe-v1').glob('*/metrics.json')).read_text());assert clock['completed'] and clock['summary']['absolute_clock_nanosleep']['max_ms']<1
m.preflight('fixed-arrival-v2')
x = m.collect(r/'validate_prefix_count_fixed_arrival_v2.py', 'prefix_count_fixed_arrival_v2', 'prefix-count-fixed-arrival-v2',
              ('--execute', '--phase', 'c4', '--probe-kind', 'moe_prefix_gmm2_count', '--run-dir', m.root, '--expected-D-CID', m.cid,
               '--cases', m.root+'/prefill-padding-cases-v1.json', '--run-tag', 'fixed-arrival-v2', '--c4-cycles', '1', '--retain-native-counters'), timeout=480)
assert x['completed'] and x['numeric_all_pass'] and len(x['blocks']) == 4
assert len([z for z in x['rows'] if z['phase'] == 'formal']) == 128
assert all(z['native_counter_boundary']['delta']['33']['vllm:generation_tokens_total'] == 3072 for z in x['blocks'])
print(json.dumps(dict(completed=True, formal128_numeric=True, fixed_arrival_control=x['fixed_arrival_control'], scope='Primary evaluates this new arrival condition; original closed-loop verdict remains unchanged.')), flush=True)
