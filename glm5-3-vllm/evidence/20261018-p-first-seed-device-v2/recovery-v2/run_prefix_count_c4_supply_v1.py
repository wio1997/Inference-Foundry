from pathlib import Path
import datetime, importlib.util, json, subprocess

r = Path(__file__).resolve().parent; repo = r.parents[3]
spec = importlib.util.spec_from_file_location('controller', r/'run_prefix_count_service_v1.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
spec = importlib.util.spec_from_file_location('lease_guard', repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
assert json.loads((r/'prefix-count-fixed-arrival-reviewed-v2.json').read_text())['decision'] == 'REJECT_ARRIVAL_FEEDBACK_AS_SUFFICIENT_EXPLANATION'
ssh = ['ssh', '-S', m.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-p', '60011', 'root@61.241.77.34']
leases = []
for role in ['P', 'D']:
    plan = json.loads((r/('prefix-count-'+role+'-plan-v1.json')).read_text()); res = plan['reservation']
    pod = json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'], timeout=25))
    assert not pod['metadata'].get('ownerReferences')
    uid, allowed, expiry = guard.fresh_lease(pod, res, datetime.datetime.now(datetime.timezone.utc))
    assert uid == res['expected_uid'] and allowed > 1800
    pods = json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'], timeout=25))['items']
    active = [z['metadata']['uid'] for z in pods if z['status']['phase'] not in ['Failed', 'Succeeded'] and any(int(c.get('resources', {}).get('requests', {}).get('huawei.com/Ascend910', 0)) for c in z['spec'].get('containers', []))]
    assert active == [uid]
    leases.append(dict(role=role, UID=uid, remaining_holder_allowed_s=allowed, expiry=expiry))
p = r/'prefix-count-c4-supply-lease-preflight-v1.json'; assert not p.exists(); p.write_text(json.dumps(leases, indent=2)+'\n')
m.preflight('c4-supply-v1')
x = m.collect(r/'profile_prefix_count_c4_supply_v1.py', 'prefix_count_c4_supply_profile_v1', 'prefix-count-c4-supply-profile-v1', timeout=360)
assert x['completed'] and x['candidate_restored_native'] and x['observer_restored_off']
assert len(x['profiles']) == 2 and all(z['profile_stopped'] and z['ranks'] == list(range(16)) for z in x['profiles'])
print(json.dumps(dict(completed=True, profiles=[dict(mode=z['mode'], ranks=z['ranks'], raw_bytes=z['source_bytes'], selected_N_by_rank=z['selected_N_by_rank']) for z in x['profiles']], next='All profiles stopped; archive sources and CPU-export rank0 only.')), flush=True)
