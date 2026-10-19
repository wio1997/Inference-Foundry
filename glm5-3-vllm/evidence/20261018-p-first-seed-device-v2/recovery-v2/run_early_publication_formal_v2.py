"""Run the frozen finite service comparison on the already loaded exact instances."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, subprocess, sys

r = Path(__file__).resolve().parent
repo = r.parents[3]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
m = module('transport', r/'run_prefix_count_service_v1.py')
g = module('guard', repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
ssh = ['ssh', '-S', m.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-p', '60011', 'root@61.241.77.34']
root = '/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live2-20261019/node33'
binding = json.loads((r/'early-sample-publication-formal-CID-binding-v2.json').read_text())
assert json.loads((r/'early-sample-publication-quality-reviewed-v2.json').read_text())['proceed_finite_formal']
assert json.loads((r/'early-sample-publication-correctness-outcome-v2.json').read_text())['completed']
leases = []
for role, node, version in [('P', 32, 1), ('D', 33, 2)]:
    plan = json.loads((r/('early-sample-publication-'+role+'-plan-v'+str(version)+'.json')).read_text())
    res = plan['reservation']
    pod = json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'], timeout=30))
    uid, allowed, expiry = g.fresh_lease(pod, res, datetime.datetime.now(datetime.timezone.utc))
    assert uid == res['expected_uid'] and not pod['metadata'].get('ownerReferences')
    pods = json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'], timeout=30))['items']
    active = [p['metadata']['uid'] for p in pods if p['status']['phase'] not in ['Failed', 'Succeeded'] and any(int(c.get('resources', {}).get('requests', {}).get('huawei.com/Ascend910', 0)) for c in p['spec']['containers'])]
    assert active == [uid]
    leases.append(dict(node=node, uid=uid, allowed_s=allowed, expiry=expiry, exclusive_requested_NPU_holder=True))
    x = m.collect(r/'preflight_early_publication_v4.py', 'early_publication_formal_preflight_v2_'+str(node), 'early-sample-publication-formal-preflight-v2-'+str(node), timeout=120, node=node)
    assert x['remaining_s'] > 1200
    print(json.dumps(dict(phase='fresh_preflight', node=node, remaining_s=x['remaining_s'], lease_uid=uid)), flush=True)
(r/'early-sample-publication-formal-lease-preflight-v2.json').write_text(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), leases=leases), indent=2)+'\n')

def phase(name, extra):
    x = m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py', 'early_publication_'+name+'_v2', 'early-sample-publication-'+name+'-v2', ('--execute', '--phase', name, '--probe-kind', 'early_sample_publication', '--run-dir', root, '--expected-D-CID', binding['D_CID'], '--cases', root+'/prefill-padding-cases-v1.json', '--run-tag', 'v1')+extra, timeout=900)
    assert x['completed'] and x['numeric_all_pass']
    print(json.dumps(dict(phase=name, rows=len(x['rows']), summary=x['summary'])), flush=True)
    return x
c1 = phase('c1', ('--retain-native-counters',))
c4 = phase('c4', ('--rotate-cases', '--c4-cycles', '3', '--retain-native-counters'))
assert len(c4['blocks']) == 12 and all(z['native_counter_boundary']['delta']['33']['vllm:generation_tokens_total'] == 3072 for z in c4['blocks'])
outcome = dict(completed=True, C1=c1['summary'], C4=c4['summary'], P_CID=binding['P_CID'], D_CID=binding['D_CID'], native_marker_restored=True, mode_owned_native_counters_each_block=True, scope='Frozen observer-off C1 and rotated ABBAx3 C4. All blocks retained. Same accepted RoPE/native operators/SDMA in both modes; not general QA or maximum SLO capacity.')
(r/'early-sample-publication-formal-outcome-v2.json').write_text(json.dumps(outcome, indent=2)+'\n')
cancel = m.collect(r/'validate_early_publication_cancel_v4.py', 'early_publication_cancel_v2', 'early-sample-publication-cancel-v2', timeout=600)
assert cancel['completed'] and cancel['client_close_requests'] == 10 and cancel['full_reuse_and_peer_responses'] == 16
print(json.dumps(dict(phase='cancel', client_close=10, full_peer_reuse=16, explicit_backend_abort=cancel['backend_abort_log_request_count'])), flush=True)
subprocess.run([sys.executable, str(r/'review_early_publication_formal_v2.py')], check=True)
