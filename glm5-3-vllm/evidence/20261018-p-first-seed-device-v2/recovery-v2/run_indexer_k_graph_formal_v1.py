"""Run the frozen finite service comparison on the already loaded exact instances."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, subprocess, sys

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

r = Path(__file__).resolve().parent
repo = r.parents[3]
harness=repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py'
assert harness.is_file() and "'indexer_k_graph_overlap'" in harness.read_text()
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value
m = module('transport', r/'run_prefix_count_service_v1.py')
g = module('guard', repo/'glm5-3-vllm/runtime-patches/state-tail-probe-v1/prepare_launch.py')
ssh = ['ssh', '-S', m.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-p', '60011', 'root@61.241.77.34']
root = '/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live2-20261019/node33'
binding = json.loads((r/'indexer-k-graph-overlap-service-CID-binding-v2.json').read_text())
quality=json.loads((r/'indexer-k-graph-quality-reviewed-v2.json').read_text())
assert quality['proceed_finite_hardware_and_formal_review']
for label in ['quality','control']:
    assert hashlib.sha256((r/quality[label+'_source']).read_bytes()).hexdigest()==quality[label+'_sha256']
paths=json.loads(next((r/'indexer-k-graph-paths-full-v2').glob('*/metrics.json')).read_text())
assert paths['all16'] and all(z['N']==[6,12,18,24] for z in paths['rows'])
hardware=json.loads((r/'indexer-k-graph-hardware-reviewed-v2.json').read_text())
assert hardware['completed'] and hardware['all16_C1_C4_new_copies_SDMA'] and len(hardware['rows'])==32 and all(z['all_added_copy_SDMA'] for z in hardware['rows'])
assert hardware['P_CID']==binding['P_CID']==quality['P_CID'] and hardware['D_CID']==binding['D_CID']==quality['D_CID']
contract=r/binding['formal_contract']
assert hashlib.sha256(contract.read_bytes()).hexdigest()==binding['formal_contract_sha256']
assert binding['workload_and_joint_gate_unchanged']
archive=hardware['archive'];assert Path(archive['local_archive']).is_file()
assert file_sha(archive['local_archive'])==archive['sha256']
assert json.loads((r/'indexer-k-graph-correctness-outcome-v2.json').read_text())['completed']
assert not (r/'indexer-k-graph-formal-outcome-v1.json').exists(), 'No same-condition rerun'
leases = []
for role, node, version in [('P', 32, 1), ('D', 33, 2)]:
    plan = json.loads((r/('indexer-k-graph-P-plan-v1.json' if node==32 else 'indexer-k-graph-D-plan-v2.json')).read_text())
    res = plan['reservation']
    pod = json.loads(subprocess.check_output(ssh+['kubectl get pod -n default '+res['name']+' -o json'], timeout=30))
    uid, allowed, expiry = g.fresh_lease(pod, res, datetime.datetime.now(datetime.timezone.utc))
    assert uid == res['expected_uid'] and not pod['metadata'].get('ownerReferences')
    pods = json.loads(subprocess.check_output(ssh+['kubectl get pods -A --field-selector spec.nodeName='+res['node_name']+' -o json'], timeout=30))['items']
    active = [p['metadata']['uid'] for p in pods if p['status']['phase'] not in ['Failed', 'Succeeded'] and any(int(c.get('resources', {}).get('requests', {}).get('huawei.com/Ascend910', 0)) for c in p['spec']['containers'])]
    assert active == [uid]
    leases.append(dict(node=node, uid=uid, allowed_s=allowed, expiry=expiry, exclusive_requested_NPU_holder=True))
    x = m.collect(r/'preflight_indexer_k_graph_formal_v1.py', 'indexer_k_graph_overlap_formal_preflight_v1_'+str(node), 'indexer-k-graph-formal-preflight-v1-'+str(node), timeout=120, node=node)
    assert x['remaining_s'] > 1800
    print(json.dumps(dict(phase='fresh_preflight', node=node, remaining_s=x['remaining_s'], lease_uid=uid)), flush=True)
(r/'indexer-k-graph-formal-lease-preflight-v2.json').write_text(json.dumps(dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), leases=leases), indent=2)+'\n')

def phase(name, extra):
    x = m.collect(repo/'glm5-3-vllm/runtime-patches/p-first-seed-v2/validate_sfa_replay_order.py', 'indexer_k_graph_overlap_'+name+'_v1', 'indexer-k-graph-'+name+'-v1', ('--execute', '--phase', name, '--probe-kind', 'indexer_k_graph_overlap', '--run-dir', root, '--expected-D-CID', binding['D_CID'], '--cases', root+'/prefill-padding-cases-v1.json', '--run-tag', 'v1')+extra, timeout=900)
    assert x['completed'] and x['numeric_all_pass']
    print(json.dumps(dict(phase=name, rows=len(x['rows']), summary=x['summary'])), flush=True)
    return x
c1 = phase('c1', ('--retain-native-counters',))
c4 = phase('c4', ('--rotate-cases', '--c4-cycles', '3', '--retain-native-counters'))
assert len(c4['blocks']) == 12 and all(z['native_counter_boundary']['delta']['33']['vllm:generation_tokens_total'] == 3072 for z in c4['blocks'])
outcome = dict(completed=True, C1=c1['summary'], C4=c4['summary'], P_CID=binding['P_CID'], D_CID=binding['D_CID'], native_marker_restored=True, mode_owned_native_counters_each_block=True, scope='Frozen observer-off C1 and rotated ABBAx3 C4. All blocks retained. Same accepted RoPE/native operators/SDMA in both modes; not general QA or maximum SLO capacity.')
(r/'indexer-k-graph-formal-outcome-v1.json').write_text(json.dumps(outcome, indent=2)+'\n')
cancel = m.collect(r/'validate_indexer_k_graph_cancel_v1.py', 'indexer_k_graph_overlap_cancel_v1', 'indexer-k-graph-cancel-v1', timeout=600)
assert cancel['completed'] and cancel['client_close_requests'] == 10 and cancel['full_reuse_and_peer_responses'] == 16
print(json.dumps(dict(phase='cancel', client_close=10, full_peer_reuse=16, explicit_backend_abort=cancel['backend_abort_log_request_count'])), flush=True)
subprocess.run([sys.executable, str(r/'review_indexer_k_graph_formal_v1.py')], check=True)
