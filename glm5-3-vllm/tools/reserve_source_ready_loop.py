#!/usr/bin/env python3
"""Replace only this task's expired empty holders with finite one-hour leases."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert not args.receipt.exists()
    ssh = ['ssh', '-S', args.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-p', '60011', 'root@61.241.77.34']
    expected = {32: '2cd9108a-eaf0-4164-856b-968bb65b79c6', 33: 'ffd5ca30-273d-4646-aa79-7b35478b9416'}
    pods = {32: 'sglang-node-access-32', 33: 'sglang-tuning-access-33-20261013'}
    before = []
    documents = []
    for node in (32, 33):
        name = f'glm53-execution-reservation-{node}-20261017'
        listing = json.loads(subprocess.check_output(ssh+[f'kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-{node} -o json']))
        for pod in listing['items']:
            count = sum(int(c.get('resources', {}).get('requests', {}).get('huawei.com/Ascend910', 0)) for c in pod['spec'].get('containers', []))
            if count:
                assert pod['status']['phase'] in ('Failed', 'Succeeded'), 'Another live NPU claim; do not preempt or stop it'
        old = json.loads(subprocess.check_output(ssh+[f'kubectl get pod -n default {name} -o json']))
        assert old['metadata']['uid'] == expected[node]
        assert old['metadata']['labels']['owner'] == 'codex-execution-20261017'
        assert old['status']['phase'] == 'Failed' and old['status'].get('reason') == 'DeadlineExceeded'
        assert not old['metadata'].get('ownerReferences')
        raw = subprocess.check_output(ssh+[f'kubectl exec -i -n default {pods[node]} -- nsenter -t 1 -m -- npu-smi info'], text=True, timeout=40)
        assert raw.count('No running processes found in NPU') == 8
        before.append({'node': node, 'name': name, 'uid': expected[node], 'phase': 'Failed', 'reason': 'DeadlineExceeded', 'empty_cards': 8})
        document = json.loads((Path(__file__).parent/f'node{node}-reservation.json').read_text())
        document['spec']['activeDeadlineSeconds'] = 3600
        document['spec']['containers'][0]['command'] = ['/bin/sleep', '3600']
        documents.append(document)
    # Both nodes passed before any mutation; only the exact expired owner UIDs are deleted.
    for old in before:
        subprocess.run(ssh+[f"kubectl delete pod -n default {old['name']} --wait=true --timeout=30s"], check=True, capture_output=True)
    created = subprocess.run(ssh+['kubectl create -f -'], input=json.dumps({'apiVersion': 'v1', 'kind': 'List', 'items': documents}).encode(), capture_output=True, check=True)
    deadline = time.monotonic()+120
    after = []
    while True:
        after = []
        for node in (32, 33):
            name = f'glm53-execution-reservation-{node}-20261017'
            pod = json.loads(subprocess.check_output(ssh+[f'kubectl get pod -n default {name} -o json']))
            after.append({'node': node, 'name': name, 'uid': pod['metadata']['uid'], 'phase': pod['status']['phase'],
                          'creation_timestamp': pod['metadata']['creationTimestamp'], 'start_time': pod['status'].get('startTime'),
                          'ascend_real': pod['metadata'].get('annotations', {}).get('huawei.com/AscendReal'),
                          'active_deadline_seconds': pod['spec']['activeDeadlineSeconds']})
        if all(x['phase'] == 'Running' and x['ascend_real'] for x in after):
            break
        assert time.monotonic() < deadline, after
        time.sleep(3)
    report = {'client_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'before': before, 'after': after,
              'scope': 'Only exact expired owner holders replaced after no live NPU claim and8 empty cards on both specified nodes. No other workload stopped. Finite3600s holder, no model loaded by this tool.'}
    args.receipt.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
