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
    parser.add_argument('--previous-receipt', type=Path)
    parser.add_argument('--lease-seconds', type=int, choices=(3600,4500,21600), default=3600)
    args = parser.parse_args()
    assert not args.receipt.exists()
    ssh = ['ssh', '-S', args.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-p', '60011', 'root@61.241.77.34']
    expected = {32: '2cd9108a-eaf0-4164-856b-968bb65b79c6', 33: 'ffd5ca30-273d-4646-aa79-7b35478b9416'}
    if args.previous_receipt:
        prior = json.loads(args.previous_receipt.read_text())
        expected = {r['node']: r['uid'] for r in prior['after']}
        assert set(expected) == {32,33}
    pods = {32: 'sglang-node-access-32', 33: 'sglang-tuning-access-33-20261013'}
    before = []
    documents = []
    for node in (32, 33):
        name = f'glm53-execution-reservation-{node}-20261017'
        listing = json.loads(subprocess.check_output(ssh+[f'kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-{node} -o json']))
        for pod in listing['items']:
            count = sum(int(c.get('resources', {}).get('requests', {}).get('huawei.com/Ascend910', 0)) for c in pod['spec'].get('containers', []))
            if count and pod['metadata']['uid'] != expected[node]:
                assert pod['status']['phase'] in ('Failed', 'Succeeded'), 'Another live NPU claim; do not preempt or stop it'
        old = json.loads(subprocess.check_output(ssh+[f'kubectl get pod -n default {name} -o json']))
        assert old['metadata']['uid'] == expected[node]
        assert old['metadata']['labels']['owner'] == 'codex-execution-20261017'
        if args.previous_receipt:
            assert old['status']['phase'] in ('Running','Failed')
            assert old['spec']['containers'][0]['command'] == ['/bin/sleep',str(old['spec']['activeDeadlineSeconds'])]
        else:
            assert old['status']['phase'] == 'Failed' and old['status'].get('reason') == 'DeadlineExceeded'
        assert not old['metadata'].get('ownerReferences')
        raw = subprocess.check_output(ssh+[f'kubectl exec -i -n default {pods[node]} -- nsenter -t 1 -m -- npu-smi info'], text=True, timeout=40)
        assert raw.count('No running processes found in NPU') == 8
        before.append({'node': node, 'name': name, 'uid': expected[node], 'phase': old['status']['phase'], 'reason': old['status'].get('reason'), 'empty_cards': 8})
        document = json.loads((Path(__file__).parent/f'node{node}-reservation.json').read_text())
        document['spec']['activeDeadlineSeconds'] = args.lease_seconds
        document['spec']['containers'][0]['command'] = ['/bin/sleep', str(args.lease_seconds)]
        documents.append(document)
    args.receipt.with_suffix('.intent.json').write_text(json.dumps({'before':before,'scope':'Exact own UIDs only,8 empty cards both,no other live NPU claim; next bounded stored-pool diagnostic'},indent=2)+'\n')
    # Both nodes passed before mutation; exact own empty holder UIDs only.
    for old in before:
        subprocess.run(ssh+[f"kubectl delete pod -n default {old['name']} --wait=false"], check=True, capture_output=True,timeout=20)
    deletion_deadline=time.monotonic()+75
    while True:
        remaining=[]
        for old in before:
            raw=subprocess.check_output(ssh+[f"kubectl get pod -n default {old['name']} --ignore-not-found -o json"],timeout=20)
            if raw.strip():
                current=json.loads(raw);assert current['metadata']['uid']==old['uid'];remaining.append(old['name'])
        if not remaining:break
        assert time.monotonic()<deletion_deadline, 'Exact own holders still terminating; inspect state, do not force'
        time.sleep(3)
    # A new foreign claim winning the release window is never preempted.
    for node in (32,33):
        listing=json.loads(subprocess.check_output(ssh+[f'kubectl get pods -A --field-selector spec.nodeName=a3-5u-atlas900rcka3-{node} -o json']))
        for pod in listing['items']:
            count=sum(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0)) for c in pod['spec'].get('containers',[]))
            if count:assert pod['status']['phase'] in ('Failed','Succeeded'), 'Other live claim after release; do not preempt'
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
              'scope': 'Only exact own empty holders replaced after no other live NPU claim and8 empty cards on both specified nodes. No other workload stopped. Finite holder sized for the current authorized native benchmark window, no model loaded by this tool.'}
    args.receipt.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
