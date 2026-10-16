#!/usr/bin/env python3
"""One-shot release of the two observed allocations, under the user's 32/33 kill authorization."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite an existing execution receipt')
    ssh = ['ssh', '-S', a.control, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
           '-p', '60011', 'root@61.241.77.34']
    report = {'authorization': 'User: 32 33 全部 可以kill; limited to workloads on these two nodes',
              'started_client_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'actions': []}

    def save():
        a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    def call(argv, capture=True):
        c = subprocess.run(ssh + [shlex.join(['kubectl'] + argv)], capture_output=True, text=True, timeout=75)
        if capture:
            report['actions'].append({'argv': ['kubectl'] + argv, 'rc': c.returncode,
                                      'stdout': c.stdout, 'stderr': c.stderr,
                                      'client_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            save()
        if c.returncode:
            raise RuntimeError('kubectl failed: ' + shlex.join(argv))
        return c.stdout

    def get(argv):
        return json.loads(call(['get'] + argv + ['-o', 'json'], capture=False))

    dep = get(['deployment', '-n', 'testportal-project', 'containerworkload-694f3t4n'])
    assert dep['metadata']['uid'] == 'b3e19d00-059f-4177-986c-45f3209164be'
    assert dep['spec']['replicas'] == 1
    pods = get(['pods', '-n', 'testportal-project', '-l', 'app=containerworkload-694f3t4n'])['items']
    live = [d for d in pods if d.get('status', {}).get('phase') not in ('Succeeded', 'Failed')]
    assert len(live) == 1 and live[0]['metadata']['name'] == 'containerworkload-694f3t4n-5cc6976cc7-7zhb2'
    assert live[0]['spec']['nodeName'] == 'a3-5u-atlas900rcka3-32', 'Controller also serves other nodes'
    hpas = get(['hpa', '-n', 'testportal-project'])['items']
    assert not any(h['spec']['scaleTargetRef']['name'] == dep['metadata']['name'] for h in hpas), 'HPA needs separate review'
    old = get(['pod', '-n', 'default', 'sglang-pd4p2d-reserve-33-20261014'])
    assert old['metadata']['uid'] == '6c2967ae-f054-41ce-8ae7-f555a9e47151'
    assert old['spec']['nodeName'] == 'a3-5u-atlas900rcka3-33'
    assert not old['metadata'].get('ownerReferences'), 'Unexpected controller'
    report['before'] = {
        'node32_deployment': {'name': dep['metadata']['name'], 'namespace': 'testportal-project',
                              'uid': dep['metadata']['uid'], 'replicas': 1,
                              'pod': live[0]['metadata']['name'], 'node': live[0]['spec']['nodeName']},
        'node33_reservation': {'name': old['metadata']['name'], 'uid': old['metadata']['uid'],
                               'node': old['spec']['nodeName'], 'ownerReferences': None,
                               'resources': old['spec']['containers'][0].get('resources', {})}}
    save()
    call(['scale', 'deployment', '-n', 'testportal-project', dep['metadata']['name'],
          '--current-replicas=1', '--replicas=0'])
    call(['delete', 'pod', '-n', 'default', old['metadata']['name'], '--wait=false'])
    call(['wait', '-n', 'testportal-project', '--for=delete', 'pod/' + live[0]['metadata']['name'], '--timeout=60s'])
    call(['wait', '-n', 'default', '--for=delete', 'pod/' + old['metadata']['name'], '--timeout=60s'])
    after = get(['deployment', '-n', 'testportal-project', dep['metadata']['name']])
    assert after['spec']['replicas'] == 0
    report['completed_client_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['node32_deployment_replicas_after'] = 0
    report['status'] = 'released_observed_allocations; system/access pods untouched'
    save()
    print(json.dumps({'status': report['status'], 'receipt': str(a.output.resolve())}))


if __name__ == '__main__':
    main()
