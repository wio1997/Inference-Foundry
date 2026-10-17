#!/usr/bin/env python3
"""Record only device state and task container state on the two authorized hosts."""
import json
import socket
import subprocess


def main():
    hostname = socket.gethostname()
    assert hostname in ('B4-33U-Atlas900RCKA3-32', 'B5-5U-Atlas900RCKA3-33')
    smi = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
    containers = subprocess.run(['docker', 'ps', '--all', '--format', '{{.ID}}\t{{.Names}}\t{{.Status}}'],
                                capture_output=True, text=True, timeout=15)
    rows = containers.stdout.splitlines()
    tasks = [row for row in rows if row.split('\t')[1].startswith('glm53-')]
    holders = [row for row in rows if 'k8s_' in row and 'glm53-execution-reservation-' in row]
    report = {'hostname': hostname, 'npu_smi_rc': smi.returncode, 'npu_smi_stdout': smi.stdout,
              'npu_smi_stderr': smi.stderr, 'task_containers': tasks, 'reservation_containers': holders}
    print(json.dumps(report, indent=2))
    assert smi.returncode == 0
    assert smi.stdout.count('No running processes found in NPU') == 8, 'A device process remains; inspect before handoff'
    assert not any('\tUp ' in row for row in tasks), 'A task container is still running'


if __name__ == '__main__':
    main()
