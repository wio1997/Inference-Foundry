#!/usr/bin/env python3
"""Read supported npu-smi/hccn getters discovered in captured on-host help."""
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    commands = [['npu-smi', 'info', '-m'], ['npu-smi', 'info', '-l'],
                ['npu-smi', 'info', '-t', 'board', '-i', '0', '-c', '0']]
    for device in range(16):
        for kind in ('ip', 'link', 'net_health', 'vnic'):
            commands.append(['hccn_tool', '-i', str(device), '-' + kind, '-g'])
    out = []
    for argv in commands:
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        t = time.monotonic()
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=6)
            rec = dict(argv=argv, rc=p.returncode, stdout=p.stdout, stderr=p.stderr)
        except subprocess.TimeoutExpired:
            rec = dict(argv=argv, rc=None, timeout=True)
        out.append(dict(rec, start_utc=start, duration_s=time.monotonic()-t))
    # Read-only NIC traffic counters: snapshots, not an isolation certificate.
    net = {}
    for p in Path('/sys/class/net').iterdir():
        if p.name.startswith(('enp', 'bond')):
            net[p.name] = {}
            for key in ('rx_bytes', 'tx_bytes', 'rx_dropped', 'tx_dropped'):
                q = p / 'statistics' / key
                try:
                    net[p.name][key] = int(q.read_text())
                except (OSError, ValueError):
                    pass
    print(json.dumps({'commands': out, 'network_counter_snapshot': net}, indent=2))


if __name__ == '__main__':
    main()
