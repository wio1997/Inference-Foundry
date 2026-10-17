#!/usr/bin/env python3
"""Supplemental getter topology and a bounded quiet-window diagnostic, no performance claim."""
import datetime
import json
from pathlib import Path
import subprocess
import time


def snapshot():
    return {'monotonic': time.monotonic(), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'diskstats': Path('/proc/diskstats').read_text(), 'netdev': Path('/proc/net/dev').read_text(),
            'cpu': Path('/proc/stat').read_text().splitlines()[0],
            'pressure': {p.name: p.read_text() for p in Path('/proc/pressure').glob('*')}}


def main():
    commands = []
    for args in [['npu-smi', 'info', '-t', 'topo'],
                 ['npu-smi', 'info', '-t', 'spod-info', '-i', '0', '-c', '0'],
                 ['npu-smi', 'info', '-t', 'hccs', '-i', '0', '-c', '0']]:
        try:
            p = subprocess.run(args, capture_output=True, text=True, timeout=10)
            commands.append(dict(argv=args, rc=p.returncode, stdout=p.stdout, stderr=p.stderr))
        except subprocess.TimeoutExpired:
            commands.append(dict(argv=args, timeout=True))
    active = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = [s.decode(errors='replace') for s in p.read_bytes().split(b'\0') if s]
            matches = [Path(s).name for s in args if Path(s).name in
                       {'node_transfer.py', 'node_rc1_worker.py', 'image_local_worker.py', 'source_rc1.py'}]
            if matches:
                active.append({'pid': int(p.parent.name), 'known_task_script_names': matches})
        except (OSError, ValueError):
            pass
    before = snapshot()
    time.sleep(5)
    after = snapshot()
    print(json.dumps({'commands': commands, 'known_transfer_process_matches': active,
                      'quiet_window_before': before, 'quiet_window_after': after,
                      'dev_nodes': [str(p) for p in Path('/dev').glob('davinci*')],
                      'note': 'Five-second background snapshot only; reassess interference during each run.'}, indent=2))


if __name__ == '__main__':
    main()
