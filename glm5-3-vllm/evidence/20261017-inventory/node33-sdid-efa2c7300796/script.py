#!/usr/bin/env python3
"""Read SDID/SuperPoD identity for each mapped die using the confirmed npu-smi 25.5.2 getter."""
import json
import subprocess
import time


def main():
    rows = []
    for physical in range(16):
        argv = ['npu-smi', 'info', '-t', 'spod-info', '-i', str(physical//2), '-c', str(physical%2)]
        start = time.monotonic()
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=6)
            fields = dict(line.strip().split(':', 1) for line in p.stdout.splitlines() if ':' in line)
            rows.append({'physical_id': physical, 'card': physical//2, 'chip': physical%2,
                         'argv': argv, 'rc': p.returncode, 'fields': {k.strip(): v.strip() for k,v in fields.items()},
                         'stdout': p.stdout, 'stderr': p.stderr, 'duration_s': time.monotonic()-start})
        except subprocess.TimeoutExpired:
            rows.append({'physical_id': physical, 'argv': argv, 'timeout': True})
    print(json.dumps({'dies': rows}, indent=2))


if __name__ == '__main__':
    main()
