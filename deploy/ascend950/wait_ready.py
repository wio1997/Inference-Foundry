#!/usr/bin/env python3
"""Wait for the real eight-rank backend, stopping on container failure or timeout."""
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--timeout', type=int, default=3600)
args = parser.parse_args()
deadline = time.monotonic() + args.timeout
while time.monotonic() < deadline:
    for name in ['deepinfer-950-engine', 'deepinfer-950-api']:
        probe = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name],
                               capture_output=True, text=True)
        if probe.returncode or probe.stdout.strip() != 'true':
            raise SystemExit('Container is not running: ' + name + '; inspect docker logs ' + name)
    try:
        with urllib.request.urlopen('http://127.0.0.1:18110/readyz', timeout=10) as reply:
            result = json.load(reply)
        if result.get('ready') is True and result.get('ranks') == 8:
            print('ENGINE_READY', json.dumps(result), flush=True)
            break
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        pass
    print('MODEL_LOADING: waiting for eight ranks; inspect docker logs deepinfer-950-engine', flush=True)
    time.sleep(5)
else:
    raise SystemExit('Readiness timed out; inspect engine/API logs. No success was recorded.')
