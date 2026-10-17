#!/usr/bin/env python3
"""Preserve native timing evidence, or close only the two exact owned models."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import socket
import os
import signal
import subprocess
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('timing', 'close'), required=True)
    parser.add_argument('--revision', type=int, choices=(1, 2), default=1)
    parser.add_argument('--expected-cid', required=True)
    args = parser.parse_args()
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    mode = 'eager' if node == 32 else 'graph'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    root = base/f'native-{mode}-node{node}-v{args.revision}'
    launch = json.loads((root/'launch.json').read_text())
    cid = launch['container_id']
    assert cid == args.expected_cid
    doc = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
    assert doc['Name'] == f'/glm53-native-{mode}-{node}-v{args.revision}-20261017'
    assert doc['Image'] == 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    test_name = f'native-{mode}-c1-v1' if args.revision == 1 else 'native-graph-c1-correction-v1'
    test_path = base/'tests'/test_name/'result.json'
    if args.phase == 'close' and args.revision == 2 and not test_path.exists():
        # The human changed priority while the pair-readiness wait was active.
        stopped = []
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = (proc/'cmdline').read_bytes().split(b'\0')
                if len(cmd) > 1 and cmd[0].endswith(b'python3') and cmd[1].startswith(b'/data/wio/glm5-3-vllm-execution/20261017-inventory/node33-native-graph-c1-correction-v1-') and cmd[1].endswith(b'/script.py'):
                    os.kill(int(proc.name), signal.SIGTERM); stopped.append(int(proc.name))
            except (FileNotFoundError, ProcessLookupError):
                pass
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text(json.dumps({'completed': False, 'passed': False,
            'cancelled_by_latest_human_priority': 'Deploy official PD connector and official proxy; do not continue exact-token graph diagnostics',
            'stopped_client_pids': stopped}, indent=2)+'\n')
    test = json.loads(test_path.read_text())
    if args.phase == 'timing':
        assert test['completed'] and test['passed']
    out = base/'consumer-audits'/f'native-{mode}-{args.phase}-v{args.revision}'
    out.mkdir(parents=True, exist_ok=False)
    report = {'node': node, 'mode': mode, 'phase': args.phase, 'container_id': cid,
              'host_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'benchmark_passed': test['passed']}
    pre = subprocess.run(['docker', 'logs', '--timestamps', cid], capture_output=True, timeout=30)
    pre_text = (pre.stdout+pre.stderr).decode(errors='replace')
    report['errors_before_action'] = [s for s in pre_text.splitlines()
        if re.search(r'ERROR|Traceback|RuntimeError|ValidationError', s)][-40:]
    if args.phase == 'close' and doc['State']['Running']:
        subprocess.run(['docker', 'stop', '-t', '90', cid], check=True, capture_output=True, timeout=100)
    elif args.phase == 'timing':
        assert doc['State']['Running']
        with urllib.request.urlopen(f'http://172.16.12.{node}:{29481 if node == 32 else 29482}/metrics', timeout=10) as response:
            (out/'metrics.txt').write_bytes(response.read())
        profiles = [str(p.relative_to(root)) for p in (root/'profile').rglob('*') if p.is_file()]
        report['profile_files_after_timing'] = profiles
        assert not profiles, 'Profiler output appeared during formal timing'
    p = subprocess.run(['docker', 'logs', '--timestamps', cid], capture_output=True, timeout=30)
    raw = p.stdout+b'\nDOCKER_STDERR_STREAM\n'+p.stderr
    (out/'console.txt.gz').write_bytes(gzip.compress(raw, compresslevel=1))
    text = raw.decode(errors='replace')
    report['effective_config'] = [s for s in text.splitlines() if 'Native benchmark startup' in s]
    report['actual_replay'] = [s for s in text.splitlines() if 'ACL graph replay is active' in s]
    report['native_errors'] = [s for s in text.splitlines() if re.search(r'ERROR|Traceback|RuntimeError|ValidationError', s)][-40:]
    report['raw_console_bytes'] = len(raw)
    report['raw_console_sha256'] = hashlib.sha256(raw).hexdigest()
    report['state'] = json.loads(subprocess.check_output(['docker', 'inspect', cid, '--format', '{{json .State}}']))
    report['device_snapshot'] = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=30).stdout
    report['docker_stats'] = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{json .}}', cid], capture_output=True, text=True, timeout=20).stdout
    report['runtime_config'] = {'args': doc['Args'], 'image': doc['Image'],
        'env': [s for s in doc['Config']['Env'] if s.startswith(('HCCL_', 'GLM53_KV_', 'MMC_', 'ASCEND_RT_VISIBLE_DEVICES='))]}
    for name in ('launch.json', 'plan.json', 'context.json'):
        if (root/name).is_file():
            (out/name).write_bytes((root/name).read_bytes())
    report['files'] = [{'path': str(p.relative_to(out)), 'bytes': p.stat().st_size,
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                       for p in sorted(out.iterdir()) if p.is_file()]
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if args.phase == 'close':
        assert not report['state']['Running'] and not report['state']['OOMKilled']
        assert report['state']['ExitCode'] in (0, 143)


if __name__ == '__main__':
    main()
