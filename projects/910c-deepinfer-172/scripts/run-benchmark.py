#!/usr/bin/env python3
"""Plan the canonical two-round benchmark in a fresh directory; never rerun by default."""
import argparse
import json
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.request

ROOT = Path('/data/tiankuan/wio/910c-deepinfer-172')
PACKAGE = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--run-id', default='bench-' + time.strftime('%Y%m%dT%H%M%S'))
    args = parser.parse_args()
    assert re.fullmatch(r'bench-[A-Za-z0-9_-]+', args.run_id)
    output = ROOT / 'benchmarks' / args.run_id
    print(json.dumps({'output': str(output), 'input_tokens': 32768, 'output_tokens': 1024,
                      'concurrency': 8, 'formal_rounds': 2, 'prefix_cached_tokens_per_round': 235904,
                      'mode': 'execute' if args.execute else 'plan'}, indent=2))
    if not args.execute:
        return
    assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-172', 'wrong server'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8172/health', timeout=3) as response:
        assert json.load(response) == {'status': 'ok', 'backend': 'DeepInfer-native'}
    # File lock is shared by all invocations of this supplied reproduction wrapper.
    import fcntl
    lock = (ROOT / 'runs' / 'reproduction-benchmark.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not output.exists(), 'never overwrite a measured run'
    output.mkdir()
    frozen = ROOT / 'final/native-v8/benchmark'
    for phase in ['measure1', 'measure2']:
        config = (frozen / (phase + '.py')).read_text()
        config = config.replace('/benchmarks/native-v8/', '/benchmarks/' + args.run_id + '/')
        config = re.sub(r"out_dir='[^']*'", 'out_dir=' + repr(str(output / phase / 'predictions') + '/'), config)
        (output / (phase + '.py')).write_text(config)
    code = (PACKAGE / 'controllers/native_v8_benchmark.py').read_text()
    code = code.replace("'benchmarks/native-v8'", repr('benchmarks/' + args.run_id))
    script = ROOT / 'tools' / (args.run_id + '.py')
    assert not script.exists()
    script.write_text(code)
    source = ROOT / 'tools/aisbench-7bcb6440da11'
    python = ROOT / 'tools/aisbench-venv-7bcb6440da11/bin/python'
    env = 'PYTHONPATH=' + ':'.join(map(str, [ROOT / 'tools/aisbench-adapter', source, source / 'plugins/prefix_cache']))
    command = ['docker', 'exec', '-e', env, '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-e', 'NO_PROXY=*', '-e', 'no_proxy=*', '-e', 'HTTP_PROXY=', '-e', 'HTTPS_PROXY=',
               '-e', 'ALL_PROXY=', '-e', 'MPLCONFIGDIR=' + str(ROOT / 'cache/matplotlib'),
               'wio-deepinfer-s172', 'timeout', '--kill-after=20', '7200', str(python), str(script)]
    with (ROOT / 'logs' / (args.run_id + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    print('benchmark_exit', result.returncode, 'summary', output / 'progress.json')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
