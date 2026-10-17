#!/usr/bin/env python3
"""Preserve and CPU-export one stopped real-request rank, without a new hardware map."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', type=int, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--after-profile', required=True)
    args = parser.parse_args()
    assert all(re.fullmatch(r'[a-z0-9-]+', value) for value in (args.label, args.after_profile))
    assert 1 <= args.revision <= 8
    node = {'B4-33U-Atlas900RCKA3-32': 32, 'B5-5U-Atlas900RCKA3-33': 33}[socket.gethostname()]
    role = 'prefill' if node == 32 else 'decode'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-official-pd')
    profile = json.loads((base / 'tests' / args.after_profile / 'result.json').read_text())
    assert profile['completed'] and all(p.get('stopped') for p in profile['profiles'])
    assert profile['decode_revision' if node == 33 else 'prefill_revision'] == args.revision
    root = base / (role + '-node' + str(node) + '-v' + str(args.revision))
    paths = list((root / 'profile').glob('dp0_pp0_tp0_dcp0_ep0_rank0_*_ascend_pt'))
    assert len(paths) == 1, paths
    selected = paths[0]
    files = sorted(p for p in selected.rglob('*') if p.is_file())
    assert files and not any(p.is_symlink() for p in files)
    assert shutil.disk_usage(root).free > 4 * 1024**3
    out = base / 'consumer-audits' / ('node' + str(node) + '-' + args.label)
    out.mkdir(parents=True, exist_ok=False)
    preserved = root / ('profile-rank0-raw-preserved-' + args.label)
    preserved.mkdir(exist_ok=False)
    manifest = []
    for source in files:
        sha = digest(source)
        target = preserved / source.relative_to(selected)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert digest(target) == sha
        manifest.append({'path': str(source.relative_to(root)), 'bytes': source.stat().st_size,
                         'sha256': sha})
    (out / 'raw-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    mounted = '/profiles/' + selected.name
    image = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    argv = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none',
            '--name', 'glm53-rank0-export-' + str(node) + '-v' + str(args.revision),
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '16',
            '--memory', '64g', '--pids-limit', '4096', '--tmpfs', '/tmp:rw,size=2g',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=1',
            '--device', '/dev/davinci_manager', '-v', str(selected) + ':' + mounted,
            '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
            '--entrypoint', '/usr/bin/timeout', image, '--signal=TERM', '--kill-after=15',
            '420', 'python3', '-c', 'from torch_npu.profiler.profiler import analyse; '
            + 'analyse(' + repr(mounted) + ", max_process_number=4, export_type='text')"]
    start = time.monotonic()
    with (out / 'stdout.txt').open('xb') as stdout, (out / 'stderr.txt').open('xb') as stderr:
        proc = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=450)
    report = {'node': node, 'revision': args.revision, 'selected_rank': 0,
              'scope': 'One real-request rank only; earlier full hardware/path map is reused. '
                       'No compute devices, no NPU microbenchmark, no all-rank claim.',
              'returncode': proc.returncode, 'elapsed_s': time.monotonic() - start,
              'argv': argv, 'compute_devices': [], 'query_devices': ['/dev/davinci_manager'],
              'preserved_raw_files': len(manifest), 'preserved_raw_sha256_verified': True,
              'raw_manifest_sha256': digest(out / 'raw-manifest.json'),
              'trace_paths': [str(p.relative_to(root)) for p in selected.rglob('trace_view.json')]}
    for name in ('stdout.txt', 'stderr.txt'):
        report[name] = {'sha256': digest(out / name),
                        'tail': (out / name).read_text(errors='replace').splitlines()[-8:]}
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert proc.returncode == 0 and report['trace_paths']


if __name__ == '__main__':
    main()
