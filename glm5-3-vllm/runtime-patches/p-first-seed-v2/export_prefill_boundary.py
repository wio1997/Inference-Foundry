#!/usr/bin/env python3
"""Freeze stopped all16 raw profiles and CPU-export one representative rank."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tarfile
import time


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    root = Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live3-20261018/node32')
    gate = json.loads((root / 'prefill-boundary-profile-v1/result.json').read_text())
    assert gate['completed'] and gate['profile_stopped'] and gate['ranks'] == list(range(16))
    assert all(row['complete_answer_pass'] for row in gate['rows'])
    dirs = [Path(p) for p in gate['new_profile_dirs']]
    assert all(p.parent == root / 'profile' and not p.is_symlink() for p in dirs)
    files = sorted(p for d in dirs for p in d.rglob('*') if p.is_file())
    assert all(not p.is_symlink() for p in files)
    assert sum(p.stat().st_size for p in files) < 4 * 1024**3
    assert shutil.disk_usage(root).free > 4 * 1024**3
    for directory in dirs:
        assert list(directory.glob('profiler_info_*.json')) and list(directory.rglob('host/end_info'))
    out = Path(__file__).resolve().parent / 'artifacts'
    out.mkdir(exist_ok=False)
    for name in ['result.json', 'cases.json']:
        shutil.copy2(root / 'prefill-boundary-profile-v1' / name, out / ('profile-' + name))
    manifest = [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    (out / 'all16-raw-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = out / 'all16-raw.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
        for p, record in zip(files, manifest):
            tar.add(p, arcname=record['path'], recursive=False)
            assert sha(p) == record['sha256'] and p.stat().st_size == record['bytes']
    source = next(p for p in dirs if '_rank0_' in p.name)
    snapshot = root / 'prefill-boundary-rank0-raw-v1' / source.name
    snapshot.mkdir(parents=True, exist_ok=False)
    by_path = {r['path']: r for r in manifest}
    for p in sorted(source.rglob('*')):
        if p.is_file():
            dest = snapshot / p.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            assert sha(dest) == by_path[str(p.relative_to(root))]['sha256']
    image = 'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
    name = 'glm53-boundary-p-rank0-export-v1'
    assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
    argv = ['docker', 'run', '--rm', '--pull', 'never', '--network', 'none', '--name', name,
            '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', '16', '--memory', '64g',
            '--pids-limit', '4096', '--tmpfs', '/tmp:rw,size=2g', '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=1',
            '--device', '/dev/davinci_manager', '-v', str(snapshot) + ':/profiles/' + source.name,
            '-v', '/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro', '--entrypoint', '/usr/bin/timeout', image,
            '--signal=TERM', '--kill-after=15', '420', 'python3', '-c',
            "from torch_npu.profiler.profiler import analyse; analyse('/profiles/" + source.name + "', max_process_number=4, export_type='text')"]
    started = time.monotonic()
    with (out / 'export-stdout.txt').open('xb') as stdout, (out / 'export-stderr.txt').open('xb') as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=450)
    names = ['trace_view.json', 'kernel_details.csv', 'op_statistic.csv', 'api_statistic.csv',
             'communication_matrix.json', 'profiler_info_0.json', 'profiler_metadata.json']
    copied = []
    for name in names:
        for p in snapshot.rglob(name):
            dest = out / p.relative_to(snapshot)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            copied.append({'path': str(dest.relative_to(out)), 'bytes': dest.stat().st_size, 'sha256': sha(dest)})
    report = {'rc': result.returncode, 'duration_s': time.monotonic() - started, 'argv': argv,
              'source_snapshot': str(snapshot), 'compute_devices': [], 'selected_rank': 0,
              'all16_raw_files': len(manifest), 'all16_raw_bytes': sum(p['bytes'] for p in manifest),
              'all16_archive': {'bytes': archive.stat().st_size, 'sha256': sha(archive)},
              'exported_files': copied, 'scope': 'Native profiler stopped before source SHA/archive. All16 raw retained; offline CPU rank0 compute/shape diagnosis, not allrank compute or formal timing. No model restarted or new NPU kernel/context.'}
    traces = list(out.rglob('trace_view.json'))
    if traces:
        try:
            data = json.loads(traces[0].read_text())
            events = data if isinstance(data, list) else data['traceEvents']
            report['valid_trace_events'] = len(events)
        except Exception as exc:
            report['trace_parse_failure'] = repr(exc)
    (out / 'export-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('GLM53_RESULT_JSON_BEGIN\n' + json.dumps(report) + '\nGLM53_RESULT_JSON_END')
    assert result.returncode == 0 and report.get('valid_trace_events', 0) > 10000
    assert any(p['path'].endswith('kernel_details.csv') for p in copied)


if __name__ == '__main__':
    main()
