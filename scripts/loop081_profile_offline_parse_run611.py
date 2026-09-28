#!/usr/bin/env python3
"""SHA-copy and parse Run610 salvaged Level0 raw after owned stop/restore."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path('/data/wio/Inference_Foundry')
RUN610 = ROOT / 'evidence/20260928_loop081_bound/run610'
ARM = RUN610 / 'live/b'
RUN611 = ROOT / 'evidence/20260928_loop081_bound/run611'
OUT = RUN611 / 'offline_parse'
CONTAINER = 'vllm-ascend26-dsv4f-w4a8'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def check_raw(manifest):
    for relative, expected in manifest['raw_sha256'].items():
        need(sha(ARM / relative) == expected, f'raw source SHA drift: {relative}')


def main():
    admission_path = RUN611 / 'raw_recovery_admission.json'
    admission = load(admission_path)
    need(admission['status'] == 'all8_cohort5_raw_profile_salvaged_after_invalid_validator_clock_gate' and
         admission['run_ts'] == 'LOOP081-RUN610-B' and
         len(admission['records']) == 8, 'Run610 profile admission')
    cleanup = dict(line.split('=', 1) for line in
                   (RUN610 / 'live/cleanup_status.txt').read_text().splitlines())
    need(set(cleanup) == {'run_exit', 'stop_exit', 'stop_verify_exit',
                          'restore_exit', 'source_sha_exit', 'source_compare_exit',
                          'script_sha_exit', 'script_compare_exit', 'final_exit'} and
         cleanup['run_exit'] == cleanup['final_exit'] == '1' and
         all(value == '0' for key, value in cleanup.items()
             if key not in ('run_exit', 'final_exit')),
         'Run610 invalid acquisition with clean stop/restore')
    need((RUN610 / 'live/source_before.sha256').read_bytes() ==
         (RUN610 / 'live/source_after.sha256').read_bytes() and
         (RUN610 / 'live/scripts_before.sha256').read_bytes() ==
         (RUN610 / 'live/scripts_after.sha256').read_bytes(),
         'Run610 source/script restoration')
    check_raw(admission)
    if OUT.exists():
        need(not (OUT / 'manifest.json').exists(), 'Run611 already complete')
    else:
        OUT.mkdir(parents=True)
    parsed = []
    for record in admission['records']:
        rank = record['rank']
        source = ARM / record['raw_session']
        target = OUT / 'input' / f'rank{rank}' / source.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        copied = [p for p in target.rglob('*') if p.is_file()]
        need(len(copied) == record['raw_file_count'] - 1,
             f'copied raw count rank {rank}')
        for path in copied:
            original_rel = str((source / path.relative_to(target)).relative_to(ARM))
            need(sha(path) == admission['raw_sha256'][original_rel],
                 f'raw copy SHA drift rank {rank}: {original_rel}')
        code = ('from torch_npu.profiler.profiler import analyse; '
                f'analyse({str(target)!r}, export_type="text")')
        done = subprocess.run(['docker', 'exec', CONTAINER, 'python3', '-c', code],
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=300)
        (OUT / f'rank{rank}_analyse.log').write_text(done.stdout)
        need(done.returncode == 0, f'offline parse rank {rank} exit {done.returncode}')
        csvs = list(target.rglob('task_time.csv'))
        traces = list(target.rglob('trace_view.json'))
        need(len(csvs) == len(traces) == 1 and csvs[0].stat().st_size > 100,
             f'parsed native trace missing rank {rank}')
        parsed.append({'rank': rank, 'copied_raw_files': len(copied),
                       'task_time_csv': str(csvs[0].relative_to(ROOT)),
                       'task_time_sha256': sha(csvs[0]),
                       'trace_view_json': str(traces[0].relative_to(ROOT)),
                       'trace_view_sha256': sha(traces[0])})
        print(json.dumps({'rank': rank, 'status': 'parsed',
                          'task_time_bytes': csvs[0].stat().st_size}), flush=True)
    check_raw(admission)
    result = {'status': 'all8_salvaged_raw_copy_offline_parse_pass',
              'scope': 'Run610 controller remains INVALID; parsed instrumented Current only, no native cycle attribution or unperturbed time yet',
              'raw_recovery_admission_sha256': sha(admission_path),
              'raw_source_immutable': True, 'parsed': parsed}
    (OUT / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'ranks': len(parsed)}))


if __name__ == '__main__':
    main()
