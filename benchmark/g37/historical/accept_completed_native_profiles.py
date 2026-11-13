"""Recheck downloaded terminal native profiles, including SQLite request timings."""
from pathlib import Path
import argparse
import ast
import csv
import datetime
import hashlib
import json
import math
import sqlite3

from analyze_aisbench_results import analyze
from analyze_stream_timeline import load_times


def timestamps(detail, rows, measured):
    intervals = []
    databases = {}
    try:
        for row in rows:
            dbpath = detail.parent / 'db_data' / row['db_name']
            if dbpath not in databases:
                databases[dbpath] = sqlite3.connect(f'file:{dbpath}?mode=ro', uri=True)
            ref = row['time_points']
            if isinstance(ref, str):
                ref = ast.literal_eval(ref)
            t = load_times(databases[dbpath], ref['__db_ref__'])
            intervals.append((t[0], t[1], t[-1]))
    finally:
        for connection in databases.values():
            connection.close()
    start = min(x[0] for x in intervals)
    end = max(x[2] for x in intervals)
    duration = end - start
    assert math.isclose(duration, measured['duration_s'], abs_tol=.001)
    events = sorted([(a, 1) for a, _, b in intervals] +
                    [(b, -1) for a, _, b in intervals])
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    assert peak == 16 and active == 0
    average = sum(b - a for a, _, b in intervals) / duration
    assert math.isclose(average, measured['average_concurrency'], abs_tol=.0001)
    assert math.isclose(sum(first - a for a, first, _ in intervals) / len(rows),
                        measured['ttft_mean_ms'] / 1000, abs_tol=.0001)
    assert math.isclose(sum(b - a for a, _, b in intervals) / len(rows),
                        measured['e2el_mean_ms'] / 1000, abs_tol=.0001)
    return {'requests': len(rows), 'duration_s': duration,
            'observed_peak_concurrency': peak, 'observed_average_concurrency': average}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('remote_verification', type=Path)
    args = parser.parse_args()
    r = Path(__file__).resolve().parent
    transport = json.loads(args.remote_verification.read_text())
    assert transport['exit_code'] == 0 and transport['remote_status_received']
    report = json.loads(next(line for line in transport['stdout'].splitlines()
                             if line.startswith('{')))
    assert report['engine'] == 'vllm'
    root = r / 'results/vllm' / report['run_id']
    accepted = []
    warmups = []
    pending_download = []
    for expected in report['formal_metrics']:
        length = expected['input_tokens']
        if not (root / f'formal_{length}').is_dir():
            pending_download.append(length)
            continue
        actual = analyze(root, length)
        assert actual == expected, length
        for relative, sha in actual['evidence'].items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == sha
        detail = next((root / f'formal_{length}').rglob('*_details.jsonl'))
        rows = [json.loads(line) for line in detail.read_text().splitlines() if line]
        inputs = [json.loads(line)['question'] for line in
                  (r / f'datasets/formal_input_{length}.jsonl').read_text().splitlines()
                  if line]
        assert sorted(row['input'] for row in rows) == sorted(inputs)
        accepted.append({'input_tokens': length, 'metrics': actual,
                         'SQLite_timestamp_contract': timestamps(detail, rows, actual),
                         'actual_inputs_match_frozen_dataset': True})
    for check in report['phase_checks']:
        if check['phase'] != 'warmup':
            continue
        detail = root / check['details']
        if not detail.is_file():
            continue
        assert hashlib.sha256(detail.read_bytes()).hexdigest() == check['details_sha256']
        rows = [json.loads(line) for line in detail.read_text().splitlines() if line]
        length = check['input_tokens']
        assert len(rows) == 16
        assert all(x['success'] and x['input_tokens'] == length and
                   x['output_tokens'] == 1024 for x in rows)
        expected_inputs = [json.loads(line)['question'] for line in
                           (r / f'datasets/warmup_input_{length}.jsonl').read_text().splitlines()
                           if line]
        assert sorted(x['input'] for x in rows) == sorted(expected_inputs)
        common_path = detail.with_name(detail.name.replace('_details.jsonl', '.json'))
        common = json.loads(common_path.read_text())
        with common_path.with_suffix('.csv').open(newline='') as f:
            stats = {x['Performance Parameters']: x for x in csv.DictReader(f)}
        measured = {'duration_s': check['duration_s'],
                    'average_concurrency': float(str(common['Concurrency']['total']).split()[0])}
        for label, key in [('TTFT', 'ttft_mean_ms'), ('E2EL', 'e2el_mean_ms')]:
            columns = [k for k in stats[label] if k.lower() in {'mean', 'average'}]
            assert len(columns) == 1
            value = stats[label][columns[0]]
            assert value.endswith(' ms')
            measured[key] = float(value.split()[0])
        warmups.append({'input_tokens': length, 'successful_requests': 16,
                        'unscored': True, 'details_sha256': check['details_sha256'],
                        'actual_inputs_match_frozen_dataset': True,
                        'SQLite_timestamp_contract': timestamps(detail, rows, measured)})
    readiness = json.loads((r / 'vllm_v2/readiness/summary.json').read_text())
    output = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'native166_downloaded_formal_profiles_raw_accepted',
              'hardware': 'BAAI166_910C_aarch64_8physical16logical',
              'run_id': report['run_id'], 'image_id': readiness['image'],
              'container_id': readiness['container_id'],
              'configuration_manifest': 'configs/native_vllm_v2_manifest.json',
              'Engram_DRAM_Rss_KiB': readiness['deduplicated_Rss_KiB'],
              'remote_verification': str(args.remote_verification),
              'remote_ownership_samples_checked': report['ownership_samples_verified'],
              'accepted_profiles': accepted,
              'accepted_unscored_warmups': warmups,
              'completed_profiles_pending_download': pending_download,
              'all4_formal_locally_accepted': len(accepted) == 4,
              'all8_remote_phases_accepted': report['all_four_profiles_complete'],
              'all8_phases_locally_accepted': len(accepted) == len(warmups) == 4,
              'repeatability_verified': False, 'whole_model_quality_verified': False,
              'scope': 'One native166 run; accepted terminal formal profiles only. '
                       'SGLang comparison, repeatability and quality remain unverified.'}
    dest = r / 'native_vllm_partial_acceptance.json'
    temp = dest.with_suffix('.tmp')
    temp.write_text(json.dumps(output, indent=2) + '\n')
    temp.replace(dest)
    print(json.dumps({'accepted_inputs': [x['input_tokens'] for x in accepted],
                      'output_tokens_per_s': [x['metrics']['aggregate_output_tokens_per_s']
                                              for x in accepted],
                      'actual_peak_concurrency': [x['SQLite_timestamp_contract'][
                          'observed_peak_concurrency'] for x in accepted],
                      'unscored_warmups_accepted': len(warmups),
                      'all4_accepted': output['all4_formal_locally_accepted']}))


if __name__ == '__main__':
    main()
