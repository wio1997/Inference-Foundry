#!/usr/bin/env python3
"""Compare first, second and last token timing from SHA-verified completed run archives."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile


def percentile(values, q):
    values = sorted(values)
    i = (len(values)-1)*q
    lo, hi = math.floor(i), math.ceil(i)
    return values[lo]+(values[hi]-values[lo])*(i-lo)


def read_runs(receipt_path, run_ids):
    receipt = json.loads(receipt_path.read_text())
    archive = Path(receipt['local_archive'])
    assert archive.stat().st_size == receipt['bytes']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['sha256']
    runs = []
    with tarfile.open(archive) as ar:
        for run_id in run_ids:
            result = json.load(ar.extractfile('tests/'+run_id+'/result.json'))
            assert result['completed'] and result['passed']
            cases = {}
            for case in result['cases']:
                rows = json.load(ar.extractfile('tests/'+run_id+'/'+case['case']+'-requests.json'))
                assert all(row['success'] and len(row['token_ids']) == 64 for row in rows)
                assert all(row['usage']['completion_tokens'] == 64 for row in rows)
                values = {'ttft_ms': [], 'time_to_second_ms': [], 'time_to_last_ms': [],
                          'first_second_gap_ms': [], 'maximum_gap_ms': [], 'average_tpot_ms': []}
                for row in rows:
                    t = row['arrivals_ns']
                    values['ttft_ms'].append((t[0]-row['start_ns'])/1e6)
                    values['time_to_second_ms'].append((t[1]-row['start_ns'])/1e6)
                    values['time_to_last_ms'].append((t[-1]-row['start_ns'])/1e6)
                    values['first_second_gap_ms'].append((t[1]-t[0])/1e6)
                    values['maximum_gap_ms'].append(max(b-a for a, b in zip(t, t[1:]))/1e6)
                    values['average_tpot_ms'].append((t[-1]-t[0])/63e6)
                cases[case['case']] = {'requests': len(rows),
                    'valid_output_tok_s': case['effective_output_tok_s'],
                    'metrics': {k: {'p50': statistics.median(v), 'p95': percentile(v, .95)}
                                for k, v in values.items()}}
            runs.append(cases)
    return runs, {'receipt': str(receipt_path), 'archive_sha256': receipt['sha256'],
                  'run_ids': run_ids}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-receipt', type=Path, required=True)
    p.add_argument('--candidate-receipt', type=Path, required=True)
    p.add_argument('--baseline-run', action='append', required=True)
    p.add_argument('--candidate-run', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    baseline, bs = read_runs(args.baseline_receipt, args.baseline_run)
    candidate, cs = read_runs(args.candidate_receipt, args.candidate_run)
    assert set(baseline[0]) == set(candidate[0])
    report = {'scope': 'Client monotonic timing of complete64-token responses. All output counts '
              'and frontend usage checked. Values are means of run percentiles, not pooled '
              'percentiles. First-to-second and completion timing must accompany TTFT.',
              'sources': {'baseline': bs, 'candidate': cs}, 'cases': []}
    for case_id in baseline[0]:
        row = {'case': case_id}
        for label, runs in [('baseline', baseline), ('candidate', candidate)]:
            cases = [run[case_id] for run in runs]
            row[label] = {'requests': sum(c['requests'] for c in cases),
                'mean_valid_output_tok_s': statistics.mean(c['valid_output_tok_s'] for c in cases),
                'metrics': {k: {q: statistics.mean(c['metrics'][k][q] for c in cases)
                               for q in ('p50', 'p95')} for k in cases[0]['metrics']}}
        row['candidate_baseline_ratios'] = {
            k: row['candidate']['metrics'][k]['p50']/row['baseline']['metrics'][k]['p50']
            for k in row['baseline']['metrics']
            if row['baseline']['metrics'][k]['p50'] != 0}
        row['output_rate_ratio'] = (row['candidate']['mean_valid_output_tok_s']
                                   / row['baseline']['mean_valid_output_tok_s'])
        report['cases'].append(row)
    report['geomean_ratios'] = {
        k: math.exp(statistics.mean(math.log(row['candidate_baseline_ratios'][k])
                                   for row in report['cases']))
        for k in report['cases'][0]['candidate_baseline_ratios']}
    report['geomean_output_rate_ratio'] = math.exp(statistics.mean(
        math.log(row['output_rate_ratio']) for row in report['cases']))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'geomean_ratios': report['geomean_ratios'],
                      'geomean_output_rate_ratio': report['geomean_output_rate_ratio']}, indent=2))


if __name__ == '__main__':
    main()
