"""Audit standard AISBench 3.1 raw outputs; never manufacture measurements."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def atomic(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def number(value):
    value = float(str(value).split()[0])
    assert math.isfinite(value), 'Nonfinite AISBench metric'
    return value


def audit_phase(directory, length, requests=32, usage_contract=None):
    root = Path(directory)
    details = list(root.rglob('*_details.jsonl'))
    assert len(details) == 1, f'Expected one raw details file, found {len(details)}'
    detail = details[0]
    rows = [json.loads(s) for s in detail.read_text().splitlines() if s.strip()]
    assert len(rows) == requests, f'Expected {requests} requests, got {len(rows)}'
    assert all(r.get('success') is True for r in rows), 'Failed request(s) retained in raw details'
    assert all(r.get('input_tokens') == length for r in rows), 'Actual input-token counts mismatch'
    assert all(r.get('output_tokens') == 1024 for r in rows), 'Actual output-token counts mismatch'
    common_path = detail.with_name(detail.name.removesuffix('_details.jsonl') + '.json')
    csv_path = common_path.with_suffix('.csv')
    common = json.loads(common_path.read_text())

    def metric(name):
        value = common[name]
        assert isinstance(value, dict) and set(value) == {'total'}, f'Unexpected stage layout for {name}'
        return number(value['total'])

    assert metric('Total Requests') == metric('Success Requests') == requests
    assert metric('Failed Requests') == 0
    assert metric('Max Concurrency') == 16, 'Observed max concurrency must equal C16'
    assert metric('Total Input Tokens') == requests * length
    assert metric('Total Generated Tokens') == requests * 1024
    raw_duration = str(common['Benchmark Duration']['total'])
    # Pinned 3.1 contract reports ms; reject explicit conflicting units.
    assert len(raw_duration.split()) == 1 or raw_duration.split()[1] == 'ms', raw_duration
    duration_s = metric('Benchmark Duration') / 1000
    assert duration_s > 0
    totals = {'Input Token Throughput': requests * length,
              'Output Token Throughput': requests * 1024,
              'Total Token Throughput': requests * (length + 1024)}
    for name, tokens in totals.items():
        assert math.isclose(metric(name), tokens / duration_s, rel_tol=1e-5, abs_tol=0.01), name
    with csv_path.open(newline='') as f:
        stats = {r['Performance Parameters']: r for r in csv.DictReader(f)}
    latencies = {}
    for name in ('TTFT', 'TPOT', 'E2EL'):
        row = stats[name]
        assert row['Stage'] == 'total'
        for label, aliases in [('mean', {'mean', 'average'}), ('p90', {'p90'}), ('p99', {'p99'})]:
            keys = [k for k in row if k.lower() in aliases]
            assert len(keys) == 1, f'Missing {name} {label}'
            value = str(row[keys[0]])
            assert value.endswith(' ms'), f'Unexpected latency units: {value}'
            latencies[f'{name.lower()}_{label}_ms'] = number(value)
    provenance = {'verified': False, 'reason': 'Raw count fields do not alone prove server usage provenance'}
    if usage_contract:
        cpath = Path(usage_contract)
        contract = json.loads(cpath.read_text())
        assert contract['input_tokens_source'] == 'server_usage.prompt_tokens'
        assert contract['output_tokens_source'] == 'server_usage.completion_tokens'
        assert contract['reviewed'] is True
        assert contract['evidence'], 'Usage contract requires actual client source evidence'
        for e in contract['evidence']:
            source = cpath.parent / e['path']
            assert sha(source) == e['sha256'], source
        provenance = {'verified': True, 'contract_sha256': sha(cpath), 'evidence': contract['evidence']}
    return {'status': 'standard_raw_counts_and_metrics_validated', 'input_tokens': length,
            'output_tokens': 1024, 'requests': requests, 'successful_requests': requests,
            'configured_concurrency': 16, 'observed_max_concurrency': metric('Max Concurrency'),
            'average_concurrency': metric('Concurrency'), 'duration_s': duration_s,
            'input_tokens_per_s': metric('Input Token Throughput'),
            'output_tokens_per_s': metric('Output Token Throughput'),
            'total_tokens_per_s': metric('Total Token Throughput'), **latencies,
            'server_usage_provenance': provenance,
            'fully_validated': provenance['verified'],
            'evidence': {str(p.relative_to(root)): sha(p) for p in (detail, common_path, csv_path)},
            'scope': 'One AISBench phase; no accuracy, repeatability or cache-hit claim'}


def audit_run(root, usage_contract=None):
    root = Path(root)
    state = json.loads((root / 'state.json').read_text())
    manifest = json.loads((root / 'run_manifest.json').read_text())
    results = []
    for length in manifest['lengths']:
        cold_meta = root / f'cold_{length}_command.json'
        hot_meta = root / f'hot_{length}_command.json'
        try:
            warmup_rows = []
            warmup_hashes = []
            phases = manifest['warmup_dataset_phases']
            assert len(phases) >= 2 and len(phases) == manifest['warmup_passes']
            for i, dataset_phase in enumerate(phases):
                name = f'warmup{i + 1}_{length}'
                command = json.loads((root / f'{name}_command.json').read_text())
                assert command['dataset_phase'] == dataset_phase
                assert command['dataset_sha256'] == manifest['warmup_dataset_sha256'][str(length)][dataset_phase]
                warmup_hashes.append(command['dataset_sha256'])
                warmup_audit = audit_phase(root / name, length, requests=16, usage_contract=usage_contract)
                after_path = root / f'after_{name}_server_snapshot.json'
                if after_path.exists():
                    after = json.loads(after_path.read_text())
                    warmup_audit['after_snapshot_posthoc'] = bool(after.get('_capture_provenance', {}).get('posthoc', False))
                warmup_rows.append(warmup_audit)
            cold = json.loads(cold_meta.read_text())
            hot = json.loads(hot_meta.read_text())
            assert cold['dataset_sha256'] == hot['dataset_sha256'], 'Cold/hot data differ'
            assert cold['config_sha256'] == hot['config_sha256'], 'Cold/hot configuration differs'
            assert cold['dataset_sha256'] == manifest['formal_dataset_sha256'][str(length)]
            cold_r = audit_phase(root / f'cold_{length}', length, usage_contract=usage_contract)
            hot_r = audit_phase(root / f'hot_{length}', length, usage_contract=usage_contract)
            if manifest.get('cold_mode', 'flush') == 'flush':
                flush = json.loads((root / f'before_cold_{length}_flush.json').read_text())
                assert set(flush) == {'prefill', 'decode'} and all(x['accepted'] is True for x in flush.values())
                cold_scope = 'Both role resets accepted before cold'
            else:
                assert manifest['gate']['formal_not_previously_submitted'] is True
                assert manifest['cold_prefix_audit_sha256']
                assert len(set(phases)) == len(phases)
                assert len(set(warmup_hashes + [cold['dataset_sha256']])) == len(warmup_hashes) + 1
                cold_scope = 'Audited novel prefixes never previously submitted in this service generation; no flush claimed'
            cache = {}
            for stage in ['cold', 'hot']:
                cache[stage] = json.loads((root / f'{stage}_{length}_cache_audit.json').read_text())
                for boundary in ['before', 'after']:
                    snapshot = json.loads((root / f'{boundary}_{stage}_{length}_server_snapshot.json').read_text())
                    assert not snapshot.get('_capture_provenance', {}).get('posthoc', False), 'Scored snapshots cannot be posthoc'
            for item in cache['cold'].values():
                if item['verified']:
                    assert item['delta'] == 0, 'Cold cache-hit counter increased'
            results.append({'input_tokens': length, 'status': 'cold_hot_raw_metrics_validated',
                            'unscored_warmup_passes': warmup_rows,
                            'cold': cold_r, 'hot_replay': hot_r,
                            'cache_counter_evidence': cache,
                            'cold_cache_counter_verified': all(x['verified'] for x in cache['cold'].values()),
                            'cache_scope': cold_scope + '; hot replays identical data without reset; actual hit efficacy/eviction comes from counters.',
                            'dataset_sha256': cold['dataset_sha256']})
        except Exception as e:
            results.append({'input_tokens': length, 'status': 'NOT_VALIDATED',
                            'error': f'{type(e).__name__}: {e}'})
    accepted = (state['phase'] == 'all_phases_terminal' and
                all(r['status'] != 'NOT_VALIDATED' and r['cold']['fully_validated'] and r['hot_replay']['fully_validated'] for r in results))
    report = {'engine': manifest['engine'], 'run_id': manifest['run_id'],
              'all_required_raw_metrics_and_usage_validated': accepted, 'profiles': results,
              'limitations': ['Cache hit efficacy is not inferred from replay label',
                              'This audit does not establish model accuracy or performance reproducibility']}
    atomic(root / 'audit_results.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_root', type=Path)
    p.add_argument('--usage-contract', type=Path)
    a = p.parse_args()
    report = audit_run(a.run_root, a.usage_contract)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['all_required_raw_metrics_and_usage_validated'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
