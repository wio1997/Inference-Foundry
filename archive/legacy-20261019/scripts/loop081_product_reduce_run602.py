#!/usr/bin/env python3
"""Compact post-run accounting for the Run602 diagnostic Product capture."""

import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arm = args.arm_dir.resolve()
    live = arm.parent
    admission = load(arm / 'product_admission.json')
    need(admission['status'] == 'product_identity_pass', 'Product admission')
    for filename, digest in admission['raw_sha256'].items():
        path = Path(filename).resolve()
        need(path.is_file() and sha(path) == digest, f'raw SHA: {filename}')
    need((live / 'source_before.sha256').read_bytes() ==
         (live / 'source_after.sha256').read_bytes(), 'source restore SHA')
    need((live / 'scripts_before.sha256').read_bytes() ==
         (live / 'scripts_after.sha256').read_bytes(), 'script SHA')
    cleanup = dict(line.split('=', 1) for line in
                   (live / 'cleanup_status.txt').read_text().splitlines())
    need(len(cleanup) == 9 and all(value == '0' for value in cleanup.values()),
         'nine cleanup exit codes')
    measured = load(arm / 'measured.log')
    need(measured['success'] == 48 and measured['fail'] == 0 and
         measured['clock']['time_namespace'] == admission['server_time_namespace'],
         'client success and clock namespace')
    rows = admission['join_rows']
    need(len(rows) == 48 and {r['cohort'] for r in rows} == set(range(5, 9)),
         'cohort rows')
    api = {payload['request_id']: payload for path in (arm / 'product').glob('api_*.json')
           for payload in [load(path)]}
    clients = {payload['row']['response_id']: payload
               for path in (arm / 'measured_client').glob('request_*.json')
               for payload in [load(path)]}
    scheduler = [load(path) for path in (arm / 'product').glob('scheduler_bulk_*.json')]
    need(len(api) == len(clients) == 48 and len(scheduler) == 4,
         'raw Product file counts')
    fields = ('scheduler_pre_actual_ids', 'scheduler_pre_placeholders',
              'bulk_incoming', 'bulk_accepted', 'api_total_ids',
              'api_yield_associated_token_ids',
              'client_received_yield_associated_ids_before_allrank_built',
              'client_total_usage')
    summaries = []
    for cohort in range(5, 9):
        subset = [row for row in rows if row['cohort'] == cohort]
        need(len(subset) == 12, 'cohort request count')
        responses = {row['response_id'] for row in subset}
        workers = [load(arm / 'product' / f'rank{rank}_cohort{cohort}.json')
                   for rank in range(8)]
        first_execute = min(worker['marks'][0]['t_ns'] for worker in workers)
        built = max(next(mark['t_ns'] for mark in worker['marks']
                         if mark['kind'] == 'runtime_built_host')
                    for worker in workers)
        synced = max(next(mark['t_ns'] for mark in worker['marks']
                          if mark['kind'] == 'serve_synced_host')
                     for worker in workers)
        relevant = [payload for payload in scheduler
                    if any(row['req_id'].startswith(next(iter(responses)) + '-')
                           for row in payload['rows'])]
        need(len(relevant) == 1, 'cohort scheduler payload')
        scheduler_end = max(row['post_ns'] for row in relevant[0]['rows'])
        api_before_done = max(api[response]['before_done_ns'] for response in responses)
        client_done = max(clients[response]['row']['sse_done_monotonic_ns']
                          for response in responses)
        need(first_execute <= built <= synced <= scheduler_end <=
             api_before_done <= client_done, 'Product Host stage order')
        summaries.append({
            'cohort': cohort,
            'counts': {field: sum(row[field] for row in subset) for field in fields},
            'first_execute_to_allrank_built_s': round((built-first_execute)/1e9, 6),
            'allrank_built_to_allrank_synced_s': round((synced-built)/1e9, 6),
            'allrank_synced_to_scheduler_end_ms': round((scheduler_end-synced)/1e6, 6),
            'allrank_synced_to_api_before_done_ms': round((api_before_done-synced)/1e6, 6),
            'allrank_synced_to_client_done_ms': round((client_done-synced)/1e6, 6),
            'first_execute_to_client_done_s': round((client_done-first_execute)/1e9, 6),
        })
    totals = {field: sum(row[field] for row in rows) for field in fields}
    need(totals['scheduler_pre_actual_ids'] + totals['bulk_accepted'] == 49152 ==
         totals['api_total_ids'] == totals['client_total_usage'],
         'measured output conservation')
    result = {
        'status': 'diagnostic_accounting_pass',
        'scope': 'Same-W0 token accounting and observed Host envelopes; no device-ready time, removable gap, formal TPS or finite Bound.',
        'run_ts': admission['run_ts'],
        'product_admission_sha256': sha(arm / 'product_admission.json'),
        'client_kernel_boot_id': measured['clock']['kernel_boot_id'],
        'shared_time_namespace': admission['server_time_namespace'],
        'cleanup': cleanup,
        'totals': totals,
        'cohorts': summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'totals': totals}))


if __name__ == '__main__':
    main()
