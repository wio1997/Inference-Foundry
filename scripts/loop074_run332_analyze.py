"""Validate passive all-rank scheduler marks and derive real prompt residual work."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def signature(row):
    data = {k: v for k, v in row.items() if k not in ('rank', 't_ns')}
    return json.dumps(data, sort_keys=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.run_dir
    bench = json.loads((root / 'measured48.json').read_text())
    warmup = json.loads((root / 'warmup48.json').read_text())
    for label, result in [('warmup', warmup), ('measured', bench)]:
        s = result['summary']
        if (s['n'], s['success'], s['fail'], s['concurrency'], s['max_tokens']) != (48, 48, 0, 12, 1024):
            raise RuntimeError(f'{label} client contract invalid')
        if any(x['output_tokens'] != 1024 for x in result['requests']):
            raise RuntimeError(f'{label} output count invalid')
    ranks = []
    for rank in range(8):
        path = root / 'marks' / f'rank{rank}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        ranks.append(rows)
    if len({len(x) for x in ranks}) != 1:
        raise RuntimeError('rank mark count mismatch')
    for step in range(len(ranks[0])):
        if len({signature(rank[step]) for rank in ranks}) != 1:
            raise RuntimeError(f'all8 scheduler/ready semantic mismatch at row{step}')
    ready_indices = [i for i, row in enumerate(ranks[0]) if row['kind'] == 'runtime_built']
    if len(ready_indices) != 4:
        raise RuntimeError(f'expected four measured runtime builds, got {len(ready_indices)}')
    for cohort in range(5, 9):
        reports = [json.loads((root / 'runtime' / f'rank{rank}_cohort{cohort}.json').read_text())
                   for rank in range(8)]
        if any(not x['pass'] or x['target_graph_mode'] != 'FULL' or
               x['generated_output_counts'] != [1024] * 12 for x in reports):
            raise RuntimeError(f'cohort{cohort} Runtime gate invalid')
        if any(x['req_ids'] != reports[0]['req_ids'] for x in reports[1:]):
            raise RuntimeError(f'cohort{cohort} rank slot mismatch')
        if ranks[0][ready_indices[cohort - 5]]['req_ids'] != reports[0]['req_ids']:
            raise RuntimeError(f'cohort{cohort} ready/runtime req_id mismatch')
    request = {}
    event_list = []
    for row in ranks[0]:
        if row['kind'] != 'execute_entry':
            continue
        t = row['t_ns']
        current = []
        for new in row['new']:
            req_id = new['req_id']
            if req_id in request:
                raise RuntimeError(f'duplicate new request {req_id}')
            if new['prompt_len'] is None:
                raise RuntimeError(f'missing prompt len {req_id}')
            request[req_id] = {'prompt_len': new['prompt_len'],
                               'first_computed': new['num_computed_tokens'],
                               'first_seen_ns': t, 'prompt_scheduled': 0,
                               'decode_or_spec_scheduled': 0, 'steps': 0}
            current.append(new)
        current.extend(row['cached'])
        for item in current:
            req_id = item['req_id']
            if req_id not in request:
                raise RuntimeError(f'cached request missing new record {req_id}')
            record = request[req_id]
            remaining_prompt = max(0, record['prompt_len'] - item['num_computed_tokens'])
            prompt = min(item['scheduled_tokens'], remaining_prompt)
            record['prompt_scheduled'] += prompt
            record['decode_or_spec_scheduled'] += item['scheduled_tokens'] - prompt
            record['steps'] += 1
        event_list.append({'t_ns': t, 'total_scheduled_tokens': row['total_scheduled_tokens'],
                           'new_count': len(row['new']), 'cached_count': len(row['cached']),
                           'prompt_scheduled': sum(min(x['scheduled_tokens'],
                               max(0, request[x['req_id']]['prompt_len'] - x['num_computed_tokens']))
                               for x in current)})
    cohorts = []
    assigned = set()
    for wave, idx in enumerate(ready_indices):
        row = ranks[0][idx]
        ids = row['req_ids']
        if any(req_id not in request for req_id in ids):
            raise RuntimeError(f'cohort{wave} missing new request')
        if assigned.intersection(ids):
            raise RuntimeError(f'cohort{wave} duplicate req ID')
        assigned.update(ids)
        r = [request[req_id] for req_id in ids]
        cohorts.append({'cohort': wave + 5, 'runtime_req_ids': ids,
                        'first_computed_tokens': [x['first_computed'] for x in r],
                        'prompt_lengths': [x['prompt_len'] for x in r],
                        'prompt_scheduled_tokens': [x['prompt_scheduled'] for x in r],
                        'decode_or_spec_scheduled_tokens': [x['decode_or_spec_scheduled'] for x in r],
                        'request_execute_steps': [x['steps'] for x in r],
                        'first_request_execute_to_runtime_built_s':
                        (row['t_ns'] - min(x['first_seen_ns'] for x in r)) / 1e9,
                        'first_request_execute_to_runtime_built_allrank_max_s':
                        max((rank[idx]['t_ns'] - min(
                            mark['t_ns'] for mark in rank[:idx]
                            if mark['kind'] == 'execute_entry' and
                            any(x['req_id'] in ids for x in mark['new']))) / 1e9
                            for rank in ranks)})
    if len(assigned) != 48 or len(request) != 48:
        raise RuntimeError(f'request coverage {len(request)}/{len(assigned)}')
    stream_ids = [x.get('stream_id') for x in bench['requests']]
    exact_stream_matches = sum(x in request for x in stream_ids if x)
    client_to_server = {}
    for item in bench['requests']:
        stream_id = item.get('stream_id')
        matches = [req_id for req_id in request if stream_id and
                   req_id.startswith(stream_id + '-')]
        if len(matches) != 1:
            raise RuntimeError(f'client {item["i"]} stream id mapping {len(matches)}')
        client_to_server[item['i']] = matches[0]
    for cohort in cohorts:
        reverse = {v: k for k, v in client_to_server.items()}
        cohort['client_indices_by_runtime_slot'] = [reverse[req_id]
                                                    for req_id in cohort['runtime_req_ids']]
    print(json.dumps({
        'status': 'passive_scheduler_host_marks_only',
        'all8_semantic_row_parity': True,
        'mark_rows_per_rank': len(ranks[0]),
        'measured_client': bench['summary'],
        'four_full_graph_cohorts_all8': True,
        'cohorts': cohorts,
        'request_record_count': len(request),
        'stream_id_exact_server_req_id_matches': exact_stream_matches,
        'stream_id_present_count': sum(bool(x) for x in stream_ids),
        'stream_id_prefix_server_req_id_matches': len(client_to_server),
        'execute_events': event_list,
        'limits': [
            'No device event or complete side-stream/HCCL join measured; runtime_built is Host readiness, not device seed-ready proof.',
            'Client stream_id may differ from internal scheduler req_id; if so client-to-slot mapping remains unknown.',
            'Existing original cohort preparation is measured; incremental refill cost and decode contention remain unknown.',
            'Diagnostic timed path with passive marks, not formal E2E or Product bound.',
        ],
    }, indent=2))


if __name__ == '__main__':
    main()
