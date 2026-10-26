#!/usr/bin/env python3
"""Admit Run602 same-W0 ownership evidence, never a formal timing result."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm-dir', required=True, type=Path)
    parser.add_argument('--run-ts', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    arm = args.arm_dir
    product = arm / 'product'
    basis_admission = load(arm / 'basis_admission.json')
    client_admission = load(arm / 'client_admission.json')
    need(basis_admission.get('status') == 'diagnostic_compact_basis_all8_admitted',
         'basis admission')
    need(basis_admission.get('run_ts') == args.run_ts, 'basis admission run identity')
    need(client_admission.get('status') == 'client_two_phase_admitted',
         'client admission')
    worker_paths = sorted(product.glob('rank*_cohort*.json'))
    scheduler_paths = sorted(product.glob('scheduler_bulk_*.json'))
    api_paths = sorted(product.glob('api_*.json'))
    need(len(worker_paths) == 32, 'worker file count')
    need(len(scheduler_paths) == 4, 'scheduler bulk count')
    need(len(api_paths) == 48, 'API file count')
    expected_runtime = {f'rank{rank}_cohort{cohort}.json'
                        for rank in range(8) for cohort in range(1, 9)}
    expected_basis = {f'rank{rank}_cohort{cohort}.json'
                      for rank in range(8) for cohort in range(5, 9)}
    expected_basis_sources = (
        {(arm / 'client_admission.json').resolve()} |
        {(arm / 'runtime' / name).resolve() for name in expected_runtime} |
        {(arm / 'basis' / name).resolve() for name in expected_basis}
    )
    need({Path(rel).resolve() for rel in basis_admission['source_sha256']} ==
         expected_basis_sources, 'basis admission exact current-arm source set')
    need({p.name for p in (arm / 'runtime').iterdir()} == expected_runtime,
         'exact Runtime file set')
    need({p.name for p in (arm / 'basis').iterdir()} == expected_basis,
         'exact measured basis file set')
    expected_clients = {f'request_{i:03d}.json' for i in range(48)} | {'summary.json'}
    for phase in ('warmup_client', 'measured_client'):
        need({p.name for p in (arm / phase).iterdir()} == expected_clients,
             f'exact {phase} file set')
    need(set(p.name for p in product.iterdir()) ==
         {'arm'} | {p.name for p in worker_paths + scheduler_paths + api_paths},
         'exact product file set')
    raw_hashes = {}
    for path in (arm / 'basis_admission.json', arm / 'client_admission.json'):
        raw_hashes[str(path)] = sha(path)
    for rel, expected in basis_admission['source_sha256'].items():
        source = Path(rel)
        need(source.is_file() and sha(source) == expected,
             f'basis admission source hash {rel}')
        raw_hashes[str(source)] = expected
    worker = {}
    for rank in range(8):
        for cohort in range(5, 9):
            path = product / f'rank{rank}_cohort{cohort}.json'
            raw_hashes[str(path)] = sha(path)
            row = load(path)
            need(row['rank'] == rank and row['cohort'] == cohort and
                 row['run_ts'] == args.run_ts, 'worker identity')
            marks = row['marks']
            entries = [m for m in marks if m['kind'] == 'execute_entry']
            need(bool(entries), 'execute marks')
            need([m['kind'] for m in marks[-3:]] ==
                 ['runtime_built_host', 'serve_start_host', 'serve_synced_host'],
                 'worker terminal marks')
            need(entries[-1]['total_scheduled_tokens'] == 96 and
                 not entries[-1]['new'], 'handoff entry')
            req_ids = marks[-3]['req_ids']
            need(req_ids == marks[-2]['req_ids'] == marks[-1]['req_ids'],
                 'worker handoff/serve IDs')
            need(req_ids == [x['req_id'] for x in entries[-1]['cached']],
                 'handoff cached order')
            runtime = load(arm / 'runtime' / f'rank{rank}_cohort{cohort}.json')
            need(req_ids == runtime['req_ids'] and runtime['pass'], 'Runtime identity')
            calls = row['calls']
            need(bool(calls), 'missing ordinary calls')
            need({call['kind'] for call in calls} ==
                 {'target_forward', 'propose_and_optional_copy'},
                 'forward/proposal call kinds')
            last_ordinal = 0
            target_ordinals = set()
            proposal_by_req = {}
            for call in calls:
                ordinal = call['execute_mark_count']
                need(1 <= ordinal < len(entries), 'call execute ordinal')
                need(call['host_start_ns'] <= call['host_end_ns'] and
                     entries[ordinal - 1]['t_ns'] <= call['host_start_ns'] <=
                     call['host_end_ns'] <= entries[ordinal]['t_ns'],
                     'call interval ownership')
                need(ordinal >= last_ordinal, 'call order')
                last_ordinal = ordinal
                need(call['current_stream_elapsed_ms'] >= 0, 'device event elapsed')
                need(math.isfinite(call['current_stream_elapsed_ms']),
                     'finite device event elapsed')
                if call['kind'] == 'target_forward':
                    target_ordinals.add(ordinal)
                else:
                    need(ordinal in target_ordinals, 'proposal without forward in execute')
                    for req_id in call['req_ids']:
                        proposal_by_req[req_id] = ordinal
            need(set(req_ids).issubset(proposal_by_req),
                 'handoff request missing observed proposal producer')
            row['last_observed_proposal_ordinal_by_req'] = {
                req_id: proposal_by_req[req_id] for req_id in req_ids
            }
            worker[(rank, cohort)] = row
    for cohort in range(5, 9):
        ref = worker[(0, cohort)]['marks'][-3]['req_ids']
        for rank in range(1, 8):
            need(worker[(rank, cohort)]['marks'][-3]['req_ids'] == ref,
                 'all8 handoff identity/order')
    server_time_namespaces = {row['time_namespace'] for row in worker.values()}
    scheduler_by_req = {}
    for path in scheduler_paths:
        raw_hashes[str(path)] = sha(path)
        payload = load(path)
        need(payload['kind'] == 'bulk_step' and payload['run_ts'] == args.run_ts,
             'scheduler bulk identity')
        server_time_namespaces.add(payload['time_namespace'])
        need(len(payload['rows']) == 12, 'bulk rows')
        for row in payload['rows']:
            req_id = row['req_id']
            need(req_id not in scheduler_by_req, 'duplicate scheduler request')
            pre, accepted, post = row['pre_ids'], row['accepted_bulk_ids'], row['post_ids']
            need(len(row['incoming_bulk_ids']) == 1024, 'incoming fixed Runtime output')
            need(pre + accepted == post, 'bulk prefix append')
            need(row['incoming_bulk_ids'][:len(accepted)] == accepted,
                 'accepted incoming prefix')
            need(len(post) == 1024, 'product stop output count')
            need(row['pre_placeholders'] >= 0 and row['post_placeholders'] >= 0,
                 'placeholder accounting')
            need(row['t_ns'] <= row['post_ns'] <= payload['write_begin_ns'],
                 'scheduler Host time order')
            scheduler_by_req[req_id] = row
    need(len(scheduler_by_req) == 48, 'scheduler request count')
    runtime_ids = {rid for cohort in range(5, 9)
                   for rid in worker[(0, cohort)]['marks'][-3]['req_ids']}
    need(set(scheduler_by_req) == runtime_ids, 'scheduler/Runtime request join')
    for cohort in range(5, 9):
        basis = load(arm / 'basis' / f'rank0_cohort{cohort}.json')
        req_ids = worker[(0, cohort)]['marks'][-3]['req_ids']
        need(basis['req_ids'] == req_ids and basis['cycles'] ==
             load(arm / 'runtime' / f'rank0_cohort{cohort}.json')['cycles'],
             'basis/worker/runtime cohort join')
        need(basis['run_ts'] == args.run_ts, 'basis record run identity')
        for slot, req_id in enumerate(req_ids):
            reconstructed = []
            for tokens, counts in zip(basis['token_history'], basis['count_history']):
                count = counts[slot]
                need(0 <= count <= 8 and all(t >= 0 for t in tokens[slot][:count]),
                     'basis token/count shape')
                reconstructed.extend(tokens[slot][:count][:1024-len(reconstructed)])
                if len(reconstructed) == 1024:
                    break
            need(reconstructed == scheduler_by_req[req_id]['incoming_bulk_ids'],
                 'Runtime basis vs scheduler incoming bulk IDs')
    clients = {}
    for path in sorted((arm / 'measured_client').glob('request_*.json')):
        raw_hashes[str(path)] = sha(path)
        item = load(path)
        row = item['row']
        need(row['error'] is None and row['output_tokens'] == 1024,
             'client success/usage')
        need(row['response_id'] not in clients, 'client duplicate response ID')
        clients[row['response_id']] = item
    for path in sorted((arm / 'warmup_client').glob('request_*.json')):
        raw_hashes[str(path)] = sha(path)
    for phase in ('warmup_client', 'measured_client'):
        path = arm / phase / 'summary.json'
        raw_hashes[str(path)] = sha(path)
    need(len(clients) == 48, 'client request count')
    need({x['response_id'] for x in client_admission['request_index']
          if x['phase'] == 'measured'} == set(clients),
         'client raw/admission response identity')
    api_by_response = {}
    join_rows = []
    for path in api_paths:
        raw_hashes[str(path)] = sha(path)
        api = load(path)
        response = api['request_id']
        need(api['kind'] == 'api_stream' and api['run_ts'] == args.run_ts,
             'API identity')
        server_time_namespaces.add(api['time_namespace'])
        need(response in clients and response not in api_by_response,
             'API/client response join')
        api_by_response[response] = api
        matches = [rid for rid in scheduler_by_req if rid.startswith(response + '-')]
        need(len(matches) == 1, 'API/scheduler ID join')
        req_id = matches[0]
        post = scheduler_by_req[req_id]['post_ids']
        events = api['events']
        flattened = [token for event in events for token in event['token_ids']]
        need(flattened == post, 'API full token IDs vs scheduler post')
        need(api['previous_num_tokens'] == [1024] and
             events[-1]['cum_output_tokens'] == 1024, 'API final count')
        running = 0
        last_seen = -1
        for event in events:
            running += len(event['token_ids'])
            need(event['choice'] == 0 and event['cum_output_tokens'] == running and
                 event['engine_seen_ns'] >= last_seen, 'API cumulative chronology')
            last_seen = event['engine_seen_ns']
            if event['sse_yield_ns'] is not None:
                need(event['sse_yield_ns'] >= event['engine_seen_ns'],
                     'API yield before output seen')
        client_events = clients[response]['events']
        pointer = 0
        yielded_tokens = 0
        received_before_handoff = 0
        cohort = next(c for c in range(5, 9)
                      if req_id in worker[(0, c)]['marks'][-3]['req_ids'])
        handoff_ns = max(worker[(rank, cohort)]['marks'][-3]['t_ns'] for rank in range(8))
        for event in events:
            if event['sse_yield_ns'] is None:
                need('sse_payload_sha256' not in event, 'suppressed API event hash')
                continue
            yielded_tokens += len(event['token_ids'])
            expected_hash = event['sse_payload_sha256']
            while pointer < len(client_events) and client_events[pointer]['payload_sha256'] != expected_hash:
                pointer += 1
            need(pointer < len(client_events), 'yielded token SSE not received')
            recv = client_events[pointer]['fragment_recv_monotonic_ns']
            need(recv >= event['sse_yield_ns'], 'SSE receive before API yield')
            if recv <= handoff_ns:
                received_before_handoff += len(event['token_ids'])
            pointer += 1
        join_rows.append({
            'response_id': response, 'req_id': req_id, 'cohort': cohort,
            'scheduler_pre_actual_ids': len(scheduler_by_req[req_id]['pre_ids']),
            'scheduler_pre_placeholders': scheduler_by_req[req_id]['pre_placeholders'],
            'bulk_incoming': 1024,
            'bulk_accepted': len(scheduler_by_req[req_id]['accepted_bulk_ids']),
            'api_total_ids': len(flattened),
            'api_yield_associated_token_ids': yielded_tokens,
            'client_received_yield_associated_ids_before_allrank_built': received_before_handoff,
            'client_total_usage': clients[response]['row']['output_tokens'],
        })
    need(len(api_by_response) == 48, 'API count')
    need(len(server_time_namespaces) == 1,
         'worker/scheduler/API time namespace mismatch')
    join_rows.sort(key=lambda x: (x['cohort'], x['req_id']))
    output = {
        'status': 'product_identity_pass',
        'scope': 'Current diagnostic W0 ownership/Host timeline only. API yield is not socket delivery; SSE text/token IDs need explicit hash join. Cross-process monotonic clocks assumed same container time namespace; no complete device-ready or numerical Bound.',
        'run_ts': args.run_ts,
        'server_time_namespace': next(iter(server_time_namespaces)),
        'raw_sha256': raw_hashes,
        'join_rows': join_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'status': output['status'], 'requests': len(join_rows),
                      'raw_files': len(raw_hashes)}))


if __name__ == '__main__':
    main()
