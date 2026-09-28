#!/usr/bin/env python3
"""Fail-closed structural admission of the Run606 ordinary dispatch packet."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frontier(worker):
    frontier = worker['frontier']
    calls = worker['calls']
    target = [c for c in calls if c['kind'] == 'target_forward']
    proposal = [c for c in calls if c['kind'] == 'propose_and_optional_copy']
    target_ord = [c['execute_mark_count'] for c in target]
    proposal_ord = [c['execute_mark_count'] for c in proposal]
    need(target_ord == proposal_ord and len(target_ord) == len(set(target_ord)),
         'ordinary Target/proposal pairing')
    need([r['ordinal'] for r in frontier['forward']] == target_ord,
         'forward context ordinal list')
    need([r['ordinal'] for r in frontier['draft_inputs']] == target_ord,
         'Draft input ordinal list')
    need([r['ordinal'] for r in frontier['context_stores']] == target_ord,
         'context store ordinal list')
    need(frontier['first_target_entry_event_present'] and
         frontier['first_draft_entry_event_present'], 'first Runtime entry events')
    need(len(frontier['forward']) <= 512 and len(frontier['draft_inputs']) <= 512
         and len(frontier['context_stores']) <= 512, 'ordinary record pool')
    for call, forward, draft, context in zip(
            target, frontier['forward'], frontier['draft_inputs'],
            frontier['context_stores']):
        need(forward['batch_num_tokens'] == call['num_tokens_padded'] and
             forward['mode'] in ('NONE', 'FULL', 'PIECEWISE', 'FULL_DECODE_ONLY'),
             'Target forward mode/shape')
        need(forward['attention'] and len(forward['attention']) <= 16 and
             any(row['attn_state'] is not None for row in forward['attention']),
             'actual attention metadata missing or truncated')
        need(draft['req_ids'] == proposal[target_ord.index(call['execute_mark_count'])]['req_ids'],
             'Draft request identity')
        need(draft['use_cuda_graph'] is False, 'ordinary DSpark unexpectedly graph')
        need(draft['batch_size'] >= len(draft['req_ids']) and
             draft['query_rows'] == draft['batch_size'] * draft['num_query_per_req'] and
             draft['sample_rows'] == draft['batch_size'] * 7 and
             draft['context_rows'] >= 0, 'Draft row geometry')
        need(context['req_ids'] == draft['req_ids'] and
             context['context_rows'] == draft['context_rows'],
             'context/Draft producer join')
        need(context['slot_list_present'], 'context slot ownership')
        for key in ('to_first_target_ms', 'to_first_draft_ms'):
            value = context[key]
            need(value is not None and math.isfinite(value) and value >= 0,
                 'finite context→Runtime event interval')
    graphs = frontier['graphs']
    need(len(graphs) <= 512 and all(g['ordinal'] in target_ord for g in graphs),
         'graph parent/pool')
    for forward in frontier['forward']:
        if forward['mode'] in ('FULL', 'FULL_DECODE_ONLY'):
            need(any(g['ordinal'] == forward['ordinal'] and
                     g['branch'] in ('capture', 'replay') for g in graphs),
                 'FULL forward missing Graph capture/replay')
    for graph in graphs:
        need(graph['branch'] in ('fallthrough', 'capture', 'replay'),
             'graph branch')
        parent = next(r for r in frontier['forward']
                      if r['ordinal'] == graph['ordinal'])
        parent_call = next(r for r in target
                           if r['execute_mark_count'] == graph['ordinal'])
        graph_end = graph.get('host_return_ns', graph.get('replay_return_ns'))
        need(graph_end is not None and
             parent_call['host_start_ns'] <= graph['host_entry_ns'] <=
             graph_end <= parent_call['host_end_ns'],
             'Graph Host span outside Target')
        need(graph['incoming_mode'] == parent['mode'] and
             graph['wrapper_mode'] in ('FULL', 'FULL_DECODE_ONLY') and
             graph['host_entry_ns'] > 0, 'Graph parent/mode/Host span')
        if graph['branch'] == 'replay':
            need(graph['incoming_mode'] == graph['wrapper_mode'],
                 'replay incoming Graph mode')
            need(graph['sync_taken'] == (not graph['enable_enpu'] and
                 graph['need_sync']), 'existing sync flag mismatch')
            need(graph['host_entry_ns'] <= graph['replay_enter_ns'] <=
                 graph['replay_return_ns'], 'graph replay Host chronology')
            if graph['sync_taken']:
                need(graph['host_entry_ns'] <= graph['sync_enter_ns'] <=
                     graph['sync_return_ns'] <= graph['replay_enter_ns'],
                     'existing graph sync chronology')
            else:
                need('sync_enter_ns' not in graph and 'sync_return_ns' not in graph,
                     'false graph sync branch')
        elif graph['branch'] == 'capture':
            need(graph['incoming_mode'] == graph['wrapper_mode'],
                 'capture incoming Graph mode')
            need(graph['host_entry_ns'] <= graph['host_return_ns'] and
                 'entry_id' in graph, 'capture Host chronology')
        else:
            need(graph['incoming_mode'] != graph['wrapper_mode'] and
                 graph['host_entry_ns'] <= graph['fallthrough_enter_ns'] <=
                 graph['host_return_ns'],
                 'fallthrough Host chronology')
    signature = {
        'forward': [(r['ordinal'], r['mode'], r['batch_num_tokens'],
                     r['batch_num_reqs'], r['attention'])
                    for r in frontier['forward']],
        'draft': [(r['ordinal'], r['req_ids'], r['batch_size'],
                   r['context_rows'], r['query_rows'], r['sample_rows'],
                   r['num_query_per_req'], r['sample_from_anchor'],
                   r['has_num_rejected'], r['group_count'])
                  for r in frontier['draft_inputs']],
        'context': [(r['ordinal'], r['req_ids'], r['context_rows'],
                     r['slot_list_present'])
                    for r in frontier['context_stores']],
        'graph': [(r['ordinal'], r['branch'], r['owner'], r['wrapper_mode'],
                   r['incoming_mode'],
                   r['enable_enpu'], r['use_eagle'], r['is_draft_model'],
                   r.get('need_sync'), r.get('sync_taken')) for r in graphs],
    }
    return {
        'ordinary_pairs': len(target_ord),
        'forward_modes': dict(Counter(r['mode'] for r in frontier['forward'])),
        'graph_branches': dict(Counter(r['branch'] for r in graphs)),
        'existing_replay_sync_count': sum(g.get('sync_taken', False) for g in graphs),
        'draft_context_rows': sum(r['context_rows'] for r in frontier['draft_inputs']),
        'draft_query_rows': sum(r['query_rows'] for r in frontier['draft_inputs']),
        'semantic_signature': signature,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm-dir', type=Path, required=True)
    parser.add_argument('--run-ts', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arm = args.arm_dir.resolve()
    product = load(arm / 'product_admission.json')
    need(product['status'] == 'product_identity_pass' and
         product['run_ts'] == args.run_ts, 'Product upstream gate')
    hashes = {}
    admitted = {str(Path(path).resolve()): digest
                for path, digest in product['raw_sha256'].items()}
    cohorts = []
    expected_returns = {f'return_rank{rank}_cohort{cohort}.json'
                        for cohort in range(5, 9) for rank in range(8)}
    need({p.name for p in (arm / 'returns').iterdir()} == expected_returns,
         'exact return file set')
    for cohort in range(5, 9):
        rank_rows = []
        for rank in range(8):
            path = arm / 'product' / f'rank{rank}_cohort{cohort}.json'
            hashes[str(path)] = sha(path)
            need(admitted.get(str(path)) == hashes[str(path)],
                 'worker/Product SHA join')
            worker = load(path)
            need((worker['rank'], worker['cohort'], worker['run_ts']) ==
                 (rank, cohort, args.run_ts), 'worker identity')
            stats = validate_frontier(worker)
            ret_path = arm / 'returns' / f'return_rank{rank}_cohort{cohort}.json'
            hashes[str(ret_path)] = sha(ret_path)
            returned = load(ret_path)
            built = next(m for m in worker['marks']
                         if m['kind'] == 'runtime_built_host')
            synced = next(m for m in worker['marks']
                          if m['kind'] == 'serve_synced_host')
            need((returned['rank'], returned['cohort'], returned['run_ts']) ==
                 (rank, cohort, args.run_ts) and
                 returned['req_ids'] == built['req_ids'] and
                 returned['model_runner_output_built_ns'] >= synced['t_ns'],
                 'worker output return identity/chronology')
            rank_rows.append(stats)
        need(all(row['semantic_signature'] == rank_rows[0]['semantic_signature']
                 for row in rank_rows), 'allrank ordinary semantic agreement')
        cohorts.append({'cohort': cohort, 'ranks': rank_rows})
    need(sum(row['ranks'][0]['ordinary_pairs'] for row in cohorts) > 0,
         'ordinary calls missing')
    output = {
        'status': 'scoped_dispatch_identity_pass',
        'scope': 'Actual ordinary branch/context geometry and current-stream markers; no complete side-stream readiness, no perturbation-controlled timing or finite Bound.',
        'run_ts': args.run_ts,
        'product_admission_sha256': sha(arm / 'product_admission.json'),
        'raw_sha256': hashes,
        'cohorts': cohorts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'status': output['status'], 'cohorts': len(cohorts),
                      'ordinary_rank0': sum(row['ranks'][0]['ordinary_pairs'] for row in cohorts)}))


if __name__ == '__main__':
    main()
