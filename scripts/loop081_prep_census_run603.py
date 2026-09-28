#!/usr/bin/env python3
"""Count ordinary preparation calls on the admitted Run602 W0; no Bound claim."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(ok, reason):
    if not ok:
        raise ValueError(reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arm = args.arm_dir.resolve()
    admission = load(arm / 'product_admission.json')
    check(admission['status'] == 'product_identity_pass', 'Product admission')
    admitted_hashes = {}
    for filename, digest in admission['raw_sha256'].items():
        resolved = str(Path(filename).resolve())
        check(resolved not in admitted_hashes or admitted_hashes[resolved] == digest,
              'duplicate admission SHA disagreement')
        admitted_hashes[resolved] = digest
    rows = []
    source_hashes = {}
    for cohort in range(5, 9):
        allrank = []
        for rank in range(8):
            path = arm / 'product' / f'rank{rank}_cohort{cohort}.json'
            source_hashes[str(path)] = sha(path)
            check(admitted_hashes.get(str(path)) == source_hashes[str(path)],
                  'Product admission source SHA')
            record = load(path)
            check(record['run_ts'] == admission['run_ts'] and
                  record['rank'] == rank and record['cohort'] == cohort,
                  'rank/cohort identity')
            marks = [m for m in record['marks'] if m['kind'] == 'execute_entry']
            targets = {c['execute_mark_count']: c for c in record['calls']
                       if c['kind'] == 'target_forward'}
            proposals = {c['execute_mark_count']: c for c in record['calls']
                         if c['kind'] == 'propose_and_optional_copy'}
            check(len(targets) == sum(c['kind'] == 'target_forward'
                                      for c in record['calls']) and
                  len(proposals) == sum(c['kind'] == 'propose_and_optional_copy'
                                        for c in record['calls']),
                  'duplicate ordinary ordinal')
            check(len(targets) == len(proposals) and
                  set(targets) == set(proposals), 'target/proposal ordinal pair')
            check(len(marks) in (16, 17, 28), 'execute count')
            ordinary = []
            no_call = []
            for ordinal, mark in enumerate(marks, 1):
                if ordinal not in targets:
                    no_call.append((ordinal, mark['total_scheduled_tokens']))
                    continue
                target, proposal = targets[ordinal], proposals[ordinal]
                check(target['host_start_ns'] <= target['host_end_ns'] <=
                      proposal['host_start_ns'] <= proposal['host_end_ns'],
                      'ordinary Host order')
                ordinary.append({
                    'ordinal': ordinal,
                    'new_requests': len(mark['new']),
                    'cached_requests': len(mark['cached']),
                    'scheduled_tokens': mark['total_scheduled_tokens'],
                    'target_num_tokens_padded': target['num_tokens_padded'],
                    'proposal_req_ids': proposal['req_ids'],
                    'target_host_ms': (target['host_end_ns']-target['host_start_ns'])/1e6,
                    'proposal_host_ms': (proposal['host_end_ns']-proposal['host_start_ns'])/1e6,
                    'target_current_stream_event_ms': target['current_stream_elapsed_ms'],
                    'proposal_current_stream_event_ms': proposal['current_stream_elapsed_ms'],
                })
            check(no_call[-1] == (len(marks), 96), 'final 96-token handoff')
            check(all(tokens == 0 for _, tokens in no_call[:-1]),
                  'non-handoff empty execute')
            allrank.append({'rank': rank, 'execute_count': len(marks),
                            'ordinary': ordinary, 'no_call': no_call})
        reference = allrank[0]
        for row in allrank[1:]:
            check(row['execute_count'] == reference['execute_count'] and
                  row['no_call'] == reference['no_call'] and
                  len(row['ordinary']) == len(reference['ordinary']),
                  'allrank execute structure')
            for left, right in zip(row['ordinary'], reference['ordinary']):
                for field in ('ordinal', 'new_requests', 'cached_requests',
                              'scheduled_tokens', 'target_num_tokens_padded',
                              'proposal_req_ids'):
                    check(left[field] == right[field], 'allrank ordinary ' + field)
        ordinary = reference['ordinary']
        pad_shapes = Counter(call['target_num_tokens_padded'] for call in ordinary)
        rows.append({
            'cohort': cohort,
            'allrank_same_dispatch_ledger': True,
            'ordinary_pairs': len(ordinary),
            'empty_execute_ordinals': [ordinal for ordinal, _ in reference['no_call'][:-1]],
            'handoff_ordinal': reference['no_call'][-1][0],
            'new_request_admissions': sum(call['new_requests'] for call in ordinary),
            'ordinary_scheduled_tokens_sum': sum(call['scheduled_tokens'] for call in ordinary),
            'ordinary_target_padded_rows_sum': sum(call['target_num_tokens_padded'] for call in ordinary),
            'target_padded_shape_histogram': dict(sorted(pad_shapes.items())),
            'rank0_host_target_s': round(sum(call['target_host_ms'] for call in ordinary)/1000, 6),
            'rank0_host_proposal_s': round(sum(call['proposal_host_ms'] for call in ordinary)/1000, 6),
            'rank0_current_stream_target_event_s': round(sum(call['target_current_stream_event_ms'] for call in ordinary)/1000, 6),
            'rank0_current_stream_proposal_event_s': round(sum(call['proposal_current_stream_event_ms'] for call in ordinary)/1000, 6),
        })
    check(sum(row['ordinary_pairs'] for row in rows) == 71 and
          sum(row['new_request_admissions'] for row in rows) == 48,
          'frozen W0 call/request census')
    result = {
        'status': 'scoped_current_work_census_pass',
        'scope': 'Observed ordinary-preparation dispatch geometry on Run602 diagnostic W0; call counts and event sums are neither compulsory work nor a critical-path/Bound latency.',
        'run_ts': admission['run_ts'],
        'product_admission_sha256': sha(arm / 'product_admission.json'),
        'source_sha256': source_hashes,
        'cohorts': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'ordinary_pairs':
                      sum(row['ordinary_pairs'] for row in rows)}))


if __name__ == '__main__':
    main()
