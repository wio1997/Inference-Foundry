#!/usr/bin/env python3
"""CPU-only mutation tests for the Run605 frontier structural validator."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from loop081_dispatch_validate_run605 import validate_frontier


def fixture():
    return {
        'calls': [
            {'kind': 'target_forward', 'execute_mark_count': 1,
             'num_tokens_padded': 8, 'host_start_ns': 0, 'host_end_ns': 10},
            {'kind': 'propose_and_optional_copy', 'execute_mark_count': 1,
             'req_ids': ['r1']},
        ],
        'frontier': {
            'first_target_entry_event_present': True,
            'first_draft_entry_event_present': True,
            'first_target_stream': 1,
            'first_draft_stream': 1,
            'forward': [{'ordinal': 1, 'mode': 'FULL', 'batch_num_tokens': 8,
                         'batch_num_reqs': 1,
                         'attention': [{'type': 'DSAMetadata', 'attn_state': 'Decode'}]}],
            'draft_inputs': [{'ordinal': 1, 'req_ids': ['r1'], 'batch_size': 1,
                              'use_cuda_graph': False, 'query_rows': 8,
                              'sample_rows': 7, 'num_query_per_req': 8,
                              'context_rows': 8, 'sample_from_anchor': False,
                              'has_num_rejected': False, 'group_count': 1}],
            'context_stores': [{'ordinal': 1, 'req_ids': ['r1'], 'context_rows': 8,
                                'slot_list_present': True, 'to_first_target_ms': 1.0,
                                'to_first_draft_ms': 2.0}],
            'graphs': [{'ordinal': 1, 'branch': 'replay', 'host_entry_ns': 1,
                        'owner': 'MockModel', 'wrapper_mode': 'FULL',
                        'incoming_mode': 'FULL',
                        'enable_enpu': False, 'use_eagle': False,
                        'is_draft_model': False, 'need_sync': True,
                        'sync_taken': True, 'sync_enter_ns': 2, 'sync_return_ns': 3,
                        'replay_enter_ns': 4, 'replay_return_ns': 5}],
        },
    }


def must_reject(name, mutate):
    row = fixture()
    mutate(row)
    try:
        validate_frontier(row)
    except ValueError:
        return name, 'rejected'
    raise AssertionError('negative accepted: ' + name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert validate_frontier(fixture())['ordinary_pairs'] == 1
    none_case = fixture()
    none_case['frontier']['forward'][0]['mode'] = 'NONE'
    none_case['frontier']['graphs'] = [
        {'ordinal': 1, 'branch': 'fallthrough', 'host_entry_ns': 1,
         'fallthrough_enter_ns': 2, 'host_return_ns': 3,
         'owner': 'MockModel',
         'wrapper_mode': 'FULL', 'incoming_mode': 'NONE',
         'enable_enpu': False, 'use_eagle': False, 'is_draft_model': False}]
    assert validate_frontier(none_case)['ordinary_pairs'] == 1
    tests = dict([
        must_reject('duplicate_target_ordinal', lambda r: r['calls'].insert(0, copy.deepcopy(r['calls'][0]))),
        must_reject('wrong_forward_ordinal', lambda r: r['frontier']['forward'][0].update(ordinal=2)),
        must_reject('draft_graph', lambda r: r['frontier']['draft_inputs'][0].update(use_cuda_graph=True)),
        must_reject('wrong_query_rows', lambda r: r['frontier']['draft_inputs'][0].update(query_rows=7)),
        must_reject('stale_context_ordinal', lambda r: r['frontier']['context_stores'][0].update(ordinal=2)),
        must_reject('missing_first_target', lambda r: r['frontier'].update(first_target_entry_event_present=False)),
        must_reject('false_sync_branch', lambda r: r['frontier']['graphs'][0].update(sync_taken=False)),
        must_reject('negative_context_elapsed', lambda r: r['frontier']['context_stores'][0].update(to_first_draft_ms=-1)),
        must_reject('missing_graph', lambda r: r['frontier'].update(graphs=[])),
        must_reject('missing_metadata', lambda r: r['frontier']['forward'][0].update(attention=[])),
        must_reject('graph_parent', lambda r: r['frontier']['graphs'][0].update(ordinal=2)),
        must_reject('graph_mode', lambda r: r['frontier']['graphs'][0].update(wrapper_mode='PIECEWISE')),
        must_reject('wrong_context_req', lambda r: r['frontier']['context_stores'][0].update(req_ids=['other'])),
        must_reject('full_fallthrough_only', lambda r: r['frontier']['graphs'][0].update(
            branch='fallthrough', incoming_mode='NONE', fallthrough_enter_ns=2)),
        must_reject('graph_outside_target', lambda r: r['frontier']['graphs'][0].update(replay_return_ns=11)),
    ])
    here = Path(__file__).resolve().parent
    validator = here / 'loop081_dispatch_validate_run605.py'
    result = {
        'status': 'pass',
        'results': {'positive': 'pass', **tests},
        'validator_sha256': hashlib.sha256(validator.read_bytes()).hexdigest(),
        'selftest_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'negative_rejections': len(tests)}))


if __name__ == '__main__':
    main()
