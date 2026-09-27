#!/usr/bin/env python3
"""Inventory accepted Run99 files for a formal-window first-position W-minus witness."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / 'evidence/20260924_loop036_metadata/run99'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    summary_file = FORMAL / 'summary.json'
    summary = json.loads(summary_file.read_text())
    if not summary['pass'] or len(summary['runs']) != 3 or summary['runtime_rank_files'] != 128:
        raise ValueError('accepted Run99 scope changed')
    if round(summary['output_tps']['median'], 3) != 571.681:
        raise ValueError('formal Current changed')
    client_files = [FORMAL / f'bench48_{i}.json' for i in (1, 2, 3)]
    client = [json.loads(p.read_text()) for p in client_files]
    if any(len(x['requests']) != 48 or any(r['error'] is not None or r['output_tokens'] != 1024 for r in x['requests']) for x in client):
        raise ValueError('formal client validity changed')
    runtime_files = sorted((FORMAL / 'runtime').glob('rank*_cohort*.json'))
    if len(runtime_files) != 128:
        raise ValueError('formal Runtime file count changed')
    runtime = [json.loads(p.read_text()) for p in runtime_files]
    seen = {(x['rank'], x['cohort']) for x in runtime}
    if seen != {(r, c) for r in range(8) for c in range(1, 17)} or any(x.get('pass') is not True for x in runtime):
        raise ValueError('formal Runtime rank/cohort admission changed')
    client_fields = set.intersection(*(set(r) for x in client for r in x['requests']))
    runtime_fields = set.intersection(*(set(x) for x in runtime))
    required = {
        'client_request_id': any(k in client_fields for k in ('request_id', 'req_id', 'id')),
        'client_raw_token_ids': any(k in client_fields for k in ('token_ids', 'output_token_ids', 'raw_token_ids')),
        'runtime_target_first_position_token': any(k in runtime_fields for k in ('target_argmax', 'first_target_token', 'sampled_token_ids')),
        'runtime_target_first_position_generation': any(k in runtime_fields for k in ('target_generation', 'selected_replay_generation')),
        'runtime_scheduler_pre_append_G': any(k in runtime_fields for k in ('scheduler_pre_append_count', 'pre_append_G')),
        'runtime_external_published_token_sequence': any(k in runtime_fields for k in ('published_token_ids', 'external_token_ids')),
        'runtime_loaded_wo_a_group_provenance': any(k in runtime_fields for k in ('last_layer_wo_a_group', 'first_position_dense_group')),
    }
    if any(required.values()):
        raise ValueError('Run99 witness field inventory changed; review actual semantics')
    files = [summary_file, *client_files, *runtime_files]
    return {
        'valid': True,
        'scope': 'accepted Run99 saved-artifact field inventory only; not a universal absence proof or fresh formal run',
        'formal_current_median_tps': summary['output_tps']['median'],
        'formal_client_repeats': 3,
        'requests_per_repeat': 48,
        'runtime_rank_cohort_files': 128,
        'client_record_fields': sorted(client_fields),
        'runtime_row_fields': sorted(runtime_fields),
        'witness_fields_present': required,
        'request_id_join_from_saved_client_to_runtime': False,
        'positive_formal_window_W_minus_certified': False,
        'reason': 'Accepted client records contain length/timing/chunk count but no request ID or raw token IDs; Runtime rows contain request IDs and aggregate counts but no same-generation first Target token, Scheduler pre-append G, typed dense-group provenance or external raw-token publication sequence.',
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in files},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'valid': True, 'inputs': len(result['inputs']), 'formal_W_minus_certified': False}))


if __name__ == '__main__':
    main()
