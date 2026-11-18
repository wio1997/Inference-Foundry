#!/usr/bin/env python3
"""Recompute conditional hybrid-cache arithmetic, not runtime capacity or concurrency."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'evidence/vllm_g4_capacity_034.json'


def candidate_interval(num_blocks, max_model_len, reported_tokens):
    """All integer denominators satisfying floor(blocks*length/denominator)==tokens."""
    vals = (num_blocks, max_model_len, reported_tokens)
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in vals):
        raise ValueError('all operands must be positive integers')
    numerator = num_blocks * max_model_len
    lower = numerator // (reported_tokens + 1) + 1
    upper = numerator // reported_tokens
    return (lower, upper) if lower <= upper else None


def reconstruct(doc):
    log = doc['historical_server_log']
    blocks = log['registered_num_blocks_per_rank']
    length = log['logged_max_model_len']
    capacity = log['logged_group_aware_capacity_tokens']
    conditional = candidate_interval(blocks, length, capacity)
    observed = {}
    for phase in ('cold', 'hot'):
        s = doc['periodic_d_scheduler_samples'][phase]
        if s['samples'] != s['running_zero'] + s['running_one'] + s['running_at_least_two']:
            raise ValueError('sample counts inconsistent')
        if not 0 <= s['waiting_zero'] <= s['samples']:
            raise ValueError('waiting count inconsistent')
        observed[phase] = {'sample_count': s['samples'],
                           'sampled_running_max': None if s['running_at_least_two'] else (1 if s['running_one'] else 0),
                           'running_ge_two_samples': s['running_at_least_two'], 'waiting_zero_samples': s['waiting_zero']}
    return {'observed': observed,
            'arithmetic': {'device_blocks': blocks, 'max_model_len': length,
                'reported_group_aware_tokens': capacity,
                'device_bound_denominator_interval': list(conditional) if conditional else None,
                'device_bound_denominator_unique': conditional[0] if conditional and conditional[0] == conditional[1] else None,
                'device_bound_assumption_proven': False,
                'actual_per_request_blocks_verified': False,
                'illustrative_2060_times_640_not_group_aware': blocks * log['logged_lcm_block_size']},
            'scope': 'conditional source-formula arithmetic only; not runtime allocation, causal attribution or concurrency proof'}


def main():
    print(json.dumps(reconstruct(json.loads(DEFAULT.read_text())), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
