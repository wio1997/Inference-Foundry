"""Recover exact per-request geometry when SGLang Prefill moments determine it."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import isqrt
from pathlib import Path


def context_moments(rows):
    """Calculate the four native CPU aggregates from (query, prefix + query)."""
    return {
        'c_sq': sum(q for q, _ in rows),
        'c_sqsq': sum(q * q for q, _ in rows),
        'c_sqsk': sum(q * k for q, k in rows),
        'c_sk': sum(k for _, k in rows),
    }


def recover_extend_geometry(step):
    """Return exact unordered request shapes or the reason they are unresolved."""
    if step['mode'] != 'EXTEND':
        return 'outside_pure_prefill', []
    fields = step['logical_fields']
    keys = ('c_sq', 'c_sqsq', 'c_sqsk', 'c_sk')
    if not all(key in fields for key in keys):
        return 'missing_context_moments', []
    q_sum, q_square_sum, qk_sum, k_sum = (fields[key] for key in keys)
    if step['batch_size'] == 1:
        raw = [(q_sum, k_sum)]
    elif step['batch_size'] == 2:
        # The query difference is sqrt(2 * (q1² + q2²) - (q1 + q2)²).
        difference_squared = 2 * q_square_sum - q_sum * q_sum
        if difference_squared < 0:
            return 'inconsistent_moments', []
        difference = isqrt(difference_squared)
        if difference * difference != difference_squared or (q_sum - difference) % 2:
            return 'inconsistent_moments', []
        q_low = (q_sum - difference) // 2
        q_high = q_sum - q_low
        if q_low <= 0 or k_sum < q_sum:
            return 'inconsistent_moments', []
        if difference:
            # Solve q_low*k_low + q_high*(k_sum-k_low) = qk_sum.
            k_low, remainder = divmod(q_high * k_sum - qk_sum, difference)
            if remainder:
                return 'inconsistent_moments', []
            raw = [(q_low, k_low), (q_high, k_sum - k_low)]
        else:
            if qk_sum != q_low * k_sum:
                return 'inconsistent_moments', []
            if k_sum - q_sum > 1:
                return 'ambiguous_equal_query_rows', []
            raw = [(q_low, q_low), (q_high, k_sum - q_low)]
    else:
        return 'batch_size_outside_reconstruction_scope', []
    if any(q <= 0 or k < q for q, k in raw) or context_moments(raw) != {k: fields[k] for k in keys}:
        return 'inconsistent_moments', []
    return 'unique', [
        {'query_tokens': q, 'prefix_tokens': k - q, 'kv_tokens_through_query': k}
        for q, k in raw
    ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('step_attribution_json', type=Path)
    args = cli.parse_args()
    raw = args.step_attribution_json.read_bytes()
    results = []
    counts = Counter()
    for step in json.loads(raw)['steps']:
        outcome, geometry = recover_extend_geometry(step)
        counts[outcome] += 1
        results.append({'step': step['step'], 'status': outcome, 'requests': geometry})
    print(json.dumps({
        'attribution_input_sha256': sha256(raw).hexdigest(),
        'status_counts': dict(counts),
        'scope': 'Unordered logical per-request shapes on pure EXTEND spans. KV length includes '
                 'the query chunk and precedes index compression or device padding.',
        'steps': results,
    }, indent=2))


if __name__ == '__main__':
    main()
