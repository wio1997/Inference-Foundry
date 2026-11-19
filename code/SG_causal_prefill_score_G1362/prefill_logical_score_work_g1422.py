"""Count logical Prefill score work for resolved contiguous request geometry."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from host_geometry import RequestRows


def sum_causal_widths_through(position, ratio):
    """Return sum(floor(i / ratio) for i in [1, position])."""
    blocks, tail = divmod(position, ratio)
    return ratio * blocks * (blocks - 1) // 2 + blocks * (tail + 1)


def request_work(row, *, ratio, heads, score_budget):
    q, prefix, total = (row[name] for name in
                        ('query_tokens', 'prefix_tokens', 'kv_tokens_through_query'))
    assert q > 0 and prefix >= 0 and total == prefix + q
    keys = total // ratio
    shapes = RequestRows(0, prefix, q).score_tiles(
        compress_ratio=ratio, heads=heads, key_count=keys, budget_bytes=score_budget
    ) if keys else ()
    assert shapes is not None
    visible = sum_causal_widths_through(total, ratio) - sum_causal_widths_through(prefix, ratio)
    dense = q * keys
    trimmed = sum((tile.end - tile.start) * tile.key_width for tile in shapes)
    assert 0 <= visible <= trimmed <= dense
    return {
        'query_rows': q,
        'full_compressed_K_rows': keys,
        'causally_eligible_pairs_before_topk': visible,
        'dense_pairs_before_mask': dense,
        'pairs_after_query_tile_suffix_trim': trimmed,
        'predicted_dense_score_calls': len(shapes),
    }


def summarize(geometry, *, ratio, heads, score_budget):
    assert ratio in (1, 2) and heads > 0 and score_budget > 0
    steps = []
    statuses = Counter()
    totals = Counter()
    for step in geometry['steps']:
        status = step['status']
        statuses[status] += 1
        if status != 'unique':
            steps.append({'step': step['step'], 'status': status})
            continue
        per_request = [
            request_work(row, ratio=ratio, heads=heads, score_budget=score_budget)
            for row in step['requests']
        ]
        assert per_request
        aggregate = Counter()
        for item in per_request:
            aggregate.update({key: value for key, value in item.items()
                              if key != 'full_compressed_K_rows'})
        totals.update(aggregate)
        steps.append({'step': step['step'], 'status': status, 'requests': per_request,
                      'step_totals': dict(aggregate)})
    return {
        'compress_ratio': ratio,
        'index_heads_per_rank': heads,
        'BF16_score_tile_budget': score_budget,
        'geometry_status_counts': dict(statuses),
        'totals_per_single_indexer_pass': dict(totals),
        'dense_score_cells_per_rank': totals['dense_pairs_before_mask'] * heads,
        'eligible_score_cells_per_rank': totals['causally_eligible_pairs_before_topk'] * heads,
        'steps': steps,
        'scope': ('Contiguous eager SGLang geometry, one indexer pass, before top-k, with the '
                  'specified native score budget. Counts model logical work. Use the actual '
                  'trace for how many passes took place, HBM traffic and device timing.'),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('geometry_json', type=Path)
    cli.add_argument('--ratio', type=int, required=True)
    cli.add_argument('--heads', type=int, required=True)
    cli.add_argument('--score-budget', type=int, required=True)
    args = cli.parse_args()
    raw = args.geometry_json.read_bytes()
    result = summarize(json.loads(raw), ratio=args.ratio, heads=args.heads,
                       score_budget=args.score_budget)
    result['input_geometry_sha256'] = sha256(raw).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
