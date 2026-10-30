"""Verify published aggregate arithmetic, not historical request identity."""
import json
from decimal import Decimal
from pathlib import Path

data = json.loads(Path(__file__).with_name('engine-spec-decode.json').read_text())
for name, engines in data['rounds'].items():
    for label, x in engines.items():
        assert x['generation_tokens'] == 4096 and x['decode_observations'] == 4
        assert x['proposed_tokens'] == 5 * x['drafts']
        mean = Decimal(x['decode_sum_seconds']) / x['decode_observations']
        accepted = 100 * Decimal(x['accepted_tokens']) / x['proposed_tokens']
        assert mean == Decimal(x['decode_mean_seconds'])
        assert abs(accepted - Decimal(x['accepted_over_proposed_percent'])) < Decimal('1e-24')
        print(f'{name} engine {label}: drafts={x["drafts"]}, acceptance={accepted:.2f}%, decode={mean:.3f} s')
print('Aggregate arithmetic only: no request pairing or causal proof.')
prefix = json.loads(Path(__file__).with_name('spec-acceptance-prefix.json').read_text())
for name, x in prefix['rounds'].items():
    edges = [x['drafts']] + x['accepted_position_counts'] + [0]
    hist = [edges[i] - edges[i+1] for i in range(6)]
    assert min(hist) >= 0 and sum(hist) == x['drafts']
    assert hist == x['conditional_exact_accepted_lengths_0_through_5']
    assert sum(i * v for i, v in enumerate(hist)) == x['accepted_total']
    assert x['drafts'] == sum(e['drafts'] for e in data['rounds'][name].values())
    assert x['accepted_total'] == sum(e['accepted_tokens'] for e in data['rounds'][name].values())
    print(f'{name}: conditional acceptance histogram 0..5 = {hist}')
