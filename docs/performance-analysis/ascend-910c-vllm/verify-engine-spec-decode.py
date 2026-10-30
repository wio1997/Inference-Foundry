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
