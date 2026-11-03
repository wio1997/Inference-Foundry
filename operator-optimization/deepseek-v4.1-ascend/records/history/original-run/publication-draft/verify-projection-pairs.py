#!/usr/bin/env python3
"""Recompute published descriptive ratios, not the profiler event binding."""
import json
import statistics
from decimal import Decimal
from pathlib import Path

data=json.loads((Path(__file__).parent/'projection-pairs.json').read_text())
pairs=data['pairs']
assert len(pairs)==96
expected={4760:(24,Decimal('1.9750795707682849')),5144:(24,Decimal('2.21600399046212')),8180:(48,Decimal('2.237629633519317'))}
for rows,(count,median) in expected.items():
    group=[p for p in pairs if p['physical_rows']==rows]
    assert len(group)==count
    ratios=[]
    for p in group:
        bf=Decimal(p['bf16_us']); fp=list(map(Decimal,p['float_us']))
        assert bf>0 and len(fp)==2 and all(x>0 for x in fp)
        ratios.extend(x/bf for x in fp)
    observed=statistics.median(ratios)
    assert abs(observed-median)<Decimal('0.00000000000001')
    print(f'M={rows}: {count} scopes; FLOAT/BF16 single-task ratios min/median/max = {min(ratios):.3f}/{observed:.3f}/{max(ratios):.3f}')
print('Arithmetic on selected profiled records only; source binding, equal work and causes are not verified here.')
full=[p for p in pairs if Decimal(p['quant_first_overlap_us'])==Decimal(p['quant_duration_us']) and Decimal(p['quant_duration_us'])>0]
long=[p for p in pairs if Decimal(p['float_gap_us'])>100]
assert len(full)==87 and len(long)==9 and all(p['quant_before_first'] for p in long)
assert all(Decimal(p['quant_second_overlap_us'])==0 for p in pairs)
print('Selected quant interval fully overlaps first FLOAT in 87/96 scopes; 9 gaps >100 us have quant ending before first FLOAT.')
print('These are exported intervals, not evidence of physical idle time, causal dependence or end-to-end savings.')
before=[p for p in pairs if p['quant_before_first']]
assert len(before)==9 and len(full)+len(before)==len(pairs)
def first_longer(group):
    return sum(Decimal(p['float_us'][0]) > Decimal(p['float_us'][1]) for p in group)
assert first_longer(pairs)==95 and first_longer(full)==86 and first_longer(before)==9
print('First FLOAT longer: 95/96 overall, 86/87 with selected quant interval inside it, 9/9 when selected quant ended before it.')
print('The selected overlap is not a necessary condition in these records; no causal attribution is implied.')
