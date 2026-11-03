#!/usr/bin/env python3
"""Recheck F16 exported percentages against a documented candidate formula, offline."""
from decimal import Decimal as D
from hashlib import sha256
from pathlib import Path
import json
import re

root = Path(__file__).resolve().parents[2]
inputs = {
    'counters': ('agents/fusion/f16-layer24-fixed-counters.json', '70f3d84b682e54766bc0b5e5ba8b04cd4fec8bee944029167e1f5e9e84f8b0b0'),
    'capacity': ('evidence/profile-device-capacity-info.json', 'f0f1e014ce728cf1c81b4f6f76a4c869c5ca9ba77340f794c2fac2d436d345e6'),
}
docs = {}
for name, (relative, expected) in inputs.items():
    data = (root / relative).read_bytes()
    actual = sha256(data).hexdigest()
    if actual != expected:
        raise SystemExit(f'STOP: SHA mismatch for {relative}: {actual}')
    docs[name] = json.loads(data)
metadata = {}
for entry in docs['capacity']['device_info']:
    for info in entry['device_info']:
        dev = int(info['id'])
        if dev in metadata:
            raise SystemExit(f'STOP: duplicate device {dev}')
        metadata[dev] = info
rows = []
seen = set()
for sample in docs['counters']['samples']:
    match = re.search(r'_ep(\d+)_', sample['rank'])
    if not match:
        raise SystemExit('STOP: unexpected rank label')
    dev = int(match.group(1))
    meta = metadata[dev]
    # The frozen profile has equal core/frequency fields on all 16 devices;
    # this is metadata lookup, not proof of hardware peak or exporter behavior.
    freq, cores = D(meta['aic_frequency']), D(meta['ai_core_num'])
    duration_us = D(sample['end_ns'] - sample['start_ns']) / D(1000)
    field = sample['fields']
    if duration_us <= 0 or (dev, sample['run'], sample['label']) in seen:
        raise SystemExit('STOP: invalid or duplicate sample')
    seen.add((dev, sample['run'], sample['label']))
    calculated = D(100) * D(field['aic_total_cycles']) / (freq * cores * duration_us)
    reported = D(field['cube_utilization(%)'])
    error = calculated - reported
    rows.append({'device': dev, 'run': sample['run'], 'label': sample['label'],
                 'duration_us': str(duration_us), 'total_cycles': field['aic_total_cycles'],
                 'frequency_mhz_metadata': str(freq), 'core_count_metadata': str(cores),
                 'reported_pct': str(reported), 'calculated_pct': str(calculated),
                 'difference_pct_points': str(error)})
if len(rows) != 96 or set(metadata) != set(range(16)):
    raise SystemExit('STOP: expected 96 samples and 16 distinct devices')
summary = {}
for label in ('first', 'second'):
    subset = [r for r in rows if r['label'] == label]
    if len(subset) != 48:
        raise SystemExit(f'STOP: expected 48 {label} rows')
    summary[label] = {'count': len(subset), 'max_abs_difference_pct_points': str(max(abs(D(r['difference_pct_points'])) for r in subset))}
output = {'source_sha256': {name: expected for name, (_, expected) in inputs.items()},
          'formula': '100 * aic_total_cycles / (aic_frequency_MHz * ai_core_num * duration_us)',
          'scope': '96 stored F16 CSV rows; candidate formula only; not exporter proof',
          'summary': summary, 'samples': rows}
# Print only: callers may redirect deliberately to a task-local file.
print(json.dumps(output, ensure_ascii=False, indent=2))
