#!/usr/bin/env python3
"""Read frozen benchmark and output only anonymous client timing aggregates."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-root', type=Path, required=True)
root = parser.parse_args().results_root
prior_path = Path(__file__).with_name('baseline-latency-anonymized.json')
prior_bytes = prior_path.read_bytes()
assert hashlib.sha256(prior_bytes).hexdigest() == '0819ff3140d285cac881f2e48141548bd08b734e516a1e91687e2108a6dce4e9'
prior = json.loads(prior_bytes)
out = {}
for phase in ('measure1', 'measure2'):
    base = root / phase
    details, databases = list(base.rglob('*details.jsonl')), list(base.rglob('*.db'))
    assert len(details) == len(databases) == 1
    for path, name in ((details[0], 'details_sha256'), (databases[0], 'db_sha256')):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == prior['phases'][phase][name]
    conn = sqlite3.connect(f'file:{databases[0]}?mode=ro', uri=True)
    arrays, refs = [], set()
    with details[0].open() as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            assert row['success'] and row['output_tokens'] == 1024
            ref = row['time_points']['__db_ref__']
            assert ref not in refs
            refs.add(ref)
            result = conn.execute('SELECT arr_blob FROM numpy_store WHERE id=?', (ref,)).fetchall()
            assert len(result) == 1
            a = np.load(io.BytesIO(result[0][0]), allow_pickle=False)
            assert a.ndim == 1 and len(a) > 2 and np.isfinite(a).all() and (np.diff(a) >= 0).all()
            arrays.append(a)
    conn.close()
    assert len(arrays) == 8
    last_first = float(max(a[1] for a in arrays))
    intervals = [(float(a[i]), float(a[i+1])) for a in arrays for i in range(1, len(a)-1)]
    long_gaps = [(s, e) for s, e in intervals if e-s > 0.1]
    later = [(e-s)*1000 for s, e in intervals if s >= last_first]
    assert later and long_gaps
    out[phase] = {
        'requests': len(arrays),
        'long_gap_count_over_100ms': len(long_gaps),
        'long_gaps_ending_at_or_before_last_first': sum(e <= last_first for s, e in long_gaps),
        'long_gaps_crossing_last_first': sum(s < last_first < e for s, e in long_gaps),
        'long_gaps_starting_at_or_after_last_first': sum(s >= last_first for s, e in long_gaps),
        'last_long_gap_end_before_last_first_ms': float((last_first-max(e for s, e in long_gaps))*1000),
        'gaps_starting_after_all_first_count': len(later),
        'max_gap_starting_after_all_first_ms': max(later),
    }
print(json.dumps(out, indent=2))
