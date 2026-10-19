#!/usr/bin/env python3
"""Join frozen real-KV integer controls by exact MemCache key and entry."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tarfile


def read_archive(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    path = Path(receipt['local_archive'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt['sha256']
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def controls(files):
    layouts, rows = {}, []
    for name, data in files.items():
        if not name.startswith('consumer-audits/payload-control-v2/kv-control-audit/'):
            continue
        if '/layout-' in name:
            value = json.loads(data)
            layouts[value['pid']] = value
        if '/buffers-' in name:
            rows.extend(json.loads(line) for line in data.splitlines())
    return layouts, rows


def signature(layout):
    result = []
    for entry in layout['groups']['0']:
        labels = []
        for item in entry['labels']:
            labels.append({key: value[1:] if key in ('shape', 'stride') else value
                           for key, value in item.items()})
        result.append({**entry, 'labels': labels})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    files = {node: read_archive(args.evidence/f'node{node}-payload-control-v2-backup.json') for node in (32, 33)}
    native = json.loads(files[32]['tests/payload-control-v2/result.json'])
    layouts_p, p = controls(files[32])
    layouts_d, d = controls(files[33])
    assert len(layouts_p) == len(layouts_d) == 16
    before, after = defaultdict(list), defaultdict(list)
    for row in p:
        (before if row['phase'] == 'put_before' else after)[row['key']].append(row)
    target = defaultdict(list)
    for row in d:
        assert row['phase'] == 'get_after'
        target[row['key']].append(row)
    waves = []
    mismatches = []
    for wave in native['waves']:
        expected = wave['expected_pool_keys']
        pairs = 0
        source_changed, payload_changed, layout_changed = [], [], []
        for key in expected:
            assert len(before[key]) == len(after[key]) == 1, key
            b, a = before[key][0], after[key][0]
            assert b['sizes'] == a['sizes'] and b['bytes'] == a['bytes'] == 6979584
            if b['checksums'] != a['checksums']:
                source_changed.append(key)
            consumers = target[key]
            assert len(consumers) == (0 if key == expected[-1] else 16), (key, len(consumers))
            for row in consumers:
                pairs += 1
                sig_p, sig_d = signature(layouts_p[a['pid']]), signature(layouts_d[row['pid']])
                if sig_p != sig_d:
                    layout_changed.append({'key': key, 'rank': layouts_d[row['pid']]['tp_rank']})
                assert row['bytes'] == 6979584 and row['sizes'] == a['sizes']
                assert len(a['checksums']) == len(row['checksums']) == 115
                differences = [i for i, (x, y) in enumerate(zip(a['checksums'], row['checksums'])) if x != y]
                if differences:
                    payload_changed.append({'key': key, 'rank': layouts_d[row['pid']]['tp_rank'], 'entries': differences})
                    mismatches.append({'wave': wave['wave'], 'key': key, 'rank': layouts_d[row['pid']]['tp_rank'],
                                       'entries': [{'entry': i, 'labels': layouts_d[row['pid']]['groups']['0'][i]['labels'],
                                                    'source': a['checksums'][i], 'target': row['checksums'][i]} for i in differences]})
        waves.append({'wave': wave['wave'], 'source_keys': len(expected), 'source_unstable_keys': source_changed,
                      'consumer_key_rank_pairs': pairs, 'differing_layouts': layout_changed,
                      'differing_controls': payload_changed,
                      'pool_matches_current_native': wave['pool_matches_current_native']})
    first = next(iter(layouts_p.values()))
    dtype_counts = Counter(e['labels'][0]['dtype'] for e in first['groups']['0'])
    report = {'completed': native['completed'], 'waves': waves, 'layouts_per_node': 16,
              'per_key_entries': len(first['groups']['0']), 'entry_dtype_counts': dict(dtype_counts),
              'per_key_payload_bytes': 6979584,
              'source_phase_record_counts': dict(Counter(r['phase'] for r in p)),
              'consumer_phase_record_counts': dict(Counter(r['phase'] for r in d)),
              'mismatches': mismatches,
              'scope': 'Four integer moments per real byte-view entry, exact key/layout join. Equality is probabilistic evidence, not cryptographic payload identity. Observer changes synchronization/timing; four passing outputs do not establish a production fix.'}
    path = args.evidence/'real-kv-control-analysis-v2.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({**report, 'mismatches': len(mismatches)}, indent=2))


if __name__ == '__main__':
    main()
