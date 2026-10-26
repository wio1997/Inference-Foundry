"""Offline two-cycle conservative write-envelope union from Run301 metadata.

Source-derived address envelopes are not native write traces.  Different typed
views on one backing are converted to byte intervals before union.
"""

import argparse
import json
from pathlib import Path


def merge(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def source_intervals(registry, sample):
    typed = {source['name']: source for source in registry['sources']}
    result = {}
    details = []
    for item in sample['sources']:
        name = item['name']
        if name not in typed:
            raise ValueError(f'missing typed source {name}')
        views = typed[name]['typed_views']
        pages = {int(page) for per_request in item['current_write_pages_by_request']
                 for page in per_request}
        for view in views:
            if len(view['shape']) != len(view['stride']):
                raise ValueError(f'bad view {name}')
            size = view['element_size']
            payload = (sum((n - 1) * step for n, step in
                           zip(view['shape'][1:], view['stride'][1:])) + 1) * size
            intervals = []
            for page in pages:
                if not 0 <= page < view['shape'][0]:
                    raise ValueError(f'page outside view {name}: {page}')
                lo = (view['storage_offset'] + page * view['stride'][0]) * size
                hi = lo + payload
                if hi > view['storage_nbytes']:
                    raise ValueError(f'interval outside backing {name}: {hi}')
                intervals.append((lo, hi))
            result.setdefault(str(view['storage_ptr']), []).extend(intervals)
            details.append({'source': name, 'view': view['path'],
                            'pages': len(pages), 'intervals': len(intervals),
                            'precision': item['write_precision']})
    return result, details


def audit(registry_dir):
    result = []
    for rank in range(8):
        registry = json.loads((registry_dir / f'rank{rank}_registry.json').read_text())
        samples = [json.loads(line) for line in
                   (registry_dir / f'rank{rank}.jsonl').read_text().splitlines() if line]
        by_key = {(s['cohort'], s['cycle']): s for s in samples}
        pairs = []
        for first in samples:
            second = by_key.get((first['cohort'], first['cycle'] + 1))
            if second is None:
                continue
            a, ad = source_intervals(registry, first)
            b, bd = source_intervals(registry, second)
            if set(a) != set(b):
                raise ValueError(f'backing set differs rank{rank} cycle{first["cycle"]}')
            backings = []
            for ptr in sorted(a):
                joined = merge(a[ptr] + b[ptr])
                backings.append({'storage_ptr': ptr, 'intervals': len(joined),
                                 'conservative_byte_union': sum(hi - lo for lo, hi in joined),
                                 'source_interval_count': len(a[ptr]) + len(b[ptr])})
            pairs.append({'cohort': first['cohort'], 'cycles': [first['cycle'], second['cycle']],
                          'first_parking': bool(first['first_parking'] or second['first_parking']),
                          'sources_per_step': [len(first['sources']), len(second['sources'])],
                          'backings': backings,
                          'source_details': {'first': ad, 'second': bd}})
        result.append({'rank': rank, 'pairs': pairs})
    if not all(len(r['pairs']) == 5 for r in result):
        raise ValueError('expected five adjacent pairs per rank')
    return {'scope': 'Run301 selected 14 layer0-5 Target source metadata address envelopes; two-cycle offline conservative byte interval union; not native writes or full Target/Draft state',
            'ranks': result,
            'limits': ['Only 14 selected sources on three backings',
                       'Metadata-derived candidate writes, not native byte traces',
                       'No Draft KV, other Target layers, Host mirrors, metadata, acceptance or Graph workspace',
                       'Different allocation from Run299; compare geometry, not storage pointers across runs']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = audit(args.registry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2) + '\n')
    pairs = [p for r in data['ranks'] for p in r['pairs']]
    print(json.dumps({'ranks': len(data['ranks']), 'pairs': len(pairs),
                      'backings_per_pair': sorted({len(p['backings']) for p in pairs}),
                      'max_union_bytes_per_rank_pair': max(sum(b['conservative_byte_union']
                                                               for b in p['backings']) for p in pairs)}))


if __name__ == '__main__':
    main()
