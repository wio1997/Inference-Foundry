"""Audit whether existing typed asset views cover each selected backing page.

This is a descriptor audit only. It does not prove the page manifest includes
every page written across successive Target/Draft cycles.
"""

import argparse
import json
from pathlib import Path


def intervals(view, page):
    size = view['element_size']
    shape = view['shape']
    stride = view['stride']
    lo = (view['storage_offset'] + page * stride[0]) * size
    hi = lo + (sum((extent - 1) * step for extent, step in zip(shape[1:], stride[1:])) + 1) * size
    return lo, hi


def union_complete(segments, lo, hi):
    cursor = lo
    for start, end in sorted(segments):
        if start > cursor:
            return False, [cursor, start]
        cursor = max(cursor, end)
    return cursor >= hi, ([] if cursor >= hi else [cursor, hi])


def audit(path):
    doc = json.loads(path.read_text())
    by_storage = {}
    for view in doc['selected_asset_views']:
        by_storage.setdefault(view['storage_ptr'], []).append(view)
    rows = []
    for storage, views in by_storage.items():
        sizes = {v['storage_nbytes'] for v in views}
        pages = {v['shape'][0] for v in views}
        if len(sizes) != 1 or len(pages) != 1:
            raise ValueError(f'inconsistent storage/page count rank{doc["rank"]}: {storage}')
        nbytes = sizes.pop()
        count = pages.pop()
        if nbytes % count:
            raise ValueError(f'nonintegral page stride rank{doc["rank"]}: {storage}')
        page_bytes = nbytes // count
        for page in (0, count // 2, count - 1):
            lo, hi = page * page_bytes, (page + 1) * page_bytes
            segs = [intervals(v, page) for v in views]
            complete, gap = union_complete(segs, lo, hi)
            rows.append({
                'storage_ptr': storage, 'page': page, 'page_bytes': page_bytes,
                'page_interval': [lo, hi],
                'views': [{'path': v['path'], 'interval': list(seg)} for v, seg in zip(views, segs)],
                'complete_if_same_page_selected_for_all_views': complete,
                'first_gap': gap,
            })
    return {'rank': doc['rank'], 'selected_asset_views': len(doc['selected_asset_views']),
            'selected_backings': len(by_storage), 'rows': rows,
            'complete': all(row['complete_if_same_page_selected_for_all_views'] for row in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    files = sorted(args.registry.glob('rank*_registry.json'))
    if len(files) != 8:
        raise SystemExit(f'expected eight rank registry files, got {len(files)}')
    results = [audit(file) for file in files]
    if sorted(row['rank'] for row in results) != list(range(8)):
        raise SystemExit('rank set incomplete')
    summary = {'scope': 'Run299 selected asset views on three layer2-related backings; descriptor coverage conditional on identical page sets per co-backing view',
               'ranks': results,
               'all_complete_conditional': all(row['complete'] for row in results),
               'not_proved': ['two-cycle write-page union', 'same selected page set across aliasing typed views',
                              'other Target and Draft mutable backings', 'metadata and Python state',
                              'native actual writes and long-lived readers']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'all_complete_conditional': summary['all_complete_conditional'],
                      'ranks': len(results), 'rows': sum(len(r['rows']) for r in results)}))


if __name__ == '__main__':
    main()
