#!/usr/bin/env python3
"""Reconstruct rank0-only histogram deltas from sanitized archive snapshots.

No service access; no summation across roles/ranks; same-sample identity is unknown.
"""
import json
import math
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'evidence/diagnostic_033/rank0_snapshots.json'
REQUIRED = {
    'p': frozenset(('prefill_bootstrap', 'prefill_forward', 'prefill_transfer_kv_cache',
                    'chunked_prefill', 'kv_transfer_bootstrap_ms', 'kv_transfer_latency_ms', 'queue_time_seconds')),
    'd': frozenset(('decode_bootstrap', 'decode_transferred', 'decode_waiting',
                    'kv_transfer_bootstrap_ms', 'queue_time_seconds')),
}


def validate_snapshot(snapshot, role):
    if set(snapshot) != {'role', 'rank', 'at_utc', 'source_status', 'source_sha256', 'series'}:
        raise ValueError('unexpected snapshot keys')
    if snapshot['role'] != role or snapshot['rank'] != 0 or snapshot['source_status'] != 200:
        raise ValueError('source role, rank or HTTP status mismatch')
    if not re.fullmatch(r'[0-9a-f]{64}', snapshot['source_sha256']):
        raise ValueError('invalid source SHA')
    time = datetime.fromisoformat(snapshot['at_utc'])
    if time.tzinfo is None:
        raise ValueError('snapshot timezone missing')
    if set(snapshot['series']) != REQUIRED[role]:
        raise ValueError('unexpected metric allowlist')
    for values in snapshot['series'].values():
        if set(values) != {'sum', 'count'}:
            raise ValueError('unexpected series keys')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in values.values()):
            raise ValueError('invalid histogram')
        if not float(values['count']).is_integer():
            raise ValueError('count is not integral')
    return time


def compare(before, after, role):
    if validate_snapshot(before, role) >= validate_snapshot(after, role):
        raise ValueError('snapshot time order invalid')
    rows = {}
    for name in sorted(REQUIRED[role]):
        a, b = before['series'][name], after['series'][name]
        count, total = b['count'] - a['count'], b['sum'] - a['sum']
        if count < 0 or total < -1e-6:
            raise ValueError('negative delta or counter reset')
        if count == 0 and abs(total) > 1e-6:
            raise ValueError('sum changed without count')
        rows[name] = {'delta_count': count, 'delta_sum': total,
                      'mean': total / count if count else None,
                      'unit': 'milliseconds' if name.endswith('_ms') else 'seconds'}
    return rows


def reconstruct(doc):
    if doc['schema_version'] != 1 or doc['rank_labels'] != {'tp_rank': 0, 'moe_ep_rank': 0, 'pp_rank': 0}:
        raise ValueError('unexpected snapshot schema or rank labels')
    if set(doc['snapshots']) != {'p', 'd'}:
        raise ValueError('expected exactly P and D')
    output = {'scope': 'rank0 histogram deltas; windows overlap; no request identity, causality or cross-role sums',
              'legacy_field_note': 'Fields containing handshake retain old schema names but refer to kv_transfer_bootstrap_ms; they do not prove remote network handshake completion or wire time.',
              'windows': {}}
    for label, index in (('partial', 1), ('to_completion', 2)):
        result = {}
        for role in ('p', 'd'):
            snapshots = doc['snapshots'][role]
            if len(snapshots) != 3:
                raise ValueError('expected three snapshots per role')
            result[role] = compare(snapshots[0], snapshots[index], role)
        whole = result['d']['decode_bootstrap']
        part = result['d']['kv_transfer_bootstrap_ms']
        same_count = whole['delta_count'] == part['delta_count'] and whole['delta_count'] > 0
        result['d_conditional_bootstrap_minus_handshake'] = {
            'whole_count': whole['delta_count'], 'handshake_count': part['delta_count'],
            'same_count': same_count, 'same_sample_identity_verified': False,
            'conditional_mean_seconds': (whole['delta_sum'] - part['delta_sum'] / 1000) / whole['delta_count'] if same_count else None,
            'condition': 'same setter samples, no reset, and scrape does not split setter updates; equal counts alone do not prove these'}
        output['windows'][label] = result
    return output


def main():
    print(json.dumps(reconstruct(json.loads(DEFAULT.read_text())), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
