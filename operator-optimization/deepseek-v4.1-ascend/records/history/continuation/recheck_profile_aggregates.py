"""Read retained aggregates only; keep capture/rank/window domains separate."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json

R = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
OUT = Path(__file__).parent / 'profile-first-20261031'
INPUTS = {
    'old': R / 'evidence/TRACE-SUMMARY.json',
    'official': R / 'publication-draft/official-operators.json',
    'index': R / 'profile-index.json',
    'consolidation': R / 'publication-draft/evidence-summary.json',
}


def read(key):
    return json.loads(INPUTS[key].read_text())


def type_table(samples, counts_key, duration_key, union_key):
    buckets = defaultdict(list)
    for identity, types in samples:
        for name, metrics in types.items():
            buckets[name].append((identity, metrics))
    return [
        {
            'type': name,
            'sample_count': len(items),
            'all_sample_call_count': sum(m[counts_key] for _, m in items),
            'per_sample_calls_range': [min(m[counts_key] for _, m in items), max(m[counts_key] for _, m in items)],
            'per_sample_duration_sum_range': [min(m[duration_key] for _, m in items), max(m[duration_key] for _, m in items)],
            'per_sample_interval_union_range': [min(m[union_key] for _, m in items), max(m[union_key] for _, m in items)],
            'sample_identities': [identity for identity, _ in items],
        }
        for name, items in sorted(buckets.items())
    ]


def main():
    OUT.mkdir(exist_ok=True)
    old, official, index, consolidated = (read(k) for k in INPUTS)
    audit = []
    for rank in old['ranks']:
        family_count = sum(m['calls'] for m in rank['families'].values())
        type_count = sum(m['calls'] for m in rank['operator_types'].values())
        audit.append({'capture': 'old', 'sample': rank['rank'], 'rows': rank['rows'],
                      'family_calls': family_count, 'type_calls': type_count,
                      'pass': rank['rows'] == family_count == type_count})
    for sample in official['samples']:
        counts = [sum(m['count'] for m in sample['types'].values()), sum(sample['core_counts'].values()), sum(sample['model_counts'].values())]
        audit.append({'capture': official['run_id'], 'sample': sample['sample_id'],
                      'window': sample['window_ordinal'], 'type_core_model_counts': counts,
                      'pass': len(set(counts)) == 1})
    old_tables = {
        key: type_table([(r['rank'], r[key]) for r in old['ranks']], 'calls', 'task_duration_sum_us', 'interval_union_us')
        for key in ['families', 'operator_types']
    }
    official_tables = {}
    for window in sorted({s['window_ordinal'] for s in official['samples']}):
        subset = [s for s in official['samples'] if s['window_ordinal'] == window]
        official_tables[str(window)] = type_table(
            [(s['sample_id'], s['types']) for s in subset], 'count', 'sum_ns', 'union_ns')
    total_old = sum(r['rows'] for r in old['ranks'])
    other = sum(r['families']['other_unattributed']['calls'] for r in old['ranks'])
    result = {
        'schema_version': 1,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS_AGGREGATE_CONSERVATION' if all(x['pass'] for x in audit) else 'AGGREGATE_MISMATCH_REQUIRES_REVIEW',
        'inputs': {k: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in INPUTS.items()},
        'old_capture': {'rank_count': len(old['ranks']), 'kernel_rows': total_old,
                        'operator_type_count': len(old_tables['operator_types']),
                        'family_other_calls': other,
                        'family_other_fraction': other / total_old,
                        'tables': old_tables, 'duration_unit': 'us',
                        'other_meaning': 'Family classifier fallback, not proof of missing operator names or unmapped semantic modules.'},
        'official_capture': {'run_id': official['run_id'], 'counts': official['counts'],
                             'total_exported_rows': sum(sum(m['count'] for m in s['types'].values()) for s in official['samples']),
                             'tables_by_window': official_tables, 'duration_unit': 'ns',
                             'sample_identity_caution': 'Use retained anonymous sample_id and window; do not invent missing explicit rank joins.'},
        'audit': audit,
        'profile_evidence_entry_points': {
            'raw_index': str(INPUTS['index']),
            'captured_target_modules': str(R / 'captured-layer-attribution/module_metrics.csv'),
            'official_shapes': str(R / 'publication-draft/official-operator-shapes.json'),
            'framework_cpu': str(R / 'publication-draft/framework-cpu.json'),
            'existing_selected_chains': str(INPUTS['consolidation']),
        },
        'capture_scope': consolidated['stage_consolidation']['official_capture_20261030'],
        'no_new_source_read': True,
        'no_remote_or_device_work': True,
        'profile_exhausted': False,
        'new_fusion_legality_proved': False,
        'whole_modules_closed': 0,
        'overall_complete': False,
        'limits': [
            'These are rechecks of existing derived aggregates, not independent raw trace reconstruction.',
            'Call counts may be summed for conservation; durations across ranks/windows/types are not additive request critical-path time.',
            'Old profile, official two-window capture, and166 observer values must not be joined as same-call evidence.',
            'Family/type names and matching shapes do not establish tensor identity, alias versions or legal numerical reassociation.',
            'Profile-only phase/branch absence does not prove a configured path is impossible.',
        ],
    }
    (OUT / 'aggregate-recheck.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    compact = {k: result[k] for k in ['status', 'no_new_source_read', 'profile_exhausted', 'whole_modules_closed', 'overall_complete']}
    compact.update({'old_rows': total_old, 'old_types': len(old_tables['operator_types']),
                    'official_exported_rows': result['official_capture']['total_exported_rows'],
                    'official_window_type_counts': {w: len(rows) for w, rows in official_tables.items()},
                    'checks': len(audit), 'failed_checks': sum(not x['pass'] for x in audit),
                    'detail': str(OUT / 'aggregate-recheck.json')})
    (OUT / 'summary.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(compact, ensure_ascii=False))


if __name__ == '__main__':
    main()
