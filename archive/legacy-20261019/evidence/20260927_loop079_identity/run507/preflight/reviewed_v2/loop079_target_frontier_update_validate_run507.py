#!/usr/bin/env python3
"""Run507 selected MLA update CPU gate; source-inferred iteration, not native event completion."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

MLA_SHA = '67b4479cca16fd164b33b91b2fa4f48e95f2474720e484abdf3b3897a8ea1545'


def gate(ok, message):
    if not ok:
        raise ValueError('Run507 update gate: ' + message)


def exact_int(x, low=0):
    return type(x) is int and x >= low


def validate(root: Path):
    capture = root / 'capture'
    paths = sorted(capture.glob('rank*_cohort*.json'))
    gate(len(paths) == 40, 'exact40 capture rows')
    summaries = []
    seen = set()
    run_ids = set()
    for path in paths:
        row = json.loads(path.read_text())
        rank, cohort = row.get('rank'), row.get('cohort')
        gate(exact_int(rank) and rank < 8 and exact_int(cohort, 1) and cohort <= 5,
             'rank/cohort')
        gate(path.name == f'rank{rank}_cohort{cohort}.json' and (rank, cohort) not in seen,
             'row filename/identity')
        seen.add((rank, cohort))
        run_ids.add(row.get('run_id'))
        replay = row.get('replay')
        update = row.get('actual_update')
        gate(type(replay) is dict and type(update) is dict and
             update.get('branch') == 'after' and update.get('return_count') == 1 and
             update.get('configured_update_stream', {}).get('stream_id') == 102,
             'selected after-update callable/return')
        for left, right in [('replay_entry_id','entry_id'),
                            ('replay_capture_generation','capture_generation'),
                            ('replay_graph_id','graph_id'),
                            ('replay_selected_ordinal','selected_observation_ordinal')]:
            gate(update.get(left) == replay.get(right) and exact_int(update.get(left), 1),
                 left + ' same-process replay join')
        captured = update.get('mla_params_capture')
        selected = update.get('mla_params_selected')
        gate(type(captured) is dict and captured == selected and
             captured.get('num_tokens') == 96 and exact_int(captured.get('params_id'), 1),
             'captured/selected MLA GraphParams identity')
        lists = captured.get('lists')
        gate(type(lists) is dict and set(lists) == {'attn_params', 'handles', 'events'},
             'exact three MLA parameter lists')
        counts = {}
        for name, record in lists.items():
            gate(type(record) is dict and exact_int(record.get('count')) and
                 type(record.get('object_ids')) is list and
                 len(record['object_ids']) == record['count'] and
                 all(exact_int(x, 1) for x in record['object_ids']),
                 name + ' object identities')
            counts[name] = record['count']
        keys = update.get('attention_key_count')
        zipped = update.get('zip_iteration_count')
        gate(exact_int(keys) and exact_int(zipped) and zipped == min(keys, *counts.values()) and
             update.get('list_counts') == counts and
             update.get('source_inferred_successful_event_records') == zipped and
             update.get('device_event_completion') == 'unobserved' and
             update.get('native_event_id') == 'unobserved' and
             type(update.get('draft_metadata_argument_is_none')) is bool,
             'key/zip/source-inferred iterations/unknown native completion')
        if cohort == 5:
            meta = json.loads((root / 'graph_dump' / f'rank{rank}_cohort5_acl_graph.meta.json').read_text())
            source = meta.get('graph_update_backend', {}).get('update_graph_params', {}).get('source', {})
            gate(source.get('sha256') == MLA_SHA and
                 source.get('qualname') == 'AscendMLAImpl.update_graph_params' and
                 meta.get('graph_update_backend', {}).get('selected_actual_update') == update and
                 meta.get('entry_id') == replay['entry_id'] and
                 meta.get('capture_generation') == replay['capture_generation'],
                 'SHA-pinned installed MLA source and exact selected graph')
        summaries.append(dict(rank=rank, cohort=cohort, attention_keys=keys,
                              counts=counts, zip_iterations=zipped,
                              params_id=captured['params_id']))
    gate(seen == {(r,c) for r in range(8) for c in range(1,6)} and len(run_ids) == 1,
         'complete all-rank cohort/run identity')
    return dict(valid=True, scope='same-process Python object identity and source-derived successful iterations only; no native event ID, device completion, timing or Bound',
                run_id=next(iter(run_ids)), rows=40, ranks=8, cohorts=5,
                source_sha256=MLA_SHA, summaries=summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'valid': True, 'rows': 40,
                      'zip_iterations': sorted({x['zip_iterations'] for x in result['summaries']})}))


if __name__ == '__main__':
    main()
