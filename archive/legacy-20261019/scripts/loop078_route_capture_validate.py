#!/usr/bin/env python3
"""Offline all-eight-rank route validator; no torch or accelerator dependency.
Fail closed: no partial result is written if any cohort/rank invariant fails.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def ints(value, n, label):
    require(isinstance(value, list) and len(value) == n, f'{label}: length != {n}')
    require(all(type(x) is int for x in value), f'{label}: noninteger')


def validate_cohort(files):
    require(len(files) == 8, 'need exactly eight ranks per cohort')
    data = [json.loads(p.read_text()) for p in files]
    require(sorted(d['rank'] for d in data) == list(range(8)), 'rank set != 0..7')
    data.sort(key=lambda d: d['rank'])
    base = data[0]
    require(base['schema'] == 1 and base['cycles'] > 65, 'schema/cycle coverage')
    for d in data:
        for field in ('schema', 'cohort', 'cycles', 'start_cycle', 'counts', 'remaining',
                      'initial_output_counts', 'initial_positions', 'slot_identity'):
            require(d[field] == base[field], f'rank metadata mismatch: {field}')
        require(set(d['records']) == {'64', '65'}, 'selected record keys')
    ints(base['remaining'], 12, 'remaining')
    require(all(x > 0 for x in base['remaining']), 'nonpositive remaining')
    require(len(base['counts']) == base['cycles'], 'history length')
    for counts in base['counts']:
        ints(counts, 12, 'counts')
        require(all(0 <= c <= 8 for c in counts), 'masked acceptance outside 0..8')
    result = dict(cohort=base['cohort'], slot_identity=base['slot_identity'], cycles={})
    for c in (64, 65):
        rec = base['records'][str(c)]
        retained = [min(base['counts'][c][s], max(0, base['remaining'][s] -
                    sum(row[s] for row in base['counts'][:c]))) for s in range(12)]
        ints(rec['target_positions'], 96, 'Target positions')
        ints(rec['target_input_ids'], 96, 'Target input IDs')
        require(len(rec['active_mask']) == 12 and
                all(type(x) is bool for x in rec['active_mask']), 'active mask shape/type')
        require(rec['absolute_cycle'] == base['start_cycle'] + c, 'cycle ordinal')
        ints(rec['raw_acceptance_counts'], 12, 'raw acceptance counts')
        require(all(1 <= x <= 8 for x in rec['raw_acceptance_counts']), 'raw acceptance outside 1..8')
        require(base['counts'][c] == [x if live else 0 for x, live in
                zip(rec['raw_acceptance_counts'], rec['active_mask'])], 'raw/masked acceptance mismatch')
        meta = rec['draft_metadata']
        q = meta['q']
        require(q == (7 if meta['sample_from_anchor'] else 8), 'Draft Q/anchor mismatch')
        ints(meta['positions'], 12*q, 'Draft positions')
        ints(meta['input_ids'], 12*q, 'Draft input IDs')
        require(meta['query_start_loc'] == [s*q for s in range(13)], 'Draft request layout')
        expected_samples = [s*q+j+(0 if meta['sample_from_anchor'] else 1)
                            for s in range(12) for j in range(7)]
        require(meta['sample_indices'] == expected_samples, 'Draft sample-index layout')
        for s in range(12):
            pos = rec['target_positions'][s*8:(s+1)*8]
            require(pos == list(range(pos[0], pos[0]+8)), 'Target request position stride')
            anchor = pos[rec['raw_acceptance_counts'][s]-1]+1
            require(meta['positions'][s*q:(s+1)*q] == list(range(anchor, anchor+q)),
                    'Draft query positions disagree with accepted Target context')
            require(not retained[s] or rec['active_mask'][s], 'useful row in inactive slot')
        cycle_result = dict(retained_by_slot=retained, useful_rows=sum(retained), models={})
        for model, ordinals, rows in (('target', list(range(43)), 96),
                                      ('draft', [43, 44, 45], 12*q)):
            all_layers = []
            for d in data:
                r = d['records'][str(c)]
                for field in ('target_positions', 'target_input_ids', 'active_mask',
                              'draft_metadata', 'draft_output', 'absolute_cycle', 'raw_acceptance_counts'):
                    require(r[field] == rec[field], f'cross-rank metadata: {field}')
                require(bool(r['replayed_graphs']), 'no recorded Target graph replay')
                layers = sorted(r[model], key=lambda x: x['ordinal'])
                require([x['ordinal'] for x in layers] == ordinals, f'{model}: layer set')
                require(len({x['name'] for x in layers}) == len(ordinals), 'duplicate names')
                all_layers.append(layers)
            active_sets, retained_sets = [], []
            for i, ordinal in enumerate(ordinals):
                reference = all_layers[0][i]
                ids = reference['ids']
                require(len(ids) == rows, f'{model}: actual rows != {rows}')
                for row in ids:
                    ints(row, 6, 'topk')
                    require(len(set(row)) == 6 and all(0 <= e < 256 for e in row),
                            'topk duplicate/out-of-range')
                histogram = Counter(e for row in ids for e in row)
                require(sum(histogram.values()) == rows*6, 'route count total')
                ep_ranks = []
                for layers in all_layers:
                    item = layers[i]
                    if model == 'target':
                        require(item['graph_key'] in data[len(ep_ranks)]['records'][str(c)]['replayed_graphs'],
                                'Target layer graph entry not replayed')
                    else:
                        require(item['graph_key'] is None, 'eager Draft has graph entry')
                    ep = item['ep_rank']
                    ep_ranks.append(ep)
                    require(item['model'] == model and item['name'] == reference['name'],
                            'layer model/name mismatch')
                    require(item['ids'] == ids, 'replicated route matrices differ')
                    expected_map = [e-ep*32 if ep*32 <= e < (ep+1)*32 else -1
                                    for e in range(256)]
                    require(item['expert_map'] == expected_map, 'EP map not contiguous frozen layout')
                    require(item['group'] == [histogram[e] for e in range(ep*32, (ep+1)*32)],
                            'source-specific group_list parity failure')
                require(sorted(ep_ranks) == list(range(8)), 'EP ownership incomplete/duplicate')
                active_sets.append(len(histogram))
                if model == 'target':
                    union = {e for s in range(12) for j in range(retained[s]) for e in ids[s*8+j]}
                    retained_sets.append(len(union))
            cycle_result['models'][model] = dict(rows=rows, layers=len(ordinals),
                active_experts_per_layer=active_sets,
                retained_experts_per_layer=retained_sets if model == 'target' else None)
        result['cycles'][str(c)] = cycle_result
    # Draft c emits candidates consumed by Target c+1. Only compare active slots:
    # parking between cycles can replace the target input state.
    a, b = base['records']['64'], base['records']['65']
    require(len(a['draft_output']) == 12, 'draft output batch')
    accepted_draft = []
    for s in range(12):
        ints(a['draft_output'][s], 7, 'Draft output')
        if b['active_mask'][s]:
            require(b['target_input_ids'][s*8+1:(s+1)*8] == a['draft_output'][s],
                    'Draft64 -> Target65 token handoff mismatch')
        # Counts include one recovery/bonus token; at most count-1 draft tokens accepted.
        available = max(0, base['remaining'][s] - sum(x[s] for x in base['counts'][:65]))
        accepted_draft.append(min(max(0, base['counts'][65][s]-1), available))
    result['draft64_accepted_tokens_by_slot'] = accepted_draft
    result['limits'] = ['Draft accepted-token attribution is not a removable-row bound: body attention is noncausal.',
        'Packed selected storage sets are not physical HBM traffic.',
        'Request identity is immutable cohort slot, not external API request ID.',
        'No source dtype metadata survives JSON; int32 is enforced at capture.']
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('capture_dir', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    groups = defaultdict(list)
    for path in sorted(args.capture_dir.glob('rank*_cohort*.json')):
        data = json.loads(path.read_text())
        groups[data['cohort']].append(path)
    require(bool(groups), 'no capture files')
    result = [validate_cohort(paths) for _, paths in sorted(groups.items())]
    args.output.write_text(json.dumps(dict(valid=True, cohorts=result), indent=2)+'\n')
    print(json.dumps(dict(valid=True, cohorts=len(result))))


if __name__ == '__main__':
    main()
