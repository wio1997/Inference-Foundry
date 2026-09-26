#!/usr/bin/env python3
"""Matched-profiling screen for the selected c4 CP fork, not an E2E verdict."""
import argparse
import glob
import json
import statistics
from pathlib import Path


def kernel_events(path):
    raw = json.loads(Path(path).read_text())
    return sorted((dict(name=e['name'], start=float(e['ts']), end=float(e['ts']) + float(e['dur']),
                        dur=float(e['dur']), stream=str(e['args']['Physic Stream Id']))
                   for e in raw if e.get('ph') == 'X' and 'dur' in e
                   and 'Physic Stream Id' in e.get('args', {})), key=lambda e: e['start'])


def first_after(events, start, pred, stop=None):
    for e in events:
        if e['start'] >= start and (stop is None or e['start'] < stop) and pred(e):
            return e
    return None


def last_before(events, start, pred, low=None):
    for e in reversed(events):
        if e['start'] < start and (low is None or e['start'] >= low) and pred(e):
            return e
    return None


def analyse_trace(path):
    events = kernel_events(path)
    qli = [e for e in events if e['name'] == 'VllmQuantLightningIndexer']
    main = max({e['stream'] for e in qli}, key=lambda stream: sum(q['stream'] == stream for q in qli))
    aux = [e for e in events if e['name'] == 'Compressor' and e['stream'] != main]
    if len(aux) != 2:
        raise ValueError(f'{path}: expected two selected-layer auxiliary Compressors, got {len(aux)}')
    rows = []
    for cycle, comp in enumerate(aux):
        meta = last_before(events, comp['start'], lambda e: e['stream'] == comp['stream'] and e['name'] == 'CompressorMetadata', comp['start'] - 40)
        if meta is None:
            raise ValueError(f'{path}: auxiliary metadata missing')
        fork = last_before(events, meta['start'], lambda e: e['stream'] == main and e['name'] == 'EVENT_RECORD', meta['start'] - 20)
        q = first_after(qli, comp['start'], lambda e: e['stream'] == main, comp['start'] + 400)
        if fork is None or q is None:
            raise ValueError(f'{path}: fork or following QLI missing')
        rotary = last_before(events, q['start'], lambda e: e['stream'] == main and e['name'] == 'InplacePartialRotaryMul', fork['start'])
        sparse = first_after(events, q['end'], lambda e: e['stream'] == main and e['name'] == 'SparseAttnSharedkv', q['end'] + 30)
        if rotary is None or sparse is None:
            raise ValueError(f'{path}: query Rotary or SparseAttn missing')
        alltoall = first_after(events, sparse['end'], lambda e: e['name'].startswith('hcom_alltoall'), sparse['end'] + 100)
        if alltoall is None:
            raise ValueError(f'{path}: following AllToAll missing')
        scatter = first_after(events, comp['end'], lambda e: e['stream'] == comp['stream'] and 'ScatterNdUpdate' in e['name'], comp['end'] + 60)
        join = first_after(events, q['end'], lambda e: e['stream'] == main and e['name'] == 'EVENT_WAIT', sparse['start'])
        overlap = max(0., min(comp['end'], rotary['end']) - max(comp['start'], rotary['start']))
        rows.append(dict(cycle=cycle, rank=int(Path(path).parts[-3].split('_')[0][4:]),
                         trace=path, fork_start_us=fork['start'], aux_metadata_us=meta['dur'],
                         aux_compressor_us=comp['dur'], aux_scatter_us=None if scatter is None else scatter['dur'],
                         rotary_us=rotary['dur'], rotary_compressor_overlap_us=overlap,
                         qli_us=q['dur'], sparse_us=sparse['dur'],
                         fork_to_qli_start_us=q['start'] - fork['start'],
                         fork_to_sparse_start_us=sparse['start'] - fork['start'],
                         fork_to_sparse_end_us=sparse['end'] - fork['start'],
                         fork_to_next_alltoall_end_us=alltoall['end'] - fork['start'],
                         main_join_us=None if join is None else join['dur']))
    return rows


def analyse_root(root):
    paths = sorted(glob.glob(str(Path(root) / 'rank*_ascend_pt' / 'ASCEND_PROFILER_OUTPUT' / 'trace_view.json')))
    if len(paths) != 8:
        raise ValueError(f'{root}: expected 8 latest exported rank traces, got {len(paths)}')
    rows = [row for path in paths for row in analyse_trace(path)]
    fields = ('aux_metadata_us', 'aux_compressor_us', 'aux_scatter_us', 'rotary_us',
              'rotary_compressor_overlap_us', 'qli_us', 'sparse_us',
              'fork_to_qli_start_us', 'fork_to_sparse_start_us',
              'fork_to_sparse_end_us', 'fork_to_next_alltoall_end_us', 'main_join_us')
    values = {field: [row[field] for row in rows if row[field] is not None] for field in fields}
    medians = {field: statistics.median(v) if v else None for field, v in values.items()}
    envelopes = []
    for cycle in range(2):
        group = [row for row in rows if row['cycle'] == cycle]
        if len(group) != 8:
            raise ValueError(f'{root}: cycle {cycle} has {len(group)} ranks')
        starts = [row['fork_start_us'] for row in group]
        ends = [row['fork_start_us'] + row['fork_to_next_alltoall_end_us'] for row in group]
        envelopes.append(dict(cycle=cycle, rank_fork_start_skew_us=max(starts)-min(starts),
                              rank_alltoall_completion_skew_us=max(ends)-min(ends),
                              earliest_fork_to_latest_alltoall_end_us=max(ends)-min(starts)))
    return dict(profile_root=root, trace_count=len(paths), selected_samples=len(rows),
                medians_us=medians, eight_rank_selected_layer_envelopes=envelopes, rows=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--immediate', required=True)
    p.add_argument('--overlap', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    immediate = analyse_root(a.immediate)
    overlap = analyse_root(a.overlap)
    result = dict(status='matched_profiler_screen_not_causal_e2e', immediate=immediate, overlap=overlap,
                  overlap_minus_immediate_median_us={key: (overlap['medians_us'][key] - immediate['medians_us'][key]
                                                        if overlap['medians_us'][key] is not None
                                                        and immediate['medians_us'][key] is not None else None)
                                                    for key in immediate['medians_us']},
                  overlap_minus_immediate_eight_rank_global_window_us=[
                      overlap['eight_rank_selected_layer_envelopes'][cycle]['earliest_fork_to_latest_alltoall_end_us']
                      - immediate['eight_rank_selected_layer_envelopes'][cycle]['earliest_fork_to_latest_alltoall_end_us']
                      for cycle in range(2)],
                  limits=['Two sampled cycles per rank in different service runs; same profiler options and selected source except wait placement',
                          'Fork-relative intervals do not equal complete Target/cycle or Product wall savings',
                          'Eight-rank trace timestamps form a selected-layer envelope only; cross-service cohort state and profiled graph timing remain confounders',
                          'Cross-run route/position and profiler interference remain possible; numerical parity not assessed'])
    Path(a.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(immediate=immediate['medians_us'], overlap=overlap['medians_us'],
                          overlap_minus_immediate=result['overlap_minus_immediate_median_us']), indent=2))


if __name__ == '__main__':
    main()
