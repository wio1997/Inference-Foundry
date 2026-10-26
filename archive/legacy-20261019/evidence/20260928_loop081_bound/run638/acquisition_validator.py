#!/usr/bin/env python3
"""Fail-closed fixed-work OFF/ON/OFF Runtime packet admission; diagnostic only."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

from runtime.bound_observer import CURRENT, REQUIRED_CURRENT, SIDE

ROOT = Path('/data/wio/Inference_Foundry')
DATASET = Path('/data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl')


def need(ok, why):
    if not ok:
        raise AssertionError(why)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def client_text_hash(path):
    item = read(path)
    text = []
    for event in item['events']:
        raw = base64.b64decode(event['payload_b64'], validate=True)
        if raw == b'[DONE]':
            continue
        data = json.loads(raw)
        for choice in data.get('choices', ()):
            delta = choice.get('delta') or {}
            text.extend(str(delta[key]) for key in ('reasoning_content', 'reasoning', 'content')
                        if delta.get(key))
    return hashlib.sha256(''.join(text).encode()).hexdigest()


def arm(arm_dir: Path, run_ts: str, on: bool):
    admission = read(arm_dir / 'client_admission.json')
    need(admission['status'] == 'client_two_phase_admitted', 'client admission')
    recheck = arm_dir / 'client_admission_recheck.json'
    subprocess.run([sys.executable,
                    str(ROOT / 'scripts/loop080_frontier_phase_validate.py'),
                    '--dataset', str(DATASET),
                    '--warmup-dir', str(arm_dir / 'warmup_client'),
                    '--measured-dir', str(arm_dir / 'measured_client'),
                    '--output', str(recheck)], check=True, capture_output=True, text=True)
    need(read(recheck) == admission, 'client raw/dataset/SSE independent recheck')
    measured = sorted((arm_dir / 'measured_client').glob('request_*.json'))
    need(len(measured) == 48, 'exact measured clients')
    by_response = {}
    for index, path in enumerate(measured):
        need(path.name == f'request_{index:03d}.json', 'measured index')
        row = read(path)['row']
        need(row['phase'] == 'measured' and row['i'] == index and
             row['http_status'] == 200 and row['error'] is None and
             row['output_tokens'] == 1024, 'client result')
        response = row['response_id']
        need(response not in by_response, 'response duplicate')
        by_response[response] = index
    need({x['response_id'] for x in admission['request_index']
          if x['phase'] == 'measured'} == set(by_response), 'client/admission identity')
    expected = {f'rank{rank}_cohort{cohort}.json'
                for rank in range(8) for cohort in range(5, 9)}
    ledger_dir, packet_dir = arm_dir / 'ledger', arm_dir / 'packet'
    need({p.name for p in ledger_dir.iterdir()} == expected | {'arm'},
         'exact all8 measured ledger')
    need({p.name for p in packet_dir.iterdir()} == (expected if on else set()),
         'exact ON/OFF packet set')
    runtime_dir = arm_dir / 'runtime'
    all_runtime = {f'rank{rank}_cohort{cohort}.json'
                   for rank in range(8) for cohort in range(1, 9)}
    need({p.name for p in runtime_dir.iterdir()} == all_runtime,
         'exact all8 warmup+measured Runtime')
    raw = {str(arm_dir / 'client_admission.json'): digest(arm_dir / 'client_admission.json')}
    raw[str(recheck)] = digest(recheck)
    for phase in ('warmup_client','measured_client'):
        phase_dir = arm_dir / phase
        expected_clients = {f'request_{i:03d}.json' for i in range(48)} | {'summary.json'}
        need({p.name for p in phase_dir.iterdir()} == expected_clients,
             'exact client raw source set: '+phase)
        for path in phase_dir.iterdir():
            raw[str(path)] = digest(path)
    for path in runtime_dir.iterdir():
        row = read(path)
        need(row['pass'] is True and row['target_graph_requested'] is True and
             row['target_graph_mode'] == 'FULL', 'warm+measured Runtime correctness/Graph')
        raw[str(path)] = digest(path)
    cohorts, times, packet_summary = {}, [], []
    wall_by_rank_cohort = {}
    server_ns = set()
    all_response = set()
    for cohort in range(5, 9):
        ref = None
        for rank in range(8):
            name = f'rank{rank}_cohort{cohort}.json'
            source = ledger_dir / name
            runtime_source = runtime_dir / name
            row, rt = read(source), read(runtime_source)
            raw[str(source)], raw[str(runtime_source)] = digest(source), digest(runtime_source)
            need(row['rank'] == rank and row['cohort'] == cohort and
                 row['run_ts'] == run_ts, 'ledger identity')
            need(rt['rank'] == rank and rt['cohort'] == cohort and rt['pass'] is True,
                 'Runtime correctness')
            need(rt['target_graph_requested'] is True and rt['target_graph_mode'] == 'FULL',
                 'actual FULL Graph mode')
            need(rt['req_ids'] == row['req_ids'] and
                 rt['cycles'] == row['runtime_basis']['cycles'] and
                 rt['generated_output_counts'] == row['generated_output_counts'] == [1024]*12,
                 'Runtime/ledger same cohort')
            need(type(row['runtime_scoped_wall_s']) in (int, float) and
                 math.isfinite(row['runtime_scoped_wall_s']) and
                 row['runtime_scoped_wall_s'] > 0, 'Runtime wall')
            need(row['runtime_scoped_wall_s'] == rt['wall_seconds'],
                 'Runtime/ledger same wall')
            basis = row['runtime_basis']
            need(all(isinstance(basis[key], str) and len(basis[key]) == 64
                     for key in ('canonical_effective_staged_trajectory_sha256',
                                 'accepted_count_history_sha256',
                                 'raw_padded_token_history_sha256')),
                 'basis digest')
            identity = (row['req_ids'], basis['cycles'],
                        basis['canonical_effective_staged_trajectory_sha256'],
                        basis['accepted_count_history_sha256'],
                        row['product_output_sha256'])
            if ref is None:
                ref = identity
            else:
                need(identity == ref, 'all8 effective trajectory/output divergence')
            server_ns.add(row['time_namespace'])
            times.append(row['runtime_scoped_wall_s'])
            wall_by_rank_cohort[f'{rank}:{cohort}'] = row['runtime_scoped_wall_s']
            if on:
                packet_path = packet_dir / name
                packet = read(packet_path)
                raw[str(packet_path)] = digest(packet_path)
                need(packet['status'] == 'instrumented_current_stream_markers_only',
                     'packet status')
                need(packet['identity'] == row and packet['sample_cycles'] == [63,64,65],
                     'packet/ledger identity')
                need(packet['post_runtime_export_uses_sync_api'] is True and
                     all(packet[key] is None for key in
                         ('strict_resource_floor_s', 'strict_scheduling_floor_s',
                          'product_e2e_ceiling_tps',
                          'numeric_current_to_credible_limit_gap')),
                     'packet scope')
                events = packet['events']
                actual = {(e['cycle'], e['label'], e['stream']) for e in events}
                required = {(c, label, 'current') for c in (63,64,65)
                            for label in REQUIRED_CURRENT} | \
                           {(c, label, 'copy') for c in (63,64,65) for label in SIDE}
                allowed = required | {(c, 'derived_target_metadata', 'current')
                                      for c in (63,64,65)}
                need(required <= actual <= allowed and len(actual) == len(events),
                     'all required selected current/copy markers')
                need(45 <= len(events) <= 48 and
                     all(e['event_generation'] == 1 and
                         math.isfinite(e['elapsed_ms_from_first_same_stream']) and
                         e['elapsed_ms_from_first_same_stream'] >= 0 and
                         type(e['host_submit_ns']) is int and e['host_submit_ns'] > 0
                         for e in events), 'packet completeness')
                need(sorted(e['issue_ordinal'] for e in events) ==
                     list(range(1,len(events)+1)),
                     'packet global issue ordinals')
                stream_ids = packet['stream_ids']
                need(set(stream_ids) == {'current','copy'} and
                     stream_ids['current'][0] == stream_ids['copy'][0] and
                     stream_ids['current'][1] != stream_ids['copy'][1],
                     'current/copy distinct same-device stream identities')
                for stream in ('current','copy'):
                    ordered = sorted((e for e in events if e['stream'] == stream),
                                     key=lambda e:e['issue_ordinal'])
                    elapsed = [e['elapsed_ms_from_first_same_stream'] for e in ordered]
                    need(elapsed[0] == 0 and elapsed == sorted(elapsed),
                         'single stream monotonic elapsed')
                by_label = {(e['cycle'],e['label']):e for e in events}
                current = {(e['cycle'],e['label']): e['elapsed_ms_from_first_same_stream']
                           for e in events if e['stream'] == 'current'}
                for c in (63,64,65):
                    waits = packet['existing_host_waits'][str(c)]
                    need(len(waits) == 1 and waits[0]['producer_cycle'] == c-1 and
                         waits[0]['end_ns'] >= waits[0]['start_ns'],
                         'existing copy wait')
                    current_ordinals = [by_label[c,label]['issue_ordinal']
                                        for label in CURRENT if (c,label) in by_label]
                    need(current_ordinals == sorted(current_ordinals),
                         'current issue sequence')
                    need(by_label[c,'proposer_before']['issue_ordinal'] <
                         by_label[c,'host_copy_before']['issue_ordinal'] <
                         by_label[c,'host_copy_after']['issue_ordinal'] <
                         by_label[c,'proposer_after']['issue_ordinal'],
                         'side copy issue order')
                    need(by_label[c,'proposer_before']['host_submit_ns'] <=
                         waits[0]['start_ns'] <= waits[0]['end_ns'] <=
                         by_label[c,'host_copy_before']['host_submit_ns'],
                         'existing host wait position')
                    need(current[c,'serve_park_after'] >= current[c,'cycle_begin'],
                         'nonnegative sampled cycle')
                    packet_summary.append({'rank':rank,'cohort':cohort,'cycle':c,
                        'current_cycle_span_ms': current[c,'serve_park_after']-
                                                 current[c,'cycle_begin'],
                        'host_existing_copy_wait_ms':
                            (waits[0]['end_ns']-waits[0]['start_ns'])/1e6})
        ids = ref[0]
        indexes = []
        for req_id in ids:
            matches = [response for response in by_response
                       if req_id.startswith(response + '-')]
            need(len(matches) == 1, 'Runtime/client request join')
            response = matches[0]
            need(response not in all_response, 'duplicate Runtime request')
            all_response.add(response)
            indexes.append(by_response[response])
        cohorts[cohort] = {'dataset_indexes_by_slot': indexes,
                           'cycles': ref[1],
                           'effective_trajectory_sha256': ref[2],
                           'accepted_count_history_sha256': ref[3],
                           'product_output_sha256': ref[4]}
    need(len(all_response) == 48 and len(server_ns) == 1, 'all48/server namespace')
    client_text = [client_text_hash(path) for path in measured]
    return {'run_ts': run_ts, 'on': on, 'cohorts': cohorts,
            'client_text_sha256_by_dataset_index': client_text,
            'client_product_measured_duration_s': admission['measured_summary']['duration_s'],
            'runtime_scoped_wall_s_all_rank_cohorts': times,
            'runtime_wall_s_by_rank_cohort': wall_by_rank_cohort,
            'packet_summary': packet_summary, 'server_time_namespace': next(iter(server_ns)),
            'raw_sha256': raw}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    names = ('off_a','on','off_b')
    rows = {name:arm(args.root/name, f'LOOP081-RUN638-{name.upper()}', name=='on')
          for name in names}
    ref = rows['off_a']['cohorts']
    for name in names[1:]:
        need(rows[name]['cohorts'] == ref, 'cross-arm fixed effective trajectory/output')
        need(rows[name]['client_text_sha256_by_dataset_index'] ==
             rows['off_a']['client_text_sha256_by_dataset_index'],
             'cross-arm client output text divergence')
    need(len(rows['on']['packet_summary']) == 8*4*3, 'ON packet cardinality')
    # Freeze a conservative observer-effect screen before looking at Event
    # phase times. Failing it keeps packets diagnostic but forbids wall transfer.
    product = {name:row['client_product_measured_duration_s'] for name,row in rows.items()}
    baseline = (product['off_a']+product['off_b'])/2
    product_off_drift = abs(product['off_b']-product['off_a'])/baseline
    product_on_effect = abs(product['on']-baseline)/baseline
    wall_effects=[]
    wall_off_drifts=[]
    for key in rows['off_a']['runtime_wall_s_by_rank_cohort']:
        a=rows['off_a']['runtime_wall_s_by_rank_cohort'][key]
        b=rows['off_b']['runtime_wall_s_by_rank_cohort'][key]
        on=rows['on']['runtime_wall_s_by_rank_cohort'][key]
        mid=(a+b)/2
        wall_off_drifts.append(abs(b-a)/mid)
        wall_effects.append(abs(on-mid)/mid)
    timing_qualified = (product_off_drift <= 0.05 and product_on_effect <= 0.05
                        and max(wall_off_drifts) <= 0.05 and max(wall_effects) <= 0.05)
    result = {'status':'diagnostic_fixed_effective_work_packet_admitted',
              'limits':'Effective/staged trajectory and Product/client output admitted. Raw parked-slot execution, full HCCL/native completion, observer-free Product wall and numeric Bound remain unproved.',
              'arms':rows,
              'observer_effect_screen':{
                  'predeclared_max_relative_drift':0.05,
                  'product_off_drift':product_off_drift,
                  'product_on_effect':product_on_effect,
                  'max_rank_cohort_off_drift':max(wall_off_drifts),
                  'max_rank_cohort_on_effect':max(wall_effects),
                  'timing_transfer_qualified':timing_qualified,
                  'scope':'Qualifies diagnostic Runtime stream-marker timing only; never a Product/Resource/Scheduling Bound.'},
              'runtime_scoped_wall_median_s':{
                  name:statistics.median(row['runtime_scoped_wall_s_all_rank_cohorts'])
                  for name,row in rows.items()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],
                      'packet_rows':len(rows['on']['packet_summary'])}))


if __name__ == '__main__':
    main()
