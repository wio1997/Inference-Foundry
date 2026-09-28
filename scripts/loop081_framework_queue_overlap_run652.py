#!/usr/bin/env python3
"""Paired same-stream Event and Host marker intervals across ON-cycle handoff."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
BASE = ROOT / 'evidence/20260928_loop081_bound'
SOURCE = BASE / 'run638/arm_scoped_recovery.json'
OUT = BASE / 'run652/host_vs_stream_handoff.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source = json.loads(SOURCE.read_text())
    assert source['status'] == 'arm_scoped_current_packet_pass_cross_arm_fixed_W0_rejected'
    pins = source['on_packet_raw_sha256']
    rows = []
    for cohort in range(5, 9):
        for rank in range(8):
            path = BASE / f'run638/live/on/packet/rank{rank}_cohort{cohort}.json'
            assert pins[str(path)] == sha(path)
            packet = json.loads(path.read_text())
            assert packet['status'] == 'instrumented_current_stream_markers_only'
            assert packet['identity']['rank'] == rank
            assert packet['identity']['cohort'] == cohort
            assert packet['identity']['run_ts'] == 'LOOP081-RUN638-ON'
            assert packet['sample_cycles'] == [63, 64, 65]
            events = {(e['cycle'], e['label']): e for e in packet['events']}
            assert len(events) == len(packet['events'])
            for cycle in (63, 64):
                a = events[(cycle, 'proposer_after')]
                b = events[(cycle + 1, 'target_before')]
                assert a['stream'] == b['stream'] == 'current'
                assert a['event_generation'] == b['event_generation'] == 1
                assert a['issue_ordinal'] < b['issue_ordinal']
                host_ms = (b['host_submit_ns'] - a['host_submit_ns']) / 1e6
                stream_ms = (b['elapsed_ms_from_first_same_stream'] -
                             a['elapsed_ms_from_first_same_stream'])
                assert host_ms > 0 and stream_ms >= 0
                rows.append({'cohort': cohort, 'rank': rank, 'from_cycle': cycle,
                             'host_marker_interval_ms': host_ms,
                             'same_stream_event_interval_ms': stream_ms,
                             'host_minus_event_ms': host_ms - stream_ms})
    assert len(rows) == 64
    def distribution(key):
        values = [x[key] for x in rows]
        return {'min_ms': min(values), 'median_ms': statistics.median(values),
                'max_ms': max(values)}
    result = {
        'status': 'within_ON_W0_paired_host_device_interval_pass',
        'source_sha256': sha(SOURCE),
        'scope': 'Run638 ON four cohorts x all8 x two adjacent cycle handoffs. Host marker submit interval and same-current-stream Event elapsed interval are separately valid rank-local durations. Their difference is not idle, overhead or removable wall. Cross-arm timing transfer invalid.',
        'rows': rows,
        'summary': {
            'n': len(rows),
            'host_marker_interval': distribution('host_marker_interval_ms'),
            'same_stream_event_interval': distribution('same_stream_event_interval_ms'),
            'host_minus_event': distribution('host_minus_event_ms'),
            'host_greater_than_event_count': sum(x['host_minus_event_ms'] > 0 for x in rows),
        },
        'framework_only_product_tps_bound': None,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))

if __name__ == '__main__':
    main()
