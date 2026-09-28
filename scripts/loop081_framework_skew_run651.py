#!/usr/bin/env python3
"""Run638 ON same-W0 all8 Host-submit skew decomposition; no timing transfer."""
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
BASE = ROOT / 'evidence/20260928_loop081_bound'
SOURCE = BASE / 'run638/arm_scoped_recovery.json'
OUT = BASE / 'run651/on_host_stage_skew.json'
LABELS = ('cycle_begin', 'target_before', 'target_forward_return',
          'target_logits_return', 'acceptance_after', 'state_advance_after',
          'proposer_before', 'host_copy_before', 'host_copy_after',
          'proposer_after', 'draft_commit_after')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    source = json.loads(SOURCE.read_text())
    assert source['status'] == 'arm_scoped_current_packet_pass_cross_arm_fixed_W0_rejected'
    pins = source['on_packet_raw_sha256']
    rows = []
    namespaces = set()
    for cohort in range(5, 9):
        packets = []
        for rank in range(8):
            path = BASE / f'run638/live/on/packet/rank{rank}_cohort{cohort}.json'
            assert pins[str(path)] == sha(path)
            packet = json.loads(path.read_text())
            assert packet['status'] == 'instrumented_current_stream_markers_only'
            assert packet['identity']['rank'] == rank
            assert packet['identity']['cohort'] == cohort
            assert packet['identity']['run_ts'] == 'LOOP081-RUN638-ON'
            assert packet['sample_cycles'] == [63, 64, 65]
            namespaces.add(packet['identity']['time_namespace'])
            packets.append(packet)
        for cycle in (63, 64, 65):
            stage = {}
            for label in LABELS:
                times = []
                for rank, packet in enumerate(packets):
                    matching = [e for e in packet['events']
                                if e['cycle'] == cycle and e['label'] == label]
                    assert len(matching) == 1, (cohort, cycle, rank, label)
                    event = matching[0]
                    assert event['event_generation'] == 1
                    times.append(event['host_submit_ns'])
                stage[label] = {
                    'spread_ms': (max(times) - min(times)) / 1e6,
                    'latest_rank': times.index(max(times)),
                    'earliest_rank': times.index(min(times)),
                }
            rows.append({
                'cohort': cohort, 'cycle': cycle, 'stage': stage,
                'target_forward_submit_skew_change_ms':
                    stage['target_forward_return']['spread_ms'] -
                    stage['target_before']['spread_ms'],
                'proposer_submit_skew_change_ms':
                    stage['proposer_after']['spread_ms'] -
                    stage['proposer_before']['spread_ms'],
            })
    assert len(namespaces) == 1 and len(rows) == 12
    def stats(label):
        values = [r['stage'][label]['spread_ms'] for r in rows]
        return {'min_ms': min(values), 'median_ms': statistics.median(values),
                'max_ms': max(values),
                'latest_rank_counts': dict(sorted(Counter(
                    r['stage'][label]['latest_rank'] for r in rows).items()))}
    summary = {label: stats(label) for label in LABELS}
    summary['paired_forward_skew_change_ms'] = {
        'median': statistics.median(r['target_forward_submit_skew_change_ms'] for r in rows),
        'negative_count': sum(r['target_forward_submit_skew_change_ms'] < 0 for r in rows),
    }
    summary['paired_proposer_skew_change_ms'] = {
        'median': statistics.median(r['proposer_submit_skew_change_ms'] for r in rows),
        'positive_count': sum(r['proposer_submit_skew_change_ms'] > 0 for r in rows),
    }
    successor_pairs = []
    by_key = {(r['cohort'], r['cycle']): r for r in rows}
    for cohort in range(5, 9):
        for cycle in (63, 64):
            prior = by_key[(cohort, cycle)]['stage']['proposer_after']
            successor = by_key[(cohort, cycle + 1)]['stage']['target_before']
            successor_pairs.append({
                'cohort': cohort, 'from_cycle': cycle,
                'proposer_after_spread_ms': prior['spread_ms'],
                'next_target_before_spread_ms': successor['spread_ms'],
                'spread_change_ms': successor['spread_ms'] - prior['spread_ms'],
                'same_latest_rank': successor['latest_rank'] == prior['latest_rank'],
            })
    summary['successor_carry'] = {
        'count': len(successor_pairs),
        'same_latest_rank_count': sum(x['same_latest_rank'] for x in successor_pairs),
        'median_spread_change_ms': statistics.median(x['spread_change_ms'] for x in successor_pairs),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        'status': 'within_ON_W0_all8_host_submit_stage_skew_pass',
        'source_sha256': sha(SOURCE), 'time_namespace': namespaces.pop(),
        'scope': 'Run638 ON 4 measured cohorts x selected cycles63-65, all8. Host-submit marker timestamps only; different W0 from Run611/606/Run99. Cross-rank host clock alignment conditional. No native producer completion or legal schedule saving.',
        'rows': rows, 'successor_pairs': successor_pairs, 'summary': summary,
        'framework_only_product_tps_bound': None,
    }, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
