#!/usr/bin/env python3
"""CPU fake-event checks for Run655 reusable bounded recorder invariants."""
from __future__ import annotations

import json

from runtime.light_stage_observer import LABELS, LightStageRecorder


class Stream:
    device = 'npu:0'
    npu_stream = 17


class FakeEvent:
    next_seq = 0
    def __init__(self):
        self.seq = None
    def record(self, stream):
        assert stream.npu_stream == 17
        FakeEvent.next_seq += 1
        self.seq = FakeEvent.next_seq
    def query(self):
        return self.seq is not None
    def elapsed_time(self, other):
        return (other.seq - self.seq) * 0.001


def expect_raises(fn, words):
    try:
        fn()
    except RuntimeError as exc:
        assert words in str(exc), str(exc)
        return
    raise AssertionError(f'missing rejection {words}')


def make(cycles=3):
    stream = Stream()
    return LightStageRecorder(
        stream, max_cycles=cycles, event_factory=FakeEvent,
        current_stream=lambda: stream)


def identity(cycles, cohort):
    return {'rank': 0, 'cohort': cohort, 'req_ids': [f'r{i}' for i in range(12)],
            'run_ts': 'RUN655-FAKE', 'time_namespace': 'time:[1]',
            'cycles': cycles, 'generated_output_counts': [1024] * 12,
            'accepted_counts': [[0] * 12 for _ in range(cycles)],
            'canonical_effective_staged_trajectory_sha256': '0' * 64,
            'count_history_sha256': '0' * 64,
            'raw_padded_token_history_sha256': '0' * 64,
            'runtime_bulk_output_sha256': '0' * 64}


def fill(rec, cycles):
    for cycle in range(cycles):
        for label in LABELS:
            rec.mark(cycle, label)
        rec.class_row(cycle, cycle, cycle + 1)
    rec.terminal(cycles - 1)


def main():
    rec = make()
    expect_raises(lambda: rec.mark(0, 'cycle_begin'), 'outside active')
    rec.begin_cohort(5)
    expect_raises(lambda: rec.mark(0, 'target_before'), 'order drift')
    fill(rec, 2)
    expect_raises(lambda: rec.export_after_existing_drain({'cycles': 2}), 'identity/coverage')
    result = rec.export_after_existing_drain(identity(2, 5))
    assert result['event_capacity'] == 16
    assert result['event_used'] == 11
    assert len(result['class_rows']) == 2
    assert result['events'][-1]['label'] == 'serve_terminal'
    assert result['events'][-1]['cycle'] == 1
    assert result['events'][0]['generation'] == 2
    expect_raises(lambda: rec.mark(2, 'cycle_begin'), 'outside active')
    rec.begin_cohort(6)
    fill(rec, 1)
    result2 = rec.export_after_existing_drain(identity(1, 6))
    assert result2['events'][0]['generation'] == 3
    assert result2['events'][-1]['generation'] == 3
    expect_raises(lambda: rec.begin_cohort(6), 'cohort reuse/order')
    bad = make(1)
    bad.begin_cohort(5)
    fill(bad, 1)
    bad.events[-1].seq = -100
    expect_raises(lambda: bad.export_after_existing_drain(identity(1, 5)), 'negative')
    overflow = make(1)
    overflow.begin_cohort(5)
    expect_raises(lambda: overflow.mark(1, 'cycle_begin'), 'overflow')
    print(json.dumps({'fake_event_checks': 'pass', 'markers_per_cycle': len(LABELS),
                      'event_capacity_test': result['event_capacity'],
                      'event_used_first_cohort': result['event_used'],
                      'event_used_second_cohort': result2['event_used'],
                      'pool_reuse_generation': result2['events'][0]['generation']}))


if __name__ == '__main__':
    main()
