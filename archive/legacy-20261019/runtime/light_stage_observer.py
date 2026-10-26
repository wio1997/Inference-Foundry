"""Dormant, bounded, reusable full-cohort current-stream stage observation.

Construct and warm-record during the service warmup lifecycle. Bind only after
the measured arm is admitted. No observer-only device wait occurs in Product.
Export after the existing terminal drain and before reusing the pool.
"""
from __future__ import annotations

import math
import time

import torch

LABELS = ('cycle_begin', 'target_before', 'target_after',
          'proposer_before', 'proposer_after')
MAX_CYCLES = 1025


class LightStageRecorder:
    def __init__(self, stream, *, max_cycles=MAX_CYCLES, event_factory=None,
                 clock_ns=None, current_stream=None):
        if type(max_cycles) is not int or not 1 <= max_cycles <= MAX_CYCLES:
            raise ValueError('invalid frozen full-cohort Event capacity')
        self.stream = stream
        self.max_cycles = max_cycles
        self.clock_ns = clock_ns or time.monotonic_ns
        self.current_stream = current_stream or torch.npu.current_stream
        make_event = event_factory or (lambda: torch.npu.Event(enable_timing=True))
        setup_begin = self.clock_ns()
        self.events = [make_event() for _ in range(max_cycles * len(LABELS) + 1)]
        create_end = self.clock_ns()
        self.stream_key = self._stream_key(stream)
        for event in self.events:
            event.record(stream)
        record_end = self.clock_ns()
        self.warm_record_end_ns = record_end
        self.setup_host_ns = record_end - setup_begin
        self.create_host_ns = create_end - setup_begin
        self.warm_record_host_ns = record_end - create_end
        self.generations = [1] * len(self.events)
        self._active = False
        # Frozen warm48 consumes four c12 cohorts before measured arm opens.
        self._last_cohort = 4
        self._next = 0
        self._terminal_cycle = None
        self.host_submit_ns = []
        self.class_rows = []

    @staticmethod
    def _stream_key(stream):
        native = getattr(stream, 'npu_stream', None)
        device = getattr(stream, 'device', None)
        if native is None or device is None:
            raise RuntimeError('light Event stream identity unavailable')
        return str(device), int(native)

    def begin_cohort(self, cohort: int):
        if self._active or type(cohort) is not int or cohort != self._last_cohort + 1:
            raise RuntimeError('light Event cohort reuse/order violation')
        if self._stream_key(self.current_stream()) != self.stream_key:
            raise RuntimeError('light Event current stream drift at cohort bind')
        # The terminal Event was the last warm record and last used record of
        # the prior cohort on this stream. Query is nonblocking, no new drain.
        if not self.events[-1].query():
            raise RuntimeError('light Event warm/prior cohort not complete')
        self._active = True
        self._last_cohort = cohort
        self._next = 0
        self._terminal_cycle = None
        self.host_submit_ns = []
        self.class_rows = []

    def _record(self, index):
        if self._stream_key(self.current_stream()) != self.stream_key:
            raise RuntimeError('light Event actual current stream drift')
        stamp = self.clock_ns()
        self.events[index].record(self.stream)
        self.generations[index] += 1
        self.host_submit_ns.append(stamp)

    def mark(self, cycle: int, label: str):
        if not self._active or self._terminal_cycle is not None:
            raise RuntimeError('light Event mark outside active cohort')
        if type(cycle) is not int or not 0 <= cycle < self.max_cycles:
            raise RuntimeError('light Event cycle overflow')
        if label not in LABELS:
            raise RuntimeError('unknown light Event label')
        index = cycle * len(LABELS) + LABELS.index(label)
        if index != self._next:
            raise RuntimeError(f'light Event order drift: expected {self._next}, got {index}')
        self._record(index)
        self._next += 1

    def class_row(self, cycle: int, parked_before: int, parked_after: int):
        if not self._active or type(cycle) is not int or cycle != len(self.class_rows):
            raise RuntimeError('light Event class cycle drift')
        if not (0 <= parked_before <= parked_after <= 12):
            raise RuntimeError('light Event parked class invalid')
        if self._next < (cycle + 1) * len(LABELS):
            raise RuntimeError('light Event class before proposer')
        self.class_rows.append({'cycle': cycle, 'parked_before': parked_before,
                                'parked_after': parked_after})

    def terminal(self, cycle: int):
        if not self._active or self._terminal_cycle is not None or cycle + 1 != len(self.class_rows):
            raise RuntimeError('light Event terminal class mismatch')
        if self._next != (cycle + 1) * len(LABELS):
            raise RuntimeError('light Event terminal before complete cycle')
        self._record(len(self.events) - 1)
        self._terminal_cycle = cycle

    def export_after_existing_drain(self, identity: dict):
        cycles = identity.get('cycles')
        counts = identity.get('accepted_counts')
        hashes = ('canonical_effective_staged_trajectory_sha256',
                  'count_history_sha256', 'raw_padded_token_history_sha256',
                  'runtime_bulk_output_sha256')
        if (not self._active or type(cycles) is not int or cycles < 1
                or cycles > self.max_cycles or self._terminal_cycle != cycles - 1
                or len(self.class_rows) != cycles
                or self._next != cycles * len(LABELS)
                or len(self.host_submit_ns) != self._next + 1
                or type(identity.get('rank')) is not int
                or not 0 <= identity['rank'] < 8
                or identity.get('cohort') != self._last_cohort
                or not isinstance(identity.get('req_ids'), list)
                or len(identity['req_ids']) != 12
                or len(set(identity['req_ids'])) != 12
                or not identity.get('run_ts') or not identity.get('time_namespace')
                or identity.get('generated_output_counts') != [1024] * 12
                or not isinstance(counts, list) or len(counts) != cycles
                or any(not isinstance(row, list) or len(row) != 12 or
                       any(type(v) is not int or not 0 <= v <= 8 for v in row)
                       for row in counts)
                or any(not isinstance(identity.get(key), str) or
                       len(identity[key]) != 64 or
                       any(ch not in '0123456789abcdef' for ch in identity[key])
                       for key in hashes)):
            raise RuntimeError('light Event identity/coverage incomplete')
        used_indices = list(range(self._next)) + [len(self.events) - 1]
        used = [self.events[index] for index in used_indices]
        if not self.events[-1].query() or not all(event.query() for event in used):
            raise RuntimeError('light Event incomplete after existing drain')
        first = used[0]
        elapsed = [first.elapsed_time(event) for event in used]
        if any(not math.isfinite(value) or value < 0 for value in elapsed):
            raise RuntimeError('light Event nonfinite or negative time')
        if any(b < a for a, b in zip(elapsed, elapsed[1:])):
            raise RuntimeError('light Event stream time reversed')
        if any(b < a for a, b in zip(self.host_submit_ns,
                                     self.host_submit_ns[1:])):
            raise RuntimeError('light Event Host order reversed')
        events = []
        for index in range(self._next):
            events.append({'cycle': index // len(LABELS),
                           'label': LABELS[index % len(LABELS)],
                           'generation': self.generations[index],
                           'host_submit_ns': self.host_submit_ns[index],
                           'elapsed_ms': elapsed[index]})
        events.append({'cycle': cycles - 1, 'label': 'serve_terminal',
                       'generation': self.generations[-1],
                       'host_submit_ns': self.host_submit_ns[-1],
                       'elapsed_ms': elapsed[-1]})
        self._active = False
        return {'status': 'instrumented_current_stream_full_cohort',
                'identity': identity, 'event_capacity': len(self.events),
                'event_used': len(used), 'stream_key': self.stream_key,
                'pre_product_warm_record_end_ns': self.warm_record_end_ns,
                'pre_product_setup_host_ns': self.setup_host_ns,
                'pre_product_create_host_ns': self.create_host_ns,
                'pre_product_warm_record_host_ns': self.warm_record_host_ns,
                'class_rows': self.class_rows, 'events': events,
                'limits': 'Current-stream stage Events include queue/wait and are not intrinsic primitive service, all-stream completion or Product savings. Warm record occurs in service warmup with no observer-only drain. Export follows the existing terminal drain but remains before Product return, requiring observer controls.'}
