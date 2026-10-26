"""Bounded, dormant event ledger for a fixed-work Extreme Runtime diagnostic.

Events are allocated and first-recorded before the measured serving loop.  A
measurement record overwrites only its own warm generation, never another
measurement.  Export is legal only after an existing terminal device drain.
"""
from __future__ import annotations

import time
import math
from collections import defaultdict

import torch


CURRENT = (
    'cycle_begin', 'prepare_target', 'derived_target_metadata',
    'target_before', 'target_forward_return', 'target_logits_return',
    'target_after', 'acceptance_after', 'state_advance_after',
    'proposer_before', 'proposer_after', 'draft_commit_after',
    'serve_stage_after', 'serve_park_after',
)
SIDE = ('host_copy_before', 'host_copy_after')
REQUIRED_CURRENT = tuple(name for name in CURRENT
                         if name != 'derived_target_metadata')


class BoundEventRecorder:
    """At most three declared adjacent cycles, with one event per marker."""

    def __init__(self, cycles: tuple[int, ...], *, current_stream, copy_stream=None):
        if not 1 <= len(cycles) <= 3 or len(set(cycles)) != len(cycles):
            raise ValueError('bound packet requires 1–3 unique cycles')
        if any(type(c) is not int or c < 0 for c in cycles):
            raise ValueError('invalid cycle')
        self.cycles = tuple(sorted(cycles))
        if any(b != a + 1 for a, b in zip(self.cycles, self.cycles[1:])):
            raise ValueError('sample cycles must be adjacent')
        self.streams = {'current': current_stream}
        if copy_stream is not None:
            self.streams['copy'] = copy_stream
        stream_ids = {key: self._stream_identity(stream)
                      for key, stream in self.streams.items()}
        if len({device for device, _ in stream_ids.values()}) != 1:
            raise ValueError('bound streams belong to different devices')
        if len(set(stream_ids.values())) != len(stream_ids):
            raise ValueError('bound copy stream aliases current stream')
        self.stream_ids = stream_ids
        self.rows = {}
        for cycle in self.cycles:
            for stream, labels in (('current', CURRENT), ('copy', SIDE)):
                if stream not in self.streams:
                    continue
                for label in labels:
                    event = torch.npu.Event(enable_timing=True)
                    # The installed torch_npu Event creates its native object
                    # at first record.  Queue this generation before the
                    # measured Runtime; the later record uses the same stream.
                    event.record(self.streams[stream])
                    self.rows[(cycle, label)] = {
                        'event': event, 'stream': stream, 'generation': 0,
                        'host_submit_ns': None,
                    }
        self.waits = defaultdict(list)
        self._wait_start = {}
        self._issue_ordinal = 0

    @staticmethod
    def _stream_identity(stream):
        native = getattr(stream, 'npu_stream', None)
        device = getattr(stream, 'device', None)
        if native is None or device is None:
            raise RuntimeError('native stream/device identity unavailable')
        return str(device), int(native)

    def mark(self, cycle: int, label: str, *, stream: str = 'current') -> None:
        if cycle not in self.cycles:
            return
        row = self.rows.get((cycle, label))
        if row is None:
            raise RuntimeError('unknown or unavailable bound event label')
        if row['stream'] != stream or row['generation'] != 0:
            raise RuntimeError('bound event stream or generation mismatch')
        actual = torch.npu.current_stream()
        if self._stream_identity(actual) != self._stream_identity(self.streams[stream]):
            raise RuntimeError('actual current stream differs from bound event stream')
        if not row['event'].query():
            raise RuntimeError('warm event generation not drained before measurement')
        row['host_submit_ns'] = time.monotonic_ns()
        row['event'].record(self.streams[stream])
        row['generation'] = 1
        self._issue_ordinal += 1
        row['issue_ordinal'] = self._issue_ordinal

    def wait_start(self, cycle: int, producer_cycle: int) -> None:
        if cycle not in self.cycles:
            return
        if producer_cycle != cycle - 1:
            raise RuntimeError('host-count copy producer generation mismatch')
        if cycle in self._wait_start:
            raise RuntimeError('duplicate existing wait start')
        self._wait_start[cycle] = (time.monotonic_ns(), producer_cycle)

    def wait_end(self, cycle: int) -> None:
        if cycle not in self._wait_start:
            if cycle in self.cycles:
                raise RuntimeError('existing wait end without start')
            return
        begin, producer_cycle = self._wait_start.pop(cycle)
        self.waits[cycle].append({
            'kind': 'existing_host_count_copy_event_synchronize',
            'producer_cycle': producer_cycle, 'start_ns': begin,
            'end_ns': time.monotonic_ns(),
        })

    def export_after_existing_drain(self, identity: dict) -> dict:
        basis = identity.get('runtime_basis')
        if (type(identity.get('rank')) is not int or
                not 0 <= identity['rank'] < 8 or
                type(identity.get('cohort')) is not int or
                identity['cohort'] < 0 or
                not isinstance(identity.get('req_ids'), list) or
                len(identity['req_ids']) != 12 or
                len(set(identity['req_ids'])) != 12 or
                not identity.get('run_ts') or
                not identity.get('time_namespace') or
                not isinstance(basis, dict) or
                type(basis.get('cycles')) is not int or
                basis['cycles'] <= max(self.cycles) or
                any(not isinstance(basis.get(key), str) or
                    len(basis[key]) != 64 or
                    any(char not in '0123456789abcdef' for char in basis[key])
                    for key in ('canonical_effective_staged_trajectory_sha256',
                                'accepted_count_history_sha256',
                                'raw_padded_token_history_sha256')) or
                not isinstance(identity.get('product_output_sha256'), str) or
                len(identity['product_output_sha256']) != 64 or
                any(char not in '0123456789abcdef'
                    for char in identity['product_output_sha256']) or
                identity.get('generated_output_counts') != [1024] * 12):
            raise RuntimeError('incomplete bound packet identity')
        if self._wait_start:
            raise RuntimeError('unclosed existing wait')
        for cycle in self.cycles:
            if any(self.rows[(cycle, label)]['generation'] != 1
                   for label in REQUIRED_CURRENT):
                raise RuntimeError('incomplete selected cycle markers')
            if 'copy' in self.streams and any(
                    self.rows[(cycle, label)]['generation'] != 1
                    for label in SIDE):
                raise RuntimeError('incomplete selected side-stream markers')
            if len(self.waits[cycle]) != 1:
                raise RuntimeError('missing or duplicate existing host mirror wait')
            if 'copy' in self.streams:
                ordinal = lambda name: self.rows[(cycle, name)]['issue_ordinal']
                if not (ordinal('proposer_before') < ordinal('host_copy_before') <
                        ordinal('host_copy_after') < ordinal('proposer_after')):
                    raise RuntimeError('side-copy Host issue order inconsistent')
                wait = self.waits[cycle][0]
                if not (self.rows[(cycle, 'proposer_before')]['host_submit_ns'] <=
                        wait['start_ns'] <= wait['end_ns'] <=
                        self.rows[(cycle, 'host_copy_before')]['host_submit_ns']):
                    raise RuntimeError('existing wait outside proposer issue chain')
            last_issue = -1
            for label in CURRENT:
                row = self.rows[(cycle, label)]
                if row['generation'] != 1:
                    continue
                if row['issue_ordinal'] <= last_issue:
                    raise RuntimeError('current-stream issue order reversed')
                last_issue = row['issue_ordinal']
        by_stream = defaultdict(list)
        order = {name: index for index, name in enumerate(CURRENT)}
        order.update({name: index for index, name in enumerate(SIDE)})
        for (cycle, label), row in sorted(
            self.rows.items(), key=lambda item: (item[0][0], order[item[0][1]])
        ):
            if row['generation'] == 1:
                if not row['event'].query():
                    raise RuntimeError('measurement event incomplete at export')
                by_stream[row['stream']].append((cycle, label, row))
        intervals = []
        extract_start_ns = time.monotonic_ns()
        for stream, items in by_stream.items():
            first = items[0][2]['event']
            last_elapsed = -1.0
            for cycle, label, row in items:
                elapsed = first.elapsed_time(row['event'])
                if not math.isfinite(elapsed) or elapsed < 0 or elapsed < last_elapsed:
                    raise RuntimeError('same-stream device marker time reversed')
                last_elapsed = elapsed
                intervals.append({
                    'cycle': cycle, 'label': label, 'stream': stream,
                    'event_generation': row['generation'],
                    'issue_ordinal': row['issue_ordinal'],
                    'host_submit_ns': row['host_submit_ns'],
                    'elapsed_ms_from_first_same_stream': elapsed,
                })
        return {
            'status': 'instrumented_current_stream_markers_only',
            'identity': identity, 'sample_cycles': self.cycles,
            'stream_ids': self.stream_ids,
            'events': intervals,
            'existing_host_waits': {str(c): w for c, w in self.waits.items()},
            'extract_host_interval_ns': [extract_start_ns, time.monotonic_ns()],
            'post_runtime_export_uses_sync_api': True,
            'limits': 'Events are stream-order markers, not proof of all HCCL/native producer completion, tensor generation, necessary work, exposed Product wall or a Bound. Hashes describe effective/staged counts and tokens; parked-slot raw acceptance is not captured, so full-workload W0 equality remains conditional. Cross-stream elapsed values are not compared. Installed torch_npu elapsed_time performs synchronization APIs even after query; caller must prove existing drain and no concurrent later work, and ON Product timing includes extraction overhead.',
            'strict_resource_floor_s': None, 'strict_scheduling_floor_s': None,
            'product_e2e_ceiling_tps': None, 'numeric_current_to_credible_limit_gap': None,
        }
