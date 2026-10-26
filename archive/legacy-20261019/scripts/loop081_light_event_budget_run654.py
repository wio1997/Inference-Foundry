#!/usr/bin/env python3
"""Idle-device capacity probe for the proposed light full-cohort Event ledger.

No model, no Runtime source edits. Timings describe the probe only.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import torch_npu  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cycles', type=int, default=1025)
    parser.add_argument('--marks', type=int, default=5)
    args = parser.parse_args()
    if args.cycles < 1 or args.marks < 1:
        raise ValueError('positive event count required')
    torch.npu.set_device(args.device)
    stream = torch.npu.current_stream()
    count = args.cycles * args.marks + 1
    t0 = time.perf_counter_ns()
    events = [torch.npu.Event(enable_timing=True) for _ in range(count)]
    t1 = time.perf_counter_ns()
    for event in events:
        event.record(stream)
    t2 = time.perf_counter_ns()
    torch.npu.synchronize()
    t3 = time.perf_counter_ns()
    if not all(event.query() for event in events):
        raise RuntimeError('warm Event not complete after drain')
    for event in events:
        event.record(stream)
    t4 = time.perf_counter_ns()
    torch.npu.synchronize()
    t5 = time.perf_counter_ns()
    # One first-relative elapsed read per marker, as intended for export.
    first = events[0]
    elapsed = [first.elapsed_time(event) for event in events]
    t6 = time.perf_counter_ns()
    if any(b < a for a, b in zip(elapsed, elapsed[1:])):
        raise RuntimeError('elapsed Event order reversed')
    print(json.dumps({
        'device': args.device, 'cycles': args.cycles, 'marks': args.marks,
        'event_count': count, 'create_ms': (t1-t0)/1e6,
        'first_record_ms': (t2-t1)/1e6, 'first_drain_ms': (t3-t2)/1e6,
        'rerecord_enqueue_ms': (t4-t3)/1e6,
        'second_drain_ms': (t5-t4)/1e6,
        'query_and_elapsed_ms': (t6-t5)/1e6,
        'first_to_last_elapsed_ms': elapsed[-1],
        'torch_version': torch.__version__,
        'torch_npu_version': torch_npu.__version__,
    }, sort_keys=True))


if __name__ == '__main__':
    main()
