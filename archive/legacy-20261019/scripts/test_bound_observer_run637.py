#!/usr/bin/env python3
"""CPU fault checks for Run637 bounded event ledger; no NPU workload."""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path

class Event:
    made = 0
    recorded = 0
    def __init__(self, enable_timing):
        assert enable_timing
        Event.made += 1
        self.sequence = None
    def record(self, stream):
        Event.recorded += 1
        self.sequence = Event.recorded
    def elapsed_time(self, other):
        return other.sequence - self.sequence
    def query(self):
        return self.sequence is not None

class Stream:
    def __init__(self, handle):
        self.npu_stream = handle
        self.device = "npu:0"

class NPU:
    Event = Event
    active = None
    @classmethod
    def current_stream(cls):
        return cls.active

sys.modules['torch'] = types.SimpleNamespace(npu=NPU())
MODULE = Path('/data/wio/Inference_Foundry/runtime/bound_observer.py')
spec = importlib.util.spec_from_file_location('bound_observer_run637', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    current, copy = Stream(1), Stream(2)
    NPU.active = current
    recorder = module.BoundEventRecorder((63,64,65), current_stream=current, copy_stream=copy)
    assert Event.made == Event.recorded == 3*(len(module.CURRENT)+len(module.SIDE))
    initial = Event.made
    for cycle in (63,64,65):
        for label in module.CURRENT:
            recorder.mark(cycle,label)
            if label == 'proposer_before':
                recorder.wait_start(cycle,cycle-1)
                recorder.wait_end(cycle)
                NPU.active = copy
                for side_label in module.SIDE:
                    recorder.mark(cycle,side_label,stream='copy')
                NPU.active = current
    assert Event.made == initial and Event.recorded == initial*2
    identity={'rank':0,'cohort':5,'req_ids':[str(i) for i in range(12)],'run_ts':'test','time_namespace':'test','runtime_basis':{'cycles':1200,'canonical_effective_staged_trajectory_sha256':'a'*64,'accepted_count_history_sha256':'b'*64,'raw_padded_token_history_sha256':'c'*64},'generated_output_counts':[1024]*12,'product_output_sha256':'d'*64}
    result = recorder.export_after_existing_drain(identity)
    assert len(result['events']) == initial
    assert result['existing_host_waits']['64'][0]['producer_cycle'] == 63
    assert all(result[k] is None for k in ('strict_resource_floor_s','strict_scheduling_floor_s','product_e2e_ceiling_tps','numeric_current_to_credible_limit_gap'))
    for action in (lambda: recorder.mark(64,'cycle_begin'),
                   lambda: recorder.mark(64,'unknown'),
                   lambda: recorder.mark(64,'host_copy_before')):
        try: action()
        except RuntimeError: pass
        else: raise AssertionError('fault accepted')
    incomplete = module.BoundEventRecorder((64,), current_stream=current)
    try: incomplete.export_after_existing_drain({})
    except RuntimeError: pass
    else: raise AssertionError('incomplete selected cycle accepted')
    for bad in ({}, {'rank':0}):
        try: recorder.export_after_existing_drain(bad)
        except RuntimeError: pass
        else: raise AssertionError('bad identity accepted')
    for bad_cycles in ((2,99),(True,2)):
        try: module.BoundEventRecorder(bad_cycles,current_stream=current)
        except ValueError: pass
        else: raise AssertionError('bad cycles accepted')
    try: recorder.wait_end(64)
    except RuntimeError: pass
    else: raise AssertionError('orphan wait end accepted')
    original_elapsed = Event.elapsed_time
    Event.elapsed_time = lambda self, other: float('nan')
    try:
        try: recorder.export_after_existing_drain(identity)
        except RuntimeError: pass
        else: raise AssertionError('NaN elapsed accepted')
    finally:
        Event.elapsed_time = original_elapsed
    print({'status':'pass','events':initial,'negative_cases':10})
finally:
    sys.modules.pop('torch', None)
