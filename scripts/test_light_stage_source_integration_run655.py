#!/usr/bin/env python3
"""Execute candidate step and serving loop with CPU fakes; no live source edit."""
from __future__ import annotations

import ast
import contextlib
import json
from types import SimpleNamespace

import loop081_light_bridge_patch_run655 as patch
from runtime.light_stage_observer import LightStageRecorder


class Stream:
    device = 'npu:0'
    npu_stream = 17


class Event:
    serial = 0
    def __init__(self): self.n = None
    def record(self, stream):
        assert stream.npu_stream == 17
        Event.serial += 1
        self.n = Event.serial
    def query(self): return self.n is not None
    def elapsed_time(self, other): return (other.n - self.n) / 1000


class Tensor:
    def __init__(self, shape=(12, 7)): self.shape = shape
    def copy_(self, other): return self


class History:
    def __getitem__(self, index): return self
    def copy_(self, other): return self


def compiled_step():
    source = patch.cycle(patch.SOURCES['cycle'][0].read_text())
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ExtremeDecodeRuntime')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'step')
    fn.decorator_list = []
    fn.returns = None
    mod = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    globals_ = {'CycleResult': lambda **kwargs: SimpleNamespace(**kwargs)}
    exec(compile(mod, '<candidate-step>', 'exec'), globals_)
    return globals_['step']


def compiled_serving_loop():
    source = patch.serving(patch.SOURCES['serving'][0].read_text())
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FixedCohortServing')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    mod = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    return compile(mod, '<candidate-serving-loop>', 'exec')


def identity():
    return {'rank':0,'cohort':5,'req_ids':[f'r{i}' for i in range(12)],
            'run_ts':'FAKE','time_namespace':'time:[1]','cycles':1,
            'generated_output_counts':[1024]*12,
            'accepted_counts':[[8]*12],
            'canonical_effective_staged_trajectory_sha256':'0'*64,
            'count_history_sha256':'0'*64,
            'raw_padded_token_history_sha256':'0'*64,
            'runtime_bulk_output_sha256':'0'*64}


def run(on):
    stream = Stream()
    probe = LightStageRecorder(stream, max_cycles=1, event_factory=Event,
                               current_stream=lambda: stream) if on else None
    if probe: probe.begin_cohort(5)
    state = SimpleNamespace(cycle_index=0, draft_tokens=Tensor(), num_sampled=Tensor())
    acceptance_value = SimpleNamespace(sampled_token_ids=Tensor(), num_sampled=Tensor())
    runtime = SimpleNamespace(
        state=state, _light_probe=probe, _diagnose=False, _diagnose_limit=0,
        _profile_dag=False, _schedule_pending=False, _schedule_mode='off',
        _schedule_fallbacks=0, target_metadata=None, kv_slot_audit=None,
        target_page_audit=None, _cycle_profiler=None, _cycle_profile_sync_target=False,
        _scope=lambda label: contextlib.nullcontext(),
        _profile_cycle_begin=lambda:None, _profile_cycle_end=lambda:None,
        _state_machine=SimpleNamespace(prepare_target_inputs=lambda:None,
                                       advance_state=lambda acceptance:None),
        target=SimpleNamespace(execute=lambda state:SimpleNamespace(logits=Tensor()),
                               state_fingerprint=lambda:'target'),
        acceptance=SimpleNamespace(execute=lambda state, output:acceptance_value),
        proposer=SimpleNamespace(execute=lambda *args:Tensor(),
                                 state_fingerprint=lambda:'draft'),
        diagnostic_events=[],
        config=SimpleNamespace(batch_size=12, target_tokens_per_request=8),
    )
    runtime.step = lambda: compiled_step()(runtime)
    serving = SimpleNamespace(runtime=runtime, max_cycles=1,
                              config=SimpleNamespace(batch_size=12),
                              _parked=[False]*12,
                              _progress_baseline=[0]*12,
                              remaining=[1024]*12)
    serving._committed_progress=lambda:[1024]*12
    serving._park_completed=lambda progress:serving._parked.__setitem__(slice(None),[True]*12)
    scope={'self':serving,'cfg':serving.config,'token_history':History(),
           'count_history':History(),'cycles':0}
    exec(compiled_serving_loop(), scope)
    assert scope['cycles']==1 and state.cycle_index==1
    if probe:
        packet=probe.export_after_existing_drain(identity())
        assert [e['label'] for e in packet['events']]==[
            'cycle_begin','target_before','target_after',
            'proposer_before','proposer_after','serve_terminal']
        assert packet['class_rows']==[{'cycle':0,'parked_before':0,'parked_after':12}]
        return packet['event_used']
    return 0


def main():
    print(json.dumps({'on_event_used':run(True),'off_event_used':run(False),
                      'candidate_step_serving_cpu':'pass'}))

if __name__=='__main__':main()
