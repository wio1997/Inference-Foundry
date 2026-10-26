#!/usr/bin/env python3
"""Offline positive and mutation gates for Run576's reversible collector."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path('/data/wio/Inference_Foundry')


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def must_fail(fn, label: str):
    try:
        fn()
    except (RuntimeError, ValueError):
        return label
    raise AssertionError(f'{label} failed to reject')


def main():
    os.environ['EXTREME_POSTPARK_CAPTURE_DIR'] = '/tmp/run576-offline-no-output'
    c = load(ROOT/'scripts/loop080_postpark_capture_run576.py', 'collector576')
    p = load(ROOT/'scripts/loop080_postpark_patch_run576.py', 'patcher576')
    manifest, originals, patched = p.prepared()
    assert len(patched) == 8
    assert all(p.MARK in value.decode() for value in patched.values())
    assert all(p.MARK not in value.decode() for value in originals.values())
    for key, value in patched.items():
        compile(value, str(p.SOURCES[key]), 'exec')
    negatives = [
        must_fail(lambda: p.patch('graph', originals['graph'].decode().replace(
            '        entry.aclgraph.replay()\n', '', 1)), 'missing_graph_replay'),
        must_fail(lambda: p.patch('dsa', originals['dsa'].decode().replace(
            '        if self.compress_ratio <= 1:\n            attn_output = attn_op(\n', '', 1)),
                  'missing_dsa_branch'),
        must_fail(lambda: p.patch('serving', originals['serving'].decode().replace(
            '            self._parked[slot] = True\n', '', 1)), 'missing_parking'),
        must_fail(lambda: p.patch('w4a8', patched['w4a8'].decode()), 'already_patched'),
        must_fail(lambda: p.patch('runner', originals['runner'].decode().replace(
            '                ).run()\n', '                )\n', 1)), 'missing_runner_handoff'),
    ]
    class Layer: pass
    x = Layer(); x.layer_name = 'model.layers.42.mlp.experts'
    assert c.identity(x) == (x.layer_name, 42, 'target')
    x.layer_name = 'mtp.2.mlp.experts'
    assert c.identity(x) == (x.layer_name, 45, 'draft')
    x.layer_name = 'model.layers.43.mlp.experts'
    negatives.append(must_fail(lambda: c.identity(x), 'extra_target_layer'))

    state = SimpleNamespace(cycle_index=0)
    runtime = SimpleNamespace(state=state, target_page_audit=None)
    serving = SimpleNamespace(runtime=runtime, max_cycles=1025,
                              _parked=[False]*12,
                              _committed_progress=lambda: [0]*12)
    for index in range(1,5):
        c.bind_cohort(index, tuple(f'r{j}' for j in range(12)),
                      'LOOP080-RUN576-TEST')
        c.cohort_start(runtime)
        state.cycle_index = 188
        serving._parked[4] = True
        c.after_park(serving, [4])
        assert c.SELECTED is None
        serving._parked[4] = False
        state.cycle_index = 0
    c.bind_cohort(5, tuple(f'r{j}' for j in range(12)),
                  'LOOP080-RUN576-TEST')
    c.cohort_start(runtime)
    state.cycle_index = 188
    c.after_park(serving, [])
    assert c.SELECTED is None
    serving._parked = [True]*12
    negatives.append(must_fail(lambda: c.after_park(serving, [4]), 'all_parked'))
    serving._parked = [False]*12
    serving._parked[4] = True
    c.after_park(serving, [4])
    assert c.COHORT == 5 and c.SELECTED == 188 and c.PARK['newly_parked_slots'] == [4]
    c.after_park(serving, [5])
    assert c.SELECTED == 188
    state.cycle_index = 187
    c.before_target(runtime)
    assert c.TRACK_TARGET is False
    state.cycle_index = 188
    c.before_target(runtime)
    assert c.TRACK_TARGET is True
    state.cycle_index = 189
    c.before_target(runtime)
    assert c.TRACK_TARGET is False

    # Execute the actual patched _park_completed method, not merely its AST.
    sys.path.insert(0,str(ROOT))
    import scripts.loop080_postpark_capture_run576 as imported
    tree=ast.parse(patched['serving'].decode())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='FixedCohortServing')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_park_completed')
    namespace={}
    class FakeTensor:
        device='cpu';dtype='int64'
        def __init__(self):self.writes=[]
        def __setitem__(self,index,value):self.writes.append((index,value))
    class FakeTorch:
        long='long'
        @staticmethod
        def tensor(value,**kw):return value
    namespace['torch']=FakeTorch
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
                 '<patched_serving_method>','exec'),namespace)
    called=[];original_hook=imported.after_park
    imported.after_park=lambda serving,slots:called.append((serving,slots))
    try:
        fake_state=SimpleNamespace(num_computed_tokens=FakeTensor(),active_mask=FakeTensor())
        fake_proposer=SimpleNamespace(park_completed_slots=lambda slots,pos:None)
        fake_runtime=SimpleNamespace(state=fake_state,proposer=fake_proposer,
                                     invalidate_scheduled_metadata=lambda:None)
        fake=SimpleNamespace(config=SimpleNamespace(batch_size=12,block_size=16),
                             _parked=[False]*12,_progress_baseline=[0]*12,
                             remaining=[1024]*12,_initial_positions=[32768]*12,
                             runtime=fake_runtime)
        progress=[0]*12;progress[4]=1024
        namespace['_park_completed'](fake,progress)
        assert len(called)==1 and called[0][1]==[4] and fake._parked[4]
        assert len(fake_state.num_computed_tokens.writes)==1
    finally:
        imported.after_park=original_hook
    negatives.append('patched_method_runtime_import_and_call_pass')

    # Temp-copy install/restore: partial restore is retryable, unknown drift
    # aborts before touching any other source. Production paths stay untouched.
    old_sources=p.SOURCES
    try:
        with tempfile.TemporaryDirectory(prefix='run576-patch-') as td:
            root=Path(td)
            p.SOURCES={key:root/f'{key}.py' for key in old_sources}
            for key,path in p.SOURCES.items(): path.write_bytes(originals[key])
            def action(name,state):
                old_argv=sys.argv
                try:
                    sys.argv=['patcher',name,'--state-dir',str(state),
                              '--record',str(root/f'{name}.json'),
                              '--offline-confirmed']
                    p.main()
                finally:
                    sys.argv=old_argv
            state=root/'state';action('install',state)
            assert all(p.SOURCES[k].read_bytes()==patched[k] for k in patched)
            first=next(iter(p.SOURCES))
            p.SOURCES[first].write_bytes(originals[first])
            action('restore',state)
            assert all(p.SOURCES[k].read_bytes()==originals[k] for k in originals)
            negatives.append('partial_restore_retry')
            state2=root/'state2';action('install',state2)
            p.SOURCES[first].write_bytes(b'unexpected drift')
            negatives.append(must_fail(lambda:action('restore',state2),
                                       'unknown_source_drift'))
            assert all(p.SOURCES[k].read_bytes()==patched[k] for k in patched if k!=first)
    finally:
        p.SOURCES=old_sources
    report = dict(status='source_only_pass', files=manifest['files'],
                  helper_sha=manifest['helper_sha'], negative_gates=negatives,
                  selected_cohort=5, selected_cycle=188,
                  claims=['reversible source preview', 'dynamic first post-park selection'],
                  not_claimed=['live readiness', 'all43 semantic row identity',
                               'fixed formal W0', 'numeric Bound', 'TPS'])
    out = ROOT/'evidence/20260928_loop080_bound/run576/preflight.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'files': len(patched),
                      'negative_gates': len(negatives)}))


if __name__ == '__main__':
    main()
