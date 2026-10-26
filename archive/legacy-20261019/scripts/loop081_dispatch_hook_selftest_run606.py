#!/usr/bin/env python3
"""Execute patched Run606 hook AST against ordinary/shim/Graph CPU mocks."""
import ast
import contextvars
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import loop081_dispatch_patch_run606 as patch


def function(src, cls, name):
    tree = ast.parse(src)
    scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in scope.body if isinstance(n, ast.FunctionDef) and n.name == name)


def top_level_slice(fn, first_name, count):
    at = next(i for i, n in enumerate(fn.body)
              if isinstance(n, ast.Assign) and
              any(isinstance(t, ast.Name) and t.id == first_name for t in n.targets))
    return fn.body[at:at + count]


def run_stmts(stmts, env):
    code = compile(ast.fix_missing_locations(ast.Module(body=stmts, type_ignores=[])),
                   '<patched-hook>', 'exec')
    exec(code, env)


class Event:
    allocations = 0

    def __init__(self, **kwargs):
        Event.allocations += 1
        self.recorded = False

    def record(self):
        self.recorded = True


class Stream:
    npu_stream = 1
    syncs = 0

    def synchronize(self):
        self.syncs += 1


def main():
    _, _, edited = patch.prepared()
    dspark = function(edited['draft'].decode(), 'AscendDSparkProposer', 'set_inputs_first_pass')
    dflash = function(edited['dflash'].decode(), 'AscendDflashProposer', 'build_model_inputs_first_pass')
    graph = function(edited['acl_graph'].decode(), 'ACLGraphWrapper', '__call__')
    ds_hook = top_level_slice(dspark, '_p606_dir', 3)
    df_hook = top_level_slice(dflash, '_p606_dir', 4)
    graph_entry = top_level_slice(graph, '_p606_parent', 3)
    stream = Stream()
    torch = SimpleNamespace(npu=SimpleNamespace(Event=Event, current_stream=lambda: stream))
    base = dict(os=os, time=time, torch=torch)
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, 'arm').touch()
        old = os.environ.get('EXTREME_RUN606_DIR')
        os.environ['EXTREME_RUN606_DIR'] = tmp
        try:
            shim = SimpleNamespace(input_batch=SimpleNamespace())
            ordinary = SimpleNamespace(input_batch=SimpleNamespace(req_ids=['r1']),
                                       _p602_marks=[1])
            owner_ctx = contextvars.ContextVar('test-run606-owner', default=None)
            fake_dspark = ModuleType('vllm_ascend.spec_decode.dspark_proposer')
            fake_dspark._p606_ordinary_owner = owner_ctx
            names = ('vllm_ascend', 'vllm_ascend.spec_decode',
                     'vllm_ascend.spec_decode.dspark_proposer')
            original_modules = {name: sys.modules.get(name) for name in names}
            for name in names[:-1]:
                package = ModuleType(name)
                package.__path__ = []
                sys.modules[name] = package
            sys.modules[names[-1]] = fake_dspark
            for bound, expected in ((None, 0), (ordinary, 1)):
                proposer = SimpleNamespace(runner=shim, _dflash_num_context=8,
                                        num_query_per_req=8, sample_from_anchor=False,
                                        draft_attn_groups=[1], use_cuda_graph=False)
                token = owner_ctx.set(bound)
                try:
                    env = dict(base, self=proposer, _p606_ordinary_owner=owner_ctx,
                               batch_size=1, num_query_total=8,
                               num_sample_total=7, has_num_rejected=False)
                    run_stmts(ds_hook, env)
                    assert len(getattr(ordinary, '_p606_draft_records', [])) == expected
                    before = Event.allocations
                    env = dict(base, self=proposer, num_context=8, _context_slots=[1])
                    run_stmts(df_hook, env)
                    assert len(getattr(ordinary, '_p606_context_events', [])) == expected
                    assert Event.allocations - before == expected
                finally:
                    owner_ctx.reset(token)
            assert owner_ctx.get() is None
            for name, original in original_modules.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
            mode = SimpleNamespace(FULL=SimpleNamespace(name='FULL'),
                                   FULL_DECODE_ONLY=SimpleNamespace(name='FULL_DECODE_ONLY'))
            var = contextvars.ContextVar('test-run606', default=None)
            records = []
            token = var.set({'records': records, 'ordinal': 1})
            try:
                wrapper = SimpleNamespace(runtime_mode=mode.FULL, runnable=lambda: None,
                                          enable_enpu=False, use_eagle=False)
                for incoming in (mode.FULL, SimpleNamespace(name='NONE')):
                    env = dict(base, self=wrapper, _p606_forward_ctx=var,
                               CUDAGraphMode=mode,
                               forward_context=SimpleNamespace(cudagraph_runtime_mode=incoming),
                               get_forward_context=lambda incoming=incoming:
                               SimpleNamespace(cudagraph_runtime_mode=incoming),
                               _EXTRA_CTX=SimpleNamespace(is_draft_model=False))
                    run_stmts(graph_entry, env)
                assert [r['incoming_mode'] for r in records] == ['FULL', 'NONE']
            finally:
                var.reset(token)
            assert var.get() is None
        finally:
            if old is None:
                os.environ.pop('EXTREME_RUN606_DIR', None)
            else:
                os.environ['EXTREME_RUN606_DIR'] = old
    assert Event.allocations == 1
    out = {'status': 'pass', 'ordinary_draft_records': 1,
           'shim_draft_records': 0, 'ordinary_context_events': 1,
           'shim_context_events': 0, 'graph_entry_modes': ['FULL', 'NONE'],
           'contextvar_reset': True}
    print(json.dumps(out))


if __name__ == '__main__':
    main()
