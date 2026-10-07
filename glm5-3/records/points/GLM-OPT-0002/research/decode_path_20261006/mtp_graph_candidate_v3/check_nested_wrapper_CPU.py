"""Run actual load-order/wrapping and Breakable dispatch AST, without devices.

The original scoped composition must attempt forbidden nested capture; the
patch must retain one outer draft graph and target/other-drafter behavior.
"""
import ast
import __future__
import itertools
import json
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / 'original_model_runner_v1.py'
PATCHED = HERE / 'model_runner_v1.py'
BREAKABLE = HERE.parent / 'stream_ordered_replay/original/breakable_cudagraph.py'
MODES = NS(NONE='NONE', FULL='FULL')
CAPTURING = False


def load_nodes(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NPUModelRunner')
    load = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'load_model')
    draft_call = next(n for n in ast.walk(load) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'load_model' and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == 'drafter')
    branch = next(n for n in ast.walk(load) if isinstance(n, ast.If) and any(isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == 'BreakableACLGraphWrapper' for v in ast.walk(n)) and isinstance(n.test, ast.BoolOp))
    assert draft_call.lineno < branch.lineno
    return draft_call, branch


class Model:
    def __init__(self, label):
        self.label, self.calls = label, 0

    def __call__(self, **kwargs):
        self.calls += 1
        return self.label


class Wrapper:
    def __init__(self, runnable, *args, **kwargs):
        self.runnable, self.entries = runnable, {}

    def _capture(self, entry, args, kwargs):
        # Execute the real pre-capture cleanup statement that failed on Run271.
        tree = ast.parse(BREAKABLE.read_text())
        cleanup = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'empty_cache')
        def empty_cache():
            if CAPTURING:
                raise RuntimeError('nested cleanup while outer graph captures')
        exec(compile(ast.Module(body=[cleanup], type_ignores=[]), str(BREAKABLE), 'exec'), dict(torch=NS(accelerator=NS(empty_cache=empty_cache))))
        return self.runnable(*args, **kwargs)


def main():
    global CAPTURING
    context = NS(batch_descriptor=object(), cudagraph_runtime_mode='FULL')
    tree = ast.parse(BREAKABLE.read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '__call__' and n.lineno > 300)
    env = dict(is_forward_context_available=lambda: True, get_forward_context=lambda: context,
               CUDAGraphMode=MODES, _BreakableEntry=lambda **kwargs: NS(capture=None, **kwargs))
    exec(compile(ast.Module(body=[call], type_ignores=[]), str(BREAKABLE), 'exec', flags=__future__.annotations.compiler_flag), env)
    Wrapper.__call__ = env['__call__']
    originals, patched = load_nodes(ORIGINAL), load_nodes(PATCHED)

    order_cases = 0
    for source, nodes in ((ORIGINAL, originals), (PATCHED, patched)):
        target, draft = Model('target'), Model('draft')
        drafter = NS(_glm_k1_graph=True)
        def draft_load(actual_target):
            assert actual_target is target and not isinstance(actual_target, Wrapper)
            drafter.model = draft
            drafter._glm_k1_acl = object()  # outer ACL is established by load
        drafter.load_model = draft_load
        runner = NS(model=target, drafter=drafter, vllm_config=object(), use_eagle=True, enable_enpu=False)
        scope = dict(self=runner, cudagraph_mode='FULL', CUDAGraphMode=MODES,
                     breakable_cudagraph=NS(is_breakable_cudagraph_enabled=lambda: True),
                     BreakableACLGraphWrapper=Wrapper, ACLGraphWrapper=Wrapper)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=nodes[0]), nodes[1]], type_ignores=[])), str(source), 'exec'), scope)
        assert isinstance(runner.model, Wrapper) and runner.model.runnable is target
        CAPTURING = True
        if source == ORIGINAL:
            assert isinstance(drafter.model, Wrapper)
            try:
                drafter.model()
            except RuntimeError as error:
                assert 'nested cleanup' in str(error)
            else:
                raise AssertionError('original must reproduce nested cleanup')
        else:
            assert drafter.model is draft and drafter.model() == 'draft'
            assert draft.calls == 1
        CAPTURING = False
        context.cudagraph_runtime_mode = 'NONE'
        assert drafter.model() == 'draft'
        context.cudagraph_runtime_mode = 'FULL'
        order_cases += 1

    scope_cases = 0
    for enabled, graph_mode, flag_kind, has_model in itertools.product((False, True), ('NONE', 'FULL'), ('absent', 'false', 'true'), (False, True)):
        target, raw = Model('target'), Model('draft')
        drafter = NS()
        if flag_kind != 'absent':
            drafter._glm_k1_graph = flag_kind == 'true'
        if has_model:
            drafter.model = raw
        runner = NS(model=target, drafter=drafter, vllm_config=object(), use_eagle=True, enable_enpu=False)
        class Mode(str):
            def has_full_cudagraphs(self):
                return self == 'FULL'
        scope = dict(self=runner, cudagraph_mode=Mode(graph_mode), CUDAGraphMode=MODES,
                     breakable_cudagraph=NS(is_breakable_cudagraph_enabled=lambda: enabled),
                     BreakableACLGraphWrapper=Wrapper, ACLGraphWrapper=Wrapper)
        exec(compile(ast.Module(body=[patched[1]], type_ignores=[]), str(PATCHED), 'exec'), scope)
        assert isinstance(runner.model, Wrapper) == (graph_mode == 'FULL')
        if has_model:
            assert isinstance(drafter.model, Wrapper) == (enabled and graph_mode == 'FULL' and flag_kind != 'true')
        scope_cases += 1
    row = dict(passed=True, CPU_only=True, model_requests=0, actual_load_order_cases=order_cases,
               wrapping_scope_cases=scope_cases, actual_Breakable_dispatch_and_cleanup_AST=True,
               original_nested_capture_reproduced=True, single_outer_draft_graph=True,
               target_wrapper_retained=True, unsupported_drafters_unchanged=True, NONE_calls_same_raw_model=True,
               runner_sha256=__import__('hashlib').sha256(PATCHED.read_bytes()).hexdigest(),
               limitations='Model/graph/allocator are doubles. Real AST reproduces host composition defect; no device capture/replay, numerical or performance claim.')
    (HERE/'nested_wrapper_CPU_result.json').write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps(row))


if __name__ == '__main__':
    main()
