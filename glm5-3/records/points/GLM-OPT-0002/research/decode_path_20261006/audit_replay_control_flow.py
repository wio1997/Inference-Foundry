"""Execute installed-source control flow with CPU doubles; never import NPU."""
from pathlib import Path
import ast
import contextlib
import dataclasses
import functools
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sources = {
    "wrapper": HERE / "graph_sources/vllm-ascend__vllm_ascend__compilation__acl_graph.py",
    "decorator": HERE / "eager_sources/vllm__vllm__compilation__breakable_cudagraph.py",
    "runner": HERE / "vllm-ascend__vllm_ascend__worker__model_runner_v1.py",
    "proposer": HERE / "graph_sources/vllm-ascend__vllm_ascend__spec_decode__llm_base_proposer.py",
    "SFA": HERE / "vllm-ascend__vllm_ascend__attention__sfa_v1.py",
    "DCP": HERE / "vllm-ascend__vllm_ascend__attention__context_parallel__sfa_cp.py",
}
trees = {name: ast.parse(path.read_text()) for name, path in sources.items()}


def function(name, symbol, owner=None):
    tree = trees[name]
    if owner:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == symbol)
    # Decorators on extracted methods are not part of their call body.
    node.decorator_list = []
    return node


def execute(nodes, namespace):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<installed-source-CPU-audit>", "exec"), namespace)


class Mode:
    NONE, FULL, PIECEWISE = range(3)


events = []
current = SimpleNamespace(batch_descriptor="capacity2", cudagraph_runtime_mode=Mode.FULL, capturing=False)


class Graph:
    def replay(self):
        events.append("graph.replay")


@contextlib.contextmanager
def capture(graph, pool):
    events.append("capture.enter")
    yield
    events.append("capture.exit")


ns = dict(get_forward_context=lambda: current, CUDAGraphMode=Mode,
          ACLGraphEntry=lambda **kwargs: SimpleNamespace(**kwargs, aclgraph=None),
          torch=SimpleNamespace(Tensor=type("Tensor", (), {}), npu=SimpleNamespace(
              NPUGraph=Graph, graph=capture,
              current_stream=lambda: SimpleNamespace(synchronize=lambda: events.append("stream.synchronize")))),
          ExitStack=contextlib.ExitStack, patch=patch,
          validate_cudagraph_capturing_enabled=lambda: events.append("validate.capture"),
          weak_ref_tensors=lambda value: value, weak_ref_workspaces=lambda value: None,
          _graph_params=None, _draft_graph_params=None, _draft_graph_prefill_params=None,
          compilation_counter=SimpleNamespace(num_cudagraph_captured=0),
          logger=SimpleNamespace(info_once=lambda *args: None), _EXTRA_CTX=SimpleNamespace(is_draft_model=False))
execute([function("wrapper", "__call__", "ACLGraphWrapper")], ns)

# The only module import executed by __call__ is the offloader; intercept it.
import builtins
old_import = builtins.__import__
offloader = SimpleNamespace(sync_prev_onload=lambda: events.append("offloader.sync"),
                            join_after_forward=lambda: events.append("offloader.join"))


def cpu_import(name, *args, **kwargs):
    if name == "vllm.model_executor.offloader.base":
        return SimpleNamespace(get_offloader=lambda: offloader)
    return old_import(name, *args, **kwargs)


def model(**kwargs):
    events.append("model.python")
    return "static-output"


wrapper = SimpleNamespace(runtime_mode=Mode.FULL, runnable=model, concrete_aclgraph_entries={},
                          aclgraph_options=SimpleNamespace(debug_log_enable=False, gc_disable=False, weak_ref_output=False),
                          graph_pool="pool", is_debugging_mode=False, enable_enpu=False, use_eagle=True)
with patch("builtins.__import__", cpu_import):
    assert ns["__call__"](wrapper, input_ids="buffer") == "static-output"
    capture_events = list(events)
    events.clear()
    current = SimpleNamespace(batch_descriptor="capacity2", cudagraph_runtime_mode=Mode.FULL, capturing=False)
    assert ns["__call__"](wrapper, input_ids="buffer") == "static-output"
    replay_events = list(events)
    assert replay_events == ["stream.synchronize", "graph.replay"]
    events.clear()
    current.cudagraph_runtime_mode = Mode.NONE
    assert ns["__call__"](wrapper, input_ids="buffer") == "static-output"
    none_events = list(events)
    assert none_events == ["model.python"]

break_capture = SimpleNamespace(_capturing=True, add_eager=lambda fn: (events.append("add_eager"), fn())[1])
decorator_ns = dict(functools=functools, is_breakable_cudagraph_enabled=lambda: True,
                    BreakableCUDAGraphCapture=SimpleNamespace(current=lambda: break_capture),
                    is_forward_context_available=lambda: True, get_forward_context=lambda: current,
                    CUDAGraphMode=Mode, torch=ns["torch"], weak_ref_tensor=lambda x: x)
execute([function("decorator", "eager_break_during_capture")], decorator_ns)
wrapped = decorator_ns["eager_break_during_capture"](lambda: events.append("MLA.python"))
events.clear()
current.cudagraph_runtime_mode = Mode.FULL
wrapped()
full_decorator_events = list(events)
assert full_decorator_events == ["MLA.python"]
events.clear()
current.cudagraph_runtime_mode = Mode.PIECEWISE
wrapped()
piecewise_decorator_events = list(events)
assert piecewise_decorator_events == ["add_eager", "MLA.python"]

# Actual runner update guard: sparse/compressed attention has no per-replay
# update_full_graph_params call. Its metadata tensors are supplied outside.
runner_ns = dict(get_forward_context=lambda: current, CUDAGraphMode=Mode,
                 partial=functools.partial, torch=ns["torch"],
                 update_full_graph_params=lambda *args: events.append("attention.update"))
execute([function("runner", "_update_full_graph_params_if_needed", "NPUModelRunner"),
         function("runner", "_model_forward", "NPUModelRunner")], runner_ns)
runner = SimpleNamespace(model=lambda **kwargs: events.append("model.call"), enable_enpu=False,
                         use_sparse=True, use_compress=False)
runner._update_full_graph_params_if_needed = functools.partial(runner_ns["_update_full_graph_params_if_needed"], runner)
events.clear()
current.cudagraph_runtime_mode = Mode.FULL
current.capturing = False
runner_ns["_model_forward"](runner, 2, input_ids="buffer")
sparse_runner_events = list(events)
assert sparse_runner_events == ["model.call"]

result = dict(CPU_only=True, imports_NPU=False, device_capture_test=False,
              source_identity={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for name, path in sources.items()},
              actual_AST_cases=dict(first_FULL_call=capture_events, repeated_FULL_call=replay_events,
                                    NONE_call=none_events, FULL_eager_break=full_decorator_events,
                                    PIECEWISE_eager_break=piecewise_decorator_events,
                                    sparse_runner_forward=sparse_runner_events),
              conclusions=[
                  "FULL replay bypasses target Python runnable including MLA/MoE/inner collectives; MLA eager-break does not add an eager segment in FULL capture.",
                  "Scheduler, input preprocessing and metadata construction, outer KV connector context, logits/sampler and GLM eager MTP remain outside target FULL wrapper.",
                  "Sparse SFA update_graph_params is a no-op; DCP builder writes persistent replicated block/slot/local seq-lens buffers before model forward. This is not proof of unsupported replay.",
                  "Actual GLM proposer sets use_cuda_graph=False. Do not enable its graph by deleting the safety branch without dynamic-input and MTP correctness evidence.",
                  "Wrapper stream synchronize orders previous replay versus next metadata update. No source-only deletion candidate established.",
              ],
              limits=["CPU doubles establish control flow only, not NPU/HCCL capture support, dynamic same-address validity, EOS/PD correctness, or performance.",
                      "Eager Run249 contains no graph replay; graph configuration gains cannot be counted as a code patch gain.",
                      "No configuration, model service, native source or kernel changed by this audit."])
(HERE / "replay_control_flow_CPU.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result["actual_AST_cases"]))
