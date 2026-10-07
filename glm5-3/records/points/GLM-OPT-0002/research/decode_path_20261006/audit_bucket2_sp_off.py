"""Actual dispatcher and dummy scalar AST, with the observed SP-off contract."""
from pathlib import Path
import ast
import dataclasses
import enum
import hashlib
import itertools
import json
import logging
from types import SimpleNamespace as S

HERE = Path(__file__).resolve().parent
sources = HERE / "graph_sources"
ns = dict(enum=enum, replace=dataclasses.replace, product=itertools.product,
          logger=logging.getLogger("actual-SP-off"), VllmConfig=object)


def execute(path, symbols):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in symbols]
    assert {node.name for node in nodes} == set(symbols)
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)


execute(sources / "vllm__vllm__config__compilation.py", ["CUDAGraphMode"])
descriptors = json.loads((sources / "descriptor_source_snippets.json").read_text())
exec("from __future__ import annotations\n" + descriptors[0]["snippets"][0]["text"], ns)
ns["BatchDescriptor"] = dataclasses.dataclass(frozen=True)(ns["BatchDescriptor"])
execute(sources / "vllm__vllm__v1__cudagraph_dispatcher.py", ["CudagraphDispatcher"])
E, Dispatcher = ns["CUDAGraphMode"], ns["CudagraphDispatcher"]
config = S(compilation_config=S(cudagraph_capture_sizes=[2], max_cudagraph_capture_size=2,
    compile_sizes=[], cudagraph_specialize_lora=False), scheduler_config=S(max_num_seqs=4),
    lora_config=None, num_speculative_tokens=1, parallel_config=S(tensor_parallel_size=16),
    speculative_config=S(num_speculative_tokens=1))
dispatcher = Dispatcher.__new__(Dispatcher)
dispatcher.vllm_config = config
dispatcher.compilation_config = config.compilation_config
dispatcher.uniform_decode_query_len = 2
dispatcher.specialize_lora_count = False
dispatcher.cudagraph_keys = {E.PIECEWISE: set(), E.FULL: set()}
dispatcher.keys_initialized = False
dispatcher.cudagraph_mode = E.NONE
dispatcher.initialize_cudagraph_keys(E.FULL_DECODE_ONLY, 2)
runtime, descriptor = dispatcher.dispatch(2, uniform_decode=True)
assert runtime == E.FULL and descriptor.num_tokens == 2 and descriptor.num_reqs == 1
outside = []
for tokens, uniform in [(1, False), (4, True), (6, True), (8, True), (58, False)]:
    mode, desc = dispatcher.dispatch(tokens, uniform_decode=uniform)
    assert mode == E.NONE
    outside.append(dict(tokens=tokens, uniform=uniform, mode=mode.name))

runner = HERE / "vllm-ascend__vllm_ascend__worker__model_runner_v1.py"
source = runner.read_text()
tree = ast.parse(source)
dummy = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_dummy_run")
branch = next(node for node in dummy.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "create_mixed_batch").orelse[0]
checks = [node for node in dummy.body if isinstance(node, ast.Assert) and
          ("sum(num_scheduled_tokens_list)" in ast.get_source_segment(source, node) or
           "len(num_scheduled_tokens_list)" in ast.get_source_segment(source, node))]
dummy_ns = dict(uniform_decode=True, max_num_reqs=4, num_tokens=2, max_query_len=2,
                cdiv=lambda a, b: (a + b - 1) // b)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.If(test=branch.test, body=branch.body, orelse=[]), *checks], type_ignores=[])), str(runner), "exec"), dummy_ns)
assert dummy_ns["num_scheduled_tokens_list"] == [2] and dummy_ns["num_reqs"] == 1

files = [sources / "vllm__vllm__config__compilation.py", sources / "descriptor_source_snippets.json",
         sources / "vllm__vllm__v1__cudagraph_dispatcher.py", runner]
result = dict(CPU_only=True, NPU_initialized=False, model_or_HCCL_capture_test=False,
              contract=dict(TP=16, max_num_seqs=4, MTP_K=1, SP=False, DSACP=False, shared_expert_DP=False),
              FULL_key_count=len(dispatcher.cudagraph_keys[E.FULL]), runtime=runtime.name,
              descriptor=dataclasses.asdict(descriptor), dummy_queries=dummy_ns["num_scheduled_tokens_list"],
              eager_fallback_cases=outside,
              source_identity={str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
              limits=["SP-off observed from actual original startup, not inferred from FlashComm flag.",
                      "Bucket2 only covers one uniform request; no 1-4 concurrency acceptance or device graph gain claimed.",
                      "Actual platform TP filter guard is false when SP/DSACP/shared-DP are off; this scalar audit does not import platform/NPU."])
(HERE / "bucket2_sp_off_CPU.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in ("FULL_key_count", "runtime", "descriptor", "dummy_queries", "eager_fallback_cases")}))
