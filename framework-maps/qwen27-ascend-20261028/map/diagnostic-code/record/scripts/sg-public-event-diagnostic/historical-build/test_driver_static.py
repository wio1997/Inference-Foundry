"""CPU-only structural check; does not import the candidate or torch."""
import ast
from pathlib import Path
p = Path(__file__).with_name("task_anchor_driver.py")
s = p.read_text()
tree = ast.parse(s)
# The module-level imports are standard-library only: torch imports stay in actions.
module_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
assert all((isinstance(n, ast.ImportFrom) and n.module in {"__future__", "pathlib"}) or
           (isinstance(n, ast.Import) and all(a.name.split(".")[0] in {
               "argparse", "hashlib", "importlib", "json", "os", "pathlib", "sys", "time", "traceback"
           } for a in n.names)) for n in module_imports)
assert "build-only" in s and "run-toy" in s
assert "with_cuda=False" in s and 'os.environ["MAX_JOBS"] = "1"' in s
assert "EXPECTED_CPP_SHA256" in s and "EXPECTED_CANN_SHA256" in s and "EXPECTED_TORCH_NPU_GIT" in s
assert '"Q27_EXPECT_TASKQUEUE_ENABLED": "1"' in s
assert '"TASK_QUEUE_ENABLE": "1"' in s
assert '"Q27_EXPECT_MODEL_QUEUES_EMPTY": "1"' in s
assert "2 * 1024**3" in s
assert "torch.npu.Stream(device=DEVICE)" in s
assert "with torch.npu.graph(graph, stream=stream" in s
assert 'anchors.append(extension.anchor("after_add"))' in s
assert 'anchors.append(extension.anchor("after_mul"))' in s
assert "stream.synchronize()" in s
assert "graph.debug_dump" in s and '"raw_dump_discarded": False' in s
assert '"raw_graph_debug.json"' in s and "raw_graph_native_ids.json" in s
assert "analyse_flag=False" in s and "native_analysis_run" in s
assert 'record_function(f"Q27_GRAPH_REPLAY_SUBMIT_{index+1}")' in s
assert '"numerical_sanity"' in s and "torch.equal(z, expected)" in s
assert "callback_thread_differs_from_producer" in s
print("CPU_STATIC_OK: gated build/run driver structure; no framework import, compile, or NPU action")
