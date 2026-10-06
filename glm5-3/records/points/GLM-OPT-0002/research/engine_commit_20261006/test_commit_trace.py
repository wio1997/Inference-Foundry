"""CPU semantic tests, using the exact installed FutureWrapper and core method.

No GPU emulation or performance claim: test native ordering, failure propagation,
and trace identity only. A real ready/commit/device diagnostic remains separate.
"""
import ast
from collections import deque
from concurrent.futures import Future, InvalidStateError
from contextlib import nullcontext, suppress
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from commit_trace import Trace, install_engine, install_worker

ROOT = Path(__file__).parent


def native_def(filename, name, environment):
    tree = ast.parse((ROOT / filename).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), environment)
    return environment[name]


class Scheduler:
    def __init__(self, counts, grammar=False):
        self.running = []
        self.current_step = 0
        self.counts = deque(counts)
        self.commits = []
        self.grammar = grammar
        self.vllm_config = NS(max_concurrent_batches=3)

    def has_requests(self):
        return bool(self.counts)

    def schedule(self, *args):
        self.current_step += 1
        n = self.counts.popleft()
        return NS(total_num_scheduled_tokens=n, num_scheduled_tokens={"r": n},
                  scheduled_spec_decode_tokens={},
                  pending_structured_output_tokens=self.grammar and self.current_step > 1,
                  batch=self.current_step)

    def get_grammar_bitmask(self, output):
        return output.batch

    def update_from_output(self, output, value):
        self.commits.append(output.batch)
        return {0: value}


class Executor:
    def __init__(self, failure=False):
        self.queue = deque()
        self.rpc = []
        self.failure = failure
        self.latest = None

    def execute_model(self, output, **kwargs):
        self.latest = output
        self.rpc.append(("execute", output.batch))
        def response():
            if self.failure:
                raise ValueError("native execute failed")
            return NS(req_ids=["r"], sampled_token_ids=[])
        return WRAPPER(self.queue, response)

    def sample_tokens(self, *args, **kwargs):
        batch = self.latest.batch
        self.rpc.append(("sample", batch))
        return WRAPPER(self.queue, lambda: None if self.failure else
                       NS(req_ids=["r"], sampled_token_ids=[[batch]]))


ENV = dict(Future=Future, deque=deque, Callable=object, Any=object,
           suppress=suppress, InvalidStateError=InvalidStateError)
# Compile with postponed annotations, retaining installed executable statements.
tree = ast.parse((ROOT / "native_multiproc_executor.py").read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FutureWrapper")
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])), "native_future", "exec"), ENV)
WRAPPER = ENV["FutureWrapper"]
step_tree = ast.parse((ROOT / "v1_engine_core.py").read_text())
step_node = next(n for n in ast.walk(step_tree) if isinstance(n, ast.FunctionDef) and n.name == "step_with_batch_queue")
env = dict(cast=lambda t, v: v, Future=Future, ModelRunnerOutput=object)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), step_node], type_ignores=[])), "native_step", "exec"), env)
STEP = env["step_with_batch_queue"]


def engine(counts, grammar=False, failure=False, cap=3):
    e = NS(scheduler=Scheduler(counts, grammar), model_executor=Executor(failure),
           batch_queue=deque(maxlen=cap), batch_queue_size=cap, is_ec_consumer=True,
           is_pooling_model=False, check_for_draft_tokens=False,
           _should_throttle_prefills=lambda: False,
           log_error_detail=lambda x: nullcontext(),
           capture_iteration_details=lambda x: nullcontext(),
           _process_aborts_queue=lambda: None,
           _attach_iteration_details=lambda *args: None)
    e.step_fn = lambda: STEP(e)
    return e


class Semantics(unittest.TestCase):
    def exercise(self, counts, grammar=False, failure=False, traced=False, cap=3):
        e = engine(counts, grammar, failure, cap)
        with tempfile.TemporaryDirectory() as folder:
            if traced:
                trace = Trace(folder, "core")
                install_engine(e, trace)
            result = []
            error = None
            for _ in range(15):
                if not e.scheduler.has_requests() and not e.batch_queue:
                    break
                try:
                    out, executed = e.step_fn()
                    result.append((bool(out), executed))
                except Exception as exc:
                    error = (type(exc), str(exc))
                    break
            events = []
            if traced:
                import os
                os.close(trace.fd)
                events = [json.loads(l) for p in Path(folder).glob("*.jsonl")
                          for l in p.read_text().splitlines()]
            return result, e.scheduler.commits, e.model_executor.rpc, error, events

    def test_fifo_nonempty_and_empty(self):
        for counts in ([3, 3, 3, 3], [3, 0, 3, 0]):
            a = self.exercise(counts)
            b = self.exercise(counts, traced=True)
            self.assertEqual(a[:4], b[:4])
            self.assertEqual(b[1], [1, 2, 3, 4])
            self.assertEqual([x["batch"] for x in b[4] if x["event"] == "commit.output"], b[1])

    def test_deferred_grammar(self):
        a = self.exercise([3, 3, 3], grammar=True)
        b = self.exercise([3, 3, 3], grammar=True, traced=True)
        self.assertEqual(a[:4], b[:4])

    def test_current_cap2_semantics(self):
        a = self.exercise([3, 0, 3, 0], cap=2)
        b = self.exercise([3, 0, 3, 0], traced=True, cap=2)
        self.assertEqual(a[:4], b[:4])

    def test_execute_failure(self):
        a = self.exercise([3, 3], failure=True)
        b = self.exercise([3, 3], failure=True, traced=True)
        self.assertEqual(a[:4], b[:4])
        self.assertEqual(b[3], (ValueError, "native execute failed"))

    def test_ready_response_does_not_set_done(self):
        q = deque()
        f = WRAPPER(q, lambda: "already reply-ready")
        self.assertFalse(f.done())
        self.assertEqual(f.result(), "already reply-ready")
        self.assertTrue(f.done())

    def test_worker_async_fifo_keeps_original_consumer_and_batch_labels(self):
        import os
        import queue
        from enum import Enum
        from types import ModuleType
        class Output:
            def __init__(self, value):
                self.value, self.calls = value, 0
            def get_output(self):
                self.calls += 1
                return self.value
        class Proc:
            class ResponseStatus(Enum):
                SUCCESS = 1
                FAILURE = 2
        env = dict(Any=object, AsyncModelRunnerOutput=Output, WorkerProc=Proc,
                   logger=NS(exception=lambda *a: None))
        for name in ("enqueue_output", "handle_output"):
            setattr(Proc, name, native_def("native_multiproc_executor.py", name, env))
        module = ModuleType("vllm.v1.executor.multiproc_executor")
        module.WorkerProc = Proc
        sent = []
        proc = Proc()
        proc.use_async_scheduling = True
        proc.async_output_queue = queue.Queue()
        proc.worker_response_mq = NS(enqueue=sent.append)
        worker = NS(rank=8, model_runner=NS(postprocess_sampled=lambda: None,
                                           speculator=None, pp_handler=None),
                    execute_model=lambda output: None, sample_tokens=lambda: Output("native"))
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", {module.__name__: module}):
            trace = Trace(folder, "rank8")
            install_worker(worker, trace)
            returned = []
            for batch in (1, 2):
                self.assertIsNone(worker.execute_model(NS(num_scheduled_tokens={"r": 3})))
                output = worker.sample_tokens()
                returned.append(output)
                proc.handle_output(output)
            # Consume both replies only after the worker has executed batch2.
            while not proc.async_output_queue.empty():
                proc.enqueue_output(proc.async_output_queue.get())
            os.close(trace.fd)
            events = [json.loads(line) for p in Path(folder).glob("*.jsonl") for line in p.read_text().splitlines()]
        self.assertEqual([x.calls for x in returned], [1, 1])
        self.assertEqual(sent, [(Proc.ResponseStatus.SUCCESS, "native")] * 2)
        self.assertEqual([e["batch"] for e in events if e["event"] == "reply.published_upper_bound"], [1, 2])
        self.assertEqual([e["batch"] for e in events if e["event"] == "get_output.begin"], [1, 2])


if __name__ == "__main__":
    unittest.main()
