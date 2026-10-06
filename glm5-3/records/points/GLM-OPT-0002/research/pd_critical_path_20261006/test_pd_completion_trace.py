import json
from collections import OrderedDict
import logging
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from pd_completion_trace import Hooks, Trace, install_core, install_worker
from reduce_pd_completion_trace import reduce_events


class MemoryTrace:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append(dict(event=event, **fields))


class KVOutput(types.SimpleNamespace):
    def __init__(self, **kwargs):
        defaults = dict(finished_sending=None, finished_recving=None,
                        invalid_block_ids=set(), expected_finished_count=0,
                        kv_connector_stats=None, kv_connector_worker_meta=None,
                        kv_cache_events=None)
        super().__init__(**dict(defaults, **kwargs))


def kv(ids):
    return KVOutput(finished_recving=set(ids))


# Execute the exact captured native class, replacing only output containers.
# This is CPU-only validation of observers, not a model/rank-device experiment.
native = json.loads((Path(__file__).parent / "native_aggregator_class.json").read_text())
namespace = dict(KVConnectorOutput=KVOutput)
exec("from __future__ import annotations\n" + native["source"], namespace)
Aggregator = namespace["KVOutputAggregator"]
tracker_source = json.loads((Path(__file__).parent / "native_tracker_class.json").read_text())
tracker_namespace = dict(OrderedDict=OrderedDict, threading=__import__("threading"),
                         time=__import__("time"), logger=logging.getLogger(__name__),
                         envs=types.SimpleNamespace(VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=60))
exec("from __future__ import annotations\n" + tracker_source["source"], tracker_namespace)
Tracker = tracker_namespace["KVCacheTaskTracker"]


class Scheduler:
    running = []
    max_num_running_reqs = 8
    max_num_scheduled_tokens = 128
    requests = {}

    def schedule(self, *args, **kwargs):
        return types.SimpleNamespace(num_scheduled_tokens={"r": 1}, total_num_scheduled_tokens=1)

    def _update_from_kv_xfer_finished(self, output):
        return None

    def _try_promote_blocked_waiting_request(self, request):
        return True

    def update_from_output(self, scheduled, model_output):
        return {}


class ObservationTests(unittest.TestCase):
    def test_worker_consumes_native_tracker_once(self):
        class Receiver:
            def __init__(self):
                self.task_tracker = Tracker()
                self.local_engine_id = "D"
                self.tp_rank = 0
            def _transfer_kv_cache_all_groups(self, meta):
                return None
            def _reformat_pending_kv_caches(self, request_id):
                return None
            def _mark_failed_recv_request(self, request_id, blocks):
                return None
        class Worker:
            tp_rank = 0
            def __init__(self):
                self.kv_recv_thread = Receiver()
                self.get_finished_calls = 0
            def start_load_kv(self, metadata):
                for request_id in metadata.requests:
                    self.kv_recv_thread.task_tracker.add_req_to_process(request_id)
            def get_finished(self):
                self.get_finished_calls += 1
                return set(), self.kv_recv_thread.task_tracker.get_and_clear_finished_requests()
        module = types.SimpleNamespace(MooncakeConnectorWorker=Worker,
                                       KVCacheTaskTracker=Tracker,
                                       KVCacheRecvingThread=Receiver)
        trace = MemoryTrace()
        hooks = install_worker(module, trace)
        try:
            worker = Worker()
            worker.start_load_kv(types.SimpleNamespace(requests={"r": object()}))
            worker.kv_recv_thread.task_tracker.update_done_task_count("r")
            value = worker.get_finished()
            self.assertEqual(value, (set(), {"r"}))
            self.assertEqual(worker.get_finished_calls, 1)
            self.assertEqual(worker.kv_recv_thread.task_tracker.get_and_clear_finished_requests(), set())
            self.assertEqual(next(e for e in trace.events if e["event"]=="get_finished.end")["recv_ids"], ["r"])
        finally:
            hooks.uninstall()

    def test_once_identity_exception_and_uninstall(self):
        class Native:
            def call(self, value):
                self.calls += 1
                if isinstance(value, BaseException):
                    raise value
                return value
        obj = Native()
        obj.calls = 0
        hooks = Hooks(MemoryTrace())
        hooks.wrap(obj, "call", before=lambda value: {"value": 1 / 0})
        value = object()
        self.assertIs(obj.call(value), value)
        error = RuntimeError("native")
        with self.assertRaises(RuntimeError) as caught:
            obj.call(error)
        self.assertIs(caught.exception, error)
        self.assertEqual(obj.calls, 2)
        hooks.uninstall()
        self.assertNotIn("call", vars(obj))

    def test_aggregator_capture_precedes_native_input_mutation(self):
        trace = MemoryTrace()
        engine = types.SimpleNamespace(scheduler=Scheduler(), batch_queue=None)
        original = Aggregator.aggregate
        hooks = install_core(engine, Aggregator, trace)
        try:
            agg = Aggregator(2)
            outputs = [types.SimpleNamespace(kv_connector_output=kv(["r"])),
                       types.SimpleNamespace(kv_connector_output=kv([]))]
            result = agg.aggregate(outputs)
            self.assertIs(result, outputs[0])
            first = next(e for e in trace.events if e["event"] == "aggregate.begin")
            self.assertEqual(first["contributors"][0]["recv_ids"], ["r"])
            self.assertIsNone(outputs[0].kv_connector_output.finished_recving)
            outputs = [types.SimpleNamespace(kv_connector_output=kv([])),
                       types.SimpleNamespace(kv_connector_output=kv(["r"]))]
            self.assertEqual(agg.aggregate(outputs).kv_connector_output.finished_recving, {"r"})
            self.assertEqual(agg._recv_remaining_count, {})
        finally:
            hooks.uninstall()
        self.assertIs(Aggregator.aggregate, original)

    def test_bound_and_io_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Trace(directory, host="D", boot_id="b", role="D", limit=1)
            trace.emit("one")
            trace.emit("two")
            trace.emit("three")
            trace.close()
            rows = [json.loads(x) for x in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
            self.assertEqual([x["event"] for x in rows], ["one", "trace.truncated"])
        with tempfile.TemporaryDirectory() as directory:
            trace = Trace(directory, host="D", boot_id="b", role="D")
            with patch("pd_completion_trace.os.write", side_effect=OSError("full")):
                trace.emit("one")
            self.assertTrue(trace.failed)
            trace.close()


def fixture():
    def e(name, t, **fields):
        return dict(event=name, monotonic_ns=t, host="D", boot_id="boot", **fields)
    return [e("get_finished.end", 10, contributor=0, recv_ids=["r"]),
            e("get_finished.end", 20, contributor=1, recv_ids=["r"]),
            e("aggregate.end", 30, expected=2, recv_ids=["r"]),
            e("_update_from_kv_xfer_finished.end", 40, recv_ids=["r"]),
            e("_try_promote_blocked_waiting_request.end", 50, request_id="r", promoted=True),
            e("schedule.end", 60, tokens={"r": 1})]


class ReductionTests(unittest.TestCase):
    manifest = dict(d_host="D", d_boot_id="boot", required_contributors=[0, 1],
                    request_ids=["r", "failed"], simulation=True)

    def result(self, events):
        return reduce_events(events, self.manifest)["requests"][0]

    def test_all_rank_host_envelope_is_not_removable_time(self):
        out = reduce_events(fixture(), self.manifest)
        self.assertTrue(out["requests"][0]["host_chain_complete"])
        self.assertFalse(out["requests"][1]["host_chain_complete"])
        self.assertEqual(len(out["requests"]), 2)
        self.assertIsNone(out["requests"][0]["removable_ms"])
        self.assertIsNone(out["requests"][0]["device_ready"])
        self.assertEqual(out["verdict"], "INCONCLUSIVE")

    def test_missing_rank_or_failed_transfer_does_not_become_ready(self):
        self.assertFalse(self.result(fixture()[1:])["host_chain_complete"])
        rows = fixture() + [dict(event="_mark_failed_recv_request.begin", monotonic_ns=5,
                                 host="D", boot_id="boot", request_id="r")]
        out = self.result(rows)
        self.assertFalse(out["host_chain_complete"])
        self.assertIsNone(out["capture_to_aggregate_ms"])

    def test_clock_domain_loss_duplicate_and_order(self):
        rows = fixture()
        rows[1] = dict(rows[1], host="P")
        self.assertFalse(self.result(rows)["host_chain_complete"])
        rows = fixture() + [dict(event="trace.truncated", monotonic_ns=70, host="D", boot_id="boot")]
        self.assertFalse(self.result(rows)["host_chain_complete"])
        self.assertFalse(self.result(fixture() + fixture()[:1])["host_chain_complete"])
        rows = fixture()
        rows[-1]["monotonic_ns"] = 45
        self.assertFalse(self.result(rows)["host_chain_complete"])

    def test_expected_count_mismatch(self):
        rows = fixture()
        rows[2]["expected"] = 16
        self.assertFalse(self.result(rows)["host_chain_complete"])


if __name__ == "__main__":
    unittest.main()
