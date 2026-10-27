import os
import json
import sys
import types
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
import q27_vllm_fx_events as q
from fx_layer_event_injector import build_task_annotation


class RIQueryOrderTests(unittest.TestCase):
    def setUp(self):
        q._GRAPHS.clear(); q._EVENTS.clear(); q._EVENT_HISTORY.clear()
        q._COMPILE_ROWS.clear(); q._TLS.capture_stack = []
        self.events = []
        self.extension = SimpleNamespace(query_model_ri=self.query)
        q._ANCHOR_EXTENSION = self.extension

    def test_annotation_is_precomputed_ascii_json_bounded_and_exact(self):
        marker="pid17-compile2:layer03:post"
        encoded=build_task_annotation(marker)
        self.assertEqual(json.loads(encoded),{"q27_marker_id":marker,"q27_marker_schema":1})
        self.assertTrue(encoded.isascii())
        self.assertLessEqual(len(encoded.encode("utf-8")),4096)
        with self.assertRaises(ValueError): build_task_annotation("")
        with self.assertRaises(ValueError): build_task_annotation("标签")

    def query(self, result):
        self.events.append(("query", result))
        return result

    def test_callback_queries_follow_drain_and_bind_marker_graph_device(self):
        graph = object(); entry = SimpleNamespace(aclgraph=None)
        fc = SimpleNamespace(batch_descriptor="bd", cudagraph_runtime_mode="target")
        sys.modules["vllm.forward_context"] = SimpleNamespace(get_forward_context=lambda: fc)
        sys.modules["vllm"] = types.ModuleType("vllm")
        events = self.events
        rows = []
        class Wrapper:
            runtime_mode = "target"
            concrete_aclgraph_entries = {"bd": entry}
            def __call__(self):
                self.concrete_aclgraph_entries["bd"].aclgraph = graph
                scope = q._TLS.capture_stack[-1]
                for i in range(64):
                    for side in ("pre", "post"):
                        marker = f"pid{os.getpid()}-compile1:layer{i:02d}:{side}"
                        result = {"model_ri_get_id_rc": 0, "get_streams_rc": 0,
                            "model_ri_id": 48, "model_ri_handle_token": "ri",
                            "physical_stream_id": 45,
                            "model_streams": [{"stream_get_id_rc": 0,"model_stream_id": 12}]}
                        scope["rows"].append({"marker_id": marker,"capture_id":scope["capture_id"],
                            "callback_capture_status":1,"callback_nonnull":True,
                            "model_ri_nonnull":True,"callback_return_codes":{
                                "callback_rc":0,"capture_rc":0,"stream_rc":0,"task_rc":0},
                            "event_ref":object(),"stream_ref":object(),"callback":result})
                return "output"
        def drain(raw):
            events.append(("drain", len(raw)))
            return raw
        emitted=[]
        def emit(name, **kwargs):
            events.append(("emit", name))
            emitted.append((name, kwargs))
        q._capture_wrapper(Wrapper, target_mode="target", drain_callbacks=drain, emit=emit,
                           device_provider=lambda _fc: "device-0")
        self.assertEqual(Wrapper()(), "output")
        drain_at = next(i for i, e in enumerate(events) if e[0] == "drain")
        first_query_at = next(i for i, e in enumerate(events) if e[0] == "query")
        first_emit_at = next(i for i, e in enumerate(events) if e[0] == "emit")
        self.assertLess(drain_at, first_query_at)
        self.assertLess(first_query_at, first_emit_at)
        self.assertEqual(sum(e[0] == "query" for e in events), 128)
        markers=[k for n,k in emitted if n == "layer_public_event_marker"]
        self.assertEqual(len(markers),128)
        first=markers[0]["model_ri_query"]
        self.assertEqual(first["marker_id"],markers[0]["marker_id"])
        self.assertEqual(first["graph_python_id"],id(graph))
        self.assertEqual(first["capture_id"],markers[0]["capture_id"])
        self.assertEqual(first["device_id"],"device-0")
        self.assertEqual(first["ri_bridge_status"],"candidate_singleton_pair_requires_same_profile_match")
        self.assertEqual(first["model_streams"],[{"stream_get_id_rc":0,"model_stream_id":12}])
        self.assertEqual(len(q.held_graphs()[0]["ri_queries"]),128)

    def test_multistream_is_preserved_but_never_mapped(self):
        graph=object(); result={"model_ri_get_id_rc":0,"get_streams_rc":0,
          "physical_stream_id":45,"model_ri_id":48,
          "model_streams":[{"stream_get_id_rc":0,"model_stream_id":12},
                            {"stream_get_id_rc":0,"model_stream_id":13}]}
        rows=[{"marker_id":"m0","capture_id":"c0","callback":result}]
        out=q._query_capture_ri(self.extension,rows,capture_id="c0",graph=graph,device_id=0)
        self.assertEqual(out[0]["ri_bridge_status"],"unknown_multiple_model_streams")
        self.assertEqual([x["model_stream_id"] for x in out[0]["model_streams"]],[12,13])
        self.assertNotIn("model_stream_id_candidate",out[0])

    def test_query_error_stays_unknown_and_does_not_drop_raw_callback(self):
        class Broken:
            def query_model_ri(self, _): raise RuntimeError("query failed")
        row={"marker_id":"m0","capture_id":"c0","callback":object(),"callback_capture_status":1}
        out=q._query_capture_ri(Broken(),[row],capture_id="c0",graph=object(),device_id=0)
        self.assertEqual(out[0]["status"],"unknown_ri_query_exception")
        self.assertEqual(out[0]["error_type"],"RuntimeError")
        self.assertEqual(row["callback_capture_status"],1)


if __name__ == "__main__":
    unittest.main()
