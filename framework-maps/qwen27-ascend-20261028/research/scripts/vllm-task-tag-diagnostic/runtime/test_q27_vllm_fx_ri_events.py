import os
import sys
import types
import unittest
from contextlib import nullcontext
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))
import q27_vllm_fx_events as q

class Tests(unittest.TestCase):
    def setUp(self):
        q._GRAPHS.clear(); q._EVENTS.clear(); q._COMPILE_ROWS.clear()
        q._EVENT_HISTORY.clear()
        q._TLS.capture_stack = []
        graph_pass=types.ModuleType("npugraph_ex._acl_concrete_graph.graph_pass")
        graph_pass._GLOBAL_SCOPE_TAG_TO_EVENT={}
        sys.modules["npugraph_ex._acl_concrete_graph.graph_pass"] = graph_pass
        fcmod = types.ModuleType("vllm.forward_context")
        fcmod.get_forward_context = lambda: CURRENT["fc"]
        sys.modules["vllm"] = types.ModuleType("vllm")
        sys.modules["vllm.forward_context"] = fcmod
        CURRENT["fc"] = SimpleNamespace(batch_descriptor="bd1", cudagraph_runtime_mode="target")

    def _rows(self, count=128):
        scope = q._TLS.capture_stack[-1]
        out = []
        for i in range(count):
            layer, edge = divmod(i, 2)
            edge = "pre" if edge == 0 else "post"
            marker = f"pid{os.getpid()}-capture-test:layer{layer:02d}:{edge}"
            # Capture ID is the active owner token; use exact source-format IDs.
            marker = f"pid{os.getpid()}-compile1:layer{layer:02d}:{edge}"
            tag = marker + ":event"
            old_event,new_event=object(),object()
            q._EVENTS[tag] = old_event
            q._EVENT_HISTORY[tag]=[old_event,new_event]
            sys.modules["npugraph_ex._acl_concrete_graph.graph_pass"]._GLOBAL_SCOPE_TAG_TO_EVENT[tag]=new_event
            scope["rows"].append({"marker_id": marker, "capture_id": scope["capture_id"],
                "callback_capture_status": 1, "model_ri_nonnull": True,
                "callback_nonnull": True,
                "callback_return_codes":{"callback_rc":0,"capture_rc":0,"stream_rc":0,"task_rc":0},
                "event_ref": new_event, "stream_ref": object()})

    def _wrapper(self, mode="target", count=128, runtime_mode="target"):
        CURRENT["fc"].cudagraph_runtime_mode = mode
        graph = object(); entry = SimpleNamespace(aclgraph=None)
        class Wrapper:
            def __init__(self): self.concrete_aclgraph_entries={"bd1":entry}; self.runtime_mode=runtime_mode
            def __call__(self, *a, **kw):
                if mode == "target" and self.runtime_mode == "target":
                    self.concrete_aclgraph_entries["bd1"].aclgraph = graph
                    self_outer._rows(count)
                return "output"
        self_outer = self
        return Wrapper(), graph, entry

    def test_mode_mismatch_does_not_bind(self):
        emits=[]; wrapper, graph, _ = self._wrapper(mode="other")
        q._capture_wrapper(type(wrapper), target_mode="target", drain_callbacks=lambda rows: rows,
                           emit=lambda *a, **k: emits.append((a,k)))
        self.assertEqual(wrapper(), "output")
        self.assertEqual(q.held_graphs(), [])
        self.assertFalse(emits)

    def test_wrapper_runtime_mode_mismatch_does_not_bind(self):
        emits=[]; wrapper, graph, _ = self._wrapper(mode="target",runtime_mode="other")
        q._capture_wrapper(type(wrapper),target_mode="target",drain_callbacks=lambda rows:rows,
                           emit=lambda *a,**k:emits.append((a,k)))
        self.assertEqual(wrapper(),"output")
        self.assertEqual(q.held_graphs(),[])
        self.assertFalse(emits)

    def test_exact_entry_graph_binds_only_full_marker_set_and_releases_with_receipt(self):
        emits=[]; wrapper, graph, entry = self._wrapper()
        q._capture_wrapper(type(wrapper), target_mode="target", drain_callbacks=lambda rows: rows,
                           emit=lambda *a, **k: emits.append((a,k)), device_provider=lambda fc: "npu0")
        self.assertEqual(wrapper(), "output")
        held=q.held_graphs()
        self.assertEqual(len(held),1)
        self.assertIs(held[0]["graph"], graph)
        self.assertIs(held[0]["entry"], entry)
        self.assertEqual(held[0]["device"], "npu0")
        self.assertEqual(len(held[0]["events"]),128)
        self.assertEqual(sum(1 for a,k in emits if a[0]=="layer_public_event_marker"),128)
        first=next(k for a,k in emits if a[0]=="layer_public_event_marker")
        self.assertEqual(first["graph_identity_status"],"unique_backend_saved_graph_object")
        self.assertEqual(first["classification"],"capture_endpoint_candidate")
        self.assertEqual(first["capture_invocation_id"],first["capture_id"])
        self.assertIn(first["marker_side"],{"pre","post"})
        self.assertIsInstance(first["layer_idx"],int)
        summary=next(k for a,k in emits if a[0]=="layer_public_event_summary")
        self.assertEqual(summary["marker_rows"],128)
        self.assertEqual(summary["capture_endpoint_rows"],128)
        self.assertFalse(q.release_graph_after_reset(graph,{"graph_python_id":id(graph),
            "reset_or_discarded":True,"device_drained_before_and_after_reset":False})["released"])
        receipt={"graph_python_id":id(graph),"reset_or_discarded":True,
                 "device_drained_before_and_after_reset":True}
        self.assertTrue(q.release_graph_after_reset(graph,receipt)["released"])
        self.assertEqual(q.held_graphs(),[])
        self.assertEqual(sys.modules["npugraph_ex._acl_concrete_graph.graph_pass"]._GLOBAL_SCOPE_TAG_TO_EVENT,{})
        self.assertEqual(q._EVENT_HISTORY,{})

    def test_missing_callbacks_fail_closed_but_keep_graph_and_event_refs(self):
        emits=[]; wrapper, graph, _ = self._wrapper(count=127)
        q._capture_wrapper(type(wrapper), target_mode="target", drain_callbacks=lambda rows: rows,
                           emit=lambda *a, **k: emits.append((a,k)))
        self.assertEqual(wrapper(), "output")
        held=q.held_graphs()
        self.assertEqual(len(held),1)
        self.assertIs(held[0]["graph"],graph)
        self.assertEqual(len(held[0]["events"]),127)
        summary=[k for a,k in emits if a[0]=="layer_public_event_summary"]
        self.assertEqual(summary[0]["status"],"unknown_capture_marker_set")
        self.assertEqual(summary[0]["marker_rows"],127)
        self.assertEqual(summary[0]["capture_endpoint_rows"],0)
        self.assertFalse(any(a[0]=="layer_public_event_marker" for a,k in emits))

    def test_runtime_replaced_event_and_gen44_fields_are_retained(self):
        marker="pid55-compile1:layer00:pre"; tag=marker+":event"
        old,new=object(),object()
        q._EVENTS[tag]=old; q._EVENT_HISTORY[tag]=[old]
        sys.modules["npugraph_ex._acl_concrete_graph.graph_pass"]._GLOBAL_SCOPE_TAG_TO_EVENT[tag]=new
        q._TLS.capture_stack=[{"capture_id":"cap1","phase":"aclgraph_capture",
                               "capture_state":True,"rows":[]}]
        torch=types.ModuleType("torch")
        torch.npu=SimpleNamespace(current_stream=lambda:"stream-object")
        torch.profiler=SimpleNamespace(record_function=lambda label:nullcontext())
        sys.modules["torch"]=torch
        ext=SimpleNamespace(anchor=lambda enum,annotation_json:SimpleNamespace(model_ri_nonnull=True))
        impl=q._anchor_impl(ext)
        annotation='{"q27_marker_id":"' + marker + '","q27_marker_schema":1}'
        impl("after_add",marker,annotation)
        row=q._TLS.capture_stack[0]["rows"][0]
        self.assertIs(row["event_ref"],new)
        self.assertIs(q._EVENTS[tag],new)
        self.assertEqual(q._EVENT_HISTORY[tag],[old,new])
        self.assertEqual(row["stream_ref"],"stream-object")
        self.assertTrue(callable(q.register_import_finder))

CURRENT={}
if __name__ == "__main__": unittest.main()
