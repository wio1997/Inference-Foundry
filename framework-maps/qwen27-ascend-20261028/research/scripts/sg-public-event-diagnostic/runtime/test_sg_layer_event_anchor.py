import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import sg_layer_event_anchor as candidate


class FakeEvent:
    def __init__(self, log): self.log=log; self.records=[]
    def record(self, stream): self.log.append(("record", stream.name)); self.records.append(stream)

class FakeStream:
    def __init__(self, name, log): self.name=name; self.log=log
    def synchronize(self): self.log.append(("sync", self.name))

class Callback:
    def __init__(self, i):
        self.acl_stream_id=44; self.task_id=i; self.stream_rc=0; self.task_rc=0
        self.capture_rc=0; self.capture_status=1; self.callback_rc=0; self.model_ri_nonnull=True

class Extension:
    def __init__(self): self.n=0
    def anchor(self, label):
        self.n+=1; return Callback(self.n)

class Layer:
    def __init__(self, i, log): self.i=i; self.log=log
    def forward(self, x): self.log.append(("forward", self.i)); return x

class CandidateTests(unittest.TestCase):
    def setUp(self):
        candidate._FAILED_EVENT_SCOPES.clear(); candidate._GRAPH_EVENT_HOLDS.clear()

    def setup_layers(self, log, ext, events, stream):
        layers=[Layer(i,log) for i in range(64)]
        candidate.install_layer_event_anchor_set(layers,["linear"]*64,ext,
            lambda event,**kw:events.append((event,kw)),stream_provider=lambda:stream)
        return layers

    def run_backend(self, *, fail=False):
        log=[]; stream=FakeStream("capture-stream",log); ext=Extension(); emitted=[]
        layers=self.setup_layers(log,ext,emitted,stream); event_objects=[]
        def event_factory():
            x=FakeEvent(log); event_objects.append(x); return x
        @contextmanager
        def scope(_name): yield
        class Backend:
            def __init__(self): self._graphs={}
            def capture_one(self, shape_key, forward_fn):
                forward_fn(); forward_fn()
                forward_fn()
                if fail: raise RuntimeError("capture broke")
                self._graphs[shape_key]=object()
                return "captured"
        backend=Backend()
        def original(self, shape_key, forward_fn):
            return Backend.capture_one(self,shape_key,forward_fn)
        out=None; error=None
        try:
            out=candidate.capture_backend_one_with_public_events(original,backend,
                ("bucket=1",lambda:[layer.forward(0) for layer in layers]),{},
                emit=lambda event,**kw:emitted.append((event,kw)),
                event_factory=event_factory,record_scope_factory=scope,expected_shape_key="bucket=1")
        except RuntimeError as exc: error=exc
        return backend,log,emitted,event_objects,out,error

    def test_only_third_forward_records_128_markers_and_graph_holds(self):
        backend,log,emitted,events,out,error=self.run_backend()
        self.assertIsNone(error); self.assertEqual(out,"captured")
        self.assertEqual(len(events),128)
        self.assertEqual(sum(x[0]=="record" for x in log),128)
        self.assertEqual(sum(x[0]=="forward" for x in log),192)
        rows=[kw for name,kw in emitted if name=="layer_public_event_marker"]
        self.assertEqual(len(rows),128)
        self.assertEqual({r["marker_side"] for r in rows},{"pre","post"})
        self.assertEqual({r["forward_call_index"] for r in rows},{3})
        self.assertEqual({i:sum(r["layer_idx"]==i for r in rows) for i in range(64)}, {i:2 for i in range(64)})
        self.assertEqual({(r["layer_idx"],r["marker_side"]) for r in rows},
                         {(i,side) for i in range(64) for side in ("pre","post")})
        self.assertEqual(len({r["marker_id"] for r in rows}),128)
        graph=backend._graphs["bucket=1"]
        self.assertIn((id(backend),id(graph)),candidate._GRAPH_EVENT_HOLDS)
        self.assertEqual(sum(x[0]=="sync" for x in log),1)
        with self.assertRaisesRegex(RuntimeError,"reset/discard receipt"):
            candidate.release_graph_event_holds(backend,graph,reset_receipt=None)
        self.assertEqual(candidate.release_graph_event_holds(backend,graph,
            reset_receipt={"graph_python_id":id(graph),"reset_or_discarded":True})["event_count"],128)

    def test_capture_exception_propagates_without_drain_or_releasing_events(self):
        _,log,emitted,events,_,error=self.run_backend(fail=True)
        self.assertRegex(str(error),"capture broke")
        self.assertEqual(sum(x[0]=="sync" for x in log),0)
        self.assertEqual(len(events),128)
        self.assertTrue(candidate._FAILED_EVENT_SCOPES)
        self.assertEqual(len(candidate._FAILED_EVENT_SCOPES[-1]["events_by_layer"]),64)

    def test_wrong_or_missing_exact_shape_key_runs_uninstrumented(self):
        log=[]; stream=FakeStream("s",log)
        class Backend:
            _graphs={}
        def original(self,key,fn): fn(); return 1
        b=Backend(); calls=[]
        out=candidate.capture_backend_one_with_public_events(original,b,("bucket=2",lambda:calls.append(1)),{},
            emit=lambda *a,**k:None,event_factory=lambda:FakeEvent(log),
            record_scope_factory=lambda _n: _Null(),expected_shape_key="bucket=1")
        self.assertEqual(out,1); self.assertEqual(calls,[1]); self.assertFalse(any(x[0]=="record" for x in log))

class _Null:
    def __enter__(self): return self
    def __exit__(self,*args): return False

if __name__=="__main__": unittest.main()
