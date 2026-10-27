import unittest
from contextlib import contextmanager

import sg_replay_outer_events as mod


class Stream:
    def __init__(self): self.synced = False
    def synchronize(self): self.synced = True


class Event:
    def __init__(self): self.stream = None
    def record(self, stream): self.stream = stream


class Callback:
    def __init__(self, stream): self.stream = stream
    def __getattr__(self, key):
        if not self.stream.synced:
            raise AssertionError("callback read before drain")
        values = {"acl_stream_id": 7, "task_id": 10, "stream_rc": 0,
                  "task_rc": 0, "capture_rc": 0, "capture_status": 0,
                  "callback_rc": 0, "model_ri_nonnull": False}
        return values.get(key)


class Extension:
    def __init__(self): self.stream = None
    def anchor(self, _label): return Callback(self.stream)


class Backend:
    def __init__(self): self._graphs = {"bucket1": object()}; self.calls = []
    def replay(self, shape_key):
        self.calls.append(("replay", shape_key)); return "replayed"
    def replay_with_input_update(self, shape_key, update):
        self.calls.append(("update", shape_key, update)); return "updated"


class Tests(unittest.TestCase):
    def setUp(self):
        mod._PENDING.clear(); mod._FAILED.clear(); mod._TLS.depth = 0
        self.stream = Stream(); self.ext = Extension(); self.ext.stream = self.stream
        self.events = []
        class E(Event):
            def __init__(inner): super().__init__(); self.events.append(inner)
        self.event_factory = E
        self.scopes = []
        @contextmanager
        def scope(name):
            self.scopes.append(name); yield
        self.scope = scope
        self.rows = []
        self.backend_cls = type("Backend", (Backend,), {})
        self.backend = self.backend_cls()
        mod.install_replay_outer_anchors(self.backend_cls, event_factory=self.event_factory,
            stream_provider=lambda: self.stream, extension=self.ext,
            emit=lambda kind, **kw: self.rows.append((kind, kw)),
            graph_key_provider=lambda _self, _method, args, kwargs: kwargs.get("shape_key", args[0] if args else None),
            work_key_provider=lambda *a: "step-9", record_scope_factory=self.scope)

    def test_two_methods_outer_boundaries_only_and_after_drain(self):
        self.assertEqual(self.backend.replay("bucket1"), "replayed")
        self.assertEqual(self.backend.replay_with_input_update("bucket1", 3), "updated")
        self.assertEqual(len(self.events), 4)
        self.assertEqual(len(self.rows), 0)
        self.assertEqual(len(self.scopes), 4)
        result = mod.drain_outer_replay_anchors(emit=lambda kind, **kw: self.rows.append((kind, kw)))
        self.assertEqual(result["rows"], 4)
        self.assertEqual([x[1]["side"] for x in self.rows], ["start", "end", "start", "end"])
        self.assertTrue(all(x[1]["classification"] == "raw_outer_replay_endpoint" for x in self.rows))
        self.assertTrue(all(x[1]["graph_python_id"] == id(self.backend._graphs["bucket1"]) for x in self.rows))

    def test_nested_backend_methods_do_not_double_mark(self):
        class NestedBackend(Backend):
            def replay_with_input_update(self, shape_key, update):
                return self.replay(shape_key)
        mod.install_replay_outer_anchors(NestedBackend, event_factory=self.event_factory,
            stream_provider=lambda: self.stream, extension=self.ext,
            emit=lambda kind, **kw: self.rows.append((kind, kw)),
            graph_key_provider=lambda _self, _method, args, kwargs: kwargs.get("shape_key", args[0] if args else None),
            record_scope_factory=self.scope)
        self.backend = NestedBackend()
        self.backend.replay_with_input_update("bucket1", 1)
        self.assertEqual(len(self.events), 2)

    def test_replay_exception_propagates_and_refs_are_retained(self):
        class FailBackend(Backend):
            def replay(self, key): raise LookupError("original")
        mod.install_replay_outer_anchors(FailBackend, event_factory=self.event_factory,
            stream_provider=lambda: self.stream, extension=self.ext,
            emit=lambda kind, **kw: self.rows.append((kind, kw)),
            graph_key_provider=lambda _self, _method, args, kwargs: kwargs.get("shape_key", args[0] if args else None),
            record_scope_factory=self.scope)
        failed = FailBackend()
        with self.assertRaisesRegex(LookupError, "original"):
            failed.replay("bucket1")
        self.assertEqual(len(mod._FAILED), 1)
        self.assertEqual(len(mod._FAILED[0]["rows"]), 1)
        self.assertEqual(len(mod._PENDING), 0)


if __name__ == "__main__": unittest.main()
