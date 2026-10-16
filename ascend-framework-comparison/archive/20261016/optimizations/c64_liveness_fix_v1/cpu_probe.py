"""Execute actual source method bodies against bounded resource fixtures.

This proves the cross-worker resource-cycle mechanism and lifecycle ordering,
not that production had exactly these holders, and not model performance.
No SGLang import, NPU allocation, SDK I/O, HTTP or installed-source mutation.
"""
import __future__
import argparse
import ast
import enum
import hashlib
import json
import types
from pathlib import Path

C = Path(__file__).resolve().parent


class Mode(enum.Enum):
    NULL = 0
    PREFILL = 1
    DECODE = 2


def load_method(source, class_name, method, namespace):
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    assert not fn.decorator_list
    tree = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    ns = dict(namespace)
    exec(compile(tree, str(source), "exec", flags=__future__.annotations.compiler_flag), ns)
    return ns[method]


class Fixture:
    def __init__(self, variant, slots=4, buffer=True, enabled=True,
                 optimistic=0, allocation="early", mode=Mode.PREFILL):
        self.events = []
        self.slots = slots
        self.staged = set()
        self.pending = []
        self.completed = []
        self.released_by_abort = []
        self.config = types.SimpleNamespace(
            optimistic_prefill_attempts=optimistic,
            disaggregation_decode_allocation_policy=allocation,
        )
        src = C / ("source" if variant == "base" else "patched")
        namespace = dict(DisaggregationMode=Mode, get_disagg=lambda: self.config)
        self.scheduler = types.SimpleNamespace(
            disaggregation_mode=mode,
            enable_hicache_storage=enabled,
            tree_cache=types.SimpleNamespace(buffer_pipeline=object() if buffer else None),
            model_config=types.SimpleNamespace(num_key_value_heads=1),
            processed_tokens_counter=0,
            waiting_queue=[],
            _set_or_validate_priority=lambda req: True,
            _abort_on_queued_limit=lambda req: False,
            _release_aborted_request=lambda req: self.abort(req.rid),
            _prefetch_kvcache=self.prefetch,
        )
        add = load_method(src / "managers/scheduler.py", "Scheduler",
                          "_add_request_to_queue", namespace)
        self.add = types.MethodType(add, self.scheduler)
        if variant == "patched":
            helper = load_method(src / "managers/scheduler.py", "Scheduler",
                                 "_defer_hicache_prefetch_until_bootstrap", namespace)
            self.scheduler._defer_hicache_prefetch_until_bootstrap = types.MethodType(
                helper, self.scheduler)
        self.requests = []
        self.metadata_ok = True
        self.queue = types.SimpleNamespace(
            scheduler=self.scheduler,
            ensure_metadata_buffer=lambda req: self.metadata_ok,
            _num_pages_to_send=lambda req, hit: (80000 - hit + 127) // 128,
        )
        self.queue.add = self.enqueue
        self.scheduler.disagg_prefill_bootstrap_queue = self.queue
        finish = load_method(src / "disaggregation/prefill.py", "PrefillBootstrapQueue",
                             "finalize_bootstrap", namespace)
        self.finalize = types.MethodType(finish, self.queue)

    def request(self, rid):
        stats = types.SimpleNamespace(
            set_prefill_bootstrap_queue_entry_time=lambda: None,
            set_wait_queue_entry_time=lambda: None,
            set_decode_prealloc_queue_entry_time=lambda: None,
            set_bootstrap_done_time=lambda: self.events.append((rid, "bootstrap_done")),
        )
        sender = types.SimpleNamespace(
            pop_decode_prefix_len=lambda: 0,
            init=lambda pages, metadata: self.events.append((rid, "sender_init")),
        )
        return types.SimpleNamespace(rid=rid, time_stats=stats,
                                     disagg_kv_sender=sender, pending_bootstrap=True,
                                     metadata_buffer_index=0)

    def enqueue(self, req, heads):
        self.requests.append(req)
        self.events.append((req.rid, "queued"))

    def prefetch(self, req):
        self.events.append((req.rid, "prefetch"))
        if len(self.staged) < self.slots:
            self.staged.add(req.rid)
        else:
            self.pending.append(req.rid)

    def drain_pending(self):
        while self.pending and len(self.staged) < self.slots:
            self.staged.add(self.pending.pop(0))

    def consume(self, rid):
        if rid not in self.staged:
            return False
        self.staged.remove(rid)
        self.completed.append(rid)
        self.drain_pending()
        return True

    def abort(self, rid):
        self.staged.discard(rid)
        self.released_by_abort.append(rid)
        self.drain_pending()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["base", "patched"], required=True)
    a = p.parse_args()
    contract = json.loads((C / "PATCH_CONTRACT.json").read_text())
    for rel, pins in contract["files"].items():
        source = C / ("source" if a.variant == "base" else "patched") / rel
        assert hashlib.sha256(source.read_bytes()).hexdigest() == pins[a.variant + "_sha256"]
    checks = []
    # Independent P/D arrival orders: A owns finite host staging but has no D
    # allocation; B already owns D HBM. Original intake stages all A before B.
    for slots in [1, 4]:
        f = Fixture(a.variant, slots=slots)
        holders = [f.request("A" + str(i)) for i in range(slots)]
        ready = f.request("B")
        for req in holders + [ready]:
            f.add(req)
        assert f.finalize(ready)
        if a.variant == "base":
            assert len(f.staged) == slots and "B" in f.pending
            assert not f.consume("B")
            f.abort(holders[0].rid)
            assert f.consume("B"), "timeout/abort must explain cycle release"
        else:
            assert f.staged == {"B"} and f.consume("B")
            for req in holders:
                assert f.finalize(req) and f.consume(req.rid)
            assert len(f.completed) == slots + 1
            assert not f.released_by_abort and not f.pending and not f.staged
            for req in holders + [ready]:
                e = [kind for rid, kind in f.events if rid == req.rid]
                assert e.index("sender_init") < e.index("prefetch")
                assert e.count("prefetch") == 1
        checks.append("resource_cycle_slots_" + str(slots))
    if a.variant == "patched":
        # A failed metadata allocation never starts a transfer or consumes L2.
        f = Fixture(a.variant)
        req = f.request("metadata")
        f.add(req)
        f.metadata_ok = False
        assert not f.finalize(req) and req.pending_bootstrap and not f.staged
        f.metadata_ok = True
        assert f.finalize(req) and not req.pending_bootstrap and f.staged == {"metadata"}
        checks.append("metadata_deferral_preserves_no_prefetch_until_success")
        for name, kwargs in [
            ("cache_mode", dict(buffer=False)),
            ("no_storage", dict(enabled=False)),
            ("optimistic", dict(optimistic=1)),
            ("prefill_complete", dict(allocation="prefill_complete")),
            ("single_engine", dict(mode=Mode.NULL)),
        ]:
            f = Fixture(a.variant, **kwargs)
            req = f.request(name)
            f.add(req)
            assert sum(kind == "prefetch" for _, kind in f.events) == 1
            if kwargs.get("mode") != Mode.NULL:
                f.finalize(req)
                assert sum(kind == "prefetch" for _, kind in f.events) == 1
            checks.append("unchanged_" + name)
        # Retraction passes the same gate and resets retry state as before.
        f = Fixture(a.variant)
        req = f.request("retracted")
        f.add(req, is_retracted=True)
        assert req.storage_prefetch_retry_attempts == 0 and not f.staged
        assert f.finalize(req) and f.consume(req.rid)
        checks.append("retracted_request_same_lifecycle")
    result = dict(
        phase="ORIGINAL_RESOURCE_CYCLE_REPRODUCED" if a.variant == "base"
        else "PATCHED_LIFECYCLE_FIXTURE_PASS_MODEL_VALIDATION_REQUIRED",
        checks=checks, variant=a.variant, model_requests=0, SDK_data_operations=0,
        explicit_NPU_allocations=0, changed_installed_sources=0,
        limitation="Real source method bodies with bounded mock host/D capacities; "
                   "does not identify exact holders in the previous production trace.",
    )
    (C / (a.variant + "-cpu-result.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
