"""Bounded E27 test-service cache using native DeepInfer sequence lifetimes.

This is the E26 leading-underscore native fork/cache lifecycle, with the
per-rank retained-parent ceiling made explicit.  A completed prefix HTTP
request retains its actual native sequence.  A later new request validates the
prefix and consumes that retained entry via native ``fork_seq``.  No token
computation is credited from a staged timed request.

The parent ceiling is read from ``E27_PREFIX_PARENT_LIMIT`` (default 32).  The
native state-slot pool must fit the retained parents, the live child sequences,
the reserved dummy state, and a bounded margin.  The official
``DEEPINFER_NUM_STATE_SLOTS_RATIO=4`` gives 65 slots for max_num_seqs=16, which
fits 32 parents + 16 children + 1 dummy + 16 slots of margin.
"""
import hashlib
import json
import os
import time

import numpy as np


def _resolve_max_num_seqs(engine):
    """Resolve the native per-rank sequence cap without inventing a value."""
    for owner_name in ("scheduler_config", "scheduler"):
        owner = getattr(engine, owner_name, None)
        value = getattr(owner, "max_num_seqs", None)
        if value is not None:
            return int(value), owner_name
    raise RuntimeError(
        "Cannot resolve native max_num_seqs for the E27 state-slot capacity "
        "check; expose engine.scheduler_config.max_num_seqs or "
        "engine.scheduler.max_num_seqs")


def install(engine, rank):
    parent_limit = int(os.environ.get("E27_PREFIX_PARENT_LIMIT", "32"))
    if parent_limit < 1:
        raise ValueError("E27_PREFIX_PARENT_LIMIT must be >= 1")

    state_slots = engine.cache_config.num_state_seqs
    if state_slots is None:
        raise RuntimeError(
            "E27 prefix cache requires an initialized native state cache "
            "(cache_config.num_state_seqs is None)")
    state_slots = int(state_slots)
    max_num_seqs, max_num_seqs_source = _resolve_max_num_seqs(engine)
    if max_num_seqs < 1:
        raise RuntimeError(
            "E27 prefix cache resolved invalid max_num_seqs=%d from %s"
            % (max_num_seqs, max_num_seqs_source))
    margin = max(8, max_num_seqs)
    required_state_slots = parent_limit + max_num_seqs + 1 + margin
    if state_slots < required_state_slots:
        raise RuntimeError(
            "E27 prefix cache needs at least %d native state slots "
            "(parents=%d + live_children=%d + dummy=1 + margin=%d), got %d; "
            "use DEEPINFER_NUM_STATE_SLOTS_RATIO=4 for max_num_seqs=16"
            % (required_state_slots, parent_limit, max_num_seqs, margin,
               state_slots))

    original_add = engine.add_request
    original_finalize = engine.finalize_seq
    warming = {}
    parents = {}

    def report(kind, **kw):
        print("[e27-prefix-cache] " + json.dumps(
            dict(event=kind, rank=rank, wall=time.time(), **kw)), flush=True)

    def add(*args, **kw):
        user = str(kw.get("user_id", ""))
        if args or not user.startswith("e27cache:"):
            return original_add(*args, **kw)
        _, action, namespace, key = user.split(":", 3)
        cache_key = (namespace, key)
        ids = np.asarray(kw["prompt_token_ids"], dtype=np.int32)
        if action == "write":
            if int(kw["sampling_params"].max_tokens) != 1:
                raise ValueError("Cache population must complete one real output token")
            if len(parents) >= parent_limit and cache_key not in parents:
                raise RuntimeError(
                    "E27 retained parent limit per rank reached (%d)" % parent_limit)
            seq = original_add(**kw)
            warming[kw["request_id"]] = (cache_key, ids.copy())
            return seq
        if action != "read":
            raise ValueError("Unknown explicit cache action")
        if cache_key not in parents:
            raise RuntimeError("Required completed-request cache entry absent")
        parent, prefix = parents.pop(cache_key)
        n = len(prefix)
        if len(ids) <= n or not np.array_equal(ids[:n], prefix):
            original_finalize(parent)
            raise ValueError("Cached prefix mismatch; no fallback computation hidden")
        if int(parent.num_cached_tokens) != n:
            original_finalize(parent)
            raise RuntimeError("Native cache length differs from retained prefix")
        temp_kw = dict(kw, enqueue=False)
        temp = original_add(**temp_kw)
        if temp is None:
            original_finalize(parent)
            return None
        source_id = parent.request_id
        parent.data = temp.data
        parent.compressed_data = temp.compressed_data
        try:
            child = engine.scheduler.fork_seq(
                parent, None, "_" + kw["request_id"], temp.sampling_params,
                temp.so_matcher, parent.data.get_last_token_id(), -1,
                temp.seq_id, time.time() * 1000)
            child.request_id = kw["request_id"]
            child.user_id = user
            child.cached_hit_tokens = n
            assert int(child.num_cached_tokens) == n
        finally:
            original_finalize(parent)
        report("reuse", request_id=child.request_id,
               source_request_id=source_id, cached_tokens=n,
               prompt_tokens=len(ids),
               native_cached_tokens=int(child.num_cached_tokens),
               prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),
               consumed=True)
        return child

    def finalize(seq):
        pending = warming.pop(seq.request_id, None)
        if pending is None:
            return original_finalize(seq)
        key, prefix = pending
        if int(seq.num_cached_tokens) != len(prefix):
            original_finalize(seq)
            raise RuntimeError(
                "Completed warm request did not preserve its entire computed prefix")
        if key in parents:
            original_finalize(parents.pop(key)[0])
        parents[key] = (seq, prefix)
        report("retained", request_id=seq.request_id,
               cached_tokens=int(seq.num_cached_tokens),
               state_id=int(seq.state_id),
               prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest())

    engine.add_request = add
    engine.finalize_seq = finalize
    report(
        "installed",
        max_retained_parents=parent_limit,
        parent_limit_env="E27_PREFIX_PARENT_LIMIT",
        state_slots=state_slots,
        max_num_seqs=max_num_seqs,
        max_num_seqs_source=max_num_seqs_source,
        dummy_state_slots=1,
        margin_state_slots=margin,
        required_state_slots=required_state_slots,
        lifecycle="retain completed -> validated native fork -> consume")
