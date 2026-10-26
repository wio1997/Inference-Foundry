"""Bounded model-cache byte qualification only; never performance evidence.

Records actual native put results, known-key metadata, and complete sampled
object readback. Source addresses come directly from the native send task.
No tensor or input text is written. Formal mode must disable this observer.
"""
import contextlib
import contextvars
import hashlib
import json
import os
import pathlib
import threading
import time

_REQUEST = contextvars.ContextVar("flash_cache_request", default=None)
_LOCK = threading.Lock()
_SAMPLED = {}
_MAX_OBJECT_BYTES = 128 * 1024 * 1024
_PIECE_BYTES = 8 * 1024 * 1024
_MAX_KEYS = 8192


@contextlib.contextmanager
def request_scope(request_id, group_id):
    assert os.environ.get("FLASH_NATIVE_CACHE_QUALIFICATION") == "1"
    assert isinstance(request_id, str) and request_id and isinstance(group_id, int)
    token = _REQUEST.set((request_id, group_id))
    try:
        yield
    finally:
        _REQUEST.reset(token)


def _emit(row):
    root = pathlib.Path("/run-evidence/cache-qualification")
    root.mkdir(exist_ok=True)
    with _LOCK:
        with (root / (str(os.getpid()) + ".jsonl")).open("a") as file:
            file.write(json.dumps(row, sort_keys=True) + "\n")


def _piece_vectors(addresses, sizes):
    pointers, lengths = [], []
    for pointer, size in zip(addresses, sizes, strict=True):
        assert type(pointer) is int and pointer > 0 and type(size) is int and size > 0
        for offset in range(0, size, _PIECE_BYTES):
            pointers.append(pointer + offset)
            lengths.append(min(_PIECE_BYTES, size - offset))
    return pointers, lengths


def _hash_pieces(device_id, addresses, sizes):
    from vllm_ascend.distributed.kv_transfer.utils.sdma_payload_hash import PayloadHasher

    hasher = PayloadHasher(device_id, list(zip(addresses, sizes, strict=True)))
    pointers, lengths = _piece_vectors(addresses, sizes)
    return [hasher.hash_ranges([pointer], [size])["sha256"] for pointer, size in zip(pointers, lengths, strict=True)]


def _readback(backend, key, addresses, sizes):
    import torch

    total = sum(sizes)
    assert 0 < total <= _MAX_OBJECT_BYTES
    source = _hash_pieces(backend.device_id, addresses, sizes)
    scratch = torch.empty(total, dtype=torch.uint8, device="npu")
    pointer = scratch.data_ptr()
    registered = int(backend.store.register_buffer(pointer, total))
    assert registered == 0, ("scratch_register", registered)
    result = {"register_rc": registered, "bytes": total}
    try:
        offsets, cursor = [], 0
        for size in sizes:
            offsets.append(pointer + cursor)
            cursor += size
        # Same native layers API and COPY_G2L=1 as MemcacheBackend.get.
        codes = [int(code) for code in backend.store.batch_get_into_layers([key], [offsets], [sizes], 1)]
        result["get_rc"] = codes
        assert codes == [0], ("readback_get", codes)
        torch.npu.synchronize()
        restored = _hash_pieces(backend.device_id, offsets, sizes)
        result.update(
            source_piece_SHA256=source,
            restored_piece_SHA256=restored,
            original_logical_range_count=len(sizes),
            piece_count=len(source),
            all_bytes_equal=source == restored,
        )
        assert source == restored, "Native cache readback bytes differ"
    finally:
        native_return = backend.store.unregister_buffer(pointer, total)
        result["unregister_return"] = native_return
        result["unregister_API_doc"] = backend.store.unregister_buffer.__doc__
        if native_return is None:
            assert "-> None" in result["unregister_API_doc"], "Unqualified void unregister return"
        else:
            assert int(native_return) == 0, ("scratch_unregister", native_return)
    return result


def record_native_put(backend, keys, addresses, sizes, results):
    assert os.environ.get("FLASH_NATIVE_CACHE_QUALIFICATION") == "1"
    context = _REQUEST.get()
    row = {
        "project_id": "glm53-flash", "kind": "native_cache_put_qualification",
        "pid": os.getpid(), "monotonic_ns": time.monotonic_ns(),
        "request_id": context[0] if context else None,
        "kv_cache_group_id": context[1] if context else None,
        "device_id": int(backend.device_id), "passed": False,
        "timing_excluded": True, "tier_name": None, "objects": [],
    }
    try:
        assert context is not None, "Put has no actual native request context"
        codes = [int(code) for code in results]
        assert 0 < len(keys) <= _MAX_KEYS
        assert len(keys) == len(addresses) == len(sizes) == len(codes)
        row["native_put_rc"] = codes
        assert all(code == 0 for code in codes), "Native put returned errors"
        deadline = time.monotonic() + 15
        while True:
            exists = [int(value) for value in backend.store.batch_is_exist(keys)]
            if exists == [1] * len(keys) or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        row["known_key_exists"] = exists
        assert exists == [1] * len(keys), "Known keys absent after successful native put"
        infos = backend.store.batch_get_key_info(keys)
        assert infos is not None and len(infos) == len(keys)
        for key, pointers, lengths, info in zip(keys, addresses, sizes, infos, strict=True):
            pointers, lengths = [int(x) for x in pointers], [int(x) for x in lengths]
            assert len(pointers) == len(lengths) > 0
            assert info is not None and int(info.size()) == sum(lengths)
            obj = {
                "key_SHA256": hashlib.sha256(key.encode()).hexdigest(),
                "size_bytes": int(info.size()),
                "native_media_type_list": [int(x) for x in info.type_list()],
                "native_GVA_count": len(info.gva_list()),
                "tier_name": None, "readback": None,
            }
            row["objects"].append(obj)
            with _LOCK:
                count = _SAMPLED.get(context, 0)
                sample = count < 2
                if sample:
                    _SAMPLED[context] = count + 1
            if sample:
                obj["readback"] = _readback(backend, key, pointers, lengths)
        row["passed"] = True
    except BaseException as error:
        row["error"] = repr(error)
        raise
    finally:
        _emit(row)


def record_shutdown_queues(worker):
    """Passive qualification snapshot; no waits, cancellation or SDK calls."""
    row = {"project_id": "glm53-flash", "kind": "native_cache_shutdown_queues",
           "pid": os.getpid(), "monotonic_ns": time.monotonic_ns(),
           "passed": False, "observation_only": True, "threads": []}
    try:
        assert os.environ.get("FLASH_NATIVE_CACHE_QUALIFICATION") == "1"
        row["native_transfer_threads_started"] = worker._transfer_threads_started
        assert row["native_transfer_threads_started"]
        for attribute in ("kv_send_thread", "kv_recv_thread"):
            thread = getattr(worker, attribute)
            if thread is None:
                continue
            with thread.request_queue.mutex:
                queued = len(thread.request_queue.queue)
                unfinished = thread.request_queue.unfinished_tasks
            with thread.done_task_lock:
                pending = dict(thread.stored_requests)
            error = getattr(thread, "_fatal_error", None)
            row["threads"].append({"attribute": attribute, "queued": queued,
                "unfinished": unfinished, "pending_save_requests": len(pending),
                "pending_save_chunks": sum(pending.values()),
                "native_fatal_error": None if error is None else repr(error),
                "alive": thread.is_alive()})
        assert row["threads"] and all(item["queued"] == item["unfinished"] == item["pending_save_requests"] == item["pending_save_chunks"] == 0 and item["native_fatal_error"] is None for item in row["threads"])
        row["passed"] = True
    except BaseException as error:
        row["error"] = repr(error)
    finally:
        _emit(row)
