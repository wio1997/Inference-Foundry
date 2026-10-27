"""Opt-in CPU-only stage hooks for frozen SGLang / vLLM request paths.

Install by passing a module namespace to install_sgl() / install_vllm(). The
hooks preserve sync/async signatures, return values, and exceptions. They log
IDs, path/type metadata, and timestamps only; never request text or tensors.
"""
from __future__ import annotations

import functools
import json
import os
import threading
import time
from typing import Any

_LOCK = threading.Lock()
_PATH = os.environ.get("Q27_FRONTEND_STAGE_JSONL")


def _event(name: str, **fields: Any) -> None:
    if not _PATH:
        return
    row = {
        "event": name,
        "pid": os.getpid(),
        "mono_ns": time.monotonic_ns(),
        "wall_ns": time.time_ns(),
        **fields,
    }
    try:
        payload = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _LOCK, open(_PATH, "a", encoding="utf-8") as stream:
            stream.write(payload)
    except Exception:
        # Diagnostics must not alter serving behavior.
        return


def _install(cls: type, method_name: str, factory) -> bool:
    original = getattr(cls, method_name, None)
    if original is None or getattr(original, "_q27_frontend_stage", False):
        return False
    wrapped = factory(original)
    wrapped._q27_frontend_stage = True
    setattr(cls, method_name, wrapped)
    return True


def install_sgl(namespace: dict[str, Any]) -> list[str]:
    """Call after tokenizer_manager/http_server modules finish loading."""
    installed: list[str] = []
    manager = namespace.get("TokenizerManager")
    if not isinstance(manager, type):
        return installed

    def generate_factory(original):
        @functools.wraps(original)
        async def wrapped(self, obj, request=None, *args, **kwargs):
            rid = getattr(obj, "rid", None)
            headers = getattr(request, "headers", None)
            external = headers.get("x-request-id") if headers is not None else None
            _event("sg_http_request_enter", rid=rid, external_request_id=external,
                   request_type=type(obj).__name__)
            try:
                # TokenizerManager.generate_request is an async generator in
                # this source snapshot. Keep that protocol instead of awaiting
                # its returned generator or converting it to a coroutine.
                linked = False
                async for item in original(self, obj, request, *args, **kwargs):
                    if not linked:
                        rid = getattr(obj, "rid", None)
                        _event("sg_request_id_link", rid=rid,
                               external_request_id=external,
                               association="same normalized manager request at first yield")
                        linked = True
                    yield item
                _event("sg_http_request_end", rid=getattr(obj, "rid", rid),
                       external_request_id=external)
            except BaseException as exc:
                _event("sg_http_request_exception", rid=rid,
                       external_request_id=external, exception_type=type(exc).__name__)
                raise
        return wrapped

    if _install(manager, "generate_request", generate_factory):
        installed.append("TokenizerManager.generate_request")

    def tokenize_factory(original):
        @functools.wraps(original)
        async def wrapped(self, obj, *args, **kwargs):
            rid = getattr(obj, "rid", None)
            if getattr(obj, "input_embeds", None) is not None:
                path = "input_embeds"
            elif getattr(obj, "input_ids", None) is not None:
                path = "pretokenized_input_ids"
            else:
                path = "text_or_multimodal_processor"
            _event("sg_tokenize_begin", rid=rid, path=path)
            try:
                result = await original(self, obj, *args, **kwargs)
            except BaseException as exc:
                _event("sg_tokenize_exception", rid=rid, path=path,
                       exception_type=type(exc).__name__)
                raise
            _event("sg_tokenize_end", rid=rid, path=path,
                   input_id_count=(len(getattr(result, "input_ids", []))
                                   if getattr(result, "input_ids", None) is not None else None))
            return result
        return wrapped

    if _install(manager, "_tokenize_one_request", tokenize_factory):
        installed.append("TokenizerManager._tokenize_one_request")

    def dispatch_factory(original):
        @functools.wraps(original)
        def wrapped(self, obj, *args, **kwargs):
            rid = getattr(obj, "rid", None)
            _event("sg_enqueue_send_begin", rid=rid, transport="sync_sock_send")
            try:
                result = original(self, obj, *args, **kwargs)
            except BaseException as exc:
                _event("sg_enqueue_send_exception", rid=rid,
                       exception_type=type(exc).__name__)
                raise
            _event("sg_enqueue_send_end", rid=rid, transport="sync_sock_send")
            return result
        return wrapped

    if _install(manager, "_dispatch_to_scheduler", dispatch_factory):
        installed.append("TokenizerManager._dispatch_to_scheduler")
    return installed


def install_vllm(namespace: dict[str, Any], input_namespace: dict[str, Any] | None = None,
                 core_namespace: dict[str, Any] | None = None) -> list[str]:
    """Call after async_llm, input_processor, and core_client modules load."""
    cls = namespace.get("AsyncLLM")
    if not isinstance(cls, type):
        return []
    installed: list[str] = []

    input_cls = (input_namespace or {}).get("InputProcessor")
    if isinstance(input_cls, type):
        proc = getattr(input_cls, "process_inputs", None)
        if proc is not None and not getattr(proc, "_q27_frontend_stage", False):
            @functools.wraps(proc)
            def process_wrapped(self, request_id, prompt, *args, **kwargs):
                path = ("already_rendered_engine_input"
                        if isinstance(prompt, dict) and "type" in prompt
                        else "raw_input_processing")
                _event("vllm_input_processing_begin", native_request_id=request_id,
                       prompt_type=type(prompt).__name__, path=path)
                try:
                    result = proc(self, request_id, prompt, *args, **kwargs)
                except BaseException as exc:
                    _event("vllm_input_processing_exception", native_request_id=request_id,
                           path=path, exception_type=type(exc).__name__)
                    raise
                _event("vllm_input_processing_end", native_request_id=request_id, path=path)
                return result
            process_wrapped._q27_frontend_stage = True
            setattr(input_cls, "process_inputs", process_wrapped)
            installed.append("InputProcessor.process_inputs")

    original = getattr(cls, "add_request", None)
    if original is None or getattr(original, "_q27_frontend_stage", False):
        return installed

    @functools.wraps(original)
    async def wrapped(self, request_id, prompt, *args, **kwargs):
        # Prompt is deliberately not inspected or serialized.
        _event("vllm_async_add_request_begin", native_request_id=request_id,
               prompt_type=type(prompt).__name__)
        try:
            result = await original(self, request_id, prompt, *args, **kwargs)
        except BaseException as exc:
            _event("vllm_async_add_request_exception", native_request_id=request_id,
                   exception_type=type(exc).__name__)
            raise
        _event("vllm_async_add_request_end", native_request_id=request_id)
        return result

    wrapped._q27_frontend_stage = True
    setattr(cls, "add_request", wrapped)
    installed.append("AsyncLLM.add_request")

    seen_classes: set[int] = set()
    for core_cls in (core_namespace or {}).values():
        if not isinstance(core_cls, type) or id(core_cls) in seen_classes:
            continue
        seen_classes.add(id(core_cls))
        core_method = core_cls.__dict__.get("add_request_async")
        if core_method is None or getattr(core_method, "_q27_frontend_stage", False):
            continue
        @functools.wraps(core_method)
        async def core_wrapped(self, request, *args, __original=core_method, **kwargs):
            rid = getattr(request, "request_id", None)
            _event("vllm_engine_core_enqueue_begin", native_request_id=rid,
                   client_type=type(self).__name__)
            try:
                result = await __original(self, request, *args, **kwargs)
            except BaseException as exc:
                _event("vllm_engine_core_enqueue_exception", native_request_id=rid,
                       client_type=type(self).__name__, exception_type=type(exc).__name__)
                raise
            _event("vllm_engine_core_enqueue_end", native_request_id=rid,
                   client_type=type(self).__name__)
            return result
        core_wrapped._q27_frontend_stage = True
        setattr(core_cls, "add_request_async", core_wrapped)
        installed.append(core_cls.__name__ + ".add_request_async")
    return installed
