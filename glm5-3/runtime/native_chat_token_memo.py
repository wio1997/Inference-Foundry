"""CPU token memo for plain native Chat; all other requests pass through.

A miss calls the SAME selected native API's /tokenize inside the HTTP request.
A hit reuses exact integer IDs bound to that API epoch. Native sampling, model
operators, validation, response bytes and state ownership remain downstream.
"""
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path
import httpx

# Any unrecognized field follows the original native request path.
PLAIN_FIELDS = {
    "model", "messages", "stream", "stream_options", "max_tokens",
    "max_completion_tokens", "temperature", "top_p", "top_k", "min_p",
    "seed", "n", "presence_penalty", "frequency_penalty", "repetition_penalty",
    "ignore_eos", "cache_salt", "return_token_ids", "chat_template_kwargs",
    "stop", "stop_token_ids", "logprobs", "top_logprobs", "allowed_token_ids",
    "min_tokens", "skip_special_tokens", "spaces_between_special_tokens",
    "include_stop_str_in_output",
}

def plain_tokenize_payload(body, model="glm-52"):
    if len(body) > 1048576:
        return None
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(value, dict) or set(value) - PLAIN_FIELDS:
        return None
    if value.get("model") != model:
        return None
    messages = value.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        return None
    message = messages[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        return None
    if message["role"] != "user" or not isinstance(message["content"], str):
        return None
    kwargs = value.get("chat_template_kwargs")
    if kwargs is not None and (
        not isinstance(kwargs, dict) or set(kwargs) - {"enable_thinking"}
        or any(type(v) is not bool for v in kwargs.values())
    ):
        return None
    if not body.rstrip().startswith(b"{") or not body.rstrip().endswith(b"}"):
        return None
    payload = dict(model=model, messages=messages, add_generation_prompt=True,
                   add_special_tokens=False, return_token_strs=False)
    if kwargs is not None:
        payload["chat_template_kwargs"] = kwargs
    return payload

class NativeChatTokenMemoTransport(httpx.AsyncBaseTransport):
    def __init__(self, base, epochs, max_bytes=33554432, audit_dir=None,
                 trace_path=None, stats_path=None, max_pending=128):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("nonnegative explicit token cache byte budget required")
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("positive bounded token lookup count required")
        self.base = base
        self.epochs = dict(epochs)
        self.max_bytes = max_bytes
        self.max_pending = max_pending
        self.cache = OrderedDict()
        self.used = 0
        self.pending = {}
        self.audit_dir = Path(audit_dir) if audit_dir else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = trace_path
        self.stats_path = Path(stats_path) if stats_path else None
        self.stats = dict(hits=0, misses=0, singleflight_waits=0, bypasses=0,
                          lookup_errors=0, evictions=0, oversize_entries=0,
                          pending_overflow=0, rewrites=0, audit_errors=0, tokenizer_CPU_calls=0,
                          tokenizer_CPU_wall_s=0.0, tokenizer_max_count=0)
        self.closed = False

    def snapshot(self):
        return dict(self.stats, entries=len(self.cache), cache_bytes=self.used,
                    cache_byte_budget=self.max_bytes, pending=len(self.pending),
                    max_pending=self.max_pending, backend_epochs=self.epochs,
                    no_model_inference_for_tokenize=True)

    def record(self, event, **fields):
        try:
            if self.trace_path:
                data = json.dumps(dict(event=event, pid=os.getpid(),
                                       monotonic_ns=time.monotonic_ns(), **fields),
                                  separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
                fd = os.open(self.trace_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
                try:
                    os.write(fd, data)
                finally:
                    os.close(fd)
            if self.stats_path:
                self.stats_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.stats_path.with_name(self.stats_path.name + ".tmp")
                tmp.write_text(json.dumps(self.snapshot(), sort_keys=True) + "\n")
                os.replace(tmp, self.stats_path)
        except OSError:
            self.stats["audit_errors"] += 1

    def artifact(self, name, data):
        if not self.audit_dir:
            return None
        path = self.audit_dir / name
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
        except OSError:
            self.stats["audit_errors"] += 1
            return None
        return dict(path=str(path), bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest())

    @staticmethod
    def metadata_size(value):
        if isinstance(value, dict):
            return sys.getsizeof(value) + sum(
                NativeChatTokenMemoTransport.metadata_size(k) +
                NativeChatTokenMemoTransport.metadata_size(v) for k, v in value.items())
        if isinstance(value, (tuple, list)):
            return sys.getsizeof(value) + sum(
                NativeChatTokenMemoTransport.metadata_size(v) for v in value)
        return sys.getsizeof(value)

    def remember(self, key, ids_json, count, source):
        # Immutable serialized IDs keep memory accounting explicit and bounded.
        size = (sys.getsizeof(key) + sys.getsizeof(ids_json) + 512
                + self.metadata_size(source))
        if size > self.max_bytes:
            self.stats["oversize_entries"] += 1
            return
        while self.cache and self.used + size > self.max_bytes:
            _, prior = self.cache.popitem(last=False)
            self.used -= prior[3]
            self.stats["evictions"] += 1
        self.cache[key] = (ids_json, count, source, size)
        self.used += size

    async def fetch(self, key, origin, epoch, payload, header_id):
        response = None
        tag = uuid.uuid4().hex
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        request_ref = self.artifact(tag + ".tokenize.body", data)
        begin = time.monotonic()
        self.stats["tokenizer_CPU_calls"] += 1
        try:
            request = httpx.Request(
                "POST", origin + "/tokenize", content=data,
                headers={"content-type": "application/json", "accept-encoding": "identity",
                         "x-request-id": header_id + "-tokenize-" + tag},
                extensions={"timeout": {"connect": 10, "read": 30, "write": 30, "pool": 30}},
            )
            response = await self.base.handle_async_request(request)
            raw = await response.aread()
            response_ref = self.artifact(tag + ".tokenize.wire", raw)
            if response.status_code != 200:
                self.stats["lookup_errors"] += 1
                return None
            value = json.loads(raw)
            ids, count = value["tokens"], value["count"]
            if (not isinstance(ids, list) or type(count) is not int
                or count != len(ids) or not 0 < count <= 144384
                or any(type(token) is not int or token < 0 for token in ids)):
                self.stats["lookup_errors"] += 1
                return None
            encoded = json.dumps(ids, separators=(",", ":")).encode()
            source = dict(origin=origin, epoch=epoch, request=request_ref,
                          response=response_ref, count=count,
                          ids_sha256=hashlib.sha256(encoded).hexdigest())
            self.stats["tokenizer_max_count"] = max(self.stats["tokenizer_max_count"], count)
            self.remember(key, encoded, count, source)
            self.record("chat_token_cache_native_tokenize", request_header_id=header_id,
                        cache_key=key, native_origin=origin, native_epoch=epoch,
                        source=source, elapsed_s=time.monotonic() - begin,
                        new_inference_requests=0, new_output_tokens=0)
            return encoded, count, source
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            self.stats["lookup_errors"] += 1
            return None
        finally:
            self.stats["tokenizer_CPU_wall_s"] += time.monotonic() - begin
            if response is not None:
                await response.aclose()

    async def lookup(self, key, origin, epoch, payload, header_id):
        if key in self.cache:
            self.stats["hits"] += 1
            value = self.cache.pop(key)
            self.cache[key] = value
            return value[:3], "hit"
        if key in self.pending:
            self.stats["singleflight_waits"] += 1
            return await asyncio.shield(self.pending[key]), "singleflight"
        if len(self.pending) >= self.max_pending:
            self.stats["pending_overflow"] += 1
            return None, "overflow"
        self.stats["misses"] += 1
        task = asyncio.create_task(self.fetch(key, origin, epoch, payload, header_id))
        self.pending[key] = task
        def finished(done):
            self.pending.pop(key, None)
            # Consume exceptions after a caller cancellation; no generation RPC is made here.
            if not done.cancelled():
                done.exception()
        task.add_done_callback(finished)
        return await asyncio.shield(task), "miss"

    async def handle_async_request(self, request):
        origin = request.url.scheme + "://" + request.url.netloc.decode("ascii")
        if (not self.max_bytes or request.method != "POST"
            or request.url.path != "/v1/chat/completions" or request.url.query
            or origin not in self.epochs
            or request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json"
            or any(name in request.headers for name in
                   ["content-md5", "digest", "signature", "x-amz-content-sha256"])):
            self.stats["bypasses"] += 1
            return await self.base.handle_async_request(request)
        body = await request.aread()
        payload = plain_tokenize_payload(body)
        if payload is None:
            self.stats["bypasses"] += 1
            return await self.base.handle_async_request(request)
        epoch = self.epochs[origin]
        try:
            canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode()
        except UnicodeError:
            self.stats["bypasses"] += 1
            return await self.base.handle_async_request(request)
        key = hashlib.sha256(origin.encode() + b"\0" + epoch.encode()
                             + b"\0" + canonical).hexdigest()
        header_id = request.headers.get("x-request-id", "")
        entry, result = await self.lookup(key, origin, epoch, payload, header_id)
        if entry is None:
            self.stats["bypasses"] += 1
            self.record("chat_token_cache_lookup_bypass", request_header_id=header_id,
                        cache_key=key, reason=result)
            return await self.base.handle_async_request(request)
        ids_json, count, source = entry
        # Preserve every original member byte, adding only native input metadata.
        native = (body.rstrip()[:-1] + b',"kv_transfer_params":{"prompt_token_ids":'
                  + ids_json + b"}}")
        tag = uuid.uuid4().hex
        original_ref = self.artifact(tag + ".original.body", body)
        native_ref = self.artifact(tag + ".native.body", native)
        self.stats["rewrites"] += 1
        self.record("chat_token_cache_native_body", request_header_id=header_id,
                    cache_result=result, cache_key=key, native_origin=origin,
                    native_epoch=epoch, original_body_sha256=hashlib.sha256(body).hexdigest(),
                    native_body_sha256=hashlib.sha256(native).hexdigest(),
                    original_body=original_ref, native_body=native_ref,
                    native_prompt_tokens=count, tokenize_source=source,
                    all_original_payload_members_preserved=True,
                    sampling_unchanged=True, output_wire_transformed=False)
        headers = [(name, value) for name, value in request.headers.multi_items()
                   if name.lower() != "content-length"]
        outgoing = httpx.Request(request.method, request.url, headers=headers,
                                 content=native, extensions=dict(request.extensions))
        return await self.base.handle_async_request(outgoing)

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        tasks = list(self.pending.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.pending.clear()
        self.record("chat_token_cache_closed", snapshot=self.snapshot())
        await self.base.aclose()
