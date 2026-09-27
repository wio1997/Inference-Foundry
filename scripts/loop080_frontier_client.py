#!/usr/bin/env python3
"""Instrumented frozen benchmark client with c12 release chronology, never formal TPS.

The JSON request body is identical to scripts/bench.py. Raw SSE data payloads
are preserved as base64 so the server's exact serialized yields can be joined
without assuming a text-to-token inverse.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import aiohttp


class SSEDecoder:
    def __init__(self):
        self.buffer = b""
        self.saw_done = False
        self.done_payload = None

    def feed(self, chunk):
        """Return exact data payloads in an incoming fragment."""
        if self.saw_done and chunk:
            raise ValueError("bytes after DONE")
        self.buffer += chunk
        payloads_out = []
        while b"\n\n" in self.buffer:
            frame, self.buffer = self.buffer.split(b"\n\n", 1)
            lines = frame.splitlines()
            payloads = [line[6:] for line in lines if line.startswith(b"data: ")]
            if len(payloads) != 1:
                raise ValueError("SSE frame must have exactly one data line")
            payload = payloads[0]
            if payload == b"[DONE]":
                self.saw_done = True
                self.done_payload = payload
                if self.buffer:
                    raise ValueError("trailing bytes after DONE")
                continue
            if self.saw_done:
                raise ValueError("payload after DONE")
            payloads_out.append(payload)
        return payloads_out

    def finish(self):
        if not self.saw_done or self.buffer:
            raise ValueError("missing DONE or residual SSE bytes")


def parse_frames(chunks):
    """Yield exact `data:` payload bytes, and require a terminal DONE frame."""
    decoder = SSEDecoder()
    for chunk in chunks:
        yield from decoder.feed(chunk)
    decoder.finish()


def completion_tokens_from_usage(usage):
    tokens = usage.get("completion_tokens")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("missing authoritative completion_tokens usage")
    return tokens


def self_test():
    good = [b'data: {"id":"x","choices":[]}\n\n', b"data: [DONE]\n\n"]
    assert list(parse_frames(good)) == [b'{"id":"x","choices":[]}']
    assert list(parse_frames([good[0][:3], good[0][3:] + good[1]])) == [b'{"id":"x","choices":[]}']
    bad = [
        [good[0]],
        [b"data: x\n\n", b"data: y\n\n", b"data: [DONE]\n\nx"],
        [b"event: x\n\n", good[1]],
        [b"data: [DONE]\n\n", good[0]],
    ]
    for fragments in bad:
        try:
            list(parse_frames(fragments))
        except ValueError:
            continue
        raise AssertionError("negative SSE framing case passed")
    assert completion_tokens_from_usage({"completion_tokens": 1024}) == 1024
    for invalid in ({}, {"completion_tokens": "1024"}, {"completion_tokens": True}):
        try:
            completion_tokens_from_usage(invalid)
        except ValueError:
            continue
        raise AssertionError("invalid usage accepted")
    print(json.dumps({"status": "pass", "positive": 3, "negative": len(bad) + 3}))


def body(prompt, max_tokens):
    return {
        "model": "dsv4",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def json_file(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


async def acquire(args):
    lines = Path(args.dataset).read_text().splitlines()
    prompts = [json.loads(line)["question"] for line in lines[args.offset:args.offset + args.limit]]
    if len(prompts) != args.limit:
        raise ValueError("dataset has fewer rows than requested")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=1800)
    clock_begin = {"wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(i, prompt):
            acquire_attempt_ns = time.monotonic_ns()
            async with sem:
                acquired_ns = time.monotonic_ns()
                request_body = body(prompt, args.max_tokens)
                start_ns = time.monotonic_ns()
                start = time.perf_counter()
                first = None
                token_events = []
                events = []
                usage = {}
                response_id = None
                status = None
                error = None
                headers = {}
                raw_chunks = []
                usage_event_count = 0
                saw_finish_reason = False
                last_payload_had_usage = False
                done_ns = None
                try:
                    async with session.post(args.url, json=request_body) as response:
                        status = response.status
                        headers = dict(response.headers)
                        if status != 200:
                            error = f"HTTP {status}: {(await response.text())[:500]}"
                        else:
                            decoder = SSEDecoder()
                            async for chunk in response.content.iter_any():
                                fragment = bytes(chunk)
                                fragment_recv_ns = time.monotonic_ns()
                                raw_chunks.append(fragment)
                                had_done = decoder.saw_done
                                for payload in decoder.feed(fragment):
                                    data = json.loads(payload)
                                    if not isinstance(data, dict) or data.get("error") is not None:
                                        raise ValueError("SSE error or non-object payload")
                                    current_id = data.get("id")
                                    if not isinstance(current_id, str) or not current_id:
                                        raise ValueError("missing SSE response id")
                                    if response_id is None:
                                        response_id = current_id
                                    elif response_id != current_id:
                                        raise ValueError("SSE response id changed")
                                    usage_value = data.get("usage")
                                    last_payload_had_usage = usage_value is not None
                                    if last_payload_had_usage:
                                        if not isinstance(usage_value, dict):
                                            raise ValueError("SSE usage is not an object")
                                        if type(usage_value.get("prompt_tokens")) is not int or usage_value["prompt_tokens"] < 0:
                                            raise ValueError("invalid authoritative prompt_tokens usage")
                                        completion_tokens_from_usage(usage_value)
                                        usage_event_count += 1
                                        usage = usage_value
                                    has_delta = False
                                    choices = data.get("choices", [])
                                    if not isinstance(choices, list) or len(choices) > 1:
                                        raise ValueError("unexpected SSE choice cardinality")
                                    for choice in choices:
                                        if choice.get("index") != 0:
                                            raise ValueError("unexpected SSE choice index")
                                        finish_reason = choice.get("finish_reason")
                                        if finish_reason == "error":
                                            raise ValueError("SSE finish_reason error")
                                        saw_finish_reason |= finish_reason is not None
                                        delta = choice.get("delta") or {}
                                        has_delta |= bool(delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"))
                                    if has_delta:
                                        now = time.perf_counter()
                                        if first is None:
                                            first = now
                                        token_events.append(now)
                                    events.append({
                                        "seq": len(events),
                                        "payload_b64": base64.b64encode(payload).decode("ascii"),
                                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                                        "response_id": current_id,
                                        "has_text_delta": has_delta,
                                        "fragment_recv_monotonic_ns": fragment_recv_ns,
                                    })
                                if decoder.saw_done and not had_done:
                                    done_ns = fragment_recv_ns
                                    events.append({
                                        "seq": len(events),
                                        "payload_b64": base64.b64encode(decoder.done_payload).decode("ascii"),
                                        "payload_sha256": hashlib.sha256(decoder.done_payload).hexdigest(),
                                        "response_id": response_id,
                                        "done": True,
                                        "fragment_recv_monotonic_ns": fragment_recv_ns,
                                    })
                            decoder.finish()
                            if response_id is None:
                                raise ValueError("missing SSE response id")
                            if usage_event_count != 1 or not last_payload_had_usage or not saw_finish_reason:
                                raise ValueError("missing terminal finish or final authoritative usage")
                            completion_tokens_from_usage(usage)
                except Exception as exc:
                    error = repr(exc)
                end = time.perf_counter()
                end_ns = time.monotonic_ns()
                tokens = usage.get("completion_tokens") if error is None else None
                row = {
                    "i": i + args.offset,
                    "phase": args.phase,
                    "dataset_row_sha256": hashlib.sha256(lines[i + args.offset].encode()).hexdigest(),
                    "request_body_sha256": hashlib.sha256(json.dumps(request_body, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                    "start": start,
                    "end": end,
                    "start_monotonic_ns": start_ns,
                    "end_monotonic_ns": end_ns,
                    "c12_acquire_attempt_monotonic_ns": acquire_attempt_ns,
                    "c12_acquired_monotonic_ns": acquired_ns,
                    "sse_done_monotonic_ns": done_ns,
                    "ttft_ms": (first - start) * 1000 if first else None,
                    "tpot_ms": (end - first) * 1000 / (tokens - 1) if first and isinstance(tokens, int) and tokens > 1 else None,
                    "output_tokens": tokens,
                    "input_tokens": usage.get("prompt_tokens"),
                    "chunks": len(token_events),
                    "sse_events": len(events),
                    "response_id": response_id,
                    "http_status": status,
                    "response_headers": headers,
                    "usage": usage,
                    "error": error,
                }
                record = {"row": row, "events": events}
                if error is not None:
                    record["wire_fragments_b64_on_error"] = [base64.b64encode(part).decode("ascii") for part in raw_chunks]
            # This mark follows asyncio.Semaphore.__aexit__ (permit release).
            # The permit is anonymous; it never identifies a unique predecessor.
            released_after_ns = time.monotonic_ns()
            record["row"]["c12_release_completed_by_monotonic_ns"] = released_after_ns
            record["row"]["c12_unique_predecessor_certified"] = False
            return record

        wall_start_ns = time.monotonic_ns()
        wall_start = time.perf_counter()
        records = await asyncio.gather(*(one(i, prompt) for i, prompt in enumerate(prompts)))
        wall_end = time.perf_counter()
        wall_end_ns = time.monotonic_ns()
    clock_end = {"wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}
    rows = [record["row"] for record in records]
    for record in records:
        json_file(args.out_dir / f"request_{record['row']['i']:03d}.json", record)
    good = [r for r in rows if r["error"] is None and r["ttft_ms"] is not None and r["tpot_ms"] is not None]
    summary = {
        "scope": "instrumented_bound_diagnostic_not_formal_tps; c12 release interval and eligible set only, no unique permit parent",
        "phase": args.phase,
        "n": len(rows),
        "success": len(good),
        "fail": len(rows) - len(good),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "wall_start_monotonic_ns": wall_start_ns,
        "wall_end_monotonic_ns": wall_end_ns,
        "clock": {
            "begin": clock_begin,
            "end": clock_end,
            "monotonic_info": str(time.get_clock_info("monotonic")),
            "perf_counter_info": str(time.get_clock_info("perf_counter")),
            "kernel_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "time_namespace": str(Path("/proc/self/ns/time").readlink()),
            "timens_offsets": Path("/proc/self/timens_offsets").read_text(),
            "pid": os.getpid(),
        },
        "duration_s": wall_end - wall_start,
        "output_tps_diagnostic_only": sum(r["output_tokens"] for r in good) / (wall_end - wall_start),
        "ttft_ms_mean": statistics.mean(r["ttft_ms"] for r in good) if good else None,
        "tpot_ms_mean": statistics.mean(r["tpot_ms"] for r in good) if good else None,
    }
    json_file(args.out_dir / "summary.json", {"summary": summary, "requests": rows})
    print(json.dumps(summary, indent=2))
    response_ids = [r["response_id"] for r in rows]
    if len(good) != len(rows) or any(r["output_tokens"] != args.max_tokens or r["http_status"] != 200 for r in rows) or len(set(response_ids)) != len(response_ids):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--phase", choices=("warmup", "measured"))
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.dataset or args.out_dir is None or not args.phase:
        parser.error("dataset, out-dir and phase are required")
    asyncio.run(acquire(args))


if __name__ == "__main__":
    main()
