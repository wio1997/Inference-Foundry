#!/usr/bin/env python3
"""One bounded R247 B diagnostic cohort: prefix000/001 x one P leg x one D leg.

Exactly four generation POSTs (two prefixes, Decode then Prefill), no smoke,
retry, flush, warm-up or formal CLI.  There is no A framework in this client.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import threading
import time
import urllib.request

import r234_prefix_manifest as manifest

B_P = "http://172.16.12.46:19030/v1/completions"
B_D = "http://172.16.12.47:19030/v1/completions"
B_BOOTSTRAP_HOST = "172.16.12.46"
B_BOOTSTRAP_PORT = 19032
ROOM_BASE = 1970400000


def _post_json(url, body, timeout):
    data = json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read(64 * 1024 * 1024))


def _stream(url, body, timeout):
    data = json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    out = {"status": None, "done": False, "finish_reason": None, "usage": None,
           "text_chars": 0, "first_text_ns": None}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        out["status"] = response.status
        pending = []
        while True:
            raw = response.readline()
            received = time.monotonic_ns()
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                if pending:
                    payload = "\n".join(pending)
                    pending = []
                    if payload == "[DONE]":
                        out["done"] = True
                        break
                    message = json.loads(payload)
                    if message.get("usage") is not None:
                        out["usage"] = message["usage"]
                    for choice in message.get("choices", []):
                        text = choice.get("text") or ""
                        if text:
                            out["text_chars"] += len(text)
                            if out["first_text_ns"] is None:
                                out["first_text_ns"] = received
                        if choice.get("finish_reason") is not None:
                            out["finish_reason"] = choice["finish_reason"]
                continue
            if line.startswith("data:"):
                pending.append(line[5:].lstrip(" "))
            elif line.startswith(":") or line.startswith("event:") or line.startswith("id:"):
                continue
            else:
                raise ValueError("malformed SSE line")
    return out


def _b_d(prefix, args, timeout, ready, counters, lock):
    with lock:
        counters["attempted"] += 1
    body = {"model": args.model, "prompt": prefix["ids"], "temperature": 0.,
            "seed": 1024, "ignore_eos": True, "max_tokens": 1, "stream": True,
            "stream_options": {"include_usage": True}, "rid": args.run_id + "-" + prefix["name"],
            "bootstrap_host": B_BOOTSTRAP_HOST, "bootstrap_port": B_BOOTSTRAP_PORT,
            "bootstrap_room": ROOM_BASE + (0 if prefix["name"].endswith("000") else 1),
            "return_cached_tokens_details": True}
    ready.set()
    start = time.monotonic_ns()
    try:
        result = _stream(B_D, body, timeout)
        with lock:
            counters["terminal"] += 1 if result.get("status") is not None else 0
        result.update(prefix=prefix["name"], leg="decode", success=False)
        usage = result.get("usage") or {}
        result["success"] = bool(result["status"] == 200 and result["done"]
                                 and result["finish_reason"] == "length"
                                 and usage.get("prompt_tokens") == prefix["input_tokens"]
                                 and usage.get("completion_tokens") == 1)
    except BaseException as exc:
        result = {"prefix": prefix["name"], "leg": "decode", "success": False,
                  "error": repr(exc), "status": None}
    result.update(issued_ns=start, terminal_ns=time.monotonic_ns())
    return result


def _b_p(prefix, args, timeout, counters, lock):
    with lock:
        counters["attempted"] += 1
    body = {"model": args.model, "prompt": prefix["ids"], "temperature": 0.,
            "seed": 1024, "ignore_eos": True, "max_tokens": 1, "stream": False,
            "rid": args.run_id + "-" + prefix["name"],
            "bootstrap_host": B_BOOTSTRAP_HOST, "bootstrap_port": B_BOOTSTRAP_PORT,
            "bootstrap_room": ROOM_BASE + (0 if prefix["name"].endswith("000") else 1),
            "return_cached_tokens_details": True}
    start = time.monotonic_ns()
    try:
        status, message = _post_json(B_P, body, timeout)
        with lock:
            counters["terminal"] += 1
        usage = message.get("usage") or {}
        success = bool(status == 200 and usage.get("prompt_tokens") == prefix["input_tokens"]
                       and usage.get("completion_tokens") == 1)
        result = {"prefix": prefix["name"], "leg": "prefill", "status": status,
                  "success": success, "usage": usage,
                  "response_id": message.get("id")}
    except BaseException as exc:
        result = {"prefix": prefix["name"], "leg": "prefill", "success": False,
                  "error": repr(exc), "status": None}
    result.update(issued_ns=start, terminal_ns=time.monotonic_ns())
    return result


def run_b(prefixes, args):
    d_ready = [threading.Event(), threading.Event()]
    results = {"decode": [None, None], "prefill": [None, None]}
    counters = {"attempted": 0, "terminal": 0}
    lock = threading.Lock()

    def d_worker(i):
        results["decode"][i] = _b_d(prefixes[i], args, args.http_seconds, d_ready[i],
                                    counters, lock)

    def p_worker(i):
        results["prefill"][i] = _b_p(prefixes[i], args, args.http_seconds, counters, lock)

    d_threads = [threading.Thread(target=d_worker, args=(i,), daemon=True) for i in range(2)]
    for thread in d_threads:
        thread.start()
    for event in d_ready:
        event.wait(5)
    p_threads = [threading.Thread(target=p_worker, args=(i,), daemon=True) for i in range(2)]
    for thread in p_threads:
        thread.start()
    deadline = time.monotonic() + args.cohort_seconds
    for thread in d_threads + p_threads:
        thread.join(max(0.1, deadline - time.monotonic()))
    ok = (all(group and group.get("success") for group in results["decode"])
          and all(group and group.get("success") for group in results["prefill"]))
    return {"framework": "sglang", "decode": results["decode"],
            "prefill": results["prefill"],
            "generation_POST_attempted": counters["attempted"],
            "HTTP_terminal": counters["terminal"],
            "all_prefixes_success": ok}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["sglang"], default="sglang")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", default="r247")
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--tokenizer-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--http-seconds", type=int, default=900)
    parser.add_argument("--cohort-seconds", type=int, default=1000)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    out = {"project_id": "glm53-flash", "unit": "r247",
           "framework": args.framework, "run_id": args.run_id,
           "case": "two_74400_cold_prefixes_one_output_each",
           "candidate_delta": True, "original_46_overlay_unchanged": False,
           "formal_requests": 0, "performance_baseline": False,
           "profile_time_excluded_from_throughput": True,
           "cold_pairs": 0, "generation_POST_expected": 4,
           "retries": 0, "smoke_requests": 0, "flush": False, "rewarm": False}
    try:
        prefixes = manifest.load_prefixes(args.private_root)
        out["cold_pairs"] = len(prefixes)
        out["prefixes"] = [{"name": p["name"], "input_tokens": p["input_tokens"],
                            "prefix_ids_sha256": p["prefix_ids_sha256"]} for p in prefixes]
        out["dataset_manifest_sha256"] = manifest.MANIFEST_SHA256
        out["tokenizer_identity"] = manifest.load_tokenizer_identity(args.tokenizer_root)
        started = time.monotonic_ns()
        result = run_b(prefixes, args)
        result["cohort_started_ns"] = started
        result["cohort_terminal_ns"] = time.monotonic_ns()
        out.update(result)
        out["passed"] = bool(result["all_prefixes_success"]
                             and result["generation_POST_attempted"] == 4
                             and result["HTTP_terminal"] == 4
                             and out["cold_pairs"] == 2)
    except BaseException as exc:
        out.update(passed=False, error=repr(exc))
    finally:
        out["status"] = "PASS" if out.get("passed") else "FAIL"
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        tmp.replace(output)
        print(json.dumps({"passed": out.get("passed"), "status": out["status"],
                          "POST": out.get("generation_POST_attempted"),
                          "HTTP": out.get("HTTP_terminal")}))
    return 0 if out.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
