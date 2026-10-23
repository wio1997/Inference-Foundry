#!/usr/bin/env python3
"""E21 pre-window idle gate.

DI: /status must report ready, ranks_ready=8, server_idle=true and
inflight/running/waiting=0 for two consecutive polls.

vLLM: /metrics must expose actual per-engine num_requests_running and
num_requests_waiting samples for all eight engines at 0.0 for two consecutive
polls. A missing gauge is never treated as zero.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request


def http_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def http_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def di_state(base: str) -> dict:
    status = http_json(base.rstrip("/") + "/status")
    ok = bool(
        status.get("ready") is True
        and status.get("ranks_ready") == 8
        and status.get("server_idle") is True
        and status.get("inflight") == 0
        and status.get("running") == 0
        and status.get("waiting") == 0
        and not status.get("fatal")
    )
    return {"ok": ok, "scope": "deepinfer_ready_idle", "status": status}


def gauge_map(text: str, name: str) -> dict:
    found = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith("vllm:" + name + "{"):
            continue
        match = re.search(r'engine="(\d+)"', line)
        if not match:
            continue
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        found[int(match.group(1))] = value
    return found


def vllm_state(base: str) -> dict:
    health = None
    health_error = None
    try:
        health = http_text(base.rstrip("/") + "/health", timeout=5.0).strip()
    except Exception as exc:  # noqa: BLE001
        health_error = repr(exc)
    text = http_text(base.rstrip("/") + "/metrics", timeout=15.0)
    running = gauge_map(text, "num_requests_running")
    waiting = gauge_map(text, "num_requests_waiting")
    engines = set(range(8))
    ok = (
        health_error is None
        and set(running) == engines
        and set(waiting) == engines
        and all(value == 0.0 for value in running.values())
        and all(value == 0.0 for value in waiting.values())
    )
    return {
        "ok": ok,
        "scope": "vllm_all8_queue_gauges_present_zero",
        "health": health,
        "health_error": health_error,
        "running": running,
        "waiting": waiting,
        "missing_running": sorted(engines - set(running)),
        "missing_waiting": sorted(engines - set(waiting)),
        "raw_tail": text[-1500:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=("di", "vllm"), required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--consecutive", type=int, default=2)
    ap.add_argument("--json-out")
    args = ap.parse_args()

    deadline = time.monotonic() + args.timeout
    started = time.time()
    consecutive = 0
    attempts = 0
    last = None
    while time.monotonic() < deadline:
        attempts += 1
        try:
            last = di_state(args.base_url) if args.framework == "di" else vllm_state(args.base_url)
        except Exception as exc:  # noqa: BLE001
            last = {"ok": False, "error": repr(exc), "scope": "request_error"}
        if last.get("ok"):
            consecutive += 1
            if consecutive >= args.consecutive:
                break
        else:
            consecutive = 0
        time.sleep(0.25)
    result = {
        "framework": args.framework,
        "base_url": args.base_url,
        "achieved": bool(last and last.get("ok") and consecutive >= args.consecutive),
        "attempts": attempts,
        "consecutive": consecutive,
        "waited_s": time.time() - started,
        "last": last,
    }
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["achieved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
