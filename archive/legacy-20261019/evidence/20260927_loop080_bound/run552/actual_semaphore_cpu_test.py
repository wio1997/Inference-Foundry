#!/usr/bin/env python3
"""Exercise the unchanged c12 client path with a 48-request fake SSE transport."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import time

source = Path(sys.argv[1])
validator = Path(sys.argv[2])
output = Path(sys.argv[3])

semaphore_trace = {}
RealSemaphore = asyncio.Semaphore
class AuditedSemaphore(RealSemaphore):
    async def acquire(self):
        result = await super().acquire()
        semaphore_trace.setdefault(asyncio.current_task(), {})["actual_acquired_by"] = time.monotonic_ns()
        return result
    def release(self):
        t = semaphore_trace.setdefault(asyncio.current_task(), {})
        t["actual_release_not_before"] = time.monotonic_ns()
        result = super().release()
        t["actual_release_completed_by"] = time.monotonic_ns()
        return result
asyncio.Semaphore = AuditedSemaphore
class Timeout:
    def __init__(self, **kwargs):
        pass

fake_aiohttp = types.SimpleNamespace(ClientTimeout=Timeout)
sys.modules["aiohttp"] = fake_aiohttp

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

client = module("frontier_client", source)
check = module("frontier_client_validate", validator)

class Response:
    status = 200
    headers = {}
    def __init__(self, rid):
        self.rid = rid
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    @property
    def content(self):
        return self
    async def iter_any(self):
        await asyncio.sleep(0.002)
        packets = [
            {"id": self.rid, "choices": [{"index": 0, "delta": {"reasoning": "x"}, "finish_reason": None}]},
            {"id": self.rid, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
            {"id": self.rid, "choices": [], "usage": {"prompt_tokens": 32851, "completion_tokens": 1024, "total_tokens": 33875}},
        ]
        wire = b"".join(b"data: " + json.dumps(p).encode() + b"\n\n" for p in packets)
        yield wire + b"data: [DONE]\n\n"

class Session:
    def __init__(self, **kwargs):
        pass
    async def __aenter__(self):
        return self
    async def __aexit__(self, *args):
        return False
    def post(self, url, json):
        semaphore_trace[asyncio.current_task()]["i"] = int(json["messages"][0]["content"][1:])
        assert url == "fake://no-network"
        assert json["temperature"] == 0 and json["max_tokens"] == 1024
        return Response(json["messages"][0]["content"])

client.aiohttp.ClientSession = Session
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    dataset = root / "dataset.jsonl"
    dataset.write_text("".join(json.dumps({"question": f"q{i}"}) + "\n" for i in range(48)))
    args = types.SimpleNamespace(dataset=str(dataset), offset=0, limit=48,
                                 out_dir=root / "out", concurrency=12, max_tokens=1024,
                                 url="fake://no-network", phase="measured")
    asyncio.run(client.acquire(args))
    summary = json.loads((args.out_dir / "summary.json").read_text())
    lineage = check.validate(summary["requests"], 12)
    assert summary["summary"]["success"] == 48
    assert lineage["successor_count"] == 36
    assert lineage["guaranteed_active_peak"] == 12
    assert lineage["unique_parent_edges_certified"] == 0
    actual = {x["i"]:x for x in semaphore_trace.values()}
    assert len(actual) == 48
    for row in summary["requests"]:
        t = actual[row["i"]]
        assert row["c12_acquire_attempt_monotonic_ns"] <= t["actual_acquired_by"] <= row["c12_acquired_monotonic_ns"]
        assert row["end_monotonic_ns"] <= t["actual_release_not_before"] <= t["actual_release_completed_by"] <= row["c12_release_completed_by_monotonic_ns"]
    # Invalid generated markers must reject even with otherwise valid SSE rows.
    import copy
    for label, mutate in [
       ("acquired_after_POST",lambda r:r.update(c12_acquired_monotonic_ns=r["start_monotonic_ns"]+1)),
       ("DONE_after_end",lambda r:r.update(sse_done_monotonic_ns=r["end_monotonic_ns"]+1)),
       ("release_upper_before_end",lambda r:r.update(c12_release_completed_by_monotonic_ns=r["end_monotonic_ns"]-1)),
       ("claim_unique_parent",lambda r:r.update(c12_unique_predecessor_certified=True)),
       ("noninteger_clock",lambda r:r.update(c12_acquired_monotonic_ns=float(r["c12_acquired_monotonic_ns"]))),
    ]:
        changed=copy.deepcopy(summary["requests"]);mutate(changed[0])
        try:check.validate(changed,12)
        except (ValueError,TypeError,KeyError):pass
        else:raise AssertionError(label)


result = {
    "actual_semaphore_acquire_release_brackets_verified_requests":48, "extra_marker_negatives":5, "status": "pass", "scope": "CPU fake transport only; no HTTP server, model, NPU or timing transfer",
    "client_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "validator_sha256": hashlib.sha256(validator.read_bytes()).hexdigest(),
    "requests": 48, "successors": lineage["successor_count"],
    "guaranteed_active_peak": lineage["guaranteed_active_peak"],
    "unique_parent_edges_certified": 0,
    "validator_synthetic": check.self_test(),
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result))
