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

source = Path(sys.argv[1])
validator = Path(sys.argv[2])
output = Path(sys.argv[3])

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

result = {
    "status": "pass", "scope": "CPU fake transport only; no HTTP server, model, NPU or timing transfer",
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
