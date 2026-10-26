"""DRAFT_PENDING_PRIMARY_REVIEW: unchanged L17 AISBench/native-ID long guards.

Six logical / 12 backend calls: two 80K known facts, cold/warm/exact-known-key
and cross-PDP storage; 2408 total output tokens if all pass.  The accepted
dataset SHA, AISBench version and native ID construction are unchanged.  Only
the run namespace and L17 source identity are new.  No extra native SDK
benchmark, SDK init, profile or model request is added here.
"""
import asyncio
import hashlib
import importlib.metadata as md
import json
import re
import struct
import time
import urllib.request
from pathlib import Path

import aiohttp
import numpy as np
import torch
from aiohttp import web
from ais_bench.benchmark.models.output import RequestOutput
from aisbench_glm53_token_ids import GLM53TokenIDAPI
from glm53_request_terminal_v1 import RequestTerminals
from glm53_sglang_pd_gateway_v3 import SGLangPDGateway, route
from glm53_storage_proof_v1 import storage_proof

RUN = Path("/run/glm53")
DATA = Path("/semantic")
BASE = "glm53-sem-sglang-L17-native-v1-20261019"
assert torch.__version__ == "2.10.0+cpu"
assert md.version("ais_bench_benchmark") == "3.1.0"
assert hashlib.sha256((DATA / "manifest.json").read_bytes()).hexdigest() == "ac8fbf5e4425c349440c85789d9a4be7137579a0b2a4072f8a2adf00c9bf6ece"


def default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def metrics(label):
    out = {}
    for node in [34, 35, 36, 41]:
        port = 30500 if node in [34, 35] else 30600
        try:
            with urllib.request.urlopen(f"http://172.16.12.{node}:{port}/metrics", timeout=5) as response:
                data = response.read()
            name = f"metrics-{label}-node{node}.txt"
            (RUN / name).write_bytes(data)
            out[node] = {
                "file": name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "selected_native_lines": [
                    line for line in data.decode().splitlines()
                    if line.startswith((
                        "sglang:prefill_effective_tokens_total{",
                        "sglang:hicache_host_used_tokens{",
                        "sglang:storage_prefetch_fulfilled_tokens_total{",
                        "sglang:storage_prefetch_dropped_tokens_total{",
                    ))
                ],
            }
        except Exception as error:
            out[node] = {"error": repr(error)}
    try:
        with urllib.request.urlopen("http://172.16.12.34:29782/metrics", timeout=5) as response:
            data = response.read()
        name = f"memcache-metrics-{label}.txt"
        (RUN / name).write_bytes(data)
        prefixes = (
            "memcache_segment_", "memcache_total_capacity_", "memcache_get_",
            "memcache_evict_", "memcache_bandwidth_", "memcache_object_",
            "memcache_allocated_",
        )
        out["MemCacheMeta"] = {
            "file": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "selected_native_lines": [line for line in data.decode().splitlines() if line.startswith(prefixes)],
        }
    except Exception as error:
        out["MemCacheMeta"] = {"error": repr(error)}
    return out


class Observed(GLM53TokenIDAPI):
    async def parse_stream_response(self, data, output):
        if data.get("id"):
            seen = output.extra_details_data.setdefault("glm53_server_response_ids", [])
            if data["id"] not in seen:
                seen.append(data["id"])
        if data.get("sglext") is not None:
            output.extra_details_data["glm53_native_sglext"] = data["sglext"]
        await super().parse_stream_response(data, output)


async def key_gate(index, record, namespace):
    request = {
        "index": index,
        "namespace": namespace,
        "prefix_ids": record["prefix_ids"],
        "prefix_sha256_le_u32": record["prefix_sha256_le_u32"],
    }
    pending = RUN / f"need-metadata-{index}.tmp"
    pending.write_text(json.dumps(request) + "\n")
    pending.replace(RUN / f"need-metadata-{index}.json")
    deadline = time.monotonic() + 180
    while not (RUN / f"metadata-gate-{index}.json").is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("Known-key metadata IPC gate timedout")
        await asyncio.sleep(0.2)
    result = json.loads((RUN / f"metadata-gate-{index}.json").read_text())
    assert result["rc"] == 0 and result["pass"], result
    return result


async def main():
    rows = []
    gateway_rows = []
    runner = None
    report = {
        "project_id": "glm53",
        "framework": "sglang",
        "functional_only": True,
        "qualification": "SEMANTIC_GUARDS_NOT_PERFORMANCE_OR_WHOLE_OUTPUT_EQUALITY",
        "namespace_base": BASE,
        "cases": rows,
        "gateway": gateway_rows,
        "metadata_gates": {},
        "cache_readback_guards": {},
        "metrics": {"before": metrics("before")},
        "native_versions": {name: md.version(name) for name in ["ais_bench_benchmark", "torch", "aiohttp"]},
        "proposed_logical_cases": 6,
        "proposed_backend_calls": 12,
        "proposed_output_tokens_if_all_pass": 2408,
        "retries": 0,
    }
    try:
        namespaces = [BASE + f"-case{i}-" + suffix for i in range(2) for suffix in ["cold", "warm"]]
        async with aiohttp.ClientSession(
            trust_env=False,
            timeout=aiohttp.ClientTimeout(total=900),
            connector=aiohttp.TCPConnector(limit=0, force_close=True),
        ) as upstream:
            terminals = RequestTerminals()
            gateway = SGLangPDGateway(upstream, namespaces, gateway_rows, terminal_registry=terminals)
            runner = web.AppRunner(gateway.application())
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            gateway_port = site._server.sockets[0].getsockname()[1]
            async with aiohttp.ClientSession(
                trust_env=False,
                timeout=aiohttp.ClientTimeout(total=900),
                connector=aiohttp.TCPConnector(limit=0, force_close=True),
            ) as client:
                for index in range(2):
                    record = json.loads((DATA / f"semantic-{index}.json").read_text())
                    ids = record["input_ids"]
                    prefix = record["prefix_ids"]
                    assert len(ids) == 80000 and len(prefix) == 74400 and ids[:74400] == prefix
                    for values, key in [(ids, "ids_sha256_le_u32"), (prefix, "prefix_sha256_le_u32")]:
                        assert hashlib.sha256(struct.pack("<" + "I" * len(values), *values)).hexdigest() == record[key]
                    base = BASE + f"-case{index}"
                    normal = index * 2
                    alternate = normal + 1
                    specs = [
                        ("cold-PD", normal, 80000, 600, base + "-cold", "127.0.0.1", gateway_port),
                        ("prefix-warmup-PD", normal, 74400, 1, base + "-warm", "127.0.0.1", gateway_port),
                        ("cross-DP-storage-PD", alternate, 80000, 600, base + "-warm", "127.0.0.1", gateway_port),
                    ]
                    for label, prefix_id, inputs, outputs, namespace, host, port in specs:
                        if label == "cross-DP-storage-PD":
                            report["metadata_gates"][index] = await key_gate(index, record, namespace)
                        model = Observed(
                            path="",
                            model="glm53-w8a8",
                            stream=True,
                            host_ip=host,
                            host_port=port,
                            run_namespace=namespace,
                            expected_input_tokens=inputs,
                            expected_output_tokens=outputs,
                        )
                        output = RequestOutput(perf_mode=True)
                        output.uuid = f"{base}-p{prefix_id:02d}-{label}"
                        terminals.reserve(output.uuid)
                        begin = time.monotonic_ns()
                        await model.generate(prefix if inputs == 74400 else ids, outputs, output, session=client)
                        end = time.monotonic_ns()
                        terminal = await terminals.wait_terminal(output.uuid, 15)
                        gateway_row = terminal["gateway"]
                        usage = output.extra_details_data.get("glm53_service_usage")
                        valid = output.extra_perf_data.get("glm53_valid_output_chunk_times", [])
                        count_pass = bool(
                            output.success
                            and usage
                            and usage.get("prompt_tokens") == inputs
                            and usage.get("completion_tokens") == outputs
                            and output.input_tokens == inputs
                            and output.output_tokens == outputs
                            and output.extra_details_data.get("glm53_finish_reason") == "length"
                            and valid
                        )
                        answers = re.findall(r"FINAL_ANSWER:\s*([0-9]{12})(?![0-9])", output.content)
                        semantic_pass = None if outputs == 1 else bool(answers and answers[0] == record["expected_answer"])
                        output.input = None
                        row = {
                            "semantic_case": record["case"],
                            "case": label,
                            "request_id": output.uuid,
                            "route_prefix_id": prefix_id,
                            "namespace": namespace,
                            "required_input": inputs,
                            "required_output": outputs,
                            "outer_dispatch_monotonic_ns": begin,
                            "outer_terminal_monotonic_ns": end,
                            "count_pass": count_pass,
                            "pd_lifecycle_pass": gateway_row["status"] == "D_STREAM_COMPLETE",
                            "gateway_terminal": terminal,
                            "expected_answer": record["expected_answer"],
                            "observed_answer_fields": answers,
                            "semantic_pass": semantic_pass,
                            "output_text_sha256": hashlib.sha256(output.content.encode()).hexdigest(),
                            "native": output.to_dict(),
                        }
                        rows.append(row)
                        if label == "cross-DP-storage-PD":
                            proof = storage_proof(
                                len(prefix),
                                128,
                                gateway_row.get("prefill_sglext"),
                                output.extra_details_data.get("glm53_native_sglext"),
                            )
                            report["cache_readback_guards"][index] = proof
                            row["native_storage_proof"] = proof
                        report["metrics"][f"case{index}-{label}"] = metrics(f"case{index}-{label}")
                        (RUN / "semantic-progress.json").write_text(json.dumps(report, default=default, indent=2) + "\n")
                        print("GLM53_SEMANTIC_CASE=" + json.dumps(row, default=default), flush=True)
                        assert count_pass and row["pd_lifecycle_pass"] and semantic_pass is not False, (
                            "Long retrieval/count/PD lifecycle gate failed; preserve raw, no baseline"
                        )
                        if label == "cross-DP-storage-PD":
                            assert proof["pass"], "Native P/D storage attribution incomplete; no cache-readback qualification"
                await gateway.wait_idle(10)
                assert not terminals.pending
                assert len(rows) == 6 and len(gateway_rows) == 6
                assert all(row["status"] == "D_STREAM_COMPLETE" for row in gateway_rows)
        report["semantic_pass"] = all(row["semantic_pass"] is not False for row in rows)
        report["whole_output_hash_equality"] = {}
        for index in range(2):
            selected = rows[index * 3:(index + 1) * 3]
            reference = selected[0]["output_text_sha256"]
            report["whole_output_hash_equality"][index] = {
                row["case"]: row["output_text_sha256"] == reference
                for row in selected
                if row["required_output"] == 600
            }
    finally:
        if runner:
            await runner.cleanup()
        report["external_backend_HTTP_attempts"] = sum(
            int("prefill_dispatch_monotonic_ns" in row) + int("decode_dispatch_monotonic_ns" in row)
            for row in gateway_rows
        )
        p_success = [
            row for row in gateway_rows
            if row.get("prefill_http_status") == 200
            and isinstance(row.get("prefill_usage"), dict)
            and row["prefill_usage"].get("completion_tokens") == 1
        ]
        d_success = [row for row in rows if row["count_pass"]]
        report["successful_backend_inferences"] = len(p_success) + len(d_success)
        report["actual_successful_output_tokens"] = len(p_success) + sum(row["required_output"] for row in d_success)
        (RUN / "semantic-result.json").write_text(json.dumps(report, default=default, indent=2) + "\n")
        print("GLM53_SEMANTIC_RESULT=" + json.dumps(report, default=default), flush=True)


asyncio.run(main())
