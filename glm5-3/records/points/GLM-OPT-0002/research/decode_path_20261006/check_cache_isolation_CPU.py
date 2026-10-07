"""Actual benchmark/API/hash AST checks; no HTTP, model import or NPU.

Tokenization, metrics arrival and model/network execution are fixture doubles.
"""
import ast
import asyncio
import errno
from datetime import datetime
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace as NS
import types
import uuid

HERE = Path(__file__).resolve().parent
ADAPTER = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parents[4] / "adapters/aisbench"


def actual_function(path, name, globals_dict):
    source = path.read_text()
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    node.decorator_list = []
    exec(compile("from __future__ import annotations\n" + ast.unparse(node), str(path), "exec"), globals_dict)
    return globals_dict[name]


def main():
    spec = importlib.util.spec_from_file_location("cache_collector_actual", ADAPTER / "hit_rate_collector.py")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    endpoints = ["P:9081", "D:9900"]
    c = collector.HitRateCollector(endpoints)
    zero = {pod: {0: dict(hbm_queries=0, hbm_hits=0, ext_queries=0, ext_hits=0)} for pod in endpoints}
    final = {"P:9081": {0: dict(hbm_queries=200, hbm_hits=186, ext_queries=200, ext_hits=0)}, "D:9900": {0: dict(hbm_queries=200, hbm_hits=0, ext_queries=200, ext_hits=200)}}
    rates = c.compute_hit_rate(zero, final)
    assert rates["per_endpoint"]["P:9081"]["dp0"]["hbm_hit_rate"] == .93
    assert rates["per_endpoint"]["D:9900"]["dp0"]["ext_hit_rate"] == 1
    assert rates["aggregated"]["hbm_hit_rate"] == .465 and rates["aggregate_mixes_endpoints"]
    assert c.compute_hit_rate(zero, zero)["per_endpoint"]["P:9081"]["dp0"]["hbm_hit_rate"] is None
    rejected = 0
    for before, after in ((final, zero), ({}, final), (zero, {"P:9081": {}})):
        try:
            c.compute_hit_rate(before, after)
        except ValueError:
            rejected += 1
    raw = "\n".join(f'vllm:{name}_total{{engine="0",model_name="glm-53"}} {value}' for name, value in (("prefix_cache_queries", 200), ("prefix_cache_hits", 186), ("external_prefix_cache_queries", 200), ("external_prefix_cache_hits", 0)))
    assert collector._parse_prefix_counters(raw) == final["P:9081"]
    for bad in (raw.rsplit("\n", 1)[0], raw + "\n" + raw.splitlines()[0], raw.replace(" 186", " -1"), raw.replace(" 186", " nan")):
        try:
            collector._parse_prefix_counters(bad)
        except ValueError:
            rejected += 1
    assert rejected == 7

    # The deployed legacy driver uses os.system; never execute its shell phase
    # from this CPU fixture. The repository driver uses execute_aisbench_phase.
    fixture_os = NS(**{k: v for k, v in vars(os).items() if k != "system"}, system=lambda *args: 0)
    g = dict(os=fixture_os, errno=errno, re=re, json=json, uuid=uuid, Path=Path, argparse=NS(), logger=logging.getLogger("CPU"), __file__=str(ADAPTER / "prefix_bench.py"))
    actual_function(ADAPTER / "prefix_bench.py", "symlink_force", g)
    modify = actual_function(ADAPTER / "prefix_bench.py", "modify_aisbench_api", g)
    build = actual_function(ADAPTER / "result_writer.py", "build_result_row", dict(datetime=datetime, json=json))
    api_globals = {"ROLE_MAP": {}}
    request_body = actual_function(HERE / "cache_sources/aisbench__vllm_custom_api_chat.py", "get_request_body", api_globals)
    configured = []
    rows = []

    def configure(**kwargs):
        modify(**kwargs)
        module = ast.parse(Path("temp_api.py").read_text())
        module.body = [n for n in module.body if not isinstance(n, ast.ImportFrom)]
        env = {"VLLMCustomAPIChatStream": object, "VLLMCustomAPIChat": object}
        exec(compile(module, "generated_actual_config", "exec"), env)
        config = env["models"][0]
        fake = NS(generation_kwargs=config["generation_kwargs"], response_anomaly_enabled=False, _resolve_lora_model_name=lambda output: None, model="glm-53", stream=True, logger=logging.getLogger("CPU"))
        body = asyncio.run(request_body(fake, "dynamic prompt", kwargs["output_len"], NS()))
        assert body["cache_salt"] == kwargs["cache_salt"] and body["max_tokens"] == kwargs["output_len"]
        configured.append(body)

    class FixtureCollector(collector.HitRateCollector):
        def __init__(self, pods):
            super().__init__(pods)
            self.index = 0
        def snapshot(self):
            value = (zero, zero, zero, final)[self.index]
            self.index += 1
            return value
        def print_hit_rate_table(self, result):
            pass

    def dataset(**kwargs):
        Path("prefix.jsonl").write_text(json.dumps({"question": "p" * 93}) + "\n")
        Path("full.jsonl").write_text("\n".join(json.dumps({"question": "p" * 93 + f"s{i:06d}"}) for i in range(2)) + "\n")
        return "prefix.jsonl", "full.jsonl"

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = NS(from_pretrained=lambda *a, **k: NS(encode=lambda text, **kw: list(text)))
    sys.modules["transformers"] = transformers
    g.update(parse_prefix_ratio=float, resolve_pod_info=lambda args: endpoints, HitRateCollector=FixtureCollector, ROUND_OVERRIDABLE_KEYS=[], create_prefix_dataset=dataset, prepare_aisbench_dataset_dir=lambda path: path, generate_aisbench_command=lambda *a: [], modify_aisbench_api=configure, link_dataset_to_aisbench=lambda *a: None, execute_aisbench_phase=lambda *a: None, parse_aisbench_log=lambda *a: ({}, "fixture"), validate_phase_details=lambda *a: None, archive_log=lambda *a: None, build_result_row=build, write_csv=lambda *a: None, write_jsonl=lambda row, target: rows.append(row))
    run = actual_function(ADAPTER / "prefix_bench.py", "run_single_round", g)
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as temp:
        os.chdir(temp)
        try:
            work = Path(temp) / "benchmark"
            (work / "ais_bench/benchmark/configs/models/vllm_api").mkdir(parents=True)
            args = NS(repeat_rate=.93, dataset=None, dataset_path=temp, model_path="tokenizer", input_len=100, data_num=2, dp=1, seed=1, prefix_num=1, length_mean=None, length_std=None, length_min=None, length_max=None, work_path=str(work), summarizer="default_perf", output_dir=temp, model_name="glm-53", host_ip="127.0.0.1", host_port=9900, url="", request_rate=0, test_type="stream", enable_think=False, concurrency=1, output_len=600, npu_num=32, result_csv="csv", result_jsonl="jsonl")
            run(args, 1)
            run(args, 2)
            saved = [json.loads(Path(f"round_{i}_cache.json").read_text()) for i in (1, 2)]
            assert saved[0]["cache_salt"] != saved[1]["cache_salt"]
            assert all(len(z["snapshots"]) == 4 and z["observed_hit_rate"] == rates["per_endpoint"] for z in saved)
        finally:
            os.chdir(original_cwd)
    assert len(configured) == len(rows) == 4
    assert configured[0]["cache_salt"] == configured[1]["cache_salt"] != configured[2]["cache_salt"] == configured[3]["cache_salt"]
    assert all(json.loads(row["cache_by_endpoint"]) for row in rows)

    h = dict(_gen_mm_extra_hash_keys=lambda request, start, end, index: ([], index), _gen_lora_extra_hash_keys=lambda request: [], _gen_prompt_embeds_extra_hash_keys=lambda request, start, end: [], NONE_HASH=b"same-initial-hash", BlockHash=lambda value: value)
    extra = actual_function(HERE / "cache_sources/vllm__vllm__v1__core__kv_cache_utils.py", "generate_block_hash_extra_keys", h)
    block = actual_function(HERE / "cache_sources/vllm__vllm__v1__core__kv_cache_utils.py", "hash_block_tokens", h)
    def chain(salt):
        parent = None
        hashes = []
        request = NS(cache_salt=salt)
        for i in range(200):
            keys, _ = extra(request, i * 128, (i + 1) * 128, 0)
            parent = block(lambda value: hashlib.sha256(repr(value).encode()).digest(), parent, range(i * 128, (i + 1) * 128), keys)
            hashes.append(parent)
        return hashes
    old_full = set(chain(None))
    assert sum(value in old_full for value in chain(None)) == 200
    warm_prefix = set(chain(configured[0]["cache_salt"])[:186])
    assert sum(value in warm_prefix for value in chain(configured[1]["cache_salt"])) == 186
    assert not set(chain(configured[2]["cache_salt"])) & set(chain(configured[0]["cache_salt"]))
    result = dict(passed=True, CPU_only=True, model_requests=0, actual_AISBench_body_preserves_salt=True, rounds=2, same_warm_full_salt=True, cross_round_salt_distinct=True, old_unsalted_full_reuse_blocks=200, controlled_same_namespace_prefix_blocks=186, cross_round_matching_blocks=0, per_endpoint_P_D_engine0_separate=True, invalid_metric_cases=rejected, limitations="Network/model/tokenizer/cache-device behavior doubled; no NPU or measured93% claim.")
    result["adapter"] = str(ADAPTER)
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "cache_isolation_CPU_result.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
