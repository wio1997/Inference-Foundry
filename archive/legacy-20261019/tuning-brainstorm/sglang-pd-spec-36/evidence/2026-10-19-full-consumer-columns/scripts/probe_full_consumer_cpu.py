#!/usr/bin/env python3
"""L005: exact Ascend sender vs frozen range sender with the real Mooncake consumer.

CPU fixture only.  The real ``MooncakeKVManager._transfer_data`` body runs, but
``self.engine.batch_transfer_sync`` is a fake no-op that never dereferences an
address, converts a native type, copies payload bytes, or touches a device.
"""

from __future__ import annotations

import ast
import concurrent.futures
import copy
import datetime
import gc
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import statistics
import sys
import time
import traceback
import types

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = pathlib.Path(__file__).resolve()
OUT = ROOT / "runs" / "L005-R001-full-consumer"
SRC = ROOT / "sources" / "installed" / "python" / "sglang" / "srt"

SESSION = "CPU_FIXTURE_NO_TRANSFER"
LAYERS = 78
RANDOM_SEED = 20261019
BOOTSTRAP_SEED = 20261019
BOOTSTRAP_RESAMPLES = 4000

INPUTS = {
    "scripts/probe_ascend_descriptor_cpu.py": ROOT / "scripts" / "probe_ascend_descriptor_cpu.py",
    "runs/L004-R001-descriptor-cpu/candidate-method.py": ROOT / "runs" / "L004-R001-descriptor-cpu" / "candidate-method.py",
    "runs/L004-R001-descriptor-cpu/result.json": ROOT / "runs" / "L004-R001-descriptor-cpu" / "result.json",
    "sources/installed/python/sglang/srt/disaggregation/ascend/conn.py": SRC / "disaggregation" / "ascend" / "conn.py",
    "sources/installed/python/sglang/srt/disaggregation/common/utils.py": SRC / "disaggregation" / "common" / "utils.py",
    "sources/installed/python/sglang/srt/disaggregation/mooncake/conn.py": SRC / "disaggregation" / "mooncake" / "conn.py",
}

SCOPE = (
    "Local CPU fixture only: PP=1, non-hybrid MLA flag false, custom_mem_pool=false, "
    "78 layers, alternating item_len 128/64, no-op envelope validation, and a fake "
    "batch_transfer_sync engine. The exact real MooncakeKVManager._transfer_data body "
    "executes zip(*transfer_blocks) and list() conversion; the fake engine does no native "
    "conversion, copy, address dereference, device access, or network I/O. No NPU, no model, "
    "no PD serving baseline. Production SGLang source is unchanged. Consumer time is Python "
    "call/boxing time, NOT network or NPU time. Envelope validation, NPU/native engine, "
    "distributed PD, and the custom_mem_pool cancellation branch are NOT tested. Overall "
    "serving evidence for task sglang-pd-spec-36 remains E0."
)

LIMITATIONS = [
    "Fake engine: batch_transfer_sync returns immediately; no payload is read or written.",
    "No envelope validation semantics are exercised; the fixture validator is a no-op.",
    "custom_mem_pool=false, so the executor/cancellation branch is not executed or tested.",
    "No NPU, Ascend transfer engine, model, network, or PD service is involved.",
    "Consumer diagnostic time excludes native transport and is not network time.",
    "Diagnostic wrappers and the gc.callbacks observer add overhead; diagnostic numbers are separate from clean samples.",
    "All 151 correctness checks use the fixed 78-layer fixture; the separate L004 replay is a transcription preflight, not a sender run.",
]

DECLARATIONS = [
    "Machine-local CPU fixture only; no NPU or device measurement.",
    "The fake engine is a controlled no-op and performs no native conversion or copy.",
    "No model weights, no network transfer, and no PD baseline were used.",
    "Production SGLang sources were not modified; input hashes are rechecked after the run.",
    "The real MooncakeKVManager._transfer_data body actually executes zip/list conversion.",
    "This script reports descriptive numbers only; it does not accept or reject an optimization.",
    "Overall serving evidence for task sglang-pd-spec-36 remains E0.",
]

RUN = {
    "stage": "start",
    "current": None,
    "out": None,
    "inputs": {},
    "checks": [],
    "bench": [],
    "diag": None,
    "result_sha256": None,
}

RUNTIME = {
    "transfer_fn": None,
    "base_sender": None,
    "candidate_sender": None,
    "real_group": None,
    "range_metadata": None,
    "base_env": None,
    "candidate_env": None,
}


def ensure(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(pathlib.Path(path).read_bytes())


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def find_top_level_function(tree, name):
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    ensure(len(matches) == 1, f"expected exactly one top-level def {name}, found {len(matches)}")
    return matches[0]


def find_method(tree, class_name, method_name):
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    ensure(len(classes) == 1, f"expected exactly one class {class_name}, found {len(classes)}")
    matches = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    ]
    ensure(
        len(matches) == 1,
        f"expected exactly one method {class_name}.{method_name}, found {len(matches)}",
    )
    return matches[0]


def ast_dump_sha256(node):
    dumped = ast.dump(node, annotate_fields=True, include_attributes=False)
    return sha256_bytes(dumped.encode("utf-8"))


def compile_nodes(nodes):
    body = [
        ast.ImportFrom(
            module="__future__",
            names=[ast.alias(name="annotations")],
            level=0,
        )
    ]
    body.extend(copy.deepcopy(node) for node in nodes)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, "<L005-isolated-ast>", "exec")


def method_metadata(label, node, path):
    ensure(not node.decorator_list, f"{label} unexpectedly has decorators")
    return {
        "label": label,
        "file": path,
        "line": node.lineno,
        "end_line": node.end_lineno,
        "ast_dump_sha256": ast_dump_sha256(node),
        "ast_dump_format": "ast.dump(annotate_fields=True, include_attributes=False)",
    }


class CaptureEngine:
    """Correctness-only fake engine; copies and keeps the three ordered lists."""

    def __init__(self, status=0):
        self.status = status
        self.calls = 0
        self.session_ids = []
        self.src_lists = []
        self.dst_lists = []
        self.length_lists = []
        self.return_codes = []
        self.three_lists_are_python_lists = True

    def batch_transfer_sync(self, mooncake_session_id, src_addrs, dst_addrs, lengths):
        self.calls += 1
        self.session_ids.append(mooncake_session_id)
        self.src_lists.append(list(src_addrs))
        self.dst_lists.append(list(dst_addrs))
        self.length_lists.append(list(lengths))
        self.return_codes.append(self.status)
        self.three_lists_are_python_lists = self.three_lists_are_python_lists and all(
            isinstance(value, list) for value in (src_addrs, dst_addrs, lengths)
        )
        return self.status


class CountOnlyEngine:
    """Timing fake engine: no copy, no iteration, only a call counter."""

    __slots__ = ("calls",)

    def __init__(self):
        self.calls = 0

    def batch_transfer_sync(self, mooncake_session_id, src_addrs, dst_addrs, lengths):
        self.calls += 1
        return 0


def make_fixture(engine, layers=LAYERS):
    transfer_fn = RUNTIME["transfer_fn"]
    ensure(transfer_fn is not None, "transfer function not extracted")
    fixture = types.SimpleNamespace(
        pp_size=1,
        is_hybrid_mla_backend=False,
        is_mla_backend=False,
        enable_custom_mem_pool=False,
    )
    fixture.kv_args = types.SimpleNamespace(
        kv_data_ptrs=[2**40 + i * 2**32 for i in range(layers)],
        kv_item_lens=[128 if i % 2 == 0 else 64 for i in range(layers)],
    )
    fixture.dst = [2**46 + i * 2**32 for i in range(layers)]
    fixture._validate_envelope_kv_layout = lambda *args: None
    fixture.engine = engine
    fixture._transfer_data = types.MethodType(transfer_fn, fixture)
    return fixture


def call_sender(fn, fixture, src, dst, hisparse=None):
    return fn(
        fixture,
        SESSION,
        src,
        fixture.dst,
        dst,
        None,
        dst_device_kv_indices=hisparse,
    )


def execute_case(fn, fixture, src, dst, hisparse=None):
    try:
        status = call_sender(fn, fixture, src, dst, hisparse)
        exception = None
    except Exception as exc:  # noqa: BLE001 - exact exception capture is the measurement
        status = None
        exception = (type(exc).__name__, str(exc))
    engine = fixture.engine
    return {
        "status": status,
        "exception": exception,
        "native_calls": engine.calls,
        "session_ids": list(engine.session_ids),
        "return_codes": list(engine.return_codes),
        "src": engine.src_lists[0] if engine.src_lists else None,
        "dst": engine.dst_lists[0] if engine.dst_lists else None,
        "lengths": engine.length_lists[0] if engine.length_lists else None,
        "three_lists_are_python_lists": engine.three_lists_are_python_lists,
    }


def oracle(fixture, src, dst):
    """Independent per-page oracle copied from the L004 frozen generation logic."""
    if not len(src) or not len(dst):
        return []
    if len(src) != len(dst):
        raise ValueError
    blocks = []
    for src_ptr, dst_ptr, width in zip(
        fixture.kv_args.kv_data_ptrs,
        fixture.dst,
        fixture.kv_args.kv_item_lens,
    ):
        previous = None
        for src_index, dst_index in zip(src, dst):
            src_index, dst_index = int(src_index), int(dst_index)
            if (
                previous is not None
                and src_index == previous[0] + 1
                and dst_index == previous[1] + 1
            ):
                last = blocks[-1]
                blocks[-1] = (last[0], last[1], last[2] + width)
            else:
                blocks.append(
                    (
                        src_ptr + src_index * width,
                        dst_ptr + dst_index * width,
                        width,
                    )
                )
            previous = (src_index, dst_index)
    return blocks


def oracle_columns(blocks):
    return (
        [block[0] for block in blocks],
        [block[1] for block in blocks],
        [block[2] for block in blocks],
    )


def build_cases():
    cases = []
    for n in (0, 1, 2, 8, 31, 4096):
        src = np.arange(10000, 10000 + n, dtype=np.int32)
        dst = np.arange(20000, 20000 + n, dtype=np.int32)
        cases.append(("contiguous-" + str(n), src, dst, LAYERS, None))
        cases.append(("fragmented-" + str(n), src * 2, dst * 3, 3, None))
        cases.append(("destination-gaps-" + str(n), src, dst * 2, 3, None))
    cases.append(
        (
            "large-contiguous-32768",
            np.arange(10000, 10000 + 32768, dtype=np.int32),
            np.arange(50000, 50000 + 32768, dtype=np.int32),
            LAYERS,
            None,
        )
    )
    cases.append(
        (
            "empty-source-nonempty-destination",
            np.array([], dtype=np.int32),
            np.array([7], dtype=np.int32),
            3,
            None,
        )
    )
    cases.append(
        (
            "nonempty-source-empty-destination",
            np.array([7], dtype=np.int32),
            np.array([], dtype=np.int32),
            3,
            None,
        )
    )
    cases.append(
        (
            "mismatched-nonempty",
            np.array([7, 8], dtype=np.int32),
            np.array([9], dtype=np.int32),
            3,
            None,
        )
    )
    cases.append(
        (
            "hisparse-rejection",
            np.array([7], dtype=np.int32),
            np.array([9], dtype=np.int32),
            3,
            np.array([9], dtype=np.int32),
        )
    )
    rng = random.Random(RANDOM_SEED)
    for i in range(128):
        n = rng.randrange(1, 100)
        src = np.array([10000 + rng.randrange(256) for _ in range(n)], dtype=np.int32)
        dst = np.array([20000 + rng.randrange(256) for _ in range(n)], dtype=np.int32)
        if i % 4 == 0:
            src = src.astype(np.int64)
            dst = dst.astype(np.int64)
        if i % 4 == 1:
            src = np.repeat(src, 2)[::2]
            dst = np.repeat(dst, 2)[::2]
            src.setflags(write=False)
            dst.setflags(write=False)
        cases.append(("random-" + str(i), src, dst, 3, None))
    ensure(len(cases) == 151, f"expected 151 L004 cases, built {len(cases)}")
    return cases


BENCH_CASES = (
    ("short-contiguous", 8, False, 128),
    ("long-contiguous", 4096, False, 16),
    ("large-contiguous", 32768, False, 4),
    ("fragmented", 4096, True, 1),
)


def bench_arrays(n, fragmented):
    src = np.arange(10000, 10000 + n, dtype=np.int32)
    dst = np.arange(100000, 100000 + n, dtype=np.int32)
    if fragmented:
        src = src * 2
        dst = dst * 3
    return src, dst


def measure_sample(fn, fixture, src, dst, repeats):
    engine = fixture.engine
    calls_before = engine.calls
    gc_before = gc.get_stats()
    wall_start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    for _ in range(repeats):
        status = call_sender(fn, fixture, src, dst)
        if status != 0:
            raise RuntimeError(f"timing call returned {status}, expected 0")
    cpu_stop = time.thread_time_ns()
    wall_stop = time.perf_counter_ns()
    gc_after = gc.get_stats()
    calls_after = engine.calls
    return {
        "wall_ns_per_call": (wall_stop - wall_start) / repeats,
        "thread_cpu_ns_per_call": (cpu_stop - cpu_start) / repeats,
        "repeats": repeats,
        "native_calls": calls_after - calls_before,
        "gc_collections_delta": [
            gc_after[gen]["collections"] - gc_before[gen]["collections"]
            for gen in range(3)
        ],
        "gc_collected_delta": [
            gc_after[gen]["collected"] - gc_before[gen]["collected"]
            for gen in range(3)
        ],
    }


def describe(samples, key):
    values = [float(sample[key]) for sample in samples]
    ensure(values, "cannot describe an empty sample list")
    q25, q75 = np.percentile(values, [25, 75])
    return {
        "n": len(values),
        "median_ns": float(statistics.median(values)),
        "iqr_ns": float(q75 - q25),
        "min_ns": float(min(values)),
        "max_ns": float(max(values)),
    }


def paired_ratio(pairs, key):
    logs = [math.log(float(a[key]) / float(b[key])) for a, b in pairs]
    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap = sorted(
        math.exp(statistics.mean(rng.choices(logs, k=len(logs))))
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "n_pairs": len(pairs),
        "ratio_of_geomeans": math.exp(statistics.mean(logs)),
        "bootstrap_95_interval": [bootstrap[99], bootstrap[3899]],
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_percentile_indices": [99, 3899],
    }


def gc_state():
    return {
        "enabled": gc.isenabled(),
        "threshold": list(gc.get_threshold()),
        "count": list(gc.get_count()),
        "callbacks": len(gc.callbacks),
        "stats": gc.get_stats(),
    }


def raw_summary(values):
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "median_ns": float(statistics.median(values)),
        "min_ns": float(min(values)),
        "max_ns": float(max(values)),
    }


def make_group_wrapper(real_group, observations):
    def wrapper(src_indices, dst_indices):
        wall_start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        result = real_group(src_indices, dst_indices)
        cpu_stop = time.thread_time_ns()
        wall_stop = time.perf_counter_ns()
        observations.append(
            {
                "wall_ns": wall_stop - wall_start,
                "thread_cpu_ns": cpu_stop - cpu_start,
            }
        )
        return result

    return wrapper


def make_consumer_wrapper(real_transfer, observations):
    def wrapper(self, mooncake_session_id, transfer_blocks):
        wall_start = time.perf_counter_ns()
        cpu_start = time.thread_time_ns()
        result = real_transfer(self, mooncake_session_id, transfer_blocks)
        cpu_stop = time.thread_time_ns()
        wall_stop = time.perf_counter_ns()
        observations.append(
            {
                "wall_ns": wall_stop - wall_start,
                "thread_cpu_ns": cpu_stop - cpu_start,
            }
        )
        return result

    return wrapper


def run_correctness(cases, l004_result):
    ensure(len(l004_result["cases"]) == len(cases), "L004 case count changed")
    capture_checks = []
    empty_cases = 0
    nonempty_cases = 0
    exception_cases = 0
    all_lists_are_python_lists = True

    for index, (label, src, dst, layers, hisparse) in enumerate(cases):
        RUN["current"] = {"stage": "correctness", "case": label}
        reference = l004_result["cases"][index]
        ensure(reference["case"] == label, f"L004 case order mismatch at {label}")
        ensure(reference["source_count"] == len(src), f"{label}: source count mismatch")
        ensure(reference["destination_count"] == len(dst), f"{label}: destination count mismatch")
        ensure(reference["layers"] == layers, f"{label}: layer count mismatch")
        reference_exception = (
            tuple(reference["exception"]) if reference["exception"] is not None else None
        )

        baseline_fixture = make_fixture(CaptureEngine(), LAYERS)
        candidate_fixture = make_fixture(CaptureEngine(), LAYERS)
        baseline = execute_case(RUNTIME["base_sender"], baseline_fixture, src, dst, hisparse)
        candidate = execute_case(
            RUNTIME["candidate_sender"], candidate_fixture, src, dst, hisparse
        )

        ensure(
            baseline["exception"] == candidate["exception"],
            f"{label}: variant exception mismatch {baseline['exception']} vs {candidate['exception']}",
        )
        ensure(
            baseline["exception"] == reference_exception,
            f"{label}: exception {baseline['exception']} does not match L004 {reference_exception}",
        )

        if baseline["exception"] is None:
            expected = oracle(baseline_fixture, src, dst)
            expected_src, expected_dst, expected_lengths = oracle_columns(expected)
            expected_calls = 1 if expected else 0
            ensure(baseline["status"] == 0 and candidate["status"] == 0, f"{label}: nonzero status")
            ensure(
                baseline["native_calls"] == candidate["native_calls"] == expected_calls,
                f"{label}: native call count {baseline['native_calls']}/{candidate['native_calls']}, "
                f"expected {expected_calls}",
            )
            ensure(
                baseline["session_ids"] == candidate["session_ids"]
                == [SESSION] * expected_calls,
                f"{label}: session id mismatch",
            )
            ensure(
                baseline["return_codes"] == candidate["return_codes"]
                == [0] * expected_calls,
                f"{label}: return code mismatch",
            )
            ensure(
                baseline["three_lists_are_python_lists"]
                and candidate["three_lists_are_python_lists"],
                f"{label}: consumer did not pass three Python lists",
            )
            if expected:
                ensure(
                    baseline["src"] == candidate["src"] == expected_src,
                    f"{label}: source address list differs from oracle",
                )
                ensure(
                    baseline["dst"] == candidate["dst"] == expected_dst,
                    f"{label}: destination address list differs from oracle",
                )
                ensure(
                    baseline["lengths"] == candidate["lengths"] == expected_lengths,
                    f"{label}: length list differs from oracle",
                )
                nonempty_cases += 1
            else:
                ensure(
                    baseline["src"] is None and candidate["src"] is None,
                    f"{label}: empty descriptor unexpectedly reached engine",
                )
                empty_cases += 1
            descriptor_count = len(expected)
            payload_bytes = int(sum(block[2] for block in expected))
            if layers == LAYERS:
                ensure(
                    descriptor_count == reference["descriptor_count"],
                    f"{label}: descriptor count {descriptor_count} != L004 {reference['descriptor_count']}",
                )
                ensure(
                    payload_bytes == reference["payload_bytes"],
                    f"{label}: payload {payload_bytes} != L004 {reference['payload_bytes']}",
                )
        else:
            expected_calls = 0
            ensure(
                baseline["native_calls"] == candidate["native_calls"] == 0,
                f"{label}: engine called on exception path",
            )
            ensure(
                not baseline["session_ids"] and not candidate["session_ids"],
                f"{label}: unexpected session id on exception path",
            )
            descriptor_count = None
            payload_bytes = None
            exception_cases += 1

        all_lists_are_python_lists = all_lists_are_python_lists and (
            baseline["three_lists_are_python_lists"]
            and candidate["three_lists_are_python_lists"]
        )
        record = {
            "case": label,
            "source_count": len(src),
            "destination_count": len(dst),
            "fixture_layers": LAYERS,
            "l004_layers": layers,
            "exception": baseline["exception"],
            "descriptor_count": descriptor_count,
            "payload_bytes": payload_bytes,
            "native_calls_expected": expected_calls,
            "baseline": {
                "sender_status": baseline["status"],
                "native_calls": baseline["native_calls"],
                "session_ids": baseline["session_ids"],
                "return_codes": baseline["return_codes"],
                "three_lists_are_python_lists": (
                    baseline["three_lists_are_python_lists"]
                    if baseline["native_calls"]
                    else None
                ),
                "lists_match_oracle": True,
            },
            "candidate": {
                "sender_status": candidate["status"],
                "native_calls": candidate["native_calls"],
                "session_ids": candidate["session_ids"],
                "return_codes": candidate["return_codes"],
                "three_lists_are_python_lists": (
                    candidate["three_lists_are_python_lists"]
                    if candidate["native_calls"]
                    else None
                ),
                "lists_match_oracle": True,
            },
        }
        capture_checks.append(record)
        RUN["checks"].append(record)

    ensure(
        empty_cases + nonempty_cases + exception_cases == len(cases),
        "case classification did not cover all cases",
    )

    RUN["current"] = {"stage": "correctness", "case": "engine-status-7"}
    src = np.arange(10000, 10008, dtype=np.int32)
    dst = np.arange(20000, 20008, dtype=np.int32)
    nonzero = {}
    for variant, fn in (
        ("baseline", RUNTIME["base_sender"]),
        ("candidate", RUNTIME["candidate_sender"]),
    ):
        fixture = make_fixture(CaptureEngine(status=7))
        result = execute_case(fn, fixture, src, dst)
        expected = oracle(fixture, src, dst)
        expected_src, expected_dst, expected_lengths = oracle_columns(expected)
        ensure(result["exception"] is None, f"{variant}: status-7 path raised {result['exception']}")
        ensure(result["status"] == 7, f"{variant}: status {result['status']} != 7")
        ensure(result["native_calls"] == 1, f"{variant}: status-7 calls {result['native_calls']} != 1")
        ensure(result["return_codes"] == [7], f"{variant}: engine return codes {result['return_codes']}")
        ensure(result["session_ids"] == [SESSION], f"{variant}: status-7 session id mismatch")
        ensure(result["src"] == expected_src, f"{variant}: status-7 source list mismatch")
        ensure(result["dst"] == expected_dst, f"{variant}: status-7 destination list mismatch")
        ensure(result["lengths"] == expected_lengths, f"{variant}: status-7 length list mismatch")
        nonzero[variant] = {
            "sender_status": result["status"],
            "native_calls": result["native_calls"],
            "engine_return_codes": result["return_codes"],
            "lists_match_oracle": True,
        }

    return {
        "status": "PASS",
        "cases_total": len(cases),
        "fixture_layers": LAYERS,
        "random_seed": RANDOM_SEED,
        "generation": "copied from the frozen L004 case generation (labels, arrays, dtype/read-only views)",
        "empty_descriptor_cases": empty_cases,
        "legal_nonempty_cases": nonempty_cases,
        "exception_cases": exception_cases,
        "empty_descriptor_engine_calls": 0,
        "legal_nonempty_engine_calls": 1,
        "exception_path_engine_calls": 0,
        "three_lists_are_python_lists": all_lists_are_python_lists,
        "workload_equivalence": {
            "descriptor_count_and_payload_bytes_match_between_variants": True,
            "native_calls_match_between_variants": True,
        },
        "engine_nonzero_status_propagation": nonzero,
        "cases": capture_checks,
    }


def run_benchmark():
    base_sender = RUNTIME["base_sender"]
    candidate_sender = RUNTIME["candidate_sender"]
    range_metadata = RUNTIME["range_metadata"]
    records = []

    for case_name, n, fragmented, repeats in BENCH_CASES:
        RUN["stage"] = "benchmark"
        RUN["current"] = {"stage": "benchmark", "case": case_name}
        src, dst = bench_arrays(n, fragmented)
        runs = len(range_metadata(src, dst))
        pointer_fixture = make_fixture(CountOnlyEngine())
        expected = oracle(pointer_fixture, src, dst)
        expected_src, expected_dst, expected_lengths = oracle_columns(expected)

        baseline_precheck_fixture = make_fixture(CaptureEngine())
        candidate_precheck_fixture = make_fixture(CaptureEngine())
        baseline_precheck = execute_case(
            base_sender, baseline_precheck_fixture, src, dst
        )
        candidate_precheck = execute_case(
            candidate_sender, candidate_precheck_fixture, src, dst
        )
        ensure(baseline_precheck["exception"] is None, f"{case_name}: baseline precheck raised")
        ensure(candidate_precheck["exception"] is None, f"{case_name}: candidate precheck raised")
        ensure(baseline_precheck["status"] == candidate_precheck["status"] == 0, f"{case_name}: precheck status")
        ensure(
            baseline_precheck["native_calls"] == candidate_precheck["native_calls"] == 1,
            f"{case_name}: precheck native calls differ",
        )
        ensure(
            baseline_precheck["src"] == candidate_precheck["src"] == expected_src,
            f"{case_name}: precheck source lists differ",
        )
        ensure(
            baseline_precheck["dst"] == candidate_precheck["dst"] == expected_dst,
            f"{case_name}: precheck destination lists differ",
        )
        ensure(
            baseline_precheck["lengths"] == candidate_precheck["lengths"] == expected_lengths,
            f"{case_name}: precheck length lists differ",
        )
        descriptor_count = len(expected)
        payload_bytes = int(sum(block[2] for block in expected))
        ensure(descriptor_count == LAYERS * runs, f"{case_name}: descriptor count mismatch")
        equivalence_precheck = {
            "baseline_lists_match_oracle": True,
            "candidate_lists_match_oracle": True,
            "baseline_and_candidate_lists_equal": True,
            "descriptor_count_baseline": descriptor_count,
            "descriptor_count_candidate": descriptor_count,
            "payload_bytes_baseline": payload_bytes,
            "payload_bytes_candidate": payload_bytes,
            "native_calls": 1,
        }
        # Release all precheck/oracle containers before the timed loops.  Their
        # presence would otherwise be an unnecessary GC root during timing.
        del expected, expected_src, expected_dst, expected_lengths
        del pointer_fixture
        del baseline_precheck_fixture, candidate_precheck_fixture
        del baseline_precheck, candidate_precheck

        baseline_fixture = make_fixture(CountOnlyEngine())
        candidate_fixture = make_fixture(CountOnlyEngine())

        for _ in range(3):
            baseline_warmup = measure_sample(
                base_sender, baseline_fixture, src, dst, repeats
            )
            candidate_warmup = measure_sample(
                candidate_sender, candidate_fixture, src, dst, repeats
            )
            ensure(
                baseline_warmup["native_calls"] == candidate_warmup["native_calls"] == repeats,
                f"{case_name}: warmup workload mismatch",
            )

        aa_pairs = []
        for _ in range(5):
            first = measure_sample(base_sender, baseline_fixture, src, dst, repeats)
            second = measure_sample(base_sender, baseline_fixture, src, dst, repeats)
            ensure(
                first["native_calls"] == second["native_calls"] == repeats,
                f"{case_name}: A/A workload mismatch",
            )
            aa_pairs.append({"order": "AA", "a": first, "b": second})

        ab_pairs = []
        for index in range(9):
            if index % 2 == 0:
                baseline_sample = measure_sample(
                    base_sender, baseline_fixture, src, dst, repeats
                )
                candidate_sample = measure_sample(
                    candidate_sender, candidate_fixture, src, dst, repeats
                )
                order = "AB"
            else:
                candidate_sample = measure_sample(
                    candidate_sender, candidate_fixture, src, dst, repeats
                )
                baseline_sample = measure_sample(
                    base_sender, baseline_fixture, src, dst, repeats
                )
                order = "BA"
            ensure(
                baseline_sample["native_calls"] == candidate_sample["native_calls"] == repeats,
                f"{case_name}: A/B workload mismatch in {order}",
            )
            ab_pairs.append({"order": order, "a": baseline_sample, "b": candidate_sample})

        record = {
            "case": case_name,
            "N_page_indices": n,
            "R_paired_contiguous_runs": runs,
            "layers": LAYERS,
            "input_dtype": str(src.dtype),
            "input_read_bytes": int(src.nbytes + dst.nbytes),
            "fixture_repeats_per_sample": repeats,
            "warmup_rounds": 3,
            "aa_pair_count": len(aa_pairs),
            "ab_pair_count": len(ab_pairs),
            "descriptor_count": descriptor_count,
            "payload_bytes": payload_bytes,
            "native_calls_per_sender_call": 1,
            "equivalence_precheck": equivalence_precheck,
            "aa_pairs": aa_pairs,
            "ab_pairs": ab_pairs,
            "outliers_removed": False,
            "gc_default_state_changed": False,
        }
        for metric in ("wall_ns_per_call", "thread_cpu_ns_per_call"):
            key = "wall" if metric.startswith("wall") else "thread_cpu"
            aa_ratio = paired_ratio(
                [(pair["a"], pair["b"]) for pair in aa_pairs], metric
            )
            ab_ratio = paired_ratio(
                [(pair["a"], pair["b"]) for pair in ab_pairs], metric
            )
            record[key] = {
                "baseline": describe([pair["a"] for pair in ab_pairs], metric),
                "candidate": describe([pair["b"] for pair in ab_pairs], metric),
                "aa": {
                    "a": describe([pair["a"] for pair in aa_pairs], metric),
                    "b": describe([pair["b"] for pair in aa_pairs], metric),
                    "ratio": aa_ratio,
                },
                "ab_ratio": ab_ratio,
                "ab_ratio_ci_excludes_1": (
                    ab_ratio["bootstrap_95_interval"][0] > 1.0
                    or ab_ratio["bootstrap_95_interval"][1] < 1.0
                ),
            }
        records.append(record)
        RUN["bench"].append(
            {
                "case": case_name,
                "baseline_median": record["wall"]["baseline"]["median_ns"],
                "candidate_median": record["wall"]["candidate"]["median_ns"],
                "ab_ratio": record["wall"]["ab_ratio"]["ratio_of_geomeans"],
            }
        )

    return {
        "clock": {
            "wall": "time.perf_counter_ns",
            "thread_cpu": "time.thread_time_ns",
            "window": "wall and thread CPU are sampled immediately before and after the same repeat loop",
        },
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "percentile_indices": [99, 3899],
        },
        "case_order": [case[0] for case in BENCH_CASES],
        "workload_equivalence": {
            "same_descriptor_count_per_case": True,
            "same_payload_bytes_per_case": True,
            "same_native_calls_per_sample": True,
            "same_repeats_per_case": True,
        },
        "cases": records,
    }


def run_diagnostics():
    RUN["stage"] = "diagnostics"
    range_metadata = RUNTIME["range_metadata"]
    transfer_fn = RUNTIME["transfer_fn"]
    gc_events = []
    context = {"sample_id": "between", "wall_start": None, "cpu_start": None}

    def observer(phase, info):
        wall_now = time.perf_counter_ns()
        cpu_now = time.thread_time_ns()
        record = {
            "sample_id": context["sample_id"],
            "phase": phase,
            "generation": info.get("generation"),
            "collected": info.get("collected"),
            "wall_ns": wall_now,
            "thread_cpu_ns": cpu_now,
        }
        if context["wall_start"] is not None:
            record["wall_offset_ns"] = wall_now - context["wall_start"]
            record["thread_cpu_offset_ns"] = cpu_now - context["cpu_start"]
        gc_events.append(record)

    callbacks_before = len(gc.callbacks)
    ensure(callbacks_before == RUN["gc_at_start"]["callbacks"], "gc.callbacks changed before diagnostics")
    gc.callbacks.append(observer)
    registered = len(gc.callbacks) - callbacks_before == 1
    ensure(registered, "exactly one diagnostic gc callback was not registered")
    removed = False
    restored = {"env_group": True, "consumer_binding": True}
    diagnostic_cases = []

    try:
        for case_name, n, fragmented, _repeats in BENCH_CASES:
            src, dst = bench_arrays(n, fragmented)
            for variant, fn, group_name in (
                ("baseline", RUNTIME["base_sender"], "group_concurrent_contiguous"),
                ("candidate", RUNTIME["candidate_sender"], "range_metadata"),
            ):
                RUN["current"] = {"stage": "diagnostics", "case": case_name, "variant": variant}
                fixture = make_fixture(CountOnlyEngine())
                env = fn.__globals__
                ensure(env is not None, f"{variant}: sender has no globals")
                original_group = env[group_name]
                original_binding = fixture._transfer_data
                group_observations = []
                consumer_observations = []
                calls = []
                env[group_name] = make_group_wrapper(original_group, group_observations)
                fixture._transfer_data = types.MethodType(
                    make_consumer_wrapper(transfer_fn, consumer_observations), fixture
                )
                try:
                    for call_index in range(3):
                        sample_id = f"{case_name}:{variant}:diag-{call_index}"
                        context["sample_id"] = sample_id
                        engine_calls_before = fixture.engine.calls
                        context["wall_start"] = time.perf_counter_ns()
                        context["cpu_start"] = time.thread_time_ns()
                        wall_start = context["wall_start"]
                        cpu_start = context["cpu_start"]
                        status = call_sender(fn, fixture, src, dst)
                        cpu_stop = time.thread_time_ns()
                        wall_stop = time.perf_counter_ns()
                        context["sample_id"] = "between"
                        context["wall_start"] = None
                        context["cpu_start"] = None
                        ensure(status == 0, f"{sample_id}: status {status} != 0")
                        ensure(
                            len(group_observations) == call_index + 1,
                            f"{sample_id}: group wrapper call count {len(group_observations)}",
                        )
                        ensure(
                            len(consumer_observations) == call_index + 1,
                            f"{sample_id}: consumer wrapper call count {len(consumer_observations)}",
                        )
                        ensure(
                            fixture.engine.calls - engine_calls_before == 1,
                            f"{sample_id}: native calls {fixture.engine.calls - engine_calls_before} != 1",
                        )
                        group_observation = group_observations[-1]
                        consumer_observation = consumer_observations[-1]
                        total = {
                            "wall_ns": wall_stop - wall_start,
                            "thread_cpu_ns": cpu_stop - cpu_start,
                        }
                        residual = {
                            "wall_ns": total["wall_ns"]
                            - group_observation["wall_ns"]
                            - consumer_observation["wall_ns"],
                            "thread_cpu_ns": total["thread_cpu_ns"]
                            - group_observation["thread_cpu_ns"]
                            - consumer_observation["thread_cpu_ns"],
                        }
                        calls.append(
                            {
                                "sample_id": sample_id,
                                "sender_status": status,
                                "total": total,
                                "group": group_observation,
                                "consumer": consumer_observation,
                                "residual": residual,
                                "group_calls": 1,
                                "consumer_calls": 1,
                                "native_calls": 1,
                            }
                        )
                finally:
                    env[group_name] = original_group
                    fixture._transfer_data = original_binding
                    context["sample_id"] = "between"
                    context["wall_start"] = None
                    context["cpu_start"] = None
                    if env[group_name] is not original_group:
                        restored["env_group"] = False
                    if fixture._transfer_data != original_binding:
                        restored["consumer_binding"] = False
                ensure(restored["env_group"], f"{case_name}/{variant}: env group not restored")
                ensure(
                    restored["consumer_binding"],
                    f"{case_name}/{variant}: consumer binding not restored",
                )
                diagnostic_cases.append(
                    {
                        "case": case_name,
                        "variant": variant,
                        "group_stage": group_name,
                        "calls_per_variant": len(calls),
                        "calls": calls,
                        "residual_summary": {
                            "wall_ns": raw_summary(
                                [call["residual"]["wall_ns"] for call in calls]
                            ),
                            "thread_cpu_ns": raw_summary(
                                [
                                    call["residual"]["thread_cpu_ns"]
                                    for call in calls
                                ]
                            ),
                        },
                    }
                )
    finally:
        if observer in gc.callbacks:
            gc.callbacks.remove(observer)
            removed = True
        context["sample_id"] = "between"
        context["wall_start"] = None
        context["cpu_start"] = None

    ensure(removed, "did not remove own diagnostic gc callback")
    ensure(
        len(gc.callbacks) == callbacks_before,
        f"gc.callbacks not restored: {len(gc.callbacks)} != {callbacks_before}",
    )
    ensure(
        range_metadata is RUNTIME["range_metadata"],
        "candidate range function changed unexpectedly",
    )

    return {
        "note": (
            "Separate post-timing diagnostic. Wrappers and the gc.callbacks observer add "
            "overhead; these values must not be compared with clean A/B samples."
        ),
        "calls_per_case_variant": 3,
        "group_stage": {
            "baseline": "group_concurrent_contiguous wrapper in the sender env",
            "candidate": "range_metadata wrapper in the sender env",
        },
        "consumer_stage": "types.MethodType wrapper around the original extracted _transfer_data; body unchanged",
        "residual_definition": "sender residual = total - group - consumer (wall and thread CPU)",
        "gc_observer": {
            "registered_one_callback": registered,
            "callbacks_before": callbacks_before,
            "callbacks_after": len(gc.callbacks),
            "removed_own_callback": removed,
            "events": gc_events,
            "event_count": len(gc_events),
        },
        "restoration": restored,
        "cases": diagnostic_cases,
    }


def run_replay_check(cases, l004_result):
    ensure(len(l004_result["cases"]) == 151, "L004 result does not contain 151 cases")
    probe_fixtures = {
        layers: make_fixture(CountOnlyEngine(), layers) for layers in (3, LAYERS)
    }
    checked = 0
    for index, (label, src, dst, layers, _hisparse) in enumerate(cases):
        reference = l004_result["cases"][index]
        ensure(reference["case"] == label, f"L004 label mismatch at index {index}")
        ensure(reference["layers"] == layers, f"{label}: L004 layer mismatch")
        ensure(reference["source_count"] == len(src), f"{label}: L004 source count mismatch")
        ensure(
            reference["destination_count"] == len(dst),
            f"{label}: L004 destination count mismatch",
        )
        if reference["exception"] is None:
            blocks = oracle(probe_fixtures[layers], src, dst)
            ensure(
                len(blocks) == reference["descriptor_count"],
                f"{label}: replay descriptor count differs from L004",
            )
            ensure(
                int(sum(block[2] for block in blocks)) == reference["payload_bytes"],
                f"{label}: replay payload differs from L004",
            )
            checked += 1
    return {
        "labels_match_L004_order": True,
        "descriptor_and_payload_match_L004_for_non_exception_cases": True,
        "layers": "L004 original per-case layer counts (3 or 78)",
        "note": "Preflight transcription check only; no sender or consumer is executed here.",
        "checked_cases": checked,
        "total_cases": len(cases),
    }


def write_failure_receipt(error_text):
    out = RUN.get("out")
    if out is None:
        return False
    try:
        if not out.exists():
            return False
        payload = {
            "schema_version": 1,
            "run_id": "L005-R001-full-consumer",
            "task_id": "sglang-pd-spec-36",
            "loop_id": "L005-full-consumer-cpu-profile",
            "timestamp_utc": utc_now(),
            "script": {
                "path": "scripts/probe_full_consumer_cpu.py",
                "sha256": sha256_file(SCRIPT),
            },
            "inputs": RUN.get("inputs", {}),
            "result": {
                "path": "runs/L005-R001-full-consumer/result.json",
                "sha256": RUN.get("result_sha256"),
            },
            "exit_code": 1,
            "correctness": "ERROR",
            "scope": SCOPE,
            "stage": RUN.get("stage"),
            "current": RUN.get("current"),
            "error_traceback": error_text,
            "partial": {
                "correctness_cases_completed": len(RUN.get("checks", [])),
                "benchmark_cases_completed": len(RUN.get("bench", [])),
                "diagnostics_started": RUN.get("diag") is not None,
            },
            "production_source_changed": False,
            "evidence_level": "E0",
        }
        (out / "receipt.json").write_text(json.dumps(payload, indent=2) + "\n")
        return True
    except Exception:  # noqa: BLE001 - failure receipt must not mask the original failure
        return False


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    RUN["out"] = OUT
    RUN["stage"] = "inputs"
    RUN["gc_at_start"] = gc_state()

    input_hashes = {name: sha256_file(path) for name, path in INPUTS.items()}
    RUN["inputs"] = input_hashes
    script_sha256 = sha256_file(SCRIPT)

    l004_result = json.loads(
        INPUTS["runs/L004-R001-descriptor-cpu/result.json"].read_text()
    )
    candidate_path = "runs/L004-R001-descriptor-cpu/candidate-method.py"
    ensure(
        input_hashes[candidate_path] == l004_result["candidate_method_sha256"],
        "frozen candidate-method.py hash does not match L004 result",
    )
    source_hash_checks = {}
    for relative, expected in l004_result["source_hashes"].items():
        actual = sha256_file(SRC / relative)
        ensure(actual == expected, f"source hash mismatch for {relative}")
        source_hash_checks[relative] = {"expected": expected, "actual": actual}

    RUN["stage"] = "extract"
    ascend_path = "sources/installed/python/sglang/srt/disaggregation/ascend/conn.py"
    utils_path = "sources/installed/python/sglang/srt/disaggregation/common/utils.py"
    mooncake_path = "sources/installed/python/sglang/srt/disaggregation/mooncake/conn.py"
    candidate_method_path = "runs/L004-R001-descriptor-cpu/candidate-method.py"

    ascend_tree = ast.parse(INPUTS[ascend_path].read_text())
    utils_tree = ast.parse(INPUTS[utils_path].read_text())
    mooncake_tree = ast.parse(INPUTS[mooncake_path].read_text())
    candidate_tree = ast.parse(INPUTS[candidate_method_path].read_text())

    send_node = find_method(ascend_tree, "AscendKVManager", "send_kvcache")
    group_node = find_top_level_function(utils_tree, "group_concurrent_contiguous")
    transfer_node = find_method(mooncake_tree, "MooncakeKVManager", "_transfer_data")
    range_node = find_top_level_function(candidate_tree, "range_metadata")
    candidate_node = find_top_level_function(candidate_tree, "candidate_sender")

    methods = {
        "ascend_send_kvcache": method_metadata(
            "AscendKVManager.send_kvcache", send_node, ascend_path
        ),
        "group_concurrent_contiguous": method_metadata(
            "group_concurrent_contiguous", group_node, utils_path
        ),
        "mooncake_transfer_data": method_metadata(
            "MooncakeKVManager._transfer_data", transfer_node, mooncake_path
        ),
        "frozen_range_metadata": method_metadata(
            "range_metadata", range_node, candidate_method_path
        ),
        "frozen_candidate_sender": method_metadata(
            "candidate_sender", candidate_node, candidate_method_path
        ),
    }

    base_env = {"np": np, "concurrent": concurrent}
    exec(compile_nodes([group_node, send_node, transfer_node]), base_env)
    candidate_env = {"np": np, "concurrent": concurrent}
    exec(compile_nodes([range_node, candidate_node]), candidate_env)

    RUNTIME["base_env"] = base_env
    RUNTIME["candidate_env"] = candidate_env
    RUNTIME["base_sender"] = base_env["send_kvcache"]
    RUNTIME["candidate_sender"] = candidate_env["candidate_sender"]
    RUNTIME["real_group"] = base_env["group_concurrent_contiguous"]
    RUNTIME["range_metadata"] = candidate_env["range_metadata"]
    RUNTIME["transfer_fn"] = base_env["_transfer_data"]
    ensure(RUNTIME["transfer_fn"] is not None, "MooncakeKVManager._transfer_data not extracted")

    snapshot_parts = [
        "# Extracted method snapshot generated at runtime by probe_full_consumer_cpu.py",
        "# Source files are not modified.",
        "",
    ]
    for metadata in methods.values():
        snapshot_parts.append(
            f"# {metadata['label']}  ({metadata['file']}:{metadata['line']}-{metadata['end_line']})"
        )
        node = {
            "AscendKVManager.send_kvcache": send_node,
            "group_concurrent_contiguous": group_node,
            "MooncakeKVManager._transfer_data": transfer_node,
            "range_metadata": range_node,
            "candidate_sender": candidate_node,
        }[metadata["label"]]
        snapshot_parts.append(ast.unparse(node))
        snapshot_parts.append("")
    snapshot_text = "\n".join(snapshot_parts) + "\n"
    snapshot_path = OUT / "extracted-methods.py"
    snapshot_path.write_text(snapshot_text)
    snapshot_sha256 = sha256_bytes(snapshot_text.encode("utf-8"))

    cases = build_cases()
    replay = run_replay_check(cases, l004_result)

    RUN["stage"] = "correctness"
    correctness = run_correctness(cases, l004_result)
    ensure(correctness["cases_total"] == 151, "correctness case count changed")

    RUN["stage"] = "benchmark"
    ensure(
        len(gc.callbacks) == RUN["gc_at_start"]["callbacks"],
        "gc.callbacks changed before clean timing",
    )
    benchmark = run_benchmark()

    RUN["stage"] = "diagnostics"
    diagnostics = run_diagnostics()
    RUN["diag"] = diagnostics

    RUN["stage"] = "finalize"
    gc_at_end = gc_state()
    ensure(
        gc_at_end["enabled"] == RUN["gc_at_start"]["enabled"],
        "GC enabled state changed",
    )
    ensure(
        gc_at_end["threshold"] == RUN["gc_at_start"]["threshold"],
        "GC threshold changed",
    )
    ensure(
        gc_at_end["callbacks"] == RUN["gc_at_start"]["callbacks"],
        "gc.callbacks count was not restored",
    )

    final_hashes = {name: sha256_file(path) for name, path in INPUTS.items()}
    ensure(final_hashes == input_hashes, "an input file changed during the run")
    ensure(sha256_file(SCRIPT) == script_sha256, "script changed during the run")
    ensure(
        ast_dump_sha256(transfer_node) == methods["mooncake_transfer_data"]["ast_dump_sha256"],
        "consumer AST changed during the run",
    )

    result = {
        "schema_version": 1,
        "run_id": "L005-R001-full-consumer",
        "task_id": "sglang-pd-spec-36",
        "loop_id": "L005-full-consumer-cpu-profile",
        "mode": "baseline",
        "timestamp_utc": utc_now(),
        "exit_code": 0,
        "script": {
            "path": "scripts/probe_full_consumer_cpu.py",
            "sha256": script_sha256,
        },
        "inputs": input_hashes,
        "input_hashes_unchanged_after_run": True,
        "l004_reference": {
            "result_sha256": input_hashes["runs/L004-R001-descriptor-cpu/result.json"],
            "source_commit_as_recorded": l004_result.get("source_commit"),
            "candidate_method_sha256_verified": True,
            "candidate_method_sha256": input_hashes[candidate_method_path],
            "source_hash_checks": source_hash_checks,
            "case_replay": replay,
        },
        "variants": {
            "baseline": "exact AscendKVManager.send_kvcache AST",
            "candidate": "frozen L004 range_metadata + candidate_sender AST",
            "consumer": "exact MooncakeKVManager._transfer_data AST bound with types.MethodType",
            "single_experimental_change": (
                "For the A/B timing experiment relative to L004, the fixture consumer is now "
                "the real _transfer_data; the two sender variants are unchanged. Per L005 spec, "
                "all correctness fixtures are fixed at 78 layers."
            ),
        },
        "methods": methods,
        "copied_methods_snapshot": {
            "path": "runs/L005-R001-full-consumer/extracted-methods.py",
            "sha256": snapshot_sha256,
        },
        "fixture": {
            "kind": "machine-local CPU fixture",
            "pp_size": 1,
            "non_hybrid": True,
            "custom_mem_pool": False,
            "timing_and_diagnostic_layers": LAYERS,
            "correctness_fixture_layers": LAYERS,
            "l004_replay_layers": "L004 original per-case 3/78 in a separate preflight integrity check",
            "layer_note": (
                "All 151 correctness checks and all timing/diagnostic checks use the specified "
                "fixed 78-layer fixture. The separate preflight L004 replay uses L004's original "
                "per-case layer counts only to verify copied case/oracle transcription."
            ),
            "item_lens": "128 for even layer, 64 for odd layer",
            "kv_data_ptr_base": 2**40,
            "kv_data_ptr_stride": 2**32,
            "dst_ptr_base": 2**46,
            "dst_ptr_stride": 2**32,
            "session_id": SESSION,
            "envelope_validation": "no-op fixture lambda; semantics not tested",
            "engine": "fake Python batch_transfer_sync; no native conversion, copy, or transfer",
            "engine_contract": {
                "correctness_capture": "CaptureEngine copies the three Python lists, session ids, call count, and return codes",
                "timing_callback": "CountOnlyEngine increments the call counter and returns 0; no list copy or traversal",
                "native_conversion_or_copy": False,
                "address_dereference": False,
                "device_or_network_io": False,
            },
        },
        "correctness": correctness,
        "benchmark": benchmark,
        "diagnostics": diagnostics,
        "gc": {
            "at_start": RUN["gc_at_start"],
            "at_end": gc_at_end,
            "enabled_unchanged": True,
            "threshold_unchanged": True,
            "script_registered_callbacks_in_clean_timing": False,
            "diagnostic_observer_registered_once": diagnostics["gc_observer"][
                "registered_one_callback"
            ],
            "diagnostic_observer_removed": diagnostics["gc_observer"][
                "removed_own_callback"
            ],
            "sample_gc_delta_read_outside_timing_window": True,
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "scope": SCOPE,
        "limitations": LIMITATIONS,
        "declarations": DECLARATIONS,
        "interpretation": "DESCRIPTIVE_ONLY",
        "production_source_changed": False,
        "evidence_level": "E0",
    }

    result_bytes = (json.dumps(result, indent=2) + "\n").encode("utf-8")
    (OUT / "result.json").write_bytes(result_bytes)
    result_sha256 = sha256_bytes(result_bytes)
    RUN["result_sha256"] = result_sha256

    receipt = {
        "schema_version": 1,
        "run_id": "L005-R001-full-consumer",
        "task_id": "sglang-pd-spec-36",
        "loop_id": "L005-full-consumer-cpu-profile",
        "timestamp_utc": utc_now(),
        "script": {
            "path": "scripts/probe_full_consumer_cpu.py",
            "sha256": script_sha256,
        },
        "inputs": input_hashes,
        "result": {
            "path": "runs/L005-R001-full-consumer/result.json",
            "sha256": result_sha256,
        },
        "exit_code": 0,
        "correctness": "PASS",
        "scope": SCOPE,
        "production_source_changed": False,
        "evidence_level": "E0",
    }
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")

    summary = {
        "run_id": "L005-R001-full-consumer",
        "exit_code": 0,
        "correctness": "PASS",
        "correctness_cases": correctness["cases_total"],
        "benchmark": [
            {
                "case": record["case"],
                "wall_ab_ratio": record["wall"]["ab_ratio"],
                "thread_cpu_ab_ratio": record["thread_cpu"]["ab_ratio"],
            }
            for record in benchmark["cases"]
        ],
        "diagnostic_calls": 4 * 2 * 3,
        "result_sha256": result_sha256,
    }
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - preserve the first failure and exit nonzero
        error_text = traceback.format_exc()
        write_failure_receipt(error_text)
        print(
            json.dumps(
                {
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "stage": RUN.get("stage"),
                    "current": RUN.get("current"),
                },
                default=str,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
