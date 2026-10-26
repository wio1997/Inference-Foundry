# SPDX-License-Identifier: Apache-2.0
"""R246 synthetic row-stride component: no model, no weights, no PD.

Stage A (rank0 only, other ranks at an HCCL barrier) runs six real-style
(page-table width, sequence length) fixtures through two implementations in
two isolated processes:

  oracle : frozen R238-B flash_kpool_score.py, TABLE_SR/table_stride_r are
           tl.constexpr;
  stride : identical file except those two parameters are runtime scalars in
           do_not_specialize.

Each implementation owns one fresh TRITON_CACHE_DIR and runs all six widths
sequentially, so cumulative compiled-variant and distinct-TTIR counts expose
whether the row stride still creates per-width compiler variants.  The two
runs must be elementwise byte-identical on every fixture.

Stage B (all 16 ranks, one native HCCL group) compares the oracle full score
against the R210 query_parallel_indexer row split calling the stride candidate,
including one all_gather_into_tensor for every query-parallel prefill.
"""

import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

R = pathlib.Path("/run-evidence")
MANIFEST_SHA = os.environ.get("R246_MANIFEST_SHA", "unset")

# Six distinct row strides drawn from the project's real sequence envelope:
# table width = ceil((sequence // 4) / 32).
VARIANTS = (
    ("w32_s4096", 4096),
    ("w63_s8064", 8064),
    ("w256_s32768", 32768),
    ("w313_s40000", 40000),
    ("w581_s74368", 74368),
    ("w582_s74400", 74400),
)

CASES = (
    dict(name="single4096_at74400", lengths=[4096], sequences=[74400], prefill=True, timing=True),
    dict(name="mixed4096_empty_req", lengths=[1536, 0, 2560], sequences=[40000, 0, 74400], prefill=True, timing=True),
    dict(name="uneven513", lengths=[33, 0, 223, 1, 256], sequences=[4096, 0, 8000, 600, 32768], prefill=True, timing=False),
    dict(name="small_prefill33_fallback", lengths=[33], sequences=[4096], prefill=True, timing=False),
    dict(name="decode1_fallback", lengths=[1], sequences=[74400], prefill=False, timing=False),
)

GATHER = "_gather_pool_cache"
SCORE = "_glm5_next_lightning_indexer_score_kernel"
EXPAND = "_expand_pool_indices"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _width(sequence):
    return ((sequence // 4) + 31) // 32


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _build_fixture(sequence, rows, torch, device):
    max_pool = sequence // 4
    width = _width(sequence)
    cache = (torch.randn((width, 32, 1, 128)) * 0.1).to(dtype=torch.bfloat16, device=device)
    table = torch.arange(width, dtype=torch.int32, device=device).reshape(1, width)
    pool_lens = torch.tensor([max_pool], dtype=torch.int32, device=device)
    ends = torch.tensor([rows], dtype=torch.int32, device=device)
    positions = torch.arange(sequence - rows, sequence, dtype=torch.int64, device=device)
    query = (torch.randn((rows, 32, 128)) * 0.1).to(dtype=torch.bfloat16, device=device)
    weights = (torch.randn((rows, 32)) * 0.05).to(dtype=torch.bfloat16, device=device)
    return dict(
        query=query,
        indexer_cache=cache,
        weights=weights,
        cum_query_lens=ends,
        indexer_seq_lens=pool_lens,
        indexer_block_table=table,
        positions=positions,
        index_topk=2048,
        index_kpool=4,
        max_pool_seq_len=max_pool,
        pack_tail=True,
    ), width, max_pool


def _build_case(spec, torch, device):
    lengths, sequences = spec["lengths"], spec["sequences"]
    total = sum(lengths)
    maximum = max(sequences) // 4
    width = (maximum + 31) // 32
    requests = len(lengths)
    cache = (torch.randn((requests * width, 32, 1, 128)) * 0.1).to(dtype=torch.bfloat16, device=device)
    table = torch.arange(requests * width, dtype=torch.int32, device=device).reshape(requests, width)
    pool_lens = torch.tensor([n // 4 for n in sequences], dtype=torch.int32, device=device)
    ends, running = [], 0
    for length in lengths:
        running += length
        ends.append(running)
    ends = torch.tensor(ends, dtype=torch.int32, device=device)
    positions = torch.tensor(
        [p for length, sequence in zip(lengths, sequences) for p in range(sequence - length, sequence)],
        dtype=torch.int64,
        device=device,
    )
    query = (torch.randn((total, 32, 128)) * 0.1).to(dtype=torch.bfloat16, device=device)
    weights = (torch.randn((total, 32)) * 0.05).to(dtype=torch.bfloat16, device=device)
    return dict(
        query=query,
        indexer_cache=cache,
        weights=weights,
        cum_query_lens=ends,
        indexer_seq_lens=pool_lens,
        indexer_block_table=table,
        positions=positions,
        index_topk=2048,
        index_kpool=4,
        max_pool_seq_len=maximum,
        pack_tail=True,
    ), width, maximum


def _cache_evidence(cache_dir):
    """Count compiled variants and distinct TTIR for each JIT kernel.

    A Triton cache entry is one gated child directory.  The score kernel can
    compile several entries with identical TTIR when a constexpr row stride
    folds away, so both counts are recorded separately.
    """
    result = {}
    files = []
    if cache_dir.exists():
        for path in cache_dir.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".ttir"):
                rel = path.relative_to(cache_dir)
                if len(rel.parts) < 2:
                    continue
                files.append(path)
    for kernel in (GATHER, SCORE, EXPAND):
        needle = kernel.encode()
        directories, json_files, ttir_files = set(), [], []
        for path in files:
            try:
                body = path.read_bytes()
            except OSError:
                continue
            if needle not in body:
                continue
            directories.add(str(path.parent))
            if path.suffix == ".json":
                json_files.append(path)
            else:
                ttir_files.append(path)
        hashes = sorted({_sha(path.read_bytes()) for path in ttir_files})
        constants = set()
        ttir = []
        for path in ttir_files:
            text = path.read_text(errors="ignore")
            constants.update(re.findall(r"arith\.constant (-?\d+) : i32", text))
            ttir.append(
                dict(
                    sha256=_sha(path.read_bytes()),
                    bytes=path.stat().st_size,
                    path=str(path.relative_to(cache_dir)),
                )
            )
        result[kernel] = dict(
            compiled_variants=len(directories),
            json_files=len(json_files),
            ttir_files=len(ttir_files),
            distinct_ttir_sha256=hashes,
            i32_constants=sorted(constants, key=lambda value: (abs(int(value)), int(value))),
            ttir=ttir,
        )
    return result


def stage_a_impl(impl):
    """One implementation, one isolated Triton cache, six fixtures in order."""
    cache_dir = R / "stageA-cache" / impl
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    (R / "stageA").mkdir(parents=True, exist_ok=True)
    result = dict(impl=impl, passed=False, manifest_sha256=MANIFEST_SHA, cache_dir=str(cache_dir), calls=[])
    try:
        import torch
        import torch_npu

        torch.set_num_threads(1)
        torch.npu.set_device(0)
        if impl == "oracle":
            from sglang.srt.hardware_backend.npu.ops.flash_kpool_score import (
                glm5_next_lightning_indexer_triton as function,
            )
        elif impl == "stride":
            from sglang.srt.hardware_backend.npu.ops.flash_kpool_score_stride import (
                glm5_next_lightning_indexer_triton as function,
            )
        else:
            raise ValueError("Unknown stage-A implementation")

        captured = {}
        original_topk = torch.topk

        def capture_topk(scores, k, dim=1, *args, **kwargs):
            captured["dtype"] = str(scores.dtype)
            captured["shape"] = list(scores.shape)
            return original_topk(scores, k, dim=dim, *args, **kwargs)

        torch.topk = capture_topk
        for index, (name, sequence) in enumerate(VARIANTS):
            torch.manual_seed(246100 + index)
            arguments, width, max_pool = _build_fixture(sequence, 64, torch, "npu:0")
            captured.clear()
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            output = function(**arguments)
            torch.npu.synchronize()
            finish = time.perf_counter_ns()
            raw = output.detach().cpu().contiguous().numpy().tobytes()
            (R / "stageA" / f"{impl}-{index}.bin").write_bytes(raw)
            result["calls"].append(
                dict(
                    variant=name,
                    index=index,
                    sequence=sequence,
                    table_width=width,
                    rows=64,
                    max_pool_seq_len=max_pool,
                    host_wall_ms=(finish - start) / 1e6,
                    cold=(index == 0),
                    output_sha256=_sha(raw),
                    output_shape=list(output.shape),
                    output_dtype=str(output.dtype),
                    query_dtype=str(arguments["query"].dtype),
                    cache_dtype=str(arguments["indexer_cache"].dtype),
                    score_dtype=captured.get("dtype"),
                    score_shape=captured.get("shape"),
                    pack_tail=True,
                )
            )
            del arguments, output
            torch.npu.empty_cache()
        torch.topk = original_topk
        result["cache"] = _cache_evidence(cache_dir)
        result["passed"] = (
            len(result["calls"]) == len(VARIANTS)
            and all(call["score_dtype"] == "torch.float32" for call in result["calls"])
            and all(call["cache_dtype"] == "torch.bfloat16" for call in result["calls"])
            and all(call["query_dtype"] == "torch.bfloat16" for call in result["calls"])
        )
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
    _write_json(R / "stageA" / f"{impl}.json", result)
    return 0 if result["passed"] else 1


def _run_stage_a():
    summary = dict(
        passed=False,
        manifest_sha256=MANIFEST_SHA,
        scope="rank0-only synthetic cold-compile and six-width bitexact component; no model/PD",
        fixtures=[
            dict(variant=name, sequence=sequence, table_width=_width(sequence), max_pool_seq_len=sequence // 4)
            for name, sequence in VARIANTS
        ],
        implementations={},
        comparisons=[],
    )
    try:
        for impl in ("oracle", "stride"):
            cache_dir = R / "stageA-cache" / impl
            cache_dir.mkdir(parents=True, exist_ok=False)
            environment = dict(os.environ)
            for key in (
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
                "TORCHELASTIC_RUN_ID",
                "TORCHELASTIC_MAX_RESTARTS",
                "TORCHELASTIC_RESTART_COUNT",
            ):
                environment.pop(key, None)
            environment.update(
                ASCEND_RT_VISIBLE_DEVICES="0",
                TRITON_CACHE_DIR=str(cache_dir),
                FLASH_PD_SCORE_ABI_AUDIT="0",
                FLASH_PD_COMPRESSION_ABI_AUDIT="0",
                R246_MANIFEST_SHA=MANIFEST_SHA,
            )
            started = time.time()
            completed = subprocess.run(
                [sys.executable, str(R / "component.py"), "stage-a-impl", impl],
                capture_output=True,
                text=True,
                env=environment,
                timeout=240,
            )
            (R / "stageA" / f"{impl}.log").write_text(completed.stdout + completed.stderr)
            artifact = R / "stageA" / f"{impl}.json"
            if artifact.exists():
                summary["implementations"][impl] = json.loads(artifact.read_text())
            else:
                summary["implementations"][impl] = dict(
                    impl=impl,
                    passed=False,
                    error="missing stage-A artifact",
                    rc=completed.returncode,
                    elapsed_s=time.time() - started,
                )

        import numpy as np

        for index, (name, sequence) in enumerate(VARIANTS):
            oracle_raw = (R / "stageA" / f"oracle-{index}.bin").read_bytes()
            stride_raw = (R / "stageA" / f"stride-{index}.bin").read_bytes()
            oracle_values = np.frombuffer(oracle_raw, dtype=np.int32)
            stride_values = np.frombuffer(stride_raw, dtype=np.int32)
            summary["comparisons"].append(
                dict(
                    variant=name,
                    index=index,
                    sequence=sequence,
                    table_width=_width(sequence),
                    elements=int(oracle_values.size),
                    bytes_equal=oracle_raw == stride_raw,
                    elementwise_equal=bool(np.array_equal(oracle_values, stride_values)),
                    oracle_sha256=_sha(oracle_raw),
                    stride_sha256=_sha(stride_raw),
                )
            )

        def observed(impl):
            cache = summary["implementations"].get(impl, {}).get("cache", {})
            return dict(
                gather_compiled=cache.get(GATHER, {}).get("compiled_variants", 0),
                score_compiled=cache.get(SCORE, {}).get("compiled_variants", 0),
                gather_distinct_ttir=len(cache.get(GATHER, {}).get("distinct_ttir_sha256", [])),
                score_distinct_ttir=len(cache.get(SCORE, {}).get("distinct_ttir_sha256", [])),
                expand_compiled=cache.get(EXPAND, {}).get("compiled_variants", 0),
            )

        def timing(impl):
            calls = summary["implementations"].get(impl, {}).get("calls", [])
            return dict(
                first_call_host_ms=calls[0]["host_wall_ms"] if calls else None,
                per_call_host_ms=[call["host_wall_ms"] for call in calls],
                calls_2_6_host_ms=[call["host_wall_ms"] for call in calls[1:]],
                sum_host_ms=sum(call["host_wall_ms"] for call in calls),
            )

        counts = {impl: observed(impl) for impl in ("oracle", "stride")}
        summary["observed"] = counts
        summary["timing"] = {impl: timing(impl) for impl in ("oracle", "stride")}
        summary["expected"] = dict(
            oracle=dict(gather_compiled=len(VARIANTS), score_compiled=len(VARIANTS)),
            stride=dict(gather_compiled=1, score_compiled=1),
        )
        summary["passed"] = (
            summary["implementations"].get("oracle", {}).get("passed") is True
            and summary["implementations"].get("stride", {}).get("passed") is True
            and len(summary["comparisons"]) == len(VARIANTS)
            and all(item["elementwise_equal"] for item in summary["comparisons"])
            and counts["oracle"]["gather_compiled"] == len(VARIANTS)
            and counts["oracle"]["score_compiled"] == len(VARIANTS)
            and counts["stride"]["gather_compiled"] == 1
            and counts["stride"]["score_compiled"] == 1
        )
        summary["interpretation"] = (
            "Oracle must accumulate one compiled variant per table width; stride candidate "
            "must accumulate one variant total. Byte equality proves the arithmetic and "
            "selected token indices are unchanged. Distinct TTIR is reported separately "
            "because a request-0 score kernel can fold the row stride out of TTIR."
        )
    except BaseException as error:
        summary.update(error=repr(error), traceback=traceback.format_exc())
    _write_json(R / "stageA-summary.json", summary)
    return summary


class NativeHCCLGroup:
    """One actual all_gather_into_tensor per query-parallel prefill."""

    def __init__(self, world, group, torch, dist):
        self.world = world
        self.group = group
        self.torch = torch
        self.dist = dist
        self.calls = 0
        self.shapes = []

    def all_gather(self, value, dim=0):
        assert dim == 0
        self.calls += 1
        self.shapes.append(list(value.shape))
        target = self.torch.empty(
            (value.shape[0] * self.world, *value.shape[1:]),
            dtype=value.dtype,
            device=value.device,
        )
        self.dist.all_gather_into_tensor(target, value.contiguous(), group=self.group)
        return target


def _call_candidate(query_parallel_indexer, arguments, native, rank, world, prefill):
    """R210 query_parallel_indexer uses positional cache/boundary names."""
    return query_parallel_indexer(
        arguments["query"],
        arguments["indexer_cache"],
        arguments["weights"],
        arguments["cum_query_lens"],
        arguments["indexer_seq_lens"],
        arguments["indexer_block_table"],
        arguments["positions"],
        group=native,
        rank=rank,
        world_size=world,
        prefill=prefill,
        index_topk=arguments["index_topk"],
        index_kpool=arguments["index_kpool"],
        max_pool_seq_len=arguments["max_pool_seq_len"],
        pack_tail=arguments["pack_tail"],
    )


def _stage_b(rank, world, group):
    import torch
    import torch.distributed as dist

    from sglang.srt.hardware_backend.npu.ops.flash_indexer_query_parallel import query_parallel_indexer
    from sglang.srt.hardware_backend.npu.ops.flash_kpool_score import glm5_next_lightning_indexer_triton as oracle

    native = NativeHCCLGroup(world, group, torch, dist)
    output = dict(
        project_id="glm53-flash",
        unit="r246",
        rank=rank,
        world_size=world,
        manifest_sha256=MANIFEST_SHA,
        passed=False,
        model_requests=0,
        weights_mounted=False,
        pd_used=False,
        collective_backend="hccl all_gather_into_tensor on dist.group.WORLD",
        cases=[],
        timings=[],
    )
    failures = []
    for case_index, spec in enumerate(CASES):
        # Every rank builds the same synthetic rows: the row split is only
        # meaningful when the whole group scores identical tensors.
        torch.manual_seed(246200 + case_index)
        arguments, width, maximum = _build_case(spec, torch, "npu:%d" % rank)
        baseline = oracle(**arguments)
        calls_before = native.calls
        changed = _call_candidate(
            query_parallel_indexer, arguments, native, rank, world, spec["prefill"]
        )
        calls = native.calls - calls_before
        torch.npu.synchronize()
        total = sum(spec["lengths"])
        expected_calls = 1 if (spec["prefill"] and total >= world * 32) else 0
        bitexact = bool(torch.equal(changed, baseline))
        output["cases"].append(
            dict(
                name=spec["name"],
                query_tokens=total,
                lengths=spec["lengths"],
                sequences=spec["sequences"],
                prefill=spec["prefill"],
                table_width=width,
                max_pool_seq_len=maximum,
                bitexact=bitexact,
                native_group_calls=calls,
                expected_native_group_calls=expected_calls,
                fallback=expected_calls == 0,
                output_sha256=_sha(changed.detach().cpu().contiguous().numpy().tobytes()),
                output_shape=list(changed.shape),
                output_dtype=str(changed.dtype),
            )
        )
        if not bitexact:
            failures.append(spec["name"] + ": token indices differ")
        if calls != expected_calls:
            failures.append(spec["name"] + ": unexpected native HCCL gather count")

        if spec["timing"] and bitexact:
            for _ in range(2):
                oracle(**arguments)
                _call_candidate(
                    query_parallel_indexer, arguments, native, rank, world, spec["prefill"]
                )
            torch.npu.synchronize()
            dist.barrier()
            samples = {"old": [], "new": []}
            for repetition in range(3):
                order = ("old", "new") if repetition % 2 == 0 else ("new", "old")
                for kind in order:
                    dist.barrier()
                    torch.npu.synchronize()
                    start_event = torch.npu.Event(enable_timing=True)
                    end_event = torch.npu.Event(enable_timing=True)
                    host_start = time.perf_counter_ns()
                    start_event.record()
                    before = native.calls
                    if kind == "old":
                        value = oracle(**arguments)
                    else:
                        value = _call_candidate(
                            query_parallel_indexer, arguments, native, rank, world, spec["prefill"]
                        )
                    end_event.record()
                    torch.npu.synchronize()
                    host_finish = time.perf_counter_ns()
                    samples[kind].append(
                        dict(
                            device_ms=start_event.elapsed_time(end_event),
                            host_wall_ms=(host_finish - host_start) / 1e6,
                            host_start_ns=host_start,
                            host_finish_ns=host_finish,
                            native_group_calls=native.calls - before,
                        )
                    )
            output["timings"].append(
                dict(
                    name=spec["name"],
                    repetitions=3,
                    samples=samples,
                    median_device_ms={
                        kind: statistics.median(item["device_ms"] for item in values)
                        for kind, values in samples.items()
                    },
                    median_host_wall_ms={
                        kind: statistics.median(item["host_wall_ms"] for item in values)
                        for kind, values in samples.items()
                    },
                    scope=(
                        "oracle full score versus runtime-stride candidate with TP16 query split; "
                        "candidate interval includes one native HCCL all_gather_into_tensor; "
                        "barrier and warmups outside intervals"
                    ),
                )
            )
        del arguments, baseline, changed
        torch.npu.empty_cache()

    output["failures"] = failures
    output["passed"] = not failures
    return output


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "stage-a-impl":
        sys.exit(stage_a_impl(sys.argv[2]))

    import torch
    import torch_npu
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 16, "R246 requires exactly 16 ranks"
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    os.environ["TRITON_CACHE_DIR"] = str(R / "triton-cache")
    (R / "triton-cache").mkdir(parents=True, exist_ok=True)

    dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=900))
    stage_a = None
    try:
        if rank == 0:
            stage_a = _run_stage_a()
        # Ranks 1..15 wait here while rank0 performs the isolated cold runs.
        dist.barrier()
        result = _stage_b(rank, world, dist.group.WORLD)
        if rank == 0:
            result["stage_a_passed"] = bool(stage_a and stage_a.get("passed"))
            result["stage_a_observed"] = (stage_a or {}).get("observed")
        _write_json(R / ("rank%d.json" % rank), result)
        dist.barrier()
        passed = result["passed"] and (rank != 0 or bool(stage_a and stage_a.get("passed")))
    except BaseException as error:
        _write_json(
            R / ("rank%d.json" % rank),
            dict(
                project_id="glm53-flash",
                unit="r246",
                rank=rank,
                manifest_sha256=MANIFEST_SHA,
                passed=False,
                error=repr(error),
                traceback=traceback.format_exc(),
            ),
        )
        passed = False
    finally:
        try:
            dist.destroy_process_group()
        except BaseException:
            pass
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
