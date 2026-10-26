"""Actual-caller negative integration probe (CPU only; no copied validation).

Run in the authorized candidate container, not on the host. This imports the
exact --caller and --helper files, stubs only external lookup/kernel modules,
and invokes the actual complete forward_flash_kpool function. Invalid metadata
must be rejected before any persistent buffer access, float activation,
projection or kernel. No NPU work or model validation is performed by this probe.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from types import ModuleType, SimpleNamespace

import torch

HELPER_MODULE = "sglang.srt.hardware_backend.npu.ops.flash_kpool_metadata_cache_npu"
CALLER_MODULE = "r4_actual_kpool_caller_probe_target"
MISSING = object()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def temporary_env(changes):
    saved = {key: os.environ.get(key) for key in changes}
    for key, value in changes.items():
        if value is None: os.environ.pop(key, None)
        else: os.environ[key] = value
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


@contextmanager
def temporary_no_bytecode():
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = saved


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


@contextmanager
def temporary_modules(entries):
    """Restore preexisting modules and package attributes after this probe."""
    saved = {name: sys.modules.get(name, MISSING) for name in entries}
    parent_attributes = []
    for name, value in entries.items():
        sys.modules[name] = value
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is not None:
                parent_attributes.append((parent, attr, getattr(parent, attr, MISSING)))
                setattr(parent, attr, value)
    try:
        yield
    finally:
        for parent, attr, value in reversed(parent_attributes):
            if value is MISSING:
                if hasattr(parent, attr): delattr(parent, attr)
            else: setattr(parent, attr, value)
        for name, value in saved.items():
            if value is MISSING: sys.modules.pop(name, None)
            else: sys.modules[name] = value


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    if "." in name:
        parent_name, attr = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None: setattr(parent, attr, loaded)
    spec.loader.exec_module(loaded)
    return loaded


class Mode:
    def __init__(self, name): self.name = name
    def is_extend(self): return self.name == "extend"
    def is_decode(self): return self.name == "decode"
    def is_target_verify(self): return False
    def is_draft_extend_v2(self): return False


class ForbiddenActivation:
    # Source caller only reads shape/device before validation.
    shape = (2, 1)
    device = torch.device("cpu")
    def __init__(self, forbidden): self.forbidden = forbidden
    def float(self):
        self.forbidden.append("activation_float_before_projection")
        raise AssertionError("Invalid metadata reached activation/projection path")


class FakeIndexer:
    skip_rope = True
    index_kpool = 4
    index_kpool_compress = True
    head_dim = 128
    n_heads = 32
    index_topk = 2048
    def __init__(self, forbidden): self.forbidden = forbidden
    @property
    def wk(self):
        self.forbidden.append("projection_weight_access")
        raise AssertionError("Invalid metadata reached projection")


class FakePool:
    def __init__(self, forbidden):
        self.forbidden = forbidden
        self.full_kv_pool = SimpleNamespace(_flash_npu_kpool=True, req_pool_size=16)
    def get_npu_kpool_buffers(self, layer_id):
        self.forbidden.append("persistent_buffer_access")
        raise AssertionError("Invalid metadata reached persistent-buffer/tailwrite path")


def make_state(mode_name, bad_kind, forbidden):
    if bad_kind == "duplicate_slots":
        seq, slots, lengths, pos = [1, 1], [3, 3], [1, 1], [0, 0]
    elif mode_name == "extend":
        seq, slots, lengths, pos = [2], [3], [2], [0, 7]
    else:
        seq, slots, lengths, pos = [2, 3], [3, 4], [1, 1], [1, 9]
    tensor = lambda values, dtype: torch.tensor(values, dtype=dtype, device="cpu")
    batch = SimpleNamespace(
        forward_mode=Mode(mode_name), spec_info=None, can_run_tbo=False,
        tbo_split_seq_index=None, forward_metadata_ready=False,
        seq_lens=tensor(seq, torch.int32),
        req_pool_indices=tensor(slots, torch.int32),
        extend_seq_lens=tensor(lengths, torch.int32))
    metadata = SimpleNamespace(block_tables=torch.zeros((len(seq), 1), dtype=torch.int32, device="cpu"))
    backend = SimpleNamespace(forward_metadata=metadata, graph_mode=False)
    return dict(
        wrapper=SimpleNamespace(full_attn_backend=backend),
        pool=FakePool(forbidden), indexer=FakeIndexer(forbidden),
        x=ForbiddenActivation(forbidden),
        q_lora=torch.zeros((2, 1), dtype=torch.bfloat16, device="cpu"),
        positions=tensor(pos, torch.int64), batch=batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caller", required=True)
    parser.add_argument("--helper", required=True)
    parser.add_argument("--output")
    parser.add_argument("--source-commit", default="UNKNOWN")
    args = parser.parse_args()
    caller_path = Path(args.caller).resolve()
    helper_path = Path(args.helper).resolve()
    active, forbidden, read_sets, events = {}, [], [], []
    def forbidden_kernel(name):
        def invoke(*args, **kwargs):
            forbidden.append(name)
            raise AssertionError("Invalid metadata reached forbidden kernel: " + name)
        return invoke
    entries = {}
    for name in ["sglang", "sglang.srt", "sglang.srt.hardware_backend",
                 "sglang.srt.hardware_backend.npu", "sglang.srt.hardware_backend.npu.ops",
                 "sglang.srt.model_executor"]:
        entries[name] = module(name, __path__=[])
    entries[HELPER_MODULE] = module(HELPER_MODULE)
    entries[CALLER_MODULE] = module(CALLER_MODULE)
    entries["sglang.srt.model_executor.forward_context"] = module(
        "sglang.srt.model_executor.forward_context",
        get_attn_backend=lambda: active["wrapper"],
        get_token_to_kv_pool=lambda: active["pool"])
    entries["sglang.srt.hardware_backend.npu.ops.flash_kpool_compression"] = module(
        "sglang.srt.hardware_backend.npu.ops.flash_kpool_compression",
        glm5_next_kpool_tail_compress_and_write_cache_triton=forbidden_kernel("compression"))
    entries["sglang.srt.hardware_backend.npu.ops.flash_indexer_query_parallel"] = module(
        "sglang.srt.hardware_backend.npu.ops.flash_indexer_query_parallel",
        query_parallel_indexer=forbidden_kernel("query_parallel_indexer"))
    entries["sglang.srt.runtime_context"] = module(
        "sglang.srt.runtime_context", get_parallel=lambda: SimpleNamespace(attn_tp_rank=0))
    results = []
    # No live service process or global environment is modified. The isolated
    # child probe suppresses optional first-hit file output from this import.
    env = {"GLM53_KPOOL_FORWARD_METADATA_CACHE": "1", "FLASH_PD_PERF_MODE": "0",
           "GLM53_EXECUTION_IDENTITY_DIR": None, "PYTHONDONTWRITEBYTECODE": "1"}
    with temporary_no_bytecode(), temporary_env(env), temporary_modules(entries):
        helper = load_file(HELPER_MODULE, helper_path)
        original_read = helper._read
        def counted_read(*xs):
            read_sets.append(1)
            return original_read(*xs)
        helper._read = counted_read
        original_snapshot = helper.read_kpool_cpu_metadata
        def observed_snapshot(*xs, **kwargs):
            output = original_snapshot(*xs, **kwargs)
            events.append(output[1])
            return output
        helper.read_kpool_cpu_metadata = observed_snapshot
        caller = load_file(CALLER_MODULE, caller_path)
        assert caller._PERFORMANCE is False, "negative probe requires original validation path"
        assert caller._KPOOL_FORWARD_METADATA_CACHE is True, "candidate flag was not enabled at import"
        assert caller.forward_flash_kpool.__code__.co_filename == str(caller_path)
        for inference in [False, True]:
            for mode_name in ["extend", "decode"]:
                for bad_kind in ["positions", "duplicate_slots"]:
                    label = f"{mode_name}_{bad_kind}_{'inference' if inference else 'normal'}"
                    start_read, start_event, start_forbidden = len(read_sets), len(events), len(forbidden)
                    detail = dict(name=label, inference_mode=inference, calls=[])
                    try:
                        with (torch.inference_mode() if inference else nullcontext()):
                            active.clear(); active.update(make_state(mode_name, bad_kind, forbidden))
                            for expected_event, expected_reads in [("miss", 1), ("hit", 1)]:
                                try:
                                    caller.forward_flash_kpool(
                                        active["indexer"], active["x"], active["q_lora"],
                                        active["positions"], active["batch"], layer_id=3)
                                except ValueError as exc:
                                    expected_message = ("Flash KPool positions must be contiguous at the actual prefix"
                                                        if bad_kind == "positions" else "Duplicate active request tail slots")
                                    assert str(exc) == expected_message, f"unexpected ValueError: {exc}"
                                    last = exc.__traceback__
                                    while last.tb_next is not None: last = last.tb_next
                                    assert last.tb_frame.f_code.co_name == "forward_flash_kpool"
                                    assert Path(last.tb_frame.f_code.co_filename).resolve() == caller_path
                                    assert events[-1] == expected_event, (events[-1], expected_event)
                                    assert len(read_sets)-start_read == expected_reads
                                    assert len(forbidden) == start_forbidden
                                    detail["calls"].append(dict(
                                        expected_cache_event=expected_event, cache_event=events[-1],
                                        read_sets_so_far=len(read_sets)-start_read,
                                        exception=type(exc).__name__, message=str(exc),
                                        actual_caller_raise_line=last.tb_lineno,
                                        traceback=traceback.format_exc()))
                                else:
                                    raise AssertionError("Actual caller accepted invalid metadata")
                        assert events[start_event:] == ["miss", "hit"]
                        detail["status"] = "PASS"
                    except Exception as exc:
                        detail.update(status="FAIL", exception=type(exc).__name__,
                                      message=str(exc), traceback=traceback.format_exc())
                    detail.update(cpu_read_sets=len(read_sets)-start_read,
                                  cache_events=events[start_event:],
                                  forbidden_operations=forbidden[start_forbidden:])
                    results.append(detail)
    output = dict(
        schema_version=1,
        scope="CPU actual-caller rejection with temporary dependency stubs; no copied validation, NPU execution, real pool, model correctness or performance claim",
        source_commit=args.source_commit,
        sources={"caller": {"path":str(caller_path),"sha256":digest(caller_path)},
                 "helper": {"path":str(helper_path),"sha256":digest(helper_path)},
                 "probe": {"path":str(Path(__file__).resolve()),"sha256":digest(__file__)}},
        expected_cases=8, expected_rejections=16,
        passed=sum(r["status"]=="PASS" for r in results),
        failed=sum(r["status"]=="FAIL" for r in results),
        total_cpu_read_sets=len(read_sets), forbidden_operations=forbidden, results=results)
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output: Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return int(output["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())

