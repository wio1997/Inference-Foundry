"""Actual registration and lookup oracle; no model or NPU imports."""
import ast
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
PATHS = (ROOT.parent / "post_full_framework_sources/rotary_embedding.py",
         ROOT / "rotary_embedding.py")


def namespace(path, raw):
    definitions = [n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name in (
                       "_record_cos_and_sin_cache_interleaved", "get_cos_and_sin_mla")]
    assert len(definitions) == 2
    ns = dict(torch=torch, _cos_cache=None, _sin_cache=None,
              _cos_mla=torch.full((32, 1, 1, raw.shape[-1]), 13, dtype=raw.dtype),
              _sin_mla=torch.full((32, 1, 1, raw.shape[-1]), 17, dtype=raw.dtype))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), ns)
    ns["_record_cos_and_sin_cache_interleaved"](raw)
    return ns


torch.set_num_threads(1)
cases = []
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    for rows in (0, 1, 8, 31):
        for width in (8, 64):
            raw = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width).to(dtype)
            a, b = (namespace(p, raw) for p in PATHS)
            for name in ("_cos_cache", "_sin_cache"):
                assert torch.equal(a[name], b[name]) and b[name].is_contiguous()
                assert a[name].shape == b[name].shape
            if rows > 1:
                old_bytes = a["_cos_cache"].untyped_storage().nbytes()
                new_bytes = b["_cos_cache"].untyped_storage().nbytes() + b["_sin_cache"].untyped_storage().nbytes()
                assert old_bytes == new_bytes
                assert a["_cos_cache"].untyped_storage().data_ptr() == a["_sin_cache"].untyped_storage().data_ptr()

            samples = [[]]
            samples += ([[0], [rows - 1, 0, rows - 1], [-rows, -1, 0], [rows], [-rows - 1]]
                        if rows else [[0], [-1]])
            for index_dtype in (torch.int32, torch.int64):
                for values in samples:
                    p = torch.tensor(values, dtype=index_dtype)
                    for use_cache in (False, True):
                        results, errors = [], []
                        for ns in (a, b):
                            try:
                                results.append(ns["get_cos_and_sin_mla"](p, use_cache))
                                errors.append(None)
                            except (IndexError, RuntimeError) as e:
                                results.append(None)
                                errors.append(type(e).__name__)
                        assert errors[0] == errors[1]
                        if not errors[0]:
                            for left, right in zip(*results):
                                assert torch.equal(left, right)
                                assert (left.shape, left.stride(), left.dtype, left.device) == (
                                    right.shape, right.stride(), right.dtype, right.device)
                            for name in ("_cos_mla", "_sin_mla"):
                                assert torch.equal(a[name], b[name])
                            if use_cache:
                                for out, name in zip(results[1], ("_cos_mla", "_sin_mla")):
                                    assert out.untyped_storage().data_ptr() == b[name].untyped_storage().data_ptr()
                        cases.append(dict(dtype=str(dtype), rows=rows, width=width,
                                          index_dtype=str(index_dtype), values=values,
                                          use_cache=use_cache, errors=errors))

            # Repeated registrations leave the first table and graph output buffers intact.
            identities = {n: (b[n], b[n].data_ptr()) for n in (
                "_cos_cache", "_sin_cache", "_cos_mla", "_sin_mla")}
            b["_record_cos_and_sin_cache_interleaved"](raw + 5)
            assert all(b[n] is obj and b[n].data_ptr() == ptr for n, (obj, ptr) in identities.items())

result = dict(passed=True, cases=len(cases), torch_version=torch.__version__,
              original_sha256=hashlib.sha256(PATHS[0].read_bytes()).hexdigest(),
              candidate_sha256=hashlib.sha256(PATHS[1].read_bytes()).hexdigest(),
              NPU_initialized=False, model_imports=0, inference_requests=0,
              claims="Registration values, persistent payload, downstream indexing/layout, rejection, repeated registration and output storage only; NPU and PD untested",
              coverage=cases)
destination = ROOT / "CPU_result.json"
assert not destination.exists()
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "coverage"}))
