"""Actual helper semantic oracle. AST extraction avoids model/NPU imports."""
import ast
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "post_full_framework_sources/rotary_embedding.py"
CANDIDATE = ROOT / "rotary_embedding.py"


def helper(path, cos, sin):
    tree = ast.parse(path.read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == "get_cos_and_sin_mla")
    ns = dict(torch=torch, _cos_cache=cos, _sin_cache=sin)
    ns["_cos_mla"] = torch.full((32, 1, 1, cos.shape[-1]), 13, dtype=cos.dtype)
    ns["_sin_mla"] = torch.full((32, 1, 1, sin.shape[-1]), 17, dtype=sin.dtype)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), ns)
    return ns


def cache(rows, width, dtype, layout):
    x = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width).to(dtype)
    if layout == "interleaved_split":
        # Exact view/repeat/chunk/squeeze construction in the installed helper.
        a, b = x.view(-1, 2, width // 2).repeat(1, 1, 2).chunk(2, dim=1)
        return a.squeeze(1), b.squeeze(1)
    if layout == "transpose":
        return x.t().contiguous().t(), (x + 1).t().contiguous().t()
    return x, x + 1


cases = []
torch.set_num_threads(1)
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    for layout in ("contiguous", "interleaved_split", "transpose"):
        for rows in (0, 1, 8, 31):
            cos, sin = cache(rows, 8, dtype, layout)
            samples = [[]]
            if rows:
                samples += [[0], [rows - 1, 0, rows - 1], [-rows, -1, 0],
                            [rows], [-rows - 1]]
            else:
                samples += [[0], [-1]]
            for index_dtype in (torch.int32, torch.int64):
                for values in samples:
                    p = torch.tensor(values, dtype=index_dtype)
                    for strided in (False, True):
                        if strided:
                            wide = torch.zeros(p.numel() * 2, dtype=index_dtype)
                            wide[::2] = p
                            p = wide[::2]
                        for use_cache in (False, True):
                            a, b = helper(ORIGINAL, cos, sin), helper(CANDIDATE, cos, sin)
                            results, errors = [], []
                            for ns in (a, b):
                                try:
                                    results.append(ns["get_cos_and_sin_mla"](p, use_cache))
                                    errors.append(None)
                                except (IndexError, RuntimeError) as e:
                                    results.append(None)
                                    errors.append(type(e).__name__)
                            assert bool(errors[0]) == bool(errors[1]), (values, errors)
                            if not errors[0]:
                                for left, right in zip(results[0], results[1]):
                                    assert torch.equal(left, right)
                                    assert (left.shape, left.stride(), left.dtype, left.device,
                                            left.is_contiguous()) == (
                                                right.shape, right.stride(), right.dtype,
                                                right.device, right.is_contiguous())
                                for name in ("_cos_mla", "_sin_mla"):
                                    assert torch.equal(a[name], b[name])
                                if use_cache:
                                    assert results[1][0].untyped_storage().data_ptr() == b["_cos_mla"].untyped_storage().data_ptr()
                                    assert results[1][1].untyped_storage().data_ptr() == b["_sin_mla"].untyped_storage().data_ptr()
                                elif p.numel():
                                    assert results[1][0].untyped_storage().data_ptr() != cos.untyped_storage().data_ptr()
                                    assert results[1][1].untyped_storage().data_ptr() != sin.untyped_storage().data_ptr()
                            cases.append(dict(dtype=str(dtype), layout=layout, rows=rows,
                                              index_dtype=str(index_dtype), values=values,
                                              strided=strided, use_cache=use_cache,
                                              error_classes=errors))

# Fallback retains differing row counts, multidimensional and boolean indexing.
for cos_rows, sin_rows, p in (
    (8, 10, torch.tensor([-1, -8], dtype=torch.int64)),
    (8, 8, torch.tensor([[0, 1], [-1, 1]], dtype=torch.int64)),
    (8, 8, torch.tensor([True, False, True, False, False, True, False, False])),
    (8, 8, torch.tensor(1, dtype=torch.int64)),
):
    cos = torch.arange(cos_rows * 8).reshape(cos_rows, 8).float()
    sin = torch.arange(sin_rows * 8).reshape(sin_rows, 8).float() + 10
    a, b = helper(ORIGINAL, cos, sin), helper(CANDIDATE, cos, sin)
    for x, y in zip(a["get_cos_and_sin_mla"](p), b["get_cos_and_sin_mla"](p)):
        assert torch.equal(x, y) and x.stride() == y.stride()
    cases.append(dict(fallback=True, cos_rows=cos_rows, sin_rows=sin_rows, index_shape=list(p.shape), index_dtype=str(p.dtype)))

result = dict(passed=True, cases=len(cases), torch_version=torch.__version__,
              original_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
              candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),
              NPU_initialized=False, model_imports=0, inference_requests=0,
              claims="CPU values/layout/storage and out-of-bounds rejection only; native NPU lowering/performance untested",
              coverage=cases)
destination = ROOT / "CPU_result.json"
assert not destination.exists()
destination.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "coverage"}))
